from libs import plots, params, models_core

from typing import Sequence, List, Tuple, Optional, Union
import gc 
import os
import io
import tempfile
import copy
import re

import pandas as pd
import numpy  as np
import math

import datetime as dt
from datetime import datetime, time
from pathlib import Path

import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as Funct

from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32   = True
torch.backends.cudnn.allow_tf32         = True
torch.backends.cudnn.benchmark          = True


######################################################################################################

class ModelClass(nn.Module):
    """
    Dual-Memory LSTM for smoothed continuous‐signal regression + binary‐flag prediction.

    Architecture steps:
      0) Conv1d + BatchNorm1d → ReLU
         • Encode local temporal patterns within each look-back window.
      1) Stateful Bidirectional “short” LSTM over each window
      2) Window-level self-attention + residual over short-LSTM outputs
      3) LayerNorm → Dropout on daily embedding (pre-norm style)
      4) Stateful Bidirectional “long” LSTM across windows
      5) Residual skip (projected short → long) + LayerNorm → Dropout
      6) Time-distributed heads in parallel:
         – regression head (1 real value per step)
         – binary head     (1 logit per step)
         – ternary head    (3 logits per step; reserved for future use)
      7) Automatic reset of hidden states at day/week boundaries
    """

    def __init__(
        self,
        n_feats:      int,
        short_units:  int,
        long_units:   int,
        dropout_short: float,
        dropout_long:  float,
        att_heads:     int,
        att_drop:      float,
        conv_k:        int = 3,
        conv_dilation: int = 1
    ):
        super().__init__()
        self.n_feats     = n_feats
        self.short_units = short_units
        self.long_units  = long_units

        # 0) Conv1d encoder + batch‐norm
        pad = (conv_k // 2) * conv_dilation
        self.conv = nn.Conv1d(n_feats, 
                              n_feats,
                              kernel_size=conv_k,
                              dilation=conv_dilation,
                              padding=pad)
        self.bn   = nn.BatchNorm1d(n_feats)

        # 1) Short‐term daily Bi‐LSTM
        assert short_units % 2 == 0
        self.short_lstm = nn.LSTM(
            input_size   = n_feats,
            hidden_size  = short_units // 2,
            batch_first  = True,
            bidirectional= True
        )

        # 2) Window‐level self‐attention
        self.attn = nn.MultiheadAttention(
            embed_dim   = short_units,
            num_heads   = att_heads,
            dropout     = att_drop,
            batch_first = True
        )

        # 3) Pre‐norm LayerNorm → Dropout on short embedding
        self.ln_short = nn.LayerNorm(short_units)
        self.do_short = nn.Dropout(dropout_short)

        # 4) Long‐term weekly Bi‐LSTM
        assert long_units % 2 == 0
        self.long_lstm = nn.LSTM(
            input_size   = short_units,
            hidden_size  = long_units // 2,
            batch_first  = True,
            bidirectional= True
        )

        # 5a) Project short → long for residual connection
        self.short2long = nn.Linear(short_units, long_units)

        # 5b) LayerNorm → Dropout on long embedding
        self.ln_long = nn.LayerNorm(long_units)
        self.do_long = nn.Dropout(dropout_long)

        # 6) Time‐distributed heads
        self.pred     = nn.Linear(long_units, 1)  # regression
        self.cls_head = nn.Linear(long_units, 1)  # binary
        self.cls_ter  = nn.Linear(long_units, 3)  # ternary (future)

        # 7) Lazy‐init hidden/cell states
        self.h_short = None
        self.c_short = None
        self.h_long  = None
        self.c_long  = None

    def _init_states(self, B: int, device: torch.device):
        # 2 directions × 1 layer
        self.h_short = torch.zeros(2, B, self.short_units // 2, device=device)
        self.c_short = torch.zeros(2, B, self.short_units // 2, device=device)
        self.h_long  = torch.zeros(2, B, self.long_units  // 2, device=device)
        self.c_long  = torch.zeros(2, B, self.long_units  // 2, device=device)

    def reset_short(self):
        if self.h_short is not None:
            B, dev = self.h_short.size(1), self.h_short.device
            self._init_states(B, dev)

    def reset_long(self):
        if self.h_long is not None:
            B, dev = self.h_long.size(1), self.h_long.device
            hs, cs = self.h_short, self.c_short
            self._init_states(B, dev)
            self.h_short, self.c_short = hs.to(dev), cs.to(dev)

    def forward(self, x: torch.Tensor):
        # reshape  extra dims → (W, S, F)
        if x.dim() > 3:
            *lead, S, F = x.shape
            x = x.view(-1, S, F)

        # ensure (W, S, n_feats)
        if x.dim() == 3 and x.size(-1) != self.n_feats:
            x = x.transpose(1, 2).contiguous()

        B, S, _ = x.size()
        dev      = x.device

        # 0) Conv1d → BatchNorm1d → ReLU
        x_conv = x.transpose(1, 2)      # (W, F, S)
        x_conv = self.conv(x_conv)
        x_conv = self.bn(x_conv)
        x_conv = Funct.relu(x_conv)
        x      = x_conv.transpose(1, 2) # back to (W, S, F)

        # lazy init/reset hidden states
        if self.h_short is None or self.h_short.size(1) != B:
            self._init_states(B, dev)

        # 1) Short‐term Bi‐LSTM
        out_short_raw, (h_s, c_s) = self.short_lstm(
            x, (self.h_short, self.c_short)
        )
        h_s.detach_(); c_s.detach_()
        self.h_short, self.c_short = h_s, c_s

        # 2) Self‐attention + residual
        attn_out, _ = self.attn(out_short_raw, out_short_raw, out_short_raw)
        out_short   = out_short_raw + attn_out

        # 3) Pre‐norm LayerNorm → Dropout
        out_short = self.ln_short(out_short)
        out_short = self.do_short(out_short)

        # 4) Long‐term Bi‐LSTM
        out_long_raw, (h_l, c_l) = self.long_lstm(
            out_short, (self.h_long, self.c_long)
        )
        h_l.detach_(); c_l.detach_()
        self.h_long, self.c_long = h_l, c_l

        # 5) Residual skip via projection + LayerNorm → Dropout
        skip     = self.short2long(out_short)
        out_long = skip + out_long_raw
        out_long = self.ln_long(out_long)
        out_long = self.do_long(out_long)

        # 6) Time‐distributed heads
        raw_reg = self.pred(out_long)     # (W, S, 1)
        raw_cls = self.cls_head(out_long) # (W, S, 1)
        raw_ter = self.cls_ter(out_long)  # (W, S, 3)

        return raw_reg, raw_cls, raw_ter



######################################################################################################


def get_metrics(device: torch.device, thr: float = 0.5):
    """Return a dict of metrics (regression, binary, ternary) placed on device."""
    return {
        # regression
        "rmse": torchmetrics.MeanSquaredError(squared=False).to(device),
        "mae":  torchmetrics.MeanAbsoluteError().to(device),
        "r2":   torchmetrics.R2Score().to(device),
        # binary
        "acc":  torchmetrics.classification.BinaryAccuracy(threshold=thr).to(device),
        "prec": torchmetrics.classification.BinaryPrecision(threshold=thr).to(device),
        "rec":  torchmetrics.classification.BinaryRecall(threshold=thr).to(device),
        "f1":   torchmetrics.classification.BinaryF1Score(threshold=thr).to(device),
        "auc":  torchmetrics.classification.BinaryAUROC().to(device),
        # ternary 
        "t_acc":  torchmetrics.classification.MulticlassAccuracy(num_classes=3).to(device),
        "t_prec": torchmetrics.classification.MulticlassPrecision(num_classes=3, average="macro").to(device),
        "t_rec":  torchmetrics.classification.MulticlassRecall(num_classes=3, average="macro").to(device),
        "t_f1":   torchmetrics.classification.MulticlassF1Score(num_classes=3, average="macro").to(device),
        "t_auc":  torchmetrics.classification.MulticlassAUROC(num_classes=3, average="macro").to(device),
    }


###################### 


def eval_on_loader(loader, model: torch.nn.Module, device: torch.device, metrics: dict, collect_preds: bool = False):
    """
    Shared evaluation helper for validation and standalone inference.

    Functionality:
      1) Reset all metric states.
      2) Move model to eval mode and clear its hidden states.
      3) Disable gradient computation for speed.
      4) For each batch in loader (progress bar):
         a) Unpack a 9-tuple batch:
            xb, y_reg, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths
         b) Move inputs to `device`.
         c) For each sequence in the batch:
            • Reset or carry LSTM states on day rollover.
            • Slice out the true window length.
            • Forward pass → raw_reg, raw_bin, raw_ter logits.
            • Sigmoid-activate regression & binary heads, softmax for ternary.
            • Update all regression, binary, and ternary metrics.
            • Optionally collect `pr` into a list.
      5) After looping, compute scalar metric results.
      6) Return `(metrics_dict, preds_array_or_None)`.
    """
    # 1) Reset metrics
    for m in metrics.values():
        m.reset()

    # Container for optional regression predictions
    preds = []

    # 2) Set model to evaluation and clear hidden states
    model.eval()
    model.h_short = model.h_long = None
    prev_day = None

    # 3) No gradients needed
    with torch.no_grad():
        # 4) Loop over all batches
        for batch in tqdm(loader, desc="eval", unit="batch"):
            # Unpack the expected 9-tuple batch structure
            xb, y_reg, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths = batch

            # Move tensors to device
            xb    = xb.to(device, non_blocking=True)
            y_reg = y_reg.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True)
            y_ter = y_ter.to(device, non_blocking=True)
            wd    = wd.to(device, non_blocking=True)

            B = xb.size(0)
            # 4.c) Process each sequence within the batch
            for i in range(B):
                W_true = lengths[i]
                day_id = int(wd[i].item())

                # Reset short‐term hidden state; reset long‐term on day rollover
                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                # Slice out the unpadded window
                x_day  = xb[i, :W_true]
                targ_r = y_reg[i, :W_true].view(-1)
                targ_b = y_bin[i, :W_true].view(-1)
                targ_t = y_ter[i, :W_true].view(-1)

                # Forward pass to get logits
                pr_logits, pb_logits, pt_logits = model(x_day)

                # Apply activations
                pr = torch.sigmoid(pr_logits[..., -1, 0])         # (W_true,)
                pb = torch.sigmoid(pb_logits[..., -1, 0])         # (W_true,)
                pt = torch.softmax(pt_logits[..., -1, :], dim=-1) # (W_true, 3)

                # Update regression metrics
                metrics["rmse"].update(pr, targ_r)
                metrics["mae"].update(pr, targ_r)
                metrics["r2"].update(pr, targ_r)
                # Update binary metrics
                metrics["acc"].update(pb,  targ_b)
                metrics["prec"].update(pb, targ_b)
                metrics["rec"].update(pb,  targ_b)
                metrics["f1"].update(pb,   targ_b)
                metrics["auc"].update(pb,  targ_b)
                # Update ternary metrics
                metrics["t_acc"].update(pt, targ_t)
                metrics["t_prec"].update(pt, targ_t)
                metrics["t_rec"].update(pt, targ_t)
                metrics["t_f1"].update(pt, targ_t)
                metrics["t_auc"].update(pt, targ_t)

                # Collect regression predictions if requested
                if collect_preds:
                    preds.append(pr.cpu().numpy())

    # 5) Compute final scalar metrics
    out = {k: m.compute().item() for k, m in metrics.items()}
    # 6) Return metrics dict and concatenated preds or None
    return out, (np.concatenate(preds, axis=0) if collect_preds and preds else None)


###################### 


# def model_training_loop(
#     model:         torch.nn.Module,
#     optimizer:     torch.optim.Optimizer,
#     cosine_sched:  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
#     plateau_sched: torch.optim.lr_scheduler.ReduceLROnPlateau,
#     scaler:        GradScaler,
#     train_loader,
#     val_loader,
#     *,
#     max_epochs:          int,
#     early_stop_patience: int,
#     clipnorm:            float,
#     device:              torch.device = torch.device("cpu"),
#     mode:                str = "train",   # "train" or "eval"
# ):
#     """
#     Train or evaluate the stateful CNN→BiLSTM→Attention→BiLSTM model.

#     This single entry point covers both training and inference:

#       • mode="train":
#         - Moves model to device and enables cuDNN benchmark.
#         - Defines losses (MSE, BCE) and smoothing hyperparameters from params.
#         - Runs a full mixed-precision, stateful training loop:
#             · Iterates epochs up to max_epochs.
#             · Per-batch: resets gradients, loops each window in a stateful fashion,
#               applies the one-sided Huber smoothing penalty, backpropagates, and updates metrics.
#             · After each batch: unscales gradients, clips norms, steps optimizer & LR (cosine warm restarts).
#             · At epoch end: computes train metrics, runs validation via shared helper, updates live plot,
#               applies plateau LR scheduler, and performs early stopping & folder-best checkpointing.
#         - At the end: saves a final checkpoint (_fin) with the best model state, metrics, and plot.

#       • mode="eval":
#         - Quickly runs a single evaluation pass over val_loader using the identical
#           validation logic: resets metric objects, walks the loader statefully,
#           updates and computes all regression, binary, and ternary metrics,
#           and returns (metrics_dict, numpy_predictions).

#     Returns:
#       - In train mode: the best validation RMSE (float).
#       - In eval mode: a tuple (metrics_dict, preds_array).
#     """
#     # Move model to target device
#     model.to(device)

#     # losses and smoothing hyperparameters (unchanged)
#     mse_loss  = nn.MSELoss()
#     bce_loss  = nn.BCEWithLogitsLoss()
#     alpha_cls = params.hparams["CLS_LOSS_WEIGHT"]
#     α = params.hparams["SMOOTH_ALPHA"]
#     β = params.hparams["SMOOTH_BETA"]
#     δ = params.hparams["SMOOTH_DELTA"]

#     # Determine execution mode
#     is_train = mode == "train"

#     # Quick evaluation path when mode!="train"
#     if not is_train:
#         metrics = get_metrics(device)
#         metrics_out, preds = eval_on_loader(val_loader, model, device, metrics, collect_preds=True)
#         return metrics_out, preds

#     # ---------------- TRAIN MODE ----------------
#     # Enable cuDNN auto-tuning for faster convs
#     torch.backends.cudnn.benchmark = True

#     # Instantiate separate metric sets for training and validation
#     train_metrics = get_metrics(device)
#     val_metrics   = get_metrics(device)

#     # Initialize checkpoint & early-stop trackers
#     best_val_rmse = float("inf")
#     best_state    = None
#     best_tr       = best_vl = {}
#     patience_ctr  = 0

#     # Live plot for RMSE progression
#     live_plot = plots.LiveRMSEPlot()

#     # Main epoch loop
#     for epoch in range(1, max_epochs + 1):
#         gc.collect()                   # free Python garbage
#         model.train()                  # set training mode
#         model.h_short = model.h_long = None  # reset hidden states

#         # Reset training metrics at epoch start
#         for m in (
#             train_metrics["rmse"], train_metrics["mae"], train_metrics["r2"],
#             train_metrics["acc"], train_metrics["prec"], train_metrics["rec"],
#             train_metrics["f1"],  train_metrics["auc"]
#         ):
#             m.reset()

#         # Batch loop with tqdm progress bar
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
#         for batch_idx, batch in enumerate(pbar):
#             (
#                 xb_days, y_sig_days, y_sig_cls_days,
#                 ret_days, y_ret_ter_days,
#                 wd_days, ts_list, lengths
#             ) = batch

#             # Move inputs to device
#             xb    = xb_days.to(device, non_blocking=True)
#             y_sig = y_sig_days.to(device, non_blocking=True)
#             y_cls = y_sig_cls_days.to(device, non_blocking=True)
#             wd    = wd_days.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)
#             prev_day = None
#             ewma     = None   # baseline for smoothing penalty

#             # Per-sample (window) loop within the batch
#             for di in range(xb.size(0)):
#                 W      = lengths[di]
#                 day_id = int(wd[di].item())

#                 x_seq   = xb[di, :W]
#                 sig_seq = y_sig[di, :W]
#                 cls_seq = y_cls[di, :W].view(-1)

#                 # Handle short‐term and long‐term hidden-state resets
#                 model.reset_short()
#                 if prev_day is not None and day_id < prev_day:
#                     model.reset_long()
#                 prev_day = day_id

#                 # Forward pass through the three-headed model
#                 pr, pc, _ = model(x_seq)

#                 with autocast(device_type=device.type):
#                     # Extract final‐step logits for regression and classification
#                     lr_logits  = pr[..., -1, 0]   # shape = (W,)
#                     cls_logits = pc[..., -1, 0]   # shape = (W,)
#                     # Sigmoid‐activate regression output to [0,1]
#                     lr = torch.sigmoid(lr_logits)

#                     targ_r = sig_seq
#                     targ_c = cls_seq

#                     # Base multi‐task loss (regression + weighted classification)
#                     loss_reg = mse_loss(lr, targ_r)
#                     loss_cls = bce_loss(cls_logits, targ_c)
#                     loss = loss_reg + alpha_cls * loss_cls

#                     # Exponential‐moving‐average smoothing penalty
#                     if ewma is None:
#                         ewma = lr.detach()
#                     else:
#                         ewma = α * lr.detach() + (1 - α) * ewma

#                     slip = torch.relu(ewma - lr)
#                     hub  = torch.where(
#                         slip <= δ,
#                         0.5 * slip**2,
#                         δ * (slip - 0.5 * δ)
#                     )
#                     loss += β * hub.mean()

#                 # Backpropagate with mixed precision
#                 scaler.scale(loss).backward()

#                 # Update training metrics (vector form)
#                 train_metrics["rmse"].update(lr, targ_r)
#                 train_metrics["mae"].update(lr, targ_r)
#                 train_metrics["r2"].update(lr, targ_r)

#                 probs = torch.sigmoid(cls_logits)
#                 train_metrics["acc"].update(probs, targ_c)
#                 train_metrics["prec"].update(probs, targ_c)
#                 train_metrics["rec"].update(probs, targ_c)
#                 train_metrics["f1"].update(probs, targ_c)
#                 train_metrics["auc"].update(probs, targ_c)

#                 # Detach hidden states to prevent graph retention
#                 for h in (model.h_short, model.c_short, model.h_long, model.c_long):
#                     h.detach_()

#             # Gradient unscale, clipping, optimizer and scheduler step
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
#             scaler.step(optimizer)
#             scaler.update()

#             frac = epoch - 1 + batch_idx / len(train_loader)
#             cosine_sched.step(frac)

#             # Update progress bar with current RMSE and LR
#             pbar.set_postfix(
#                 train_rmse=train_metrics["rmse"].compute().item(),
#                 lr=optimizer.param_groups[0]["lr"],
#                 refresh=False
#             )

#         # Gather final train metrics for this epoch
#         tr = {
#             "rmse":  train_metrics["rmse"].compute().item(),
#             "mae":   train_metrics["mae"].compute().item(),
#             "r2":    train_metrics["r2"].compute().item(),
#             "acc":   train_metrics["acc"].compute().item(),
#             "prec":  train_metrics["prec"].compute().item(),
#             "rec":   train_metrics["rec"].compute().item(),
#             "f1":    train_metrics["f1"].compute().item(),
#             "auroc": train_metrics["auc"].compute().item(),
#         }

#         # Validation using the shared eval helper (no preds collected here)
#         vl, _ = eval_on_loader(val_loader, model, device, val_metrics, collect_preds=False)

#         # e) Maybe save folder-best checkpoint and update best state/metrics
#         models_dir = Path(params.models_folder)
#         best_val_rmse, maybe_state, maybe_tr, maybe_vl, best_existing = \
#             models_core.maybe_save_chkpt(
#                 models_dir, model, vl["rmse"],
#                 best_val_rmse, tr, vl, live_plot, params
#             )

#         if maybe_state is not None:
#             best_state, best_tr, best_vl = maybe_state, maybe_tr, maybe_vl
#             patience_ctr = 0
#         else:
#             patience_ctr += 1
#             if patience_ctr >= early_stop_patience:
#                 print("Early stopping at epoch", epoch)
#                 break

#         # f) Logging, live-plot update, and plateau LR scheduler
#         live_plot.update(tr["rmse"], vl["rmse"])
#         print(f"Epoch {epoch:03d}")
#         print(
#             f'TRAIN→ RMSE={tr["rmse"]:.4f} MAE={tr["mae"]:.4f} R2={tr["r2"]:.4f} | '
#             f'Acc={tr["acc"]:.4f} Prec={tr["prec"]:.4f} Rec={tr["rec"]:.4f} '
#             f'F1={tr["f1"]:.4f} AUROC={tr["auroc"]:.4f}'
#         )
#         print(
#             f'VALID→ RMSE={vl["rmse"]:.4f} MAE={vl["mae"]:.4f} R2={vl["r2"]:.4f} | '
#             f'Acc={vl["acc"]:.4f} Prec={vl["prec"]:.4f} Rec={vl["rec"]:.4f} '
#             f'F1={vl["f1"]:.4f} AUROC={vl["auc"]:.4f}'
#         )

#         if epoch > params.hparams["LR_EPOCHS_WARMUP"]:
#             plateau_sched.step(vl["rmse"])

#     # Final checkpoint using the best state captured during training
#     if best_state is not None:
#         models_dir = Path(params.models_folder)
#         models_core.save_final_chkpt(
#             models_dir, best_state, best_val_rmse,
#             params, best_tr, best_vl, live_plot, suffix="_fin"
#         )

#     return best_val_rmse

def model_training_loop(
    model:         torch.nn.Module,
    optimizer:     torch.optim.Optimizer,
    cosine_sched:  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    plateau_sched: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler:        GradScaler,
    train_loader,
    val_loader,
    *,
    max_epochs:          int,
    early_stop_patience: int,
    clipnorm:            float,
    device:              torch.device = torch.device("cpu"),
    mode:                str = "train",   # "train" or "eval"
):
    """
    Train or evaluate the stateful CNN→BiLSTM→Attention→BiLSTM model.

    Modes:
      • train:
        - Mixed-precision, per-window stateful loops with smoothing penalty.
        - CosineAnnealingWarmRestarts + ReduceLROnPlateau schedulers.
        - Early stopping, live RMSE plotting, folder‐best and final checkpointing.
        - Expects each train batch as a 9-tuple:
          (xb_days, y_sig_days, y_sig_cls_days,
           ret_days, y_ret_ter_days,
           rc_days,  # raw_close slice, unused in loss/metrics
           wd_days, ts_list, lengths)
      • eval:
        - Single-pass evaluation over val_loader via shared `eval_on_loader`.
        - Returns (metrics_dict, preds_array).

    Returns:
      - train mode: best validation RMSE (float)
      - eval mode: (metrics_dict, preds_array)
    """
    model.to(device)

    # Loss functions & smoothing hyperparameters
    mse_loss  = nn.MSELoss()
    bce_loss  = nn.BCEWithLogitsLoss()
    alpha_cls = params.hparams["CLS_LOSS_WEIGHT"]
    α = params.hparams["SMOOTH_ALPHA"]
    β = params.hparams["SMOOTH_BETA"]
    δ = params.hparams["SMOOTH_DELTA"]

    is_train = (mode == "train")
    if not is_train:
        metrics_out, preds = eval_on_loader(
            val_loader, model, device, get_metrics(device), collect_preds=True
        )
        return metrics_out, preds

    # ---------------- TRAIN MODE ----------------
    torch.backends.cudnn.benchmark = True
    train_metrics = get_metrics(device)
    val_metrics   = get_metrics(device)

    best_val_rmse = float("inf")
    best_state    = None
    best_tr = best_vl = {}
    patience_ctr  = 0
    live_plot     = plots.LiveRMSEPlot()

    for epoch in range(1, max_epochs + 1):
        gc.collect()
        model.train()
        model.h_short = model.h_long = None

        # Reset metrics at epoch start
        for m in (
            train_metrics["rmse"], train_metrics["mae"], train_metrics["r2"],
            train_metrics["acc"], train_metrics["prec"], train_metrics["rec"],
            train_metrics["f1"],  train_metrics["auc"]
        ):
            m.reset()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            (
                xb_days, y_sig_days, y_sig_cls_days,
                ret_days, y_ret_ter_days,
                rc_days,                # raw_close slice, not used here
                wd_days, ts_list, lengths
            ) = batch

            xb    = xb_days.to(device, non_blocking=True)
            y_sig = y_sig_days.to(device, non_blocking=True)
            y_cls = y_sig_cls_days.to(device, non_blocking=True)
            wd    = wd_days.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prev_day = None
            ewma     = None

            # Per-sample stateful loop
            for di in range(xb.size(0)):
                W      = lengths[di]
                day_id = int(wd[di].item())

                x_seq   = xb[di, :W]
                sig_seq = y_sig[di, :W]
                cls_seq = y_cls[di, :W].view(-1)

                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                pr, pc, _ = model(x_seq)

                with autocast(device_type=device.type):
                    lr_logits  = pr[..., -1, 0]
                    cls_logits = pc[..., -1, 0]
                    lr = torch.sigmoid(lr_logits)

                    targ_r = sig_seq
                    targ_c = cls_seq

                    loss_reg = mse_loss(lr, targ_r)
                    loss_cls = bce_loss(cls_logits, targ_c)
                    loss = loss_reg + alpha_cls * loss_cls

                    if ewma is None:
                        ewma = lr.detach()
                    else:
                        ewma = α * lr.detach() + (1 - α) * ewma

                    slip = torch.relu(ewma - lr)
                    hub  = torch.where(
                        slip <= δ,
                        0.5 * slip**2,
                        δ * (slip - 0.5 * δ)
                    )
                    loss += β * hub.mean()

                scaler.scale(loss).backward()

                # Update regression metrics
                train_metrics["rmse"].update(lr, targ_r)
                train_metrics["mae"].update(lr, targ_r)
                train_metrics["r2"].update(lr, targ_r)

                # Update binary classification metrics
                probs = torch.sigmoid(cls_logits)
                train_metrics["acc"].update(probs, targ_c)
                train_metrics["prec"].update(probs, targ_c)
                train_metrics["rec"].update(probs, targ_c)
                train_metrics["f1"].update(probs, targ_c)
                train_metrics["auc"].update(probs, targ_c)

                # Detach hidden states after each sample
                for h in (model.h_short, model.c_short, model.h_long, model.c_long):
                    h.detach_()

            # Optimizer & schedulers step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()
            frac = epoch - 1 + batch_idx / len(train_loader)
            cosine_sched.step(frac)

            pbar.set_postfix(
                train_rmse=train_metrics["rmse"].compute().item(),
                lr=optimizer.param_groups[0]["lr"],
                refresh=False
            )

        # ---- Collect only the metrics actually updated in training
        tr = {
            "rmse":  train_metrics["rmse"].compute().item(),
            "mae":   train_metrics["mae"].compute().item(),
            "r2":    train_metrics["r2"].compute().item(),
            "acc":   train_metrics["acc"].compute().item(),
            "prec":  train_metrics["prec"].compute().item(),
            "rec":   train_metrics["rec"].compute().item(),
            "f1":    train_metrics["f1"].compute().item(),
            "auroc": train_metrics["auc"].compute().item(),
        }

        # Validation pass
        vl, _ = eval_on_loader(val_loader, model, device, val_metrics, collect_preds=False)

        # Folder-best checkpoint & early stopping
        models_dir = Path(params.models_folder)
        best_val_rmse, maybe_state, maybe_tr, maybe_vl, _ = \
            models_core.maybe_save_chkpt(
                models_dir, model, vl["rmse"],
                best_val_rmse, tr, vl, live_plot, params
            )

        if maybe_state is not None:
            best_state, best_tr, best_vl = maybe_state, maybe_tr, maybe_vl
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print("Early stopping at epoch", epoch)
                break

        # Logging, plotting, and plateau scheduler
        live_plot.update(tr["rmse"], vl["rmse"])
        print(f"Epoch {epoch:03d}")
        print(
            f'TRAIN→ RMSE={tr["rmse"]:.4f} MAE={tr["mae"]:.4f} R2={tr["r2"]:.4f} | '
            f'Acc={tr["acc"]:.4f} Prec={tr["prec"]:.4f} Rec={tr["rec"]:.4f} '
            f'F1={tr["f1"]:.4f} AUROC={tr["auroc"]:.4f}'
        )
        print(
            f'VALID→ RMSE={vl["rmse"]:.4f} MAE={vl["mae"]:.4f} R2={vl["r2"]:.4f} | '
            f'Acc={vl["acc"]:.4f} Prec={vl["prec"]:.4f} Rec={vl["rec"]:.4f} '
            f'F1={vl["f1"]:.4f} AUROC={vl["auc"]:.4f}'
        )

        if epoch > params.hparams["LR_EPOCHS_WARMUP"]:
            plateau_sched.step(vl["rmse"])

    # Final checkpoint save if best_state exists
    if best_state is not None:
        models_core.save_final_chkpt(
            Path(params.models_folder),
            best_state, best_val_rmse, params,
            best_tr, best_vl, live_plot, suffix="_fin"
        )

    return best_val_rmse
