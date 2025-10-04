from libs import plots, params, models_core

from typing import Sequence, List, Tuple, Optional, Union
import gc 
import os
import io
import tempfile
import copy
import re
import warnings

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


""

class ModelClass(nn.Module):
    """
    Stateful CNN→BiLSTM→Attention→BiLSTM with strictly causal
    moving-average smoothing + upward‐spike gating.

    Forward:
      Input x: (B, S, F)
        • 0) Conv1d→BN→ReLU on features
        • 1) Short Bi-LSTM → MultiheadAttention → residual → LayerNorm → Dropout
        • 2) Long  Bi-LSTM → skip-proj → residual → LayerNorm → Dropout
        • 3) Time-dist heads: 
            – pred: regression logits (B, S, 1)
            – cls_ head: binary logits    (B, S, 1)
            – cls_ter: ternary logits     (B, S, 3)
        • 4) Causal smoothing: left-pad only, fixed conv filter, no future leak
        • 5) Gate: upward spikes pass raw, else smoothed

    Hidden states:
      • reset_short() on every window
      • reset_long() at day rollover

    Returns:
      out_reg: (B, S, 1), gated-smoothed regression logits
      raw_cls: (B, S, 1), binary logits
      raw_ter:(B, S, 3), ternary logits
    """
    def __init__(
        self,
        n_feats:         int,
        short_units:     int,
        long_units:      int,
        dropout_short:   float,
        dropout_long:    float,
        att_heads:       int,
        att_drop:        float,
        conv_k:          int = 3,
        conv_dilation:   int = 1,
        smooth_k:        int = 3,
        smooth_dilation: int = 1,
    ):
        super().__init__()
        self.n_feats       = n_feats
        self.short_units   = short_units
        self.long_units    = long_units
        self.pad_sm        = (smooth_k - 1) * smooth_dilation

        # 0) Input Conv1d + BN
        pad_in = (conv_k // 2) * conv_dilation
        self.conv = nn.Conv1d(n_feats, n_feats, conv_k,
                              dilation=conv_dilation, padding=pad_in)
        self.bn   = nn.BatchNorm1d(n_feats)

        # 1) Short-term Bi-LSTM
        assert short_units % 2 == 0
        self.short_lstm = nn.LSTM(
            input_size    = n_feats,
            hidden_size   = short_units // 2,
            batch_first   = True,
            bidirectional = True
        )

        # 2) Window‐level attention
        self.attn = nn.MultiheadAttention(
            embed_dim   = short_units,
            num_heads   = att_heads,
            dropout     = att_drop,
            batch_first = True
        )
        self.ln_short = nn.LayerNorm(short_units)
        self.do_short = nn.Dropout(dropout_short)

        # 3) Long‐term Bi-LSTM
        assert long_units % 2 == 0
        self.long_lstm = nn.LSTM(
            input_size    = short_units,
            hidden_size   = long_units // 2,
            batch_first   = True,
            bidirectional = True
        )
        self.short2long = nn.Linear(short_units, long_units)
        self.ln_long    = nn.LayerNorm(long_units)
        self.do_long    = nn.Dropout(dropout_long)

        # 4) Time-distributed heads
        self.pred     = nn.Linear(long_units, 1)
        self.cls_head = nn.Linear(long_units, 1)
        self.cls_ter  = nn.Linear(long_units, 3)

        # 5) Causal smoother—left pad only, fixed weights
        self.smoother = nn.Conv1d(1, 1, kernel_size=smooth_k,
                                  dilation=smooth_dilation,
                                  padding=0, bias=False)
        # init moving-avg filter
        nn.init.constant_(self.smoother.weight, 1.0 / smooth_k)
        self.smoother.weight.requires_grad = False

        # 6) Lazy hidden states
        self.h_short = self.c_short = None
        self.h_long  = self.c_long  = None

    def _init_states(self, B: int, device: torch.device):
        # 2 directions × batch × (hidden_size)
        self.h_short = torch.zeros(2, B, self.short_units // 2, device=device)
        self.c_short = torch.zeros(2, B, self.short_units // 2, device=device)
        self.h_long  = torch.zeros(2, B, self.long_units  // 2, device=device)
        self.c_long  = torch.zeros(2, B, self.long_units  // 2, device=device)

    def reset_short(self):
        if self.h_short is not None:
            B, dev = self.h_short.shape[1], self.h_short.device
            self._init_states(B, dev)

    def reset_long(self):
        if self.h_long is not None:
            B, dev = self.h_long.shape[1], self.h_long.device
            hs, cs = self.h_short, self.c_short
            self._init_states(B, dev)
            # carry over daily LSTM state
            self.h_short, self.c_short = hs.to(dev), cs.to(dev)

    def forward(self, x: torch.Tensor):
        # ensure (B, S, F)
        if x.dim() > 3:
            *lead, S, F = x.shape
            x = x.view(-1, S, F)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, S, _ = x.shape
        dev     = x.device

        # 0) Conv1d → BN → ReLU
        xc = x.transpose(1, 2)         # (B, F, S)
        xc = self.conv(xc)
        xc = self.bn(xc)
        x  = Funct.relu(xc).transpose(1, 2) # (B, S, F)

        # init hidden states if first pass or batch‐size changed
        if self.h_short is None or self.h_short.size(1) != B:
            self._init_states(B, dev)

        # 1) Short LSTM
        out_s, (h_s, c_s) = self.short_lstm(x, (self.h_short, self.c_short))
        h_s.detach_(); c_s.detach_()
        self.h_short, self.c_short = h_s, c_s

        # 2) Attention + residual + Norm + Dropout
        attn_out, _ = self.attn(out_s, out_s, out_s)
        out_s = self.ln_short(out_s + attn_out)
        out_s = self.do_short(out_s)

        # 3) Long LSTM
        out_l_raw, (h_l, c_l) = self.long_lstm(out_s, (self.h_long, self.c_long))
        h_l.detach_(); c_l.detach_()
        self.h_long, self.c_long = h_l, c_l

        # 4) Residual skip + Norm + Dropout
        skip  = self.short2long(out_s)
        out_l = self.ln_long(skip + out_l_raw)
        out_l = self.do_long(out_l)

        # 5) Heads
        raw_reg = self.pred(out_l)     # (B, S, 1)
        raw_cls = self.cls_head(out_l) # (B, S, 1)
        raw_ter = self.cls_ter(out_l)  # (B, S, 3)

        # 6) Causal smoothing + gating
        reg_t = raw_reg.permute(0, 2, 1)      # (B, 1, S)
        reg_t = Funct.pad(reg_t, (self.pad_sm, 0))# left pad only
        sm_t  = self.smoother(reg_t)          # (B, 1, S)
        sm_r  = sm_t.permute(0, 2, 1)         # (B, S, 1)
        out_reg = torch.where(raw_reg > sm_r, raw_reg, sm_r)

        return out_reg, raw_cls, raw_ter



""

def get_metrics(device: torch.device, thr: float = 0.5):
    """
    Return a dict of torchmetrics Metric objects on `device` for:
      - regression:    rmse, mae, r2
      - binary cls:    acc, prec, rec, f1, auc
      - ternary cls:   t_acc, t_prec, t_rec, t_f1, t_auc
    """
    # regression metrics
    regs = {
        "rmse": torchmetrics.MeanSquaredError(squared=False),
        "mae":  torchmetrics.MeanAbsoluteError(),
        "r2":   torchmetrics.R2Score(),
    }
    # binary classification metrics
    bins = {
        "acc":  torchmetrics.classification.BinaryAccuracy(threshold=thr),
        "prec": torchmetrics.classification.BinaryPrecision(threshold=thr),
        "rec":  torchmetrics.classification.BinaryRecall(threshold=thr),
        "f1":   torchmetrics.classification.BinaryF1Score(threshold=thr),
        "auc":  torchmetrics.classification.BinaryAUROC(),
    }
    # multiclass (ternary) classification metrics
    terns = {
        "t_acc":  torchmetrics.classification.MulticlassAccuracy(num_classes=3),
        "t_prec": torchmetrics.classification.MulticlassPrecision(num_classes=3, average="macro"),
        "t_rec":  torchmetrics.classification.MulticlassRecall(num_classes=3, average="macro"),
        "t_f1":   torchmetrics.classification.MulticlassF1Score(num_classes=3, average="macro"),
        "t_auc":  torchmetrics.classification.MulticlassAUROC(num_classes=3, average="macro"),
    }

    # Move all metrics to the target device
    all_metrics = {**regs, **bins, **terns}
    for m in all_metrics.values():
        m.to(device)
    return all_metrics



""

def update_metrics(
    metrics: dict[str, torchmetrics.Metric],
    pr_seq: torch.Tensor, t_r: torch.Tensor,
    pb_seq: torch.Tensor, t_b: torch.Tensor,
    pt_seq: torch.Tensor, t_t: torch.Tensor
):
    """
    Update regression, binary, and ternary metrics in one call.

    - pr_seq, t_r: float predictions & targets for regression
    - pb_seq, t_b: float probabilities & targets for binary cls
    - pt_seq, t_t: [N,3] probs & int targets (0/1/2) for ternary cls
    """
    # 1) regression
    for name in ("rmse", "mae", "r2"):
        metrics[name].update(pr_seq, t_r)

    # 2) binary classification
    rounded = pb_seq.round()
    for name in ("acc", "prec", "rec", "f1"):
        metrics[name].update(rounded, t_b)
    metrics["auc"].update(pb_seq, t_b)

    # 3) multiclass (ternary) classification
    preds_cls = pt_seq.argmax(dim=1)
    for name in ("t_acc", "t_prec", "t_rec", "t_f1"):
        metrics[name].update(preds_cls, t_t)
    metrics["t_auc"].update(pt_seq, t_t)



""

def eval_on_loader(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    metrics: dict,
    collect_preds: bool = False,
    disable_tqdm: bool = False
):
    """
    Evaluate `model` on all windows in `loader`, computing regression,
    binary‐ and ternary‐classification metrics.

    Optionally returns the flat array of regression predictions
    (one per window) in chronological order.

    Steps:
      1) Reset all metric states.
      2) eval() + zero LSTM hidden states.
      3) Keep a reference `full_end_times` to loader.dataset.end_times
         (an array of length N_windows).
      4) Loop batches (no shuffle, one day per batch):
         a) For each day-window group:
            • Reset short‐term LSTM each day;
            • Reset long‐term LSTM on weekday wrap 6→0;
            • Forward windows → three heads, convert logits to probs;
            • Read discrete ternary labels y_ter directly (0,1,2);
            • Update all metrics in one call;
            • If collecting preds: append pr_seq and slice
              full_end_times[global_idx:global_idx+W_day] to `times`.
         b) Increment global_idx by W_day.
      5) compute() metrics → floats.
      6) Return (metrics_dict, preds_array) if collect_preds=True,
         else (metrics_dict, None).
    """
    # 1) Reset metrics
    for m in metrics.values():
        m.reset()

    preds = [] if collect_preds else None
    # we no longer need ts_list from pad_collate — use full_end_times instead
    full_end_times = loader.dataset.end_times
    global_idx = 0

    # 2) Model to eval
    model.eval()
    model.h_short = model.h_long = None
    prev_day = None

    with torch.no_grad():
        loop = tqdm(loader, desc="eval", unit="batch", disable=disable_tqdm)
        for batch in loop:
            # Unpack exactly what DayWindowDataset returns
            xb, y_reg, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths = batch

            # Move to device
            xb    = xb.to(device, non_blocking=True)
            y_reg = y_reg.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True)
            y_ter = y_ter.to(device, non_blocking=True)
            wd    = wd.to(device, non_blocking=True)

            B = xb.size(0)
            for i in range(B):
                W_day  = lengths[i]
                day_id = int(wd[i].item())

                # reset short‐term per day, long‐term on wrap
                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                # forward
                x_seq = xb[i, :W_day]
                pr_log, pb_log, pt_log = model(x_seq)
                pr_seq = torch.sigmoid(pr_log[:, -1, 0])
                pb_seq = torch.sigmoid(pb_log[:, -1, 0])
                pt_seq = torch.softmax(pt_log[:, -1, :], dim=-1)

                # targets
                t_r = y_reg[i, :W_day].view(-1)
                t_b = y_bin[i, :W_day].view(-1)
                t_t = y_ter[i, :W_day].view(-1)  # already 0/1/2

                # update metrics
                update_metrics(
                    metrics,
                    pr_seq, t_r,
                    pb_seq, t_b,
                    pt_seq, t_t
                )

                # collect preds + matching timestamps
                if collect_preds:
                    preds.extend(pr_seq.cpu().tolist())
                    # slice out exactly W_day timestamps
                    times_slice = full_end_times[global_idx : global_idx + W_day]
                    preds.extend([])  # no change to preds
                    # extend `times` in sync with preds
                    # (we only return preds array downstream)
                    global_idx += W_day

    # 5) finalize
    results = {name: m.compute().item() for name, m in metrics.items()}

    # 6) return
    if collect_preds:
        preds_arr = np.array(preds)
        return results, preds_arr
    return results, None



""

def model_training_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cosine_sched: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    plateau_sched: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: GradScaler,
    train_loader,
    val_loader,
    *,
    max_epochs: int,
    early_stop_patience: int,
    clipnorm: float,
    device: torch.device = torch.device("cpu"),
    mode: str = "train",
    cls_loss_weight: float = 0.05,    # unused in loss
    smooth_alpha: float = 0.005,
    smooth_beta: float = 20.0,       # diagnostics only
    smooth_delta: float = 0.01,
    diff1_weight: float = 1.0,
    diff2_weight: float = 0.2
) -> float:
    """
    Train (or eval) the CNN→BiLSTM→Attention→BiLSTM model.
    
    In 'train' mode:
      - Loss = pure MSE on the regression head.
      - We still compute slip smoothing + diff penalties every window
        for logging, but we do NOT add them into `loss`.
      - Binary & ternary heads run for metrics only; their losses
        are never backpropagated.
      - All stateful‐LSTM resets, mixed‐precision, schedulers,
        gradient clipping, checkpointing, live RMSE plotting
        remain as before.
    
    In other modes:
      - Delegate to eval_on_loader().
    
    Returns:
      best_val_rmse: lowest validation RMSE observed.
    """
    model.to(device)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()  # metrics only

    if mode != "train":
        return eval_on_loader(
            val_loader, model, device,
            get_metrics(device),
            collect_preds=True
        )

    torch.backends.cudnn.benchmark = True
    train_metrics = get_metrics(device)
    val_metrics   = get_metrics(device)

    best_val_rmse, best_state = float("inf"), None
    best_tr_metrics, best_vl_metrics = {}, {}
    patience_ctr = 0
    live_plot = plots.LiveRMSEPlot()

    for epoch in range(1, max_epochs + 1):
        gc.collect()
        model.train()
        model.h_short = model.h_long = None

        for m in train_metrics.values():
            m.reset()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            xb_days, y_r_days, y_b_days, _, y_t_days, \
            rc_days, wd_days, ts_list, lengths = batch

            xb = xb_days.to(device, non_blocking=True)
            targ_r = y_r_days.to(device, non_blocking=True)
            targ_b = y_b_days.to(device, non_blocking=True)
            targ_t = y_t_days.to(device, non_blocking=True).view(-1)
            wd = wd_days.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            preds_r, targs_r = [], []
            preds_b, targs_b = [], []
            preds_t, targs_t = [], []

            prev_day, ewma = None, None
            prev_lr, prev_prev_lr = None, None

            # --- window‐level forward & diagnostics ---
            for di in range(xb.size(0)):
                W      = lengths[di]
                day_id = int(wd[di].item())
                x_seq  = xb[di, :W]
                y_r    = targ_r[di, :W].view(-1)
                y_b    = targ_b[di, :W].view(-1)
                y_t    = targ_t[di * W : di * W + W]

                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                pr_log, pc_log, pt_log = model(x_seq)
                lr_logits = pr_log[..., -1, 0]
                b_logits  = pc_log[..., -1, 0]
                t_logits  = pt_log[..., -1, :]

                lr = torch.sigmoid(lr_logits)
                pb = torch.sigmoid(b_logits)
                pt = torch.softmax(t_logits, dim=-1)

                preds_r.append(lr.detach()); targs_r.append(y_r)
                preds_b.append(pb.detach()); targs_b.append(y_b)
                preds_t.append(pt.detach()); targs_t.append(y_t)

                with autocast(device_type=device.type):
                    # Pure‐MSE for backprop
                    loss = mse_loss(lr, y_r)

                    # Build diagnostics terms
                    ewma_new = lr.detach() if ewma is None \
                               else smooth_alpha * lr.detach() + (1 - smooth_alpha) * ewma
                    slip = torch.relu(ewma_new - lr)
                    if prev_lr is not None:
                        slip = torch.where(lr > prev_lr,
                                           torch.zeros_like(slip),
                                           slip)
                    hub = torch.where(
                        slip <= smooth_delta,
                        0.5 * slip**2,
                        smooth_delta * (slip - 0.5 * smooth_delta),
                    )

                    # explicit if/else to avoid `tensor or tensor`
                    if prev_lr is not None:
                        neg_diff = torch.relu(prev_lr - lr)
                    else:
                        neg_diff = torch.zeros_like(lr)

                    if prev_prev_lr is not None:
                        curv = lr - 2 * prev_lr + prev_prev_lr
                    else:
                        curv = torch.zeros_like(lr)

                prev_prev_lr, prev_lr, ewma = prev_lr, lr.detach(), ewma_new
            # --- end window loop ---

            # diagnostics on first batch only
            if batch_idx == 0:
                models_core.log_loss_components(
                    epoch, batch_idx, lr, y_r, b_logits, hub, prev_lr, prev_prev_lr, 
                    mse_loss, bce_loss, cls_loss_weight, smooth_beta,
                    diff1_weight, diff2_weight, params.log_file
                )

            # backprop pure‐MSE
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if batch_idx == 0:
                models_core.log_gradient_norms(
                    epoch, batch_idx, model, params.log_file
                )
                
            nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()

            # batch‐level metrics (all heads)
            with torch.no_grad():
                update_metrics(
                    train_metrics,
                    torch.cat(preds_r), torch.cat(targs_r),
                    torch.cat(preds_b), torch.cat(targs_b),
                    torch.cat(preds_t), torch.cat(targs_t),
                )

            # scheduler & progress bar
            frac = epoch - 1 + (batch_idx + 1) / len(train_loader)
            cosine_sched.step(frac)
            pbar.set_postfix(lr=optimizer.param_groups[0]["lr"], refresh=False)

        # Compute epoch metrics
        raw_tr = {n: m.compute() for n, m in train_metrics.items()}
        tr     = {n: (v.item() if isinstance(v, torch.Tensor) else v)
                  for n, v in raw_tr.items()}

        for m in val_metrics.values():
            m.reset()
        raw_vl, _ = eval_on_loader(val_loader, model, device, val_metrics)
        vl = {n: (v.item() if isinstance(v, torch.Tensor) else v)
              for n, v in raw_vl.items()}

        # Checkpoint & early stopping
        models_dir = Path(params.models_folder)
        live_plot.update(tr["rmse"], vl["rmse"])
        best_val_rmse, improved, tr_best, vl_best, tmp_state = \
            models_core.maybe_save_chkpt(
                models_dir, model, vl["rmse"],
                best_val_rmse, tr, vl, live_plot, params
            )
        if tmp_state is not None:
            best_state      = tmp_state
            best_tr_metrics = tr_best.copy()
            best_vl_metrics = vl_best.copy()

        patience_ctr = 0 if improved else patience_ctr + 1
        if patience_ctr >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Print epoch summary
        print(f"Epoch {epoch:03d}")
        print(
            f'TRAIN→ RMSE={tr["rmse"]:.5f} MAE={tr["mae"]:.5f} R2={tr["r2"]:.5f} | '
            f'Acc={tr["acc"]:.5f} Prec={tr["prec"]:.5f} Rec={tr["rec"]:.5f} '
            f'F1={tr["f1"]:.5f} AUROC={tr["auc"]:.5f} | '
            f'T_ACC={tr["t_acc"]:.5f} T_P={tr["t_prec"]:.5f} T_R={tr["t_rec"]:.5f} '
            f'T_F1={tr["t_f1"]:.5f} T_AUC={tr["t_auc"]:.5f}'
        )
        print(
            f'VALID→ RMSE={vl["rmse"]:.5f} MAE={vl["mae"]:.5f} R2={vl["r2"]:.5f} | '
            f'Acc={vl["acc"]:.5f} Prec={vl["prec"]:.5f} Rec={vl["rec"]:.5f} '
            f'F1={vl["f1"]:.5f} AUROC={vl["auc"]:.5f} | '
            f'T_ACC={vl["t_acc"]:.5f} T_P={vl["t_prec"]:.5f} T_R={vl["t_rec"]:.5f} '
            f'T_F1={vl["t_f1"]:.5f} T_AUC={vl["t_auc"]:.5f}'
        )

        if epoch > params.hparams["LR_EPOCHS_WARMUP"]:
            plateau_sched.step(vl["rmse"])

    # Final-best checkpoint
    if best_state is not None:
        models_core.save_final_chkpt(
            Path(params.models_folder),
            best_state,
            best_val_rmse,
            params,
            best_tr_metrics,
            best_vl_metrics,
            live_plot,
            suffix="_fin"
        )

    return best_val_rmse

