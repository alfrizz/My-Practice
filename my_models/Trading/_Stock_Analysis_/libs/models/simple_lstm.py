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


######################################################################################################


# class ModelClass(nn.Module):
#     """
#     Stateful CNN → BiLSTM → BiLSTM simplified model for regression.

#     Changes relative to the original design:
#       - Removed window-level MultiheadAttention, binary and ternary classification heads, causal smoothing convolution and gating.
#       - Keeps: input conv+BN+ReLU, short (bidirectional) LSTM, long (bidirectional) LSTM,
#                skip projection short->long, layer-norms, dropouts, and stateful hidden-state handling.

#     Forward:
#       Input x: (B, S, F) or (S, F) or with extra leading dims (will be flattened to (B, S, F))
#         0) Conv1d → BatchNorm1d → ReLU on features
#         1) Short Bi-LSTM (stateful, reset via reset_short)
#         2) Long  Bi-LSTM (stateful, reset via reset_long)
#         3) Skip projection from short->long, residual addition, LayerNorm, Dropout
#         4) Regression head: time-distributed linear -> raw regression logits (B, S, 1)
#       Hidden states:
#         - reset_short() on each window
#         - reset_long() at day rollover
#     Returns:
#       raw_reg: (B, S, 1) regression logits (no smoothing or gating applied)
#     """
#     def __init__(
#         self,
#         n_feats:         int,
#         short_units:     int,
#         long_units:      int,
#         dropout_short:   float,
#         dropout_long:    float,
#         # att_heads and att_drop parameters are kept in the signature for backward compatibility,
#         # but they are not used in this simplified implementation.
#         att_heads:       int = 1,
#         att_drop:        float = 0.0,
#         conv_k:          int = 3,
#         conv_dilation:   int = 1,
#     ):
#         super().__init__()

#         # store sizes
#         self.n_feats     = n_feats
#         self.short_units = short_units
#         self.long_units  = long_units

#         # 0) Input Conv1d + BN
#         pad_in = (conv_k // 2) * conv_dilation
#         self.conv = nn.Conv1d(n_feats, n_feats, conv_k,
#                               dilation=conv_dilation, padding=pad_in)
#         self.bn = nn.BatchNorm1d(n_feats)

#         # 1) Short-term Bi-LSTM (bidirectional)
#         assert short_units % 2 == 0, "short_units must be divisible by 2"
#         self.short_lstm = nn.LSTM(
#             input_size    = n_feats,
#             hidden_size   = short_units // 2,  # because bidirectional
#             batch_first   = True,
#             bidirectional = True
#         )
#         # short-level normalization + dropout retained
#         self.ln_short = nn.LayerNorm(short_units)
#         self.do_short = nn.Dropout(dropout_short)

#         # 2) Long-term Bi-LSTM (bidirectional)
#         assert long_units % 2 == 0, "long_units must be divisible by 2"
#         self.long_lstm = nn.LSTM(
#             input_size    = short_units,
#             hidden_size   = long_units // 2,
#             batch_first   = True,
#             bidirectional = True
#         )
#         # projection from short -> long, normalization + dropout retained
#         self.short2long = nn.Linear(short_units, long_units)
#         self.ln_long = nn.LayerNorm(long_units)
#         self.do_long = nn.Dropout(dropout_long)

#         # 3) Regression head (time-distributed)
#         self.pred = nn.Linear(long_units, 1)

#         # 4) Lazy hidden states (stateful handling kept)
#         self.h_short = self.c_short = None
#         self.h_long  = self.c_long  = None

#     def _init_states(self, B: int, device: torch.device):
#         # initialize hidden states for bidirectional LSTMs
#         # shapes: (num_directions=2, batch, hidden_size_per_direction)
#         self.h_short = torch.zeros(2, B, self.short_units // 2, device=device)
#         self.c_short = torch.zeros(2, B, self.short_units // 2, device=device)
#         self.h_long  = torch.zeros(2, B, self.long_units  // 2, device=device)
#         self.c_long  = torch.zeros(2, B, self.long_units  // 2, device=device)

#     def reset_short(self):
#         """Reset short-term LSTM hidden state (re-init for same B on device)."""
#         if self.h_short is not None:
#             B, dev = self.h_short.shape[1], self.h_short.device
#             self._init_states(B, dev)

#     def reset_long(self):
#         """
#         Reset long-term LSTM hidden state while carrying over the short LSTM
#         state into the newly initialized buffers (same semantics as original).
#         """
#         if self.h_long is not None:
#             B, dev = self.h_long.shape[1], self.h_long.device
#             hs, cs = self.h_short, self.c_short
#             self._init_states(B, dev)
#             # carry over daily short LSTM state into the new buffers to preserve daily continuity
#             self.h_short, self.c_short = hs.to(dev), cs.to(dev)

#     def forward(self, x: torch.Tensor):
#         # Accept inputs with optional leading dims and collapse them to (B, S, F)
#         if x.dim() > 3:
#             *lead, S, F = x.shape
#             x = x.view(-1, S, F)
#         if x.dim() == 2:
#             x = x.unsqueeze(0)

#         B, S, _ = x.shape
#         dev     = x.device

#         # 0) Conv1d → BN → ReLU
#         xc = x.transpose(1, 2)          # (B, F, S)
#         xc = self.conv(xc)
#         xc = self.bn(xc)
#         x = Funct.relu(xc).transpose(1, 2)  # (B, S, F)

#         # Initialize states on first pass or when batch-size changes
#         if self.h_short is None or self.h_short.size(1) != B:
#             self._init_states(B, dev)

#         # 1) Short LSTM (stateful)
#         out_s, (h_s, c_s) = self.short_lstm(x, (self.h_short, self.c_short))
#         # detach carried states in-place (preserve no-grad link to past)
#         h_s.detach_(); c_s.detach_()
#         self.h_short, self.c_short = h_s, c_s

#         # optional normalization + dropout (attenuation of noise)
#         out_s = self.ln_short(out_s)
#         out_s = self.do_short(out_s)

#         # 2) Long LSTM (stateful)
#         out_l_raw, (h_l, c_l) = self.long_lstm(out_s, (self.h_long, self.c_long))
#         h_l.detach_(); c_l.detach_()
#         self.h_long, self.c_long = h_l, c_l

#         # 3) Residual skip from short->long + Norm + Dropout
#         skip  = self.short2long(out_s)
#         out_l = self.ln_long(skip + out_l_raw)
#         out_l = self.do_long(out_l)

#         # 4) Regression head (time-distributed)
#         raw_reg = self.pred(out_l)  # (B, S, 1)

#         # Return raw regression logits (no smoothing/gating/cls heads)
#         return raw_reg

class ModelClass(nn.Module):
    """
    Stateful CNN → BiLSTM (short) → projection → head.

    Notes
      - Long Bi-LSTM removed. short2long remains as the projection from
        short-LSTM outputs into the final feature space consumed by pred.
      - reset_long left as a no-op to preserve external training-loop calls.
      - All other behavior (conv, short_lstm, ln/do, stateful handling, pred)
        preserved as in your last version so training loop and serialization
        remain compatible.
    """
    def __init__(
        self,
        n_feats:         int,
        short_units:     int,
        long_units:      int,            # kept name for compatibility; used as proj dim
        dropout_short:   float,
        dropout_long:    float,          # retained name, used as dropout after projection
        att_heads:       int = 1,
        att_drop:        float = 0.0,
        conv_k:          int = 3,
        conv_dilation:   int = 1,
    ):
        super().__init__()

        # store sizes
        self.n_feats     = n_feats
        self.short_units = short_units
        # long_units now defines projection/feature dim that pred expects
        self.long_units  = long_units

        # 0) Input Conv1d + BN
        pad_in = (conv_k // 2) * conv_dilation
        self.conv = nn.Conv1d(n_feats, n_feats, conv_k,
                              dilation=conv_dilation, padding=pad_in)
        self.bn = nn.BatchNorm1d(n_feats)

        # 1) Short-term Bi-LSTM (bidirectional)
        assert short_units % 2 == 0, "short_units must be divisible by 2"
        self.short_lstm = nn.LSTM(
            input_size    = n_feats,
            hidden_size   = short_units // 2,  # because bidirectional
            batch_first   = True,
            bidirectional = True
        )
        self.ln_short = nn.LayerNorm(short_units)
        self.do_short = nn.Dropout(dropout_short)

        # 2) Remove long LSTM: keep projection short->long (short2long)
        #    short2long maps short_units -> long_units (same dim pred expects)
        self.short2long = nn.Linear(short_units, long_units)
        # keep ln_long/do_long names for minimal code changes in forward/training
        self.ln_long = nn.LayerNorm(long_units)
        self.do_long = nn.Dropout(dropout_long)

        # 3) Regression head (time-distributed)
        self.pred = nn.Linear(long_units, 1)

        # 4) Lazy hidden states (only short LSTM now)
        self.h_short = self.c_short = None
        # keep long buffers but unused (to avoid breaking external code)
        self.h_long  = self.c_long  = None

    def _init_states(self, B: int, device: torch.device):
        # only short LSTM states required now
        self.h_short = torch.zeros(2, B, self.short_units // 2, device=device)
        self.c_short = torch.zeros(2, B, self.short_units // 2, device=device)
        # reserve placeholders for long (kept but unused)
        self.h_long  = None
        self.c_long  = None

    def reset_short(self):
        """Reset short-term LSTM hidden state (re-init for same B on device)."""
        if self.h_short is not None:
            B, dev = self.h_short.shape[1], self.h_short.device
            self._init_states(B, dev)

    def reset_long(self):
        """No-op kept for compatibility with training loop day-rollover calls."""
        # intentionally no state carried; long LSTM removed
        return

    def forward(self, x: torch.Tensor):
        # Accept inputs with optional leading dims and collapse them to (B, S, F)
        if x.dim() > 3:
            *lead, S, F = x.shape
            x = x.view(-1, S, F)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, S, _ = x.shape
        dev     = x.device

        # 0) Conv1d → BN → ReLU
        xc = x.transpose(1, 2)          # (B, F, S)
        xc = self.conv(xc)
        xc = self.bn(xc)
        x = Funct.relu(xc).transpose(1, 2)  # (B, S, F)

        # Initialize short states on first pass or when batch-size changes
        if self.h_short is None or self.h_short.size(1) != B:
            self._init_states(B, dev)

        # 1) Short LSTM (stateful)
        out_s, (h_s, c_s) = self.short_lstm(x, (self.h_short, self.c_short))
        h_s.detach_(); c_s.detach_()
        self.h_short, self.c_short = h_s, c_s

        out_s = self.ln_short(out_s)
        out_s = self.do_short(out_s)

        # 2) Projection (short -> "long" feature space)
        skip = self.short2long(out_s)   # (B, S, long_units)

        # apply layer-norm and dropout previously applied after long LSTM
        out_l = self.ln_long(skip)
        out_l = self.do_long(out_l)

        # 3) Regression head (time-distributed)
        raw_reg = self.pred(out_l)  # (B, S, 1)

        return raw_reg

######################################################################################################


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


#######################


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


####################### 


def eval_on_loader(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    metrics: dict,
    collect_preds: bool = False,
    disable_tqdm: bool = False
):
    """
    Evaluate model on loader and update provided metrics.

    Behavior:
      - Resets provided metric objects, runs model in eval() mode, and keeps
        model stateful semantics (reset_short / reset_long) used by the dataset.
      - Accepts model.forward that returns either:
          * raw_reg (Tensor) or
          * (raw_reg, raw_cls, raw_ter) tuple/list.
      - Converts logits to probabilities for metrics:
          * regression: sigmoid(raw_reg[..., -1, 0])
          * binary:    sigmoid(raw_cls[..., -1, 0]) if present, otherwise zeros
          * ternary:   softmax(raw_ter[..., -1, :]) if present, otherwise zeros
      - Optionally collects flat reg predictions in chronological order and
        returns them alongside final metrics.
    Returns:
      (results_dict, preds_array or None)
    """
    # 1) Reset metrics
    for m in metrics.values():
        m.reset()

    preds = [] if collect_preds else None
    expected_total_preds = 0  # strict counter for collected predictions

    full_end_times = getattr(loader.dataset, "end_times", None)

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
                W_day  = int(lengths[i])
                day_id = int(wd[i].item())

                # reset short‐term per day, long‐term on wrap
                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                # forward; accept either single-tensor or tuple output
                x_seq = xb[i, :W_day]
                out = model(x_seq)

                if isinstance(out, (tuple, list)):
                    raw_reg = out[0]
                    raw_cls = out[1] if len(out) > 1 else None
                    raw_ter = out[2] if len(out) > 2 else None
                else:
                    raw_reg = out
                    raw_cls = None
                    raw_ter = None

                # robustly index final time step logits
                if raw_reg is None:
                    raise RuntimeError("Model returned None for regression output")
                pr_seq = torch.sigmoid(raw_reg[..., -1, 0]) if raw_reg.dim() >= 2 else torch.sigmoid(raw_reg[..., 0])

                if raw_cls is not None:
                    pb_seq = torch.sigmoid(raw_cls[..., -1, 0]) if raw_cls.dim() >= 2 else torch.sigmoid(raw_cls[..., 0])
                else:
                    pb_seq = torch.zeros_like(pr_seq)

                if raw_ter is not None:
                    t_logits = raw_ter[..., -1, :] if raw_ter.dim() >= 2 else raw_ter
                    pt_seq = torch.softmax(t_logits, dim=-1)
                else:
                    pt_seq = torch.zeros(pr_seq.size(0), 3, device=pr_seq.device)

                # targets
                t_r = y_reg[i, :W_day].view(-1)
                t_b = y_bin[i, :W_day].view(-1)
                t_t = y_ter[i, :W_day].view(-1)

                # update metrics
                update_metrics(
                    metrics,
                    pr_seq, t_r,
                    pb_seq, t_b,
                    pt_seq, t_t
                )

                # collect preds (if requested) and enforce strict length matching
                if collect_preds:
                    # append predictions for this window
                    preds.extend(pr_seq.cpu().tolist())
                    expected_total_preds += W_day
                    # sanity check: ensure we haven't lost/duplicated entries so far
                    if len(preds) != expected_total_preds:
                        raise RuntimeError(
                            f"Prediction-collection mismatch: collected {len(preds)} items, "
                            f"expected {expected_total_preds} after processing batch index with W_day={W_day}"
                        )

    # finalize metrics
    results = {name: m.compute().item() for name, m in metrics.items()}

    # final assertion for collect_preds: total matches sum(lengths)
    if collect_preds:
        # derive dataset-wide expected total if possible
        if full_end_times is not None:
            total_from_dataset = len(full_end_times)
            if len(preds) != total_from_dataset:
                raise AssertionError(
                    f"Collected preds ({len(preds)}) != dataset end_times length ({total_from_dataset})."
                )
        # otherwise rely on the running counter
        # (expected_total_preds holds the total number of collected windows)
        if len(preds) != expected_total_preds:
            raise AssertionError(
                f"Final collected preds ({len(preds)}) != expected_total_preds ({expected_total_preds})."
            )
        preds_arr = np.array(preds)
        return results, preds_arr

    return results, None



###################### 


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
    cls_loss_weight: float = 0.05,    # kept for backward compatibility (unused)
    smooth_alpha: float = 0.005,      # diagnostics only
    smooth_beta: float = 20.0,        # diagnostics only
    smooth_delta: float = 0.01,       # diagnostics only
    diff1_weight: float = 1.0,        # diagnostics only
    diff2_weight: float = 0.2         # diagnostics only
) -> float:
    """
    Train or evaluate the simplified CNN→BiLSTM→BiLSTM regression model.

    Behavior:
      - 'train' mode:
          * Loss = pure MSE on model's regression output.
          * Diagnostics (slip smoothing, diff penalties) are computed per-window
            for logging only and are NOT added to the backpropagated loss.
          * Classification heads (if present) are treated as optional: their
            outputs are consumed for metrics only when available.
          * Stateful LSTM resets (reset_short / reset_long) remain unchanged.
          * Mixed-precision, gradient clipping, CosineAnnealingWarmRestarts,
            ReduceLROnPlateau, early stopping, checkpointing, and plotting
            remain unchanged.
      - non-'train' mode: delegates to eval_on_loader() and returns its result.

    Returns:
      best_val_rmse (float): lowest validation RMSE observed during training.
    """
    model.to(device)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()  # diagnostics/metrics only

    if mode != "train":
        return eval_on_loader(
            val_loader, model, device,
            get_metrics(device),
            collect_preds=True
        )

    torch.backends.cudnn.benchmark = True
    train_metrics = get_metrics(device)
    val_metrics = get_metrics(device)

    best_val_rmse, best_state = float("inf"), None
    best_tr_metrics, best_vl_metrics = {}, {}
    patience_ctr = 0
    live_plot = plots.LiveRMSEPlot()

    for epoch in range(1, max_epochs + 1):
        gc.collect()
        model.train()
        model.h_short = model.h_long = None

        # reset train metrics
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

            # buffers for metrics
            preds_r, targs_r = [], []
            preds_b, targs_b = [], []
            preds_t, targs_t = [], []

            prev_day, ewma = None, None
            prev_lr, prev_prev_lr = None, None

            # accumulator for the batch loss (aggregate over windows)
            batch_loss = torch.tensor(0.0, device=device)
            n_windows = xb.size(0)

            # per-window forward + diagnostics
            for di in range(n_windows):
                W = lengths[di]
                day_id = int(wd[di].item())
                x_seq = xb[di, :W]
                y_r = targ_r[di, :W].view(-1)
                y_b = targ_b[di, :W].view(-1)
                y_t = targ_t[di * W: di * W + W]

                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                # Forward: robustly accept either single-tensor or 3-tuple outputs
                out = model(x_seq)
                if isinstance(out, (tuple, list)):
                    raw_reg = out[0]
                    raw_cls = out[1] if len(out) > 1 else None
                    raw_ter = out[2] if len(out) > 2 else None
                else:
                    raw_reg = out
                    raw_cls = None
                    raw_ter = None

                # Index final timestep logits robustly
                lr_logits = raw_reg[..., -1, 0] if raw_reg.dim() >= 3 else raw_reg[..., 0]

                # classification logits (if present)
                b_logits = None
                t_logits = None
                if raw_cls is not None:
                    b_logits = raw_cls[..., -1, 0] if raw_cls.dim() >= 3 else raw_cls[..., 0]
                if raw_ter is not None:
                    t_logits = raw_ter[..., -1, :] if raw_ter.dim() >= 3 else raw_ter

                # probabilities used for metric collection
                lr = torch.sigmoid(lr_logits)
                pb = torch.sigmoid(b_logits) if b_logits is not None else torch.zeros_like(lr)
                pt = torch.softmax(t_logits, dim=-1) if t_logits is not None else torch.zeros(lr.size(0), 3, device=lr.device)

                preds_r.append(lr.detach()); targs_r.append(y_r)
                preds_b.append(pb.detach()); targs_b.append(y_b)
                preds_t.append(pt.detach()); targs_t.append(y_t)

                # diagnostics (computed but NOT added to loss directly)
                with autocast(device_type=device.type):
                    win_loss = mse_loss(lr, y_r)  # per-window MSE

                    ewma_new = lr.detach() if ewma is None else smooth_alpha * lr.detach() + (1 - smooth_alpha) * ewma
                    slip = torch.relu(ewma_new - lr)
                    if prev_lr is not None:
                        slip = torch.where(lr > prev_lr, torch.zeros_like(slip), slip)
                    hub = torch.where(
                        slip <= smooth_delta,
                        0.5 * slip ** 2,
                        smooth_delta * (slip - 0.5 * smooth_delta)
                    )

                    if prev_lr is not None:
                        neg_diff = torch.relu(prev_lr - lr)
                    else:
                        neg_diff = torch.zeros_like(lr)

                    if prev_prev_lr is not None:
                        curv = lr - 2 * prev_lr + prev_prev_lr
                    else:
                        curv = torch.zeros_like(lr)

                prev_prev_lr, prev_lr, ewma = prev_lr, lr.detach(), ewma_new

                # accumulate window loss (simple average over windows)
                batch_loss = batch_loss + win_loss

            # finalize batch loss
            batch_loss = batch_loss / float(n_windows)

            # diagnostics logging for first batch (use last-window tensors where needed)
            if batch_idx == 0:
                b_log_for_diag = b_logits if b_logits is not None else torch.zeros_like(lr)
                models_core.log_loss_components(
                    epoch, batch_idx, lr, y_r, b_log_for_diag, hub, prev_lr, prev_prev_lr, mse_loss, bce_loss, 
                    cls_loss_weight, smooth_beta, diff1_weight, diff2_weight, model, params.log_file
                )

            # backprop aggregated batch loss
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)

            # now that grads exist and are unscaled, log gradient norms (first batch)
            if batch_idx == 0:
                models_core.log_gradient_norms(
                    epoch, batch_idx, model, params.log_file
                )

            nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()

            # batch-level metric update (regression always present; cls/ter used if available)
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

        # --- epoch-level metrics & validation ---
        raw_tr = {n: m.compute() for n, m in train_metrics.items()}
        tr = {n: (v.item() if isinstance(v, torch.Tensor) else v) for n, v in raw_tr.items()}

        for m in val_metrics.values():
            m.reset()
        raw_vl, _ = eval_on_loader(val_loader, model, device, val_metrics)
        vl = {n: (v.item() if isinstance(v, torch.Tensor) else v) for n, v in raw_vl.items()}

        # checkpointing, plotting & early stopping (unchanged)
        models_dir = Path(params.models_folder)
        live_plot.update(tr["rmse"], vl["rmse"])
        best_val_rmse, improved, tr_best, vl_best, tmp_state = \
            models_core.maybe_save_chkpt(
                models_dir, model, vl["rmse"],
                best_val_rmse, tr, vl, live_plot, params
            )
        if tmp_state is not None:
            best_state = tmp_state
            best_tr_metrics = tr_best.copy()
            best_vl_metrics = vl_best.copy()

        patience_ctr = 0 if improved else patience_ctr + 1
        if patience_ctr >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # epoch summary print (unchanged)
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

    # final-best checkpoint (unchanged)
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
