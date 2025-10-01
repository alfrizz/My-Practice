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
#     Dual-Memory LSTM for smoothed continuous‐signal regression + binary‐flag prediction.

#     Architecture:
#       0) Conv1d + BatchNorm1d → ReLU on raw features
#       1) Stateful Bidirectional “short” LSTM per look-back window
#       2) Window-level self-attention + residual
#       3) LayerNorm → Dropout on short embedding
#       4) Stateful Bidirectional “long” LSTM across windows
#       5) Residual skip (projected short → long) + LayerNorm → Dropout
#       6) Time-distributed heads:
#          – regression head (1 logit per step)
#          – binary head     (1 logit per step)
#          – ternary head    (3 logits per step)
#       7) Causal smoothing Conv1d on regression logits:
#          • kernel_size = smooth_k
#          • dilation    = smooth_dilation
#          • padding     = (smooth_k–1)*smooth_dilation (symmetric)
#          • slice off the first pad_sm outputs to realign
#          • gate upward jumps so raw logit passes through instantly
#       8) Automatic reset of hidden states at day/week boundaries
#     """
#     def __init__(
#         self,
#         n_feats:         int,
#         short_units:     int,
#         long_units:      int,
#         dropout_short:   float,
#         dropout_long:    float,
#         att_heads:       int,
#         att_drop:        float,
#         conv_k:          int = 3,
#         conv_dilation:   int = 1,
#         smooth_k:        int = 3,
#         smooth_dilation: int = 1,
#     ):
#         super().__init__()
#         self.n_feats         = n_feats
#         self.short_units     = short_units
#         self.long_units      = long_units
#         self.smooth_k        = smooth_k
#         self.smooth_dilation = smooth_dilation

#         # 0) Input Conv1d + BN
#         pad_in = (conv_k // 2) * conv_dilation
#         self.conv = nn.Conv1d(
#             in_channels  = n_feats,
#             out_channels = n_feats,
#             kernel_size  = conv_k,
#             dilation     = conv_dilation,
#             padding      = pad_in
#         )
#         self.bn = nn.BatchNorm1d(n_feats)

#         # 1) Short‐term Bi‐LSTM
#         assert short_units % 2 == 0
#         self.short_lstm = nn.LSTM(
#             input_size    = n_feats,
#             hidden_size   = short_units // 2,
#             batch_first   = True,
#             bidirectional = True
#         )

#         # 2) Window‐level self‐attention
#         self.attn = nn.MultiheadAttention(
#             embed_dim   = short_units,
#             num_heads   = att_heads,
#             dropout     = att_drop,
#             batch_first = True
#         )

#         # 3) LayerNorm → Dropout on short embedding
#         self.ln_short = nn.LayerNorm(short_units)
#         self.do_short = nn.Dropout(dropout_short)

#         # 4) Long‐term Bi‐LSTM
#         assert long_units % 2 == 0
#         self.long_lstm = nn.LSTM(
#             input_size    = short_units,
#             hidden_size   = long_units // 2,
#             batch_first   = True,
#             bidirectional = True
#         )

#         # 5a) Project short→long for residual
#         self.short2long = nn.Linear(short_units, long_units)

#         # 5b) LayerNorm → Dropout on long embedding
#         self.ln_long = nn.LayerNorm(long_units)
#         self.do_long = nn.Dropout(dropout_long)

#         # 6) Time-distributed heads
#         self.pred     = nn.Linear(long_units, 1)  # regression logits
#         self.cls_head = nn.Linear(long_units, 1)  # binary logits
#         self.cls_ter  = nn.Linear(long_units, 3)  # ternary logits

#         # 7) Causal smoothing conv on regression logits
#         #    symmetric padding = pad_sm on both sides
#         pad_sm = (smooth_k - 1) * smooth_dilation
#         self.smoother = nn.Conv1d(
#             in_channels  = 1,
#             out_channels = 1,
#             kernel_size  = smooth_k,
#             dilation     = smooth_dilation,
#             padding      = pad_sm,
#             bias         = False
#         )
#         # initialize uniform average
#         nn.init.constant_(self.smoother.weight, 1.0 / smooth_k)

#         # 8) Lazy-init hidden states
#         self.h_short = self.c_short = None
#         self.h_long  = self.c_long  = None

#     def _init_states(self, B: int, device: torch.device):
#         self.h_short = torch.zeros(2, B, self.short_units // 2, device=device)
#         self.c_short = torch.zeros(2, B, self.short_units // 2, device=device)
#         self.h_long  = torch.zeros(2, B, self.long_units  // 2, device=device)
#         self.c_long  = torch.zeros(2, B, self.long_units  // 2, device=device)

#     def reset_short(self):
#         if self.h_short is not None:
#             B, dev = self.h_short.size(1), self.h_short.device
#             self._init_states(B, dev)

#     def reset_long(self):
#         if self.h_long is not None:
#             B, dev = self.h_long.size(1), self.h_long.device
#             hs, cs = self.h_short, self.c_short
#             self._init_states(B, dev)
#             self.h_short, self.c_short = hs.to(dev), cs.to(dev)

#     def forward(self, x: torch.Tensor):
#         # reshape extra dims → (W, S, F)
#         if x.dim() > 3:
#             *lead, S, F = x.shape
#             x = x.view(-1, S, F)

#         # ensure (W, S, n_feats)
#         if x.dim() == 3 and x.size(-1) != self.n_feats:
#             x = x.transpose(1, 2).contiguous()

#         B, S, _ = x.size()
#         dev     = x.device

#         # 0) Conv1d → BN → ReLU
#         x_conv = x.transpose(1, 2)      # (W, F, S)
#         x_conv = self.conv(x_conv)
#         x_conv = self.bn(x_conv)
#         x      = Funct.relu(x_conv).transpose(1, 2)  # back to (W, S, F)

#         # init/reset hidden states
#         if self.h_short is None or self.h_short.size(1) != B:
#             self._init_states(B, dev)

#         # 1) Short‐term Bi‐LSTM
#         out_short_raw, (h_s, c_s) = self.short_lstm(
#             x, (self.h_short, self.c_short)
#         )
#         h_s.detach_(); c_s.detach_()
#         self.h_short, self.c_short = h_s, c_s

#         # 2) Attention + residual
#         attn_out, _ = self.attn(out_short_raw, out_short_raw, out_short_raw)
#         out_short   = out_short_raw + attn_out

#         # 3) LayerNorm → Dropout
#         out_short = self.ln_short(out_short)
#         out_short = self.do_short(out_short)

#         # 4) Long‐term Bi‐LSTM
#         out_long_raw, (h_l, c_l) = self.long_lstm(
#             out_short, (self.h_long, self.c_long)
#         )
#         h_l.detach_(); c_l.detach_()
#         self.h_long, self.c_long = h_l, c_l

#         # 5) Residual skip → LayerNorm → Dropout
#         skip     = self.short2long(out_short)
#         out_long = skip + out_long_raw
#         out_long = self.ln_long(out_long)
#         out_long = self.do_long(out_long)

#         # 6) Time-distributed heads
#         raw_reg = self.pred(out_long)     # (W, S, 1)
#         raw_cls = self.cls_head(out_long) # (W, S, 1)
#         raw_ter = self.cls_ter(out_long)  # (W, S, 3)

#         # 7) Causal smoothing + gating
#         #    permute to (batch_windows, 1, time)
#         reg_t   = raw_reg.permute(1, 2, 0)          # (S, 1, W)
#         pad_sm  = (self.smooth_k - 1) * self.smooth_dilation
#         # conv output length = W + pad_sm*2 - dilation*(k-1)
#         # we slice off exactly pad_sm entries from the left to realign
#         sm_t    = self.smoother(reg_t)[:, :, pad_sm:]
#         sm_reg  = sm_t.permute(2, 0, 1)             # (W, S, 1)
#         # gate upward spikes: if raw_reg > sm_reg, pass raw
#         out_reg = torch.where(raw_reg > sm_reg, raw_reg, sm_reg)

#         # 8) return gated-smoothed regression + raw classification
#         return out_reg, raw_cls, raw_ter



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
        # init moving‐avg filter
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



######################################################################################################


# def get_metrics(device: torch.device, thr: float = 0.5):
#     """Return a dict of metrics (regression, binary, ternary) placed on device."""
#     return {
#         # regression
#         "rmse": torchmetrics.MeanSquaredError(squared=False).to(device),
#         "mae":  torchmetrics.MeanAbsoluteError().to(device),
#         "r2":   torchmetrics.R2Score().to(device),
#         # binary
#         "acc":  torchmetrics.classification.BinaryAccuracy(threshold=thr).to(device),
#         "prec": torchmetrics.classification.BinaryPrecision(threshold=thr).to(device),
#         "rec":  torchmetrics.classification.BinaryRecall(threshold=thr).to(device),
#         "f1":   torchmetrics.classification.BinaryF1Score(threshold=thr).to(device),
#         "auc":  torchmetrics.classification.BinaryAUROC().to(device),
#         # ternary 
#         "t_acc":  torchmetrics.classification.MulticlassAccuracy(num_classes=3).to(device),
#         "t_prec": torchmetrics.classification.MulticlassPrecision(num_classes=3, average="macro").to(device),
#         "t_rec":  torchmetrics.classification.MulticlassRecall(num_classes=3, average="macro").to(device),
#         "t_f1":   torchmetrics.classification.MulticlassF1Score(num_classes=3, average="macro").to(device),
#         "t_auc":  torchmetrics.classification.MulticlassAUROC(num_classes=3, average="macro").to(device),
#     }

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


# def eval_on_loader(
#     loader,
#     model: torch.nn.Module,
#     device: torch.device,
#     metrics: dict,
#     collect_preds: bool = False
# ):
#     """
#     Evaluate a stateful model over a DataLoader of per-day sliding-window batches.

#     1) Reset all metrics’ internal state.
#     2) Switch model to eval() and zero its short- & long-term hidden states.
#     3) Disable gradient computation.
#     4) Loop over each batch of “per-day” groups:
#        a) Unpack the 9-tuple returned by pad_collate:
#           (xb, y_reg, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths)
#        b) Move xb, y_reg, y_bin, y_ter, wd to the specified device.
#        c) For each sample i in the batch:
#           • Determine W_day = lengths[i]  (number of sliding windows that day).
#           • Reset short-term LSTM; reset long-term on calendar-day rollover.
#           • SLICE & forward a batch of W_day windows:
#                 x_seq = xb[i, :W_day]           # shape: (W_day, look_back, F)
#                 pr_log, pb_log, pt_log = model(x_seq)
#           • EXTRACT exactly one forecast per sliding window by taking the **last**
#             time-step of each of the W_day outputs:
#                 pr_seq = sigmoid(pr_log[:, -1, 0])    # → (W_day,)
#                 pb_seq = sigmoid(pb_log[:, -1, 0])    # → (W_day,)
#                 pt_seq = softmax(pt_log[:, -1, :], -1) # → (W_day, C)
#           • SLICE true targets to length W_day:
#                 t_r = y_reg[i, :W_day].view(-1)
#                 t_b = y_bin[i, :W_day].view(-1)
#                 t_t = (y_ter[i, :W_day].argmax(dim=-1).view(-1)
#                        if y_ter.dim()==3
#                        else y_ter[i, :W_day].view(-1))
#           • UPDATE all metrics *vectorized* on the full sequences:
#                 metrics["rmse"].update(pr_seq, t_r)
#                 metrics["mae"].update(pr_seq, t_r)
#                 metrics["r2"].update(pr_seq, t_r)
#                 metrics["acc"].update(pb_seq, t_b)
#                 metrics["prec"].update(pb_seq, t_b)
#                 metrics["rec"].update(pb_seq, t_b)
#                 metrics["f1"].update(pb_seq, t_b)
#                 metrics["auc"].update(pb_seq, t_b)
#                 metrics["t_acc"].update(pt_seq, t_t)
#                 metrics["t_prec"].update(pt_seq, t_t)
#                 metrics["t_rec"].update(pt_seq, t_t)
#                 metrics["t_f1"].update(pt_seq, t_t)
#                 metrics["t_auc"].update(pt_seq, t_t)
#           • IF collect_preds:
#                 - Append all pr_seq scalars to `preds`.
#                 - Append the matching slice of global timestamps
#                   `loader.dataset.end_times[global_idx:global_idx+W_day]` to `times`.
#                 - Increment `global_idx` by W_day.
#     5) After iterating all batches, call `.compute()` on each metric to get final scalars.
#     6) RETURN:
#        - `(metrics_out, None)` if collect_preds=False
#        - `(metrics_out, preds_array, times_array)` if collect_preds=True

#     Notes:
#       - We do **not** reshape or unsqueeze the batch-axis here—this matches the
#         training forward logic, where batch_size = W_day and seq_len = look_back.
#       - `lengths` is a plain Python list of ints, so no `.item()` is required.
#       - `loader.dataset.end_times` must be the flat NumPy array of all window timestamps.
#     """
#     # 1) Reset all metrics
#     for m in metrics.values():
#         m.reset()

#     # 2) Prepare prediction/time collectors
#     preds = [] if collect_preds else None
#     times = [] if collect_preds else None

#     # Flat array of every sliding-window timestamp
#     full_end_times = loader.dataset.end_times
#     global_idx     = 0

#     # 3) Switch to eval mode & clear hidden states
#     model.eval()
#     model.h_short = model.h_long = None
#     prev_day = None

#     # 4) Disable gradients for evaluation
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="eval", unit="batch"):
#             xb, y_reg, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths = batch

#             # Move tensors to device
#             xb    = xb.to(device, non_blocking=True)
#             y_reg = y_reg.to(device, non_blocking=True)
#             y_bin = y_bin.to(device, non_blocking=True)
#             y_ter = y_ter.to(device, non_blocking=True)
#             wd    = wd.to(device, non_blocking=True)

#             B = xb.size(0)
#             for i in range(B):
#                 W_day  = lengths[i]            # how many windows today
#                 day_id = int(wd[i].item())

#                 # Reset LSTM states as in training
#                 model.reset_short()
#                 if prev_day is not None and day_id < prev_day:
#                     model.reset_long()
#                 prev_day = day_id

#                 # 4.c.2) Slice & forward a batch of W_day windows
#                 x_seq = xb[i, :W_day]  # shape (W_day, look_back, F)
#                 pr_log, pb_log, pt_log = model(x_seq)

#                 # 4.c.3) Extract one forecast per window at the final time-step
#                 pr_seq = torch.sigmoid(pr_log[:, -1, 0])      # (W_day,)
#                 pb_seq = torch.sigmoid(pb_log[:, -1, 0])      # (W_day,)
#                 pt_seq = torch.softmax(pt_log[:, -1, :], dim=-1)  # (W_day, C)

#                 # 4.c.4) Slice targets to match W_day
#                 t_r = y_reg[i, :W_day].view(-1)
#                 t_b = y_bin[i, :W_day].view(-1)
#                 if y_ter.dim() == 3:
#                     t_t = y_ter[i, :W_day].argmax(dim=-1).view(-1)
#                 else:
#                     t_t = y_ter[i, :W_day].view(-1)

#                 # 4.c.5) Vectorized metric updates
#                 metrics["rmse"].update(pr_seq, t_r)
#                 metrics["mae"].update(pr_seq, t_r)
#                 metrics["r2"].update(pr_seq, t_r)

#                 metrics["acc"].update(pb_seq, t_b)
#                 metrics["prec"].update(pb_seq, t_b)
#                 metrics["rec"].update(pb_seq, t_b)
#                 metrics["f1"].update(pb_seq, t_b)
#                 metrics["auc"].update(pb_seq, t_b)

#                 metrics["t_acc"].update(pt_seq, t_t)
#                 metrics["t_prec"].update(pt_seq, t_t)
#                 metrics["t_rec"].update(pt_seq, t_t)
#                 metrics["t_f1"].update(pt_seq, t_t)
#                 metrics["t_auc"].update(pt_seq, t_t)

#                 # 4.c.6) Collect predictions & timestamps if requested
#                 if collect_preds:
#                     preds.extend(pr_seq.cpu().tolist())
#                     times.extend(
#                         full_end_times[global_idx : global_idx + W_day].tolist()
#                     )
#                     global_idx += W_day

#     # 5) Compute final metric values
#     metrics_out = {k: m.compute().item() for k, m in metrics.items()}

#     # 6) Return based on collect_preds flag
#     if collect_preds:
#         preds_arr = np.array(preds)
#         times_arr = np.array(pd.to_datetime(times), dtype="datetime64[ns]")
#         return metrics_out, preds_arr, times_arr

#     return metrics_out, None

def eval_on_loader(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    metrics: dict,
    collect_preds: bool = False,
    disable_tqdm: bool = False
):
    """
    Shared eval helper for train/val/test.

    1) Reset metrics.  
    2) model.eval() + zero hidden state.  
    3) torch.no_grad(), loop batches with optional progress bar.  
    4) For each day in batch:
       - reset_short(); reset_long() on rollover  
       - slice out W_day windows → x_seq  
       - forward → (reg, bin, ter) logits  
       - pr_seq = sigmoid(reg[:, -1,0]), pb_seq, pt_seq  
       - slice targets t_r, t_b, t_t  
       - metrics[k].update(...) vectorized  
       - if collect_preds: append pr_seq & matching timestamps  
    5) compute() all metrics → scalars  
    6) return (metrics_dict, preds/times or None)
    """
    for m in metrics.values():
        m.reset()

    preds = [] if collect_preds else None
    times = [] if collect_preds else None
    full_end_times = loader.dataset.end_times
    global_idx = 0

    model.eval()
    model.h_short = model.h_long = None
    prev_day = None

    with torch.no_grad():
        loop = tqdm(loader, desc="eval", unit="batch", disable=disable_tqdm)
        for batch in loop:
            xb, y_reg, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths = batch
            xb    = xb.to(device, non_blocking=True)
            y_reg = y_reg.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True)
            y_ter = y_ter.to(device, non_blocking=True)
            wd    = wd.to(device, non_blocking=True)

            B = xb.size(0)
            for i in range(B):
                W_day  = lengths[i]
                day_id = int(wd[i].item())

                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                x_seq = xb[i, :W_day]
                pr_log, pb_log, pt_log = model(x_seq)

                pr_seq = torch.sigmoid(pr_log[:, -1, 0])
                pb_seq = torch.sigmoid(pb_log[:, -1, 0])
                pt_seq = torch.softmax(pt_log[:, -1, :], dim=-1)

                t_r = y_reg[i, :W_day].view(-1)
                t_b = y_bin[i, :W_day].view(-1)
                if y_ter.dim() == 3:
                    t_t = y_ter[i, :W_day].argmax(dim=-1).view(-1)
                else:
                    t_t = y_ter[i, :W_day].view(-1)

                # vectorized updates
                update_metrics(
                    metrics,
                    pr_seq, t_r,
                    pb_seq, t_b,
                    pt_seq, t_t
                )

                if collect_preds:
                    preds.extend(pr_seq.cpu().tolist())
                    times.extend(
                        full_end_times[global_idx : global_idx + W_day].tolist()
                    )
                    global_idx += W_day

    out = {k: m.compute().item() for k, m in metrics.items()}

    if collect_preds:
        preds_arr = np.array(preds)
        times_arr = np.array(pd.to_datetime(times), dtype="datetime64[ns]")
        return out, preds_arr, times_arr

    return out, None

###################### 




# def model_training_loop(
#     model:               torch.nn.Module,
#     optimizer:           torch.optim.Optimizer,
#     cosine_sched:        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
#     plateau_sched:       torch.optim.lr_scheduler.ReduceLROnPlateau,
#     scaler:              GradScaler,
#     train_loader,
#     val_loader,
#     *,
#     max_epochs:          int,
#     early_stop_patience: int,
#     clipnorm:            float,
#     device:              torch.device = torch.device("cpu"),
#     mode:                str = "train",
#     cls_loss_weight:     float = 0.05,
#     smooth_alpha:        float = 0.005,
#     smooth_beta:         float = 20.0,
#     smooth_delta:        float = 0.01,
#     diff1_weight:        float = 1.0,
#     diff2_weight:        float = 0.2
# ):
#     """
#     Train or evaluate the stateful CNN→BiLSTM→Attention→BiLSTM network.

#     TRAIN mode:
#       1) Mixed-precision, per-window forward/backward with:
#          - Causal-conv smoothing on regression head
#          - One-way EWMA + Huber penalty on downward slip
#          - L2 on first- & second-differences
#       2) Joint regression (MSE) + binary classification (BCE) loss
#       3) Per-window metric tracking (vectorized updates for all windows):
#          - Regression: RMSE, MAE, R2
#          - Classification: Acc, Prec, Rec, F1, AUC
#       4) Stateful LSTM resets: short-term every window, long-term on day rollover
#       5) CosineAnnealingWarmRestarts + ReduceLROnPlateau + early stopping
#       6) Live RMSE plotting & best-model checkpointing

#     EVAL mode:
#       - Single-pass evaluation via eval_on_loader
#       - Returns (metrics_dict, preds_array)

#     Returns:
#       - TRAIN → best validation RMSE (float)
#       - EVAL  → (metrics_dict, preds_array)
#     """
#     model.to(device)
#     mse_loss = nn.MSELoss()
#     bce_loss = nn.BCEWithLogitsLoss()

#     # 1) Eval-only path
#     if mode != "train":
#         return eval_on_loader(
#             val_loader, model, device,
#             get_metrics(device), collect_preds=True
#         )

#     # 2) Setup training & validation metrics, scheduler, state
#     torch.backends.cudnn.benchmark = True
#     train_metrics = get_metrics(device)
#     val_metrics   = get_metrics(device)
#     best_val_rmse = float("inf")
#     patience_ctr  = 0
#     live_plot     = plots.LiveRMSEPlot()

#     # 3) Epoch loop
#     for epoch in range(1, max_epochs + 1):
#         gc.collect()
#         model.train()
#         model.h_short = model.h_long = None

#         # Reset train metrics each epoch
#         for m in train_metrics.values():
#             m.reset()

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
#         for batch_idx, batch in enumerate(pbar):
#             # Unpack per-day sliding-window batch
#             xb_days, y_sig_days, y_sig_cls_days, *_rest, wd_days, _, lengths = batch
#             xb     = xb_days.to(device,   non_blocking=True)
#             targ_r = y_sig_days.to(device, non_blocking=True)
#             targ_c = y_sig_cls_days.to(device, non_blocking=True)
#             wd     = wd_days.to(device,    non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)
#             prev_day = None
#             ewma     = None
#             prev_lr  = None
#             prev_prev_lr = None

#             # Per-day loop
#             for di in range(xb.size(0)):
#                 W      = lengths[di]           # actual windows that day
#                 day_id = int(wd[di].item())
#                 x_seq  = xb[di, :W]            # → (W, look_back, F)
#                 y_r    = targ_r[di, :W].view(-1)
#                 y_c    = targ_c[di, :W].view(-1)

#                 # LSTM state resets
#                 model.reset_short()
#                 if prev_day is not None and day_id < prev_day:
#                     model.reset_long()
#                 prev_day = day_id

#                 # Forward
#                 pr_log, pc_log, _ = model(x_seq)
#                 # Last-step logits
#                 lr_logits  = pr_log[..., -1, 0]   # → (W,)
#                 cls_logits = pc_log[..., -1, 0]   # → (W,)
#                 lr         = torch.sigmoid(lr_logits)

#                 # Mixed-precision losses
#                 with autocast(device_type=device.type):
#                     loss = mse_loss(lr, y_r) \
#                          + cls_loss_weight * bce_loss(cls_logits, y_c)

#                     # EWMA + Huber slip penalty
#                     ewma = lr.detach() if ewma is None \
#                            else smooth_alpha * lr.detach() + (1 - smooth_alpha) * ewma
#                     slip = torch.relu(ewma - lr)
#                     slip = torch.where(lr > prev_lr, torch.zeros_like(slip), slip) \
#                            if prev_lr is not None else slip
#                     hub  = torch.where(
#                               slip <= smooth_delta,
#                               0.5 * slip**2,
#                               smooth_delta * (slip - 0.5 * smooth_delta)
#                             )
#                     loss += smooth_beta * hub.mean()

#                     # First- & second-difference L2
#                     if prev_lr is not None:
#                         neg_diff = torch.relu(prev_lr - lr)
#                         loss    += diff1_weight * neg_diff.pow(2).mean()
#                     if prev_prev_lr is not None:
#                         curvature = lr - 2 * prev_lr + prev_prev_lr
#                         loss     += diff2_weight * curvature.pow(2).mean()

#                 # Update history & backprop
#                 prev_prev_lr, prev_lr = prev_lr, lr.detach()
#                 scaler.scale(loss).backward()

#                 # ——— VECTORIZED METRIC UPDATES ———
#                 # Regression metrics over *all* W windows
#                 train_metrics["rmse"].update(lr,     y_r)
#                 train_metrics["mae"].update(lr,     y_r)
#                 train_metrics["r2"].update(lr,     y_r)

#                 # Classification metrics over *all* W windows
#                 prob_seq = torch.sigmoid(cls_logits)  # shape (W,)
#                 for key in ("acc", "prec", "rec", "f1", "auc"):
#                     train_metrics[key].update(prob_seq, y_c)

#                 # Detach LSTM hidden states to avoid backprop through history
#                 for h in (model.h_short, model.c_short, model.h_long, model.c_long):
#                     h.detach_()

#             # Optimizer & scheduler step
#             scaler.unscale_(optimizer)
#             nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
#             scaler.step(optimizer)
#             scaler.update()
#             frac = epoch - 1 + (batch_idx+1) / len(train_loader)
#             cosine_sched.step(frac)

#             # Live train-RMSE in the progress bar
#             pbar.set_postfix(
#                 train_rmse = train_metrics["rmse"].compute().item(),
#                 lr         = optimizer.param_groups[0]["lr"],
#                 refresh    = False
#             )

#         # 4) Epoch-end train metrics (already computed on-the-fly)
#         tr = {k: train_metrics[k].compute().item() for k in
#               ("rmse","mae","r2","acc","prec","rec","f1","auc")}

#         # 5) Validation via shared eval_on_loader
#         for m in val_metrics.values():
#             m.reset()
#         vl, _ = eval_on_loader(
#             val_loader, model, device, val_metrics, collect_preds=False
#         )

#         # 6) Checkpoint, early stopping & live-plot update
#         models_dir = Path(params.models_folder)
#         best_val_rmse, maybe_state, maybe_tr, maybe_vl, _ = \
#             models_core.maybe_save_chkpt(
#                 models_dir, model, vl["rmse"],
#                 best_val_rmse, tr, vl, live_plot, params
#             )
#         patience_ctr = 0 if maybe_state else patience_ctr + 1
#         if patience_ctr >= early_stop_patience:
#             print("Early stopping at epoch", epoch)
#             break

#         live_plot.update(tr["rmse"], vl["rmse"])
#         print(f"Epoch {epoch:03d}")
#         print(
#             f'TRAIN→ RMSE={tr["rmse"]:.5f} MAE={tr["mae"]:.5f} R2={tr["r2"]:.5f} | '
#             f'Acc={tr["acc"]:.5f} Prec={tr["prec"]:.5f} Rec={tr["rec"]:.5f} '
#             f'F1={tr["f1"]:.5f} AUROC={tr["auc"]:.5f}'
#         )
#         print(
#             f'VALID→ RMSE={vl["rmse"]:.5f} MAE={vl["mae"]:.5f} R2={vl["r2"]:.5f} | '
#             f'Acc={vl["acc"]:.5f} Prec={vl["prec"]:.5f} Rec={vl["rec"]:.5f} '
#             f'F1={vl["f1"]:.5f} AUROC={vl["auc"]:.5f}'
#         )
#         if epoch > params.hparams["LR_EPOCHS_WARMUP"]:
#             plateau_sched.step(vl["rmse"])

#     # 7) Final checkpoint if any
#     if best_state is not None:
#         models_core.save_final_chkpt(
#             Path(params.models_folder),
#             best_state, best_val_rmse, params,
#             best_tr, best_vl, live_plot, suffix="_fin"
#         )

#     return best_val_rmse


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
    cls_loss_weight: float = 0.05,
    smooth_alpha: float = 0.005,
    smooth_beta: float = 20.0,
    smooth_delta: float = 0.01,
    diff1_weight: float = 1.0,
    diff2_weight: float = 0.2
):
    """
    Train or evaluate the CNN→BiLSTM→Attention→BiLSTM network with three heads.

    Modes:
      - "train":
          1) Mixed-precision, per-window forward/backward with slip smoothing
          2) MSE regression + weighted BCE binary head
          3) No loss on ternary head (metrics only)
          4) Stateful LSTMs reset per window/day rollover
          5) Gradient clipping, CosineAnnealingWarmRestarts,
             ReduceLROnPlateau, early stopping, live‐plot & checkpointing
          6) Batch-level metric updates for regression, binary, ternary
      - others: single-pass eval_on_loader on val_loader

    Returns:
      TRAIN → best validation RMSE (float)
      EVAL  → (metrics_dict, preds_array)
    """
    model.to(device)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # Eval-only shortcut
    if mode != "train":
        return eval_on_loader(
            val_loader, model, device,
            get_metrics(device), collect_preds=True
        )

    torch.backends.cudnn.benchmark = True
    train_metrics = get_metrics(device)
    val_metrics   = get_metrics(device)
    best_val_rmse = float("inf")
    patience_ctr  = 0
    live_plot     = plots.LiveRMSEPlot()

    # Define ternary discretization bins once
    flat_delta = 0.001
    bins = torch.tensor([-flat_delta, flat_delta], device=device)

    for epoch in range(1, max_epochs + 1):
        gc.collect()
        model.train()
        model.h_short = model.h_long = None

        # Reset all train‐metrics
        for m in train_metrics.values():
            m.reset()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            xb_days, y_sig_days, y_sig_cls_days, y_sig_tern_days, *_r, wd_days, _, lengths = batch
            xb     = xb_days.to(device,   non_blocking=True)
            targ_r = y_sig_days.to(device, non_blocking=True)
            targ_b = y_sig_cls_days.to(device, non_blocking=True)
            # Discretize continuous ternary target → classes {0,1,2}
            raw_t = y_sig_tern_days.to(device, non_blocking=True)
            targ_t = torch.bucketize(raw_t, bins).view(-1)
            wd      = wd_days.to(device,    non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Buffers for preds & targets
            preds_r, targs_r = [], []
            preds_b, targs_b = [], []
            preds_t, targs_t = [], []

            prev_day     = None
            ewma         = None
            prev_lr      = None
            prev_prev_lr = None

            # Per-window forward/backward + stash preds
            for di in range(xb.size(0)):
                W      = lengths[di]
                day_id = int(wd[di].item())
                x_seq  = xb[di, :W]
                y_r    = targ_r[di, :W].view(-1)
                y_b    = targ_b[di, :W].view(-1)
                # Slice the flattened class labels
                y_t    = targ_t[di * W : di * W + W]

                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                # Forward through three heads
                pr_log, pc_log, pt_log = model(x_seq)
                lr_logits = pr_log[..., -1, 0]
                b_logits  = pc_log[..., -1, 0]
                t_logits  = pt_log[..., -1, :]  # shape [W,3]

                lr = torch.sigmoid(lr_logits)
                pb = torch.sigmoid(b_logits)
                pt = torch.softmax(t_logits, dim=-1)

                # Stash detached preds/targets
                preds_r.append(lr.detach()); targs_r.append(y_r)
                preds_b.append(pb.detach()); targs_b.append(y_b)
                preds_t.append(pt.detach()); targs_t.append(y_t)

                # Compute loss under autocast
                with autocast(device_type=device.type):
                    loss = mse_loss(lr, y_r) \
                         + cls_loss_weight * bce_loss(b_logits, y_b)

                    ewma_new = lr.detach() if ewma is None \
                               else smooth_alpha * lr.detach() + (1 - smooth_alpha) * ewma
                    slip = torch.relu(ewma_new - lr)
                    if prev_lr is not None:
                        slip = torch.where(lr > prev_lr, torch.zeros_like(slip), slip)
                    hub = torch.where(
                        slip <= smooth_delta,
                        0.5 * slip**2,
                        smooth_delta * (slip - 0.5 * smooth_delta),
                    )
                    loss = loss + smooth_beta * hub.mean()

                    if prev_lr is not None:
                        neg_diff = torch.relu(prev_lr - lr)
                        loss    = loss + diff1_weight * neg_diff.pow(2).mean()
                    if prev_prev_lr is not None:
                        curv = lr - 2 * prev_lr + prev_prev_lr
                        loss    = loss + diff2_weight * curv.pow(2).mean()

                prev_prev_lr, prev_lr, ewma = prev_lr, lr.detach(), ewma_new
                scaler.scale(loss).backward()

                # Detach LSTM hidden states
                for h in (model.h_short, model.c_short, model.h_long, model.c_long):
                    h.detach_()

            # End per-window

            # Optimizer step
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()

            # Batch-level metric update
            with torch.no_grad():
                pr_cat = torch.cat(preds_r)
                tr_cat = torch.cat(targs_r)
                pb_cat = torch.cat(preds_b)
                tb_cat = torch.cat(targs_b)
                pt_cat = torch.cat(preds_t)
                tt_cat = torch.cat(targs_t)

                update_metrics(
                    train_metrics,
                    pr_cat, tr_cat,
                    pb_cat, tb_cat,
                    pt_cat, tt_cat
                )

            # Scheduler & progress bar
            frac = epoch - 1 + (batch_idx + 1) / len(train_loader)
            cosine_sched.step(frac)
            pbar.set_postfix(lr=optimizer.param_groups[0]["lr"], refresh=False)

        # 1) Compute raw train‐metrics (might be Tensor or float)
        raw_tr = {name: m.compute() for name, m in train_metrics.items()}

        # 2) Safely convert each to a Python float
        tr = {}
        for name, val in raw_tr.items():
            if isinstance(val, torch.Tensor):
                # Tensor.item() works on both CPU and CUDA
                tr[name] = val.item()
            else:
                # already a float
                tr[name] = val

        # Validation pass (full sweep)
        for m in val_metrics.values():
            m.reset()
        raw_vl, _ = eval_on_loader(
            val_loader, model, device, val_metrics,
            collect_preds=False, disable_tqdm=False
        )

        # Safely convert validation metrics too
        vl = {}
        for name, val in raw_vl.items():
            if isinstance(val, torch.Tensor):
                vl[name] = val.item()
            else:
                vl[name] = val

        # Checkpoint, early stop
        models_dir = Path(params.models_folder)
        best_val_rmse, st, tr_best, vl_best, best_state = models_core.maybe_save_chkpt(
            models_dir, model, vl["rmse"],
            best_val_rmse, tr, vl, live_plot, params
        )
        patience_ctr = 0 if st else patience_ctr + 1
        if patience_ctr >= early_stop_patience:
            print("Early stopping at epoch", epoch)
            break

        # Update live plot with pure floats
        live_plot.update(tr["rmse"], vl["rmse"])

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

        # Step ReduceLROnPlateau after warmup
        if epoch > params.hparams["LR_EPOCHS_WARMUP"]:
            plateau_sched.step(vl["rmse"])

    # End epoch loop

    # Final checkpoint if available
    if best_state is not None:
        models_core.save_final_chkpt(
            Path(params.models_folder),
            best_state, best_val_rmse, params,
            tr_best, vl_best, live_plot, suffix="_fin"
        )

    return best_val_rmse
