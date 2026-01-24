from libs import plots, params, models_core  

from typing import Optional, Dict, Tuple, List, Sequence, Union, Any

import gc 
import os
import io
import tempfile
import copy
import re
import warnings
import sys

import pandas as pd
import numpy  as np
import math

import datetime as dt
from datetime import datetime, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torchmetrics

from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32   = True
torch.backends.cudnn.allow_tf32         = True
torch.backends.cudnn.benchmark          = True


#####################################################################################################


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encodings so the Transformer knows each time‐step index.
    """
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

############### 


def _allocate_lstm_states(batch_size: int,
                          hidden_size: int,
                          bidirectional: bool,
                          device: torch.device):
    """
    Allocate hidden (h) and cell (c) buffers for an LSTM.
    Returns two tensors shaped (num_directions, batch_size, hidden_size).
    """
    num_directions = 2 if bidirectional else 1
    h = torch.zeros(num_directions, batch_size, hidden_size, device=device)
    c = torch.zeros(num_directions, batch_size, hidden_size, device=device)
    return h, c


###############


class ModelClass(nn.Module):
    """
    Minimal configurable backbone:
    - optional Conv1d -> optional TCN
    - optional short Bi-LSTM -> optional Transformer path (feature_proj + pos_enc)
    - short2long projection -> optional long Bi-LSTM -> pooling/attn -> MLP head
    - canonical input_proj registered so diagnostics map to raw features
    Naming, attribute names and forward semantics preserved from your last version.
    """

    def layer_projection(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def __init__(
        self,
        n_feats: int,
        short_units: int,
        long_units: int,
        transformer_d_model: int,
        transformer_layers: int,
        dropout_short: float,
        dropout_long: float,
        dropout_trans: float,
        pred_hidden: int,
        look_back: int,
        use_conv: bool,
        use_tcn: bool,
        use_short_lstm: bool,
        use_transformer: bool,
        use_long_lstm: bool,
        use_delta: bool,
        flatten_mode: str,
    ):
        super().__init__()
        self.look_back = look_back
        self.short_units = short_units
        self.long_units = long_units
        self.use_conv = use_conv
        self.use_tcn = use_tcn
        self.use_short_lstm = use_short_lstm
        self.use_transformer = use_transformer
        self.use_long_lstm = use_long_lstm
        self.use_delta = use_delta

        assert short_units % 2 == 0 and long_units % 2 == 0

        # 0) Conv
        CONV_CHANNELS = params.hparams["CONV_CHANNELS"]
        if use_conv:
            conv_k, conv_dilation = params.hparams["CONV_K"], params.hparams["CONV_DILATION"]
            padding = (conv_k // 2) * conv_dilation
            self.conv = nn.Conv1d(n_feats, CONV_CHANNELS, kernel_size=conv_k, dilation=conv_dilation, padding=padding)
            self.bn = nn.GroupNorm(8, CONV_CHANNELS)
        else:
            self.conv = nn.Identity(); self.bn = nn.Identity()
        self.relu = nn.ReLU()

        # 1) TCN
        TCN_CHANNELS = params.hparams["TCN_CHANNELS"]
        if use_tcn:
            layers, k = params.hparams["TCN_LAYERS"], params.hparams["TCN_KERNEL"]
            blocks, in_ch = [], (CONV_CHANNELS if use_conv else n_feats)
            for i in range(layers):
                d = 2 ** i; pad = (k // 2) * d
                blocks += [nn.Conv1d(in_ch, TCN_CHANNELS, k, dilation=d, padding=pad),
                           nn.GroupNorm(8, TCN_CHANNELS), nn.ReLU()]
                in_ch = TCN_CHANNELS
            self.tcn = nn.Sequential(*blocks)
        else:
            self.tcn = nn.Identity()

        # 2) Short LSTM
        short_in = (TCN_CHANNELS if use_tcn else (CONV_CHANNELS if use_conv else n_feats))
        if use_short_lstm:
            self.short_lstm = nn.LSTM(input_size=short_in, hidden_size=short_units // 2, batch_first=True, bidirectional=True)
            self.ln_short = nn.LayerNorm(short_units); self.do_short = nn.Dropout(dropout_short)
        else:
            self.short_lstm = None; self.ln_short = nn.Identity(); self.do_short = nn.Identity()
        self.h_short = None; self.c_short = None

        upstream_dim = short_units if use_short_lstm else short_in

        # canonical input projection raw->upstream (registered for diagnostics)
        self.input_proj = nn.Linear(n_feats, upstream_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        if getattr(self.input_proj, "bias", None) is not None:
            nn.init.zeros_(self.input_proj.bias)

        # 3) Transformer adapter + modules
        if use_transformer:
            d_model = transformer_d_model
            heads = params.hparams["TRANSFORMER_HEADS"]
            ff_dim = d_model * params.hparams["TRANSFORMER_FF_MULT"]
            self.feature_proj = self.layer_projection(upstream_dim, d_model)
            self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout_trans, max_len=look_back)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=ff_dim, dropout=dropout_trans)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        else:
            self.feature_proj = nn.Linear(upstream_dim, upstream_dim)
            nn.init.eye_(self.feature_proj.weight) if upstream_dim == upstream_dim else None
            if getattr(self.feature_proj, "bias", None) is not None:
                nn.init.zeros_(self.feature_proj.bias)
            self.pos_enc = nn.Identity(); self.transformer = nn.Identity()

        # 4) Projection -> LayerNorm -> Dropout
        proj_in = transformer_d_model if use_transformer else upstream_dim
        self.short2long = self.layer_projection(proj_in, long_units)
        self.ln_proj = nn.LayerNorm(long_units); self.do_proj = nn.Dropout(dropout_long)

        # 5) Long LSTM
        if use_long_lstm:
            self.long_lstm = nn.LSTM(input_size=long_units, hidden_size=long_units // 2, batch_first=True, bidirectional=True)
            self.ln_long = nn.LayerNorm(long_units); self.do_long = nn.Dropout(dropout_long)
        else:
            self.long_lstm = None; self.ln_long = nn.Identity(); self.do_long = nn.Identity()
        self.h_long = None; self.c_long = None

        # 6) Head
        assert flatten_mode in ("flatten", "last", "pool", "attn")
        self.flatten_mode = flatten_mode
        flat_dim = (look_back * long_units if flatten_mode == "flatten" else long_units)
        self.ln_flat = nn.LayerNorm(flat_dim)
        self.head_flat = nn.Sequential(weight_norm(nn.Linear(flat_dim, pred_hidden)), nn.ReLU(), weight_norm(nn.Linear(pred_hidden, 1)))
        nn.init.zeros_(self.head_flat[-1].bias)
        self.attn_pool = nn.Linear(long_units, 1); nn.init.zeros_(self.attn_pool.bias)

        # 7) Delta head
        if self.use_delta:
            _d = nn.Linear(flat_dim, 1); nn.init.zeros_(_d.weight); nn.init.zeros_(_d.bias); self.delta_head = _d

    def forward(self, x: torch.Tensor):
        if x.dim() > 3:
            *lead, T, F = x.shape; x = x.view(-1, T, F)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, T, _ = x.shape

        xc = self.conv(x.transpose(1,2)); xc = self.bn(xc); xc = self.relu(xc); x = xc.transpose(1,2)
        tcn_out = self.tcn(x.transpose(1,2)).transpose(1,2)

        if self.use_short_lstm:
            if self.h_short is None or self.h_short.size(1) != B:
                self.h_short, self.c_short = _allocate_lstm_states(B, self.short_units // 2, True, x.device)
            out_s, (h_s, c_s) = self.short_lstm(tcn_out, (self.h_short, self.c_short))
            self.h_short, self.c_short = h_s.detach(), c_s.detach()
            out_s = self.ln_short(out_s); out_s = self.do_short(out_s)
        else:
            out_s = tcn_out

        # Transformer wiring: apply canonical input_proj only when its expected in_dim matches tensor last-dim
        if hasattr(self, "input_proj"):
            in_dim_expected = self.input_proj.weight.shape[1]
            if out_s.size(-1) == in_dim_expected:
                pre_tr = self.input_proj(out_s)
            else:
                pre_tr = out_s
        else:
            pre_tr = out_s

        tr_in = self.feature_proj(pre_tr)
        tr_in = self.pos_enc(tr_in)
        tr_out = self.transformer(tr_in.transpose(0,1).contiguous())
        out_t = tr_out.transpose(0,1).contiguous()

        out_p = self.short2long(out_t); out_p = self.ln_proj(out_p); out_p = self.do_proj(out_p)

        if self.use_long_lstm:
            if self.h_long is None or self.h_long.size(1) != B:
                self.h_long, self.c_long = _allocate_lstm_states(B, self.long_units // 2, True, out_p.device)
            out_l, (h_l, c_l) = self.long_lstm(out_p, (self.h_long, self.c_long))
            self.h_long, self.c_long = h_l.detach(), c_l.detach()
            out_l = self.ln_long(out_l); out_l = self.do_long(out_l)
        else:
            out_l = out_p

        if self.flatten_mode == "flatten":
            flat = out_l.reshape(B, -1)
        elif self.flatten_mode == "last":
            flat = out_l[:, -1, :]
        elif self.flatten_mode == "attn":
            scores = self.attn_pool(out_l).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            flat = (out_l * weights.unsqueeze(-1)).sum(dim=1)
        else:
            flat = out_l.mean(dim=1)

        norm_flat = self.ln_flat(flat)
        base_out = self.head_flat(norm_flat)
        base = base_out.squeeze(-1)
        delta = self.delta_head(norm_flat).squeeze(-1) if self.use_delta else torch.zeros_like(base)
        return base.unsqueeze(-1), delta.unsqueeze(-1)


######################################################################################################


def compute_baselines(flat_targets: np.ndarray, lengths: Sequence[int]) -> Tuple[float, float]:
    """
    Compute per-sample baselines from flattened targets and per-day lengths.

    Parameters
    - flat_targets: 1D array-like of scalar targets flattened in the same order
      used for model predictions (shape (N,) or convertible to that).
    - lengths: sequence of integers giving the number of windows per day in the
      same order that produced flat_targets. The sum(lengths) must equal len(flat_targets).
      Days with L <= 0 are ignored; days with L == 1 contribute to the mean baseline
      but not to the persistence baseline.
    """
    arr = np.asarray(flat_targets).ravel()
    N = arr.size
    if N == 0:
        return float("nan"), float("nan")

    ss_tot = float(((arr - arr.mean())**2).sum())
    mean_rmse = float(np.sqrt(ss_tot / N))

    start = 0
    sum_sq_persist = 0.0
    count_persist = 0
    for L in lengths:
        if L <= 1:
            start += max(0, L)
            continue
        end = start + L
        day = arr[start:end]
        err2 = (float(day[-1]) - float(day[-2])) ** 2
        sum_sq_persist += err2 * (L - 1)
        count_persist += (L - 1)
        start = end

    persist_rmse = float(np.sqrt(sum_sq_persist / count_persist)) if count_persist else float("nan")
    return mean_rmse, persist_rmse


############### 


def _reset_states(
    model:    nn.Module,
    wd_i:     torch.Tensor,
    prev_day: int | None
) -> int:
    """
    Reset short-term on any day change, and long-term weekly, ie when the day index
    wraps around (i.e. new_day < prev_day).

    Args:
      model     : ModelClass instance (implements reset_short/reset_long)
      wd_i      : scalar tensor with day-of-week (0=Mon,…,6=Sun)
      prev_day  : last seen day-of-week or None

    Returns:
      the new day-of-week for next call
    """
    if not hasattr(model, "_reset_log"): model._reset_log = []
    day = int(wd_i)

    # daily reset if the day changed
    if prev_day is None or day != prev_day:
        # clear short LSTM states so forward will allocate fresh states
        model.h_short = None
        model.c_short = None
        model._reset_log.append(("short", day))

        # weekly reset if we've wrapped past the end of the week
        if prev_day is not None and day < prev_day:
            model.h_long = None
            model.c_long = None
            model._reset_log.append(("long", day))

    return day


############################ 


class CustomMSELoss(nn.Module):
    """
    Level + slope loss.

    Total loss = level_loss + effective_alpha * slope_loss

    - alpha: base weight for slope (derivative) MSE (same as before).
    - warmup_steps: integer number of steps/epochs to linearly ramp slope weight from 0 -> alpha.
                    If 0, no warmup (effective_alpha == alpha).
    - use_huber: if True use Huber (smooth L1) for the level term; otherwise use MSE exactly.
    - huber_delta: delta for Huber (transition threshold). Note: Huber in PyTorch uses 0.5 * e^2
                   in the quadratic region (not identical to MSE), so keep use_huber=False to
                   preserve exact previous MSE behavior.
    forward signature:
      forward(preds, targs, seq_lengths, step=None)
      - step: optional integer used for warmup (epoch or global step depending on caller).
    """
    def __init__(
        self,
        alpha: float = 0.0,
        warmup_steps: int = 0,
        use_huber: bool = False,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.warmup_steps = int(warmup_steps)
        self.use_huber = bool(use_huber)
        self.huber_delta = float(huber_delta)

    def _effective_alpha(self, step: Optional[int]) -> float:
        if self.alpha <= 0.0:
            return 0.0
        if self.warmup_steps <= 0 or step is None:
            return self.alpha
        t = min(1.0, float(step) / float(self.warmup_steps))
        return self.alpha * t

    def forward(
        self,
        preds: torch.Tensor,
        targs: torch.Tensor,
        seq_lengths: List[int],
        step: Optional[int] = None,
    ) -> torch.Tensor:
        assert preds.shape == targs.shape, "preds and targs must have identical shapes"

        # Level term: MSE or Huber depending on flag
        if self.use_huber:
            # PyTorch's huber_loss uses delta and returns per-element losses; set reduction="mean"
            L1 = torch.nn.functional.huber_loss(preds, targs, reduction="mean", delta=self.huber_delta)
        else:
            L1 = torch.nn.functional.mse_loss(preds, targs, reduction="mean")

        # If no slope penalty requested, return level loss
        if self.alpha <= 0.0:
            return L1

        eff_alpha = self._effective_alpha(step)
        if eff_alpha <= 0.0:
            return L1

        # Build per-sequence one-step diffs, avoid crossing sequence boundaries
        parts_p = []
        parts_t = []
        s = 0
        for L in seq_lengths:
            if L > 1:
                e = s + L
                p = preds[s:e]
                r = targs[s:e]
                parts_p.append(p[1:] - p[:-1])
                parts_t.append(r[1:] - r[:-1])
            s += L

        if not parts_p:
            return L1

        dp = torch.cat(parts_p, dim=0)
        dt = torch.cat(parts_t, dim=0)
        L2 = torch.nn.functional.mse_loss(dp, dt, reduction="mean")

        return L1 + eff_alpha * L2

############################


def _compute_metrics(preds: np.ndarray, targs: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, and R² on flat NumPy arrays of predictions vs. targets.
    """
    if preds.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    diff = preds - targs
    mse  = float((diff ** 2).mean())
    mae  = float(np.abs(diff).mean())
    rmse = float(np.sqrt(mse))

    # R² = 1 − RSS/TSS
    if preds.size < 2 or np.isclose(targs.var(), 0.0):
        r2 = float("nan")
    else:
        tss = float(((targs - targs.mean()) ** 2).sum())
        rss = float((diff ** 2).sum())
        r2  = float("nan") if tss == 0.0 else 1.0 - (rss / tss)

    return {"rmse": rmse, "mae": mae, "r2": r2}


#################### 


# def _prepare_windows_and_targets_batch(x_batch: torch.Tensor,
#                                        y_signal: torch.Tensor,
#                                        seq_lengths: list,
#                                        wd_batch: torch.Tensor,
#                                        reset_state_fn,
#                                        model,
#                                        prev_day=None):
#     """
#     Build (windows_tensor, targets_tensor, prev_day) from a flattened collate.

#     Assumes:
#       - x_batch: Tensor shape (N, T, F)  where N = sum(seq_lengths)
#       - y_signal: Tensor shape (N,)
#       - seq_lengths: list/tuple of per-day window counts [W0, W1, ...] summing to N
#       - wd_batch: per-day weekday tensor (len == B)

#     Behaviour:
#       - Calls reset_state_fn once per day (using wd_batch[day_idx]) before that day's windows.
#       - Returns windows_tensor: (N, T, F), targets_tensor: (N,)
#       - If no windows present returns (None, None, prev_day)
#     """
#     device = x_batch.device

#     # Quick sanity
#     if not (isinstance(seq_lengths, (list, tuple)) and sum(seq_lengths) == x_batch.size(0)):
#         raise RuntimeError(f"_prepare_windows_and_targets_batch expects seq_lengths summing to x_batch.size(0); got seq_lengths={seq_lengths} x_batch.shape={tuple(x_batch.shape)}")

#     windows_list = []
#     targets_list = []
#     start = 0

#     for day_idx, L in enumerate(seq_lengths):
#         prev_day = reset_state_fn(model, wd_batch[day_idx], prev_day)  

#         if L <= 0:
#             start += L
#             continue

#         end = start + L
#         day_windows = x_batch[start:end]     # (L, T, F)
#         day_targets = y_signal[start:end]    # (L,)

#         windows_list.append(day_windows)
#         targets_list.append(day_targets)
#         start = end

#     if not windows_list:
#         return None, None, prev_day

#     windows_tensor = torch.cat(windows_list, dim=0).to(device).float()   # (N, T, F)
#     targets_tensor = torch.cat(targets_list, dim=0).to(device).float()   # (N,)
#     return windows_tensor, targets_tensor, prev_day


def _prepare_windows_and_targets_batch(x_batch: torch.Tensor,
                                       y_signal: torch.Tensor,
                                       seq_lengths: list,
                                       wd_batch: torch.Tensor,
                                       reset_state_fn,
                                       model,
                                       prev_day=None):
    """
    Prepare per-window tensors from a padded flattened collate (minimal, no format checks).

    Assumes pad_collate produced x_batch flattened as B blocks of W_max each:
      x_batch.shape == (B * W_max, T, F)
      y_signal.shape == (B * W_max,)
      seq_lengths is a sequence of length B with true window counts per day.
    Behaviour:
      - Calls reset_state_fn(model, wd_batch[day_idx], prev_day) once per day.
      - For day i takes the first L = seq_lengths[i] entries from block
        x_batch[i*W_max : (i+1)*W_max].
      - Returns (windows_tensor, targets_tensor, prev_day) where windows_tensor
        has shape (N, T, F) and targets_tensor has shape (N,), with N = sum(seq_lengths).
      - Minimal implementation: no layout detection or format checks.
    """
    device = x_batch.device

    B = len(seq_lengths)
    W_max = x_batch.size(0) // B

    windows_list = []
    targets_list = []

    for day_idx, L in enumerate(seq_lengths):
        prev_day = reset_state_fn(model, wd_batch[day_idx], prev_day)
        if L <= 0:
            continue
        base = day_idx * W_max
        windows_list.append(x_batch[base : base + L])   # (L, T, F)
        targets_list.append(y_signal[base : base + L])  # (L,)

    if not windows_list:
        return None, None, prev_day

    windows_tensor = torch.cat(windows_list, dim=0).to(device).float()
    targets_tensor = torch.cat(targets_list, dim=0).to(device).float()
    return windows_tensor, targets_tensor, prev_day


########################


def eval_on_loader(loader, model: nn.Module) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model validation over a loader and return metrics and predictions.
    
    Returns: (metrics_dict, preds_array, base_preds_array, targets_array)
    - Moves model to current parameter device and runs in eval mode with no grad.
    - Preserves model LSTM state across windows using prev_day and _prepare_windows_and_targets_batch.
    - Calls model(windows) -> (base, delta) and composes total = base + delta.
    - Collects CPU numpy arrays: tot_preds, base_preds, and targets for metrics.
    - Stores last_val_tot_preds, last_val_targs on the model for logging-
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.h_short = model.h_long = None
    prev_day = None
    val_base_preds, val_tot_preds, val_targs, val_lengths = [], [], [], []

    with torch.no_grad():
        for x_batch, y_signal, rc, wd, ts_list, seq_lengths in \
                tqdm(loader, desc="eval", leave=False):

            x_batch = x_batch.to(device, non_blocking=True)
            y_signal = y_signal.to(device, non_blocking=True)

            # prepare windows and per-window scalar targets (single helper)
            windows_tensor, targets_tensor, prev_day = _prepare_windows_and_targets_batch(
                x_batch, y_signal, seq_lengths, wd, _reset_states, model, prev_day
            )
            if windows_tensor is None:
                continue
   
            base_tensor, delta_tensor = model(windows_tensor)
            total_tensor = base_tensor + delta_tensor
            assert total_tensor.dim() == 2 and total_tensor.size(1) == 1
            
            # total and base preds as CPU 1D lists
            val_base_preds.extend(base_tensor.reshape(total_tensor.size(0), -1)[:, 0].detach().cpu().tolist())
            val_tot_preds.extend(total_tensor.reshape(total_tensor.size(0), -1)[:, 0].detach().cpu().tolist())
            val_targs.extend(targets_tensor.cpu().tolist())
            val_lengths.extend([L for L in seq_lengths if L > 0])
            
    val_base_preds = np.array(val_base_preds, dtype=np.float64)
    val_tot_preds = np.array(val_tot_preds, dtype=np.float64)
    val_targs = np.array(val_targs, dtype=np.float64)

    model.bl_val_mean, model.bl_val_pers = compute_baselines(np.asarray(val_targs), val_lengths)

    model.last_val_tot_preds  = torch.from_numpy(val_tot_preds).float()
    model.last_val_targs = torch.from_numpy(val_targs).float()
   
    return _compute_metrics(val_tot_preds, val_targs), _compute_metrics(val_base_preds, val_targs), val_tot_preds, val_base_preds, val_targs


###################################################################################################### 

def model_training_loop(
    model:                nn.Module,
    optimizer:            torch.optim.Optimizer,
    scheduler:            torch.optim.lr_scheduler.OneCycleLR,
    scaler:               torch.amp.GradScaler,
    train_loader,
    val_loader,
    all_features = False
) -> float:
    """
    Train the model with a baseline head and a detached residual (delta) head using AMP.
    
    Behavior summary
    - model(windows) -> (base, delta); prediction total = base + delta.
    - base is supervised by base_loss = CustomMSELoss(base, targets) smoothed.
    - delta is trained to match detached residual delta_target = (targets - base).detach(); delta_loss = CustomMSELoss(delta, delta_target) not smoothed.
    - Combined objective: total_loss = base_loss + lambda_delta * delta_loss.
    - Uses torch.amp.autocast (mixed precision) + GradScaler; unscale -> clip_grad_norm -> scaler.step/update; scheduler.step guarded to only run when optimizer.step executed.
    - Aggregates epoch metrics on total predictions and runs validation via eval_on_loader.
    
    Returns best_val (best validation RMSE).
    """
    # zero-init delta head bias so delta starts neutral
    if getattr(model, "delta_head", None) is not None:
        torch.nn.init.zeros_(model.delta_head.bias)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_feats = params.features_cols_tick
    model_hparams = params.hparams

    gc.collect()
    torch.cuda.empty_cache()
    model.to(device)

    live_plot = plots.LiveRMSEPlot()
    best_val, best_state, patience = float("inf"), None, 0
    models_dir = Path(params.models_folder)

    MSE_base = CustomMSELoss(
        alpha=float(model_hparams["ALPHA_SMOOTH"]),
        warmup_steps=int(model_hparams["WARMUP_STEPS"]),
        use_huber=bool(model_hparams["USE_HUBER"]),
        huber_delta=float(model_hparams["HUBER_DELTA"]),
    )
    
    # delta head is trained to match detached residual; keep no slope penalty there
    MSE_delta = CustomMSELoss(alpha=0.0, warmup_steps=0, use_huber=False)

    for epoch in range(1, model_hparams["MAX_EPOCHS"] + 1):

        model.train()
        model.h_short = model.h_long = None

        tr_base_preds, tr_tot_preds, tr_targs, tr_delta_preds, tr_delta_targs, tr_lengths  = [], [], [], [], [], []           
        prev_day = None

        epoch_total_loss_sum = 0.0
        epoch_base_loss_sum = 0.0
        epoch_delta_loss_sum = 0.0
        epoch_loss_count = 0
        
        epoch_start = datetime.utcnow().timestamp()
        epoch_samples = 0

        for x_batch, y_signal, rc, wd, ts_list, seq_lengths in \
                tqdm(train_loader, desc=f"Epoch {epoch} ▶ Train", leave=False):

            x_batch = x_batch.to(device, non_blocking=True)
            y_signal = y_signal.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # prepare windows and per-window scalar targets (single helper)
            windows_tensor, targets_tensor, prev_day = _prepare_windows_and_targets_batch(
                x_batch, y_signal, seq_lengths, wd, _reset_states, model, prev_day
            )
            if windows_tensor is None:
                continue

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
 
                # model now returns primitive outputs: base (B,1) and delta (B,1)
                base_tensor, delta_tensor = model(windows_tensor)  # base, delta each (B,1)
                total_tensor = base_tensor + delta_tensor 
                
                base_tensor  = base_tensor.reshape(base_tensor.size(0), -1)[:, 0]
                delta_tensor = delta_tensor.reshape(delta_tensor.size(0), -1)[:, 0]
                total_tensor = total_tensor.reshape(total_tensor.size(0), -1)[:, 0]
                assert total_tensor.shape[0] == targets_tensor.shape[0], "mismatch between model outputs and assembled targets"

                # base loss: supervise the baseline head only
                base_loss = MSE_base(base_tensor, targets_tensor, seq_lengths)
                
                # teach delta to predict the detached residual
                if model.use_delta and model_hparams["LAMBDA_DELTA"] > 0:
                    delta_target = (targets_tensor - base_tensor).detach()
                    delta_loss = MSE_delta(delta_tensor, delta_target, seq_lengths)
                    tr_delta_preds.extend(delta_tensor.detach().cpu().tolist())
                    tr_delta_targs.extend(delta_target.cpu().tolist())
                else:
                    delta_loss = torch.tensor(0.0, device=base_tensor.device)

                tr_base_preds.extend(base_tensor.detach().cpu().tolist())
                tr_tot_preds.extend(total_tensor.detach().cpu().tolist())
                tr_targs.extend(targets_tensor.cpu().tolist())
                tr_lengths.extend([L for L in seq_lengths if L > 0])
                epoch_samples += int(targets_tensor.size(0))

                # combined training objective (final scalar used for backward)
                total_loss = base_loss + model_hparams["LAMBDA_DELTA"] * delta_loss

            # Backward (AMP) with event timing and a single host sync (for logging)
            if torch.cuda.is_available():
                be0 = torch.cuda.Event(enable_timing=True); be1 = torch.cuda.Event(enable_timing=True)
                be0.record()
                scaler.scale(total_loss).backward()
                be1.record()
                torch.cuda.synchronize()    # single explicit sync to complete events
                backward_ms = be0.elapsed_time(be1)
            else:
                t0b = perf_counter()
                scaler.scale(total_loss).backward()
                backward_ms = (perf_counter() - t0b) * 1000.0
            model._last_backward_ms = backward_ms

            scaler.unscale_(optimizer)

            # Clip and step (operate on pre-built params_with_grad list to avoid repeated generator overhead)
            params_with_grad = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            if params_with_grad:
                torch.nn.utils.clip_grad_norm_(params_with_grad, model_hparams["CLIPNORM"])

            # Guard scheduler so it only advances when optimizer.step actually ran (prevents LR drift if GradScaler skipped the update)
            _called = {"v": False}
            _real = optimizer.step
            optimizer.step = lambda *a, **k: (_called.__setitem__("v", True), _real(*a, **k))[1]
            try:
                scaler.step(optimizer)
            finally:
                scaler.update()
            optimizer.step = _real
            if _called["v"]:
                scheduler.step()
  
            # After step: minimal host work (convert scalars/lists once)
            epoch_total_loss_sum += float(total_loss.detach().cpu())
            epoch_base_loss_sum += float(base_loss.detach().cpu())
            epoch_delta_loss_sum += float(delta_loss.detach().cpu())
            epoch_loss_count += 1

        # compute baselines for logging
        model.bl_tr_mean, model.bl_tr_pers = compute_baselines(np.asarray(tr_targs), tr_lengths)
        
        # Metrics & validation
        tr_tot_metrics = _compute_metrics(np.array(tr_tot_preds, dtype=np.float64), np.array(tr_targs, dtype=np.float64))
        tr_base_metrics = _compute_metrics(np.array(tr_base_preds, dtype=np.float64), np.array(tr_targs, dtype=np.float64))  
        if len(tr_delta_preds) > 0:
            tr_delta_metrics = _compute_metrics(np.array(tr_delta_preds, dtype=np.float64), np.array(tr_delta_targs, dtype=np.float64))
        else:
            tr_delta_metrics = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
        tr_tot_rmse, tr_tot_r2 = tr_tot_metrics["rmse"], tr_tot_metrics["r2"]
        
        val_tot_metrics, val_base_metrics, val_tot_preds, val_base_preds, val_targs = eval_on_loader(val_loader, model)
        val_tot_rmse, val_tot_r2 = val_tot_metrics["rmse"], val_tot_metrics["r2"]

        avg_loss = epoch_total_loss_sum / max(1, epoch_loss_count)
        avg_base_loss = epoch_base_loss_sum / max(1, epoch_loss_count)
        avg_delta_loss = epoch_delta_loss_sum / max(1, epoch_loss_count)

        epoch_elapsed = datetime.utcnow().timestamp() - epoch_start
        model._last_epoch_elapsed = float(epoch_elapsed)
        model._last_epoch_samples = int(epoch_samples)
    
        models_core.log_epoch_summary(
            epoch,
            model,
            optimizer,
            tr_tot_metrics=tr_tot_metrics,
            tr_base_metrics=tr_base_metrics,
            tr_delta_metrics=tr_delta_metrics,
            val_tot_metrics=val_tot_metrics,
            val_base_metrics=val_base_metrics,
            val_tot_preds=val_tot_preds,
            val_base_preds=val_base_preds,
            val_targs=val_targs,
            avg_base_loss=avg_base_loss,
            avg_delta_loss=avg_delta_loss,
            log_file=params.log_file,
            hparams=model_hparams,
        )
        live_plot.update(tr_tot_rmse, val_tot_rmse)

        best_val, improved, *_ = models_core.maybe_save_chkpt(
            models_dir, model, val_tot_rmse, best_val, model_feats, model_hparams,
            {"rmse": tr_tot_rmse}, {"rmse": val_tot_rmse}, live_plot
        )

        model._last_epoch_checkpoint = bool(improved)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}  "
            f"TRAIN→ RMSE={tr_tot_rmse:.5f}, R²={tr_tot_r2:.3f} |  "
            f"VALID→ RMSE={val_tot_rmse:.5f}, R²={val_tot_r2:.3f} |  "
            f"lr={current_lr:.2e} |  "
            f"loss={avg_loss:.5e} |  "
            f"improved={bool(improved)}"
        )

        if improved:
            best_state, patience = {
                k: v.cpu() for k, v in model.state_dict().items()
            }, 0
        else:
            patience += 1
            if patience >= model_hparams["EARLY_STOP_PATIENCE"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        models_core.save_final_chkpt(
            models_dir, best_state, best_val, model_feats, model_hparams, tr_tot_metrics, val_tot_metrics, live_plot, 
            suffix="_all" if all_features else "_fin"
        )

    for attr in ("_last_epoch_elapsed","_last_epoch_samples","_last_epoch_checkpoint","_first_batch_snapshot"):
        try:
            delattr(model, attr)
        except Exception:
            pass

    return best_val


######################################################################################################


# def add_preds_and_split(
#     df: pd.DataFrame,
#     train_preds: np.ndarray,
#     val_preds:   np.ndarray,
#     test_preds:  np.ndarray,
#     end_times_tr:  np.ndarray,
#     end_times_val: np.ndarray,
#     end_times_te:  np.ndarray
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Add per-window predictions into a minute-bar DataFrame and return
#     (train+val, test) subsets where pred_signal exists.

#     Generic: stamps predictions onto the minute bars, aligns timezones,
#     drops duplicate timestamps keeping the last, and returns two DataFrames.
#   """
#     # Validate lengths for train/val/test predictions vs their end times
#     for name, preds, times in (
#           ("train", train_preds, end_times_tr),
#           ("val",   val_preds,   end_times_val),
#           ("test",  test_preds,  end_times_te),
#     ):
#         # ensure each preds array matches its timestamps length
#         if len(np.asarray(preds).ravel()) != len(np.asarray(times)):
#             raise ValueError(f"{name} preds length != times length: {len(preds)} != {len(times)}")

#     # Work on a copy of the input minute-bar DataFrame to avoid mutating caller data
#     df = df.copy()
#     # initialize prediction column with NaN
#     df["pred_signal"] = np.nan
#     # compute synthetic ask/bid from close_raw and configured spread
#     df['ask'] = df['close_raw'] * (1 + params.bidask_spread_pct/100)
#     df['bid'] = df['close_raw'] * (1 - params.bidask_spread_pct/100)

#     # capture timezone of the minute-bar index (may be None)
#     tz_df = getattr(df.index, "tz", None)

#     # helper: build a timezone-aligned Series from preds + times
#     def _series(preds, times):
#         # flatten preds to 1D array
#         arr = np.asarray(preds).ravel()
#         # parse timestamps into DatetimeIndex
#         idx = pd.to_datetime(times)
#         # align timezone of parsed times to df index timezone if needed
#         if tz_df is not None and getattr(idx, "tz", None) is None:
#             idx = idx.tz_localize(tz_df)
#         elif tz_df is None and getattr(idx, "tz", None) is not None:
#             idx = idx.tz_convert(None)
#         # build series with datetime index
#         s = pd.Series(arr, index=pd.DatetimeIndex(idx))
#         # if duplicate timestamps exist, keep the last value (most recent window)
#         if s.index.has_duplicates:
#             s = s[~s.index.duplicated(keep="last")]
#         return s

#     # build series for train, val and test (no side-effects)
#     s_tr  = _series(train_preds, end_times_tr)
#     s_val = _series(val_preds,   end_times_val)
#     s_te  = _series(test_preds,  end_times_te)

#     # compute total work so the progress bar reflects the whole stamping phase
#     total = sum(len(s.index.intersection(df.index)) for s in (s_tr, s_val, s_te))
#     if total:
#         chunk = 1000
#         bar_fmt = "{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
#         with tqdm(total=total, desc="Applying prediction series", leave=True, dynamic_ncols=True, bar_format=bar_fmt) as pbar:
#             for s in (s_tr, s_val, s_te):
#                 common = s.index.intersection(df.index)
#                 if common.empty:
#                     continue
#                 common_list = list(common)
#                 for i in range(0, len(common_list), chunk):
#                     chunk_idx = common_list[i : i + chunk]
#                     df.loc[chunk_idx, "pred_signal"] = s.loc[chunk_idx].values
#                     pbar.update(len(chunk_idx))

#     # compute indices for train+val and test that exist in the minute-bar index
#     idx_trval = s_tr.index.union(s_val.index).intersection(df.index)
#     idx_te    = s_te.index.intersection(df.index)

#     # slice out rows that actually have a pred_signal and return
#     df_trainval = df.loc[idx_trval].dropna(subset=["pred_signal"])
#     df_test     = df.loc[idx_te].dropna(subset=["pred_signal"])

#     return df_trainval, df_test





def add_preds_and_split(
    df: pd.DataFrame,
    train_preds: np.ndarray,
    val_preds:   np.ndarray,
    test_preds:  np.ndarray,
    end_times_tr:  np.ndarray,
    end_times_val: np.ndarray,
    end_times_te:  np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stamp per-window predictions onto minute bars and return (train+val, test) rows with pred_signal.
    Fail-fast checks: matching lengths, no duplicate prediction timestamps, and all timestamps present in df.index.
    Adds simple synthetic bid/ask from close_raw.
    Assumes tz-consistent or tz-naive timestamps.
    """
    # length checks
    for name, preds, times in (
        ("train", train_preds, end_times_tr),
        ("val",   val_preds,   end_times_val),
        ("test",  test_preds,  end_times_te),
    ):
        if len(np.asarray(preds).ravel()) != len(np.asarray(times)):
            raise ValueError(f"{name} preds length != times length: {len(preds)} != {len(times)}")

    df = df.copy()
    df["pred_signal"] = np.nan
    df["ask"] = df["close_raw"] * (1 + params.bidask_spread_pct / 100.0)
    df["bid"] = df["close_raw"] * (1 - params.bidask_spread_pct / 100.0)

    def _series(preds, times, name):
        idx = pd.to_datetime(times)
        if pd.Index(idx).has_duplicates:
            dup = pd.Index(idx)[pd.Index(idx).duplicated(keep=False)][:5]
            raise ValueError(f"{name} timestamps contain duplicates, e.g. {dup.tolist()}")
        return pd.Series(np.asarray(preds).ravel(), index=idx)

    # build series and validate
    s_tr  = _series(train_preds, end_times_tr, "train")
    s_val = _series(val_preds,   end_times_val, "val")
    s_te  = _series(test_preds,  end_times_te,  "test")

    # stamp predictions with a simple progress bar
    for desc, s in (("train", s_tr), ("val", s_val), ("test", s_te)):
        missing = s.index.difference(df.index)
        if not missing.empty:
            raise KeyError(f"{desc} timestamps not in df.index; e.g. {missing[:5].tolist()} (total {len(missing)})")
        for ts in tqdm(s.index, desc=f"Stamping {desc}", unit="ts"):
            df.at[ts, "pred_signal"] = s.at[ts]

    # split outputs where pred_signal exists
    idx_trval = s_tr.index.union(s_val.index)
    idx_te    = s_te.index
    df_trainval = df.loc[idx_trval].dropna(subset=["pred_signal"])
    df_test     = df.loc[idx_te].dropna(subset=["pred_signal"])

    return df_trainval, df_test