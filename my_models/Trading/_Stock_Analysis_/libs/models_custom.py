from libs import plots, params, models_core

from typing import Optional, Dict, Tuple, List, Sequence, Union, Any

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


######################################################################################################

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


class ModelClass(nn.Module):
    """
    Initialize a flexible, stateful sequence regression model composed of optional
    Conv1d/TCN/Short-LSTM/Transformer/Projection/Long-LSTM blocks with a
    small MLP prediction head and an optional delta correction head.

    Architecture summary
    - Input shape accepted: (B, T, F) or with extra leading dims (collapsed to (B, T, F)).
    - Stage 0: Optional Conv1d + BatchNorm1d + ReLU applied across the feature dimension.
    - Stage 1: Optional TCN implemented as stacked Conv1d + BatchNorm + ReLU blocks.
    - Stage 2: Optional short (bidirectional) LSTM producing `short_units` features.
      * Followed by LayerNorm and Dropout.
      * Internal LSTM states held in self.h_short / self.c_short (stateful across calls).
    - Stage 3: Optional Transformer encoder stack (positional encoding + linear feature_proj).
    - Stage 4: Linear projection short->long dimension (short2long) + LayerNorm + Dropout.
    - Stage 5: Optional long (bidirectional) LSTM producing `long_units` features.
      * Followed by LayerNorm and Dropout.
      * Internal LSTM states held in self.h_long / self.c_long (stateful across calls).
    - Stage 6: Flattening strategy (one of "flatten", "last", "pool") producing a flat feature
      vector of dimension `flat_dim` which equals `window_len*long_units` for "flatten" else `long_units`.
      * LayerNorm applied to the flattened vector.
    - Head(s):
      * head_flat: MLP (weight_norm Linear -> ReLU -> weight_norm Linear) producing a 1-D baseline.
      * baseline_lin: a small linear baseline per-feature (kept for legacy/aux uses).
      * delta_head: optional linear head producing corrective residual; when disabled returns Identity
        and training sets delta to zero.
    - Returns from forward: (final, delta, base) each shaped (B, 1).
    """
    def __init__(
        self,
        n_feats: int,
        short_units: int,
        long_units: int,
        dropout_short: float,
        dropout_long: float,
        pred_hidden: int,
        window_len: int,
        use_conv: bool,
        use_tcn: bool,
        use_short_lstm: bool,
        use_transformer: bool,
        use_long_lstm: bool,
        use_delta: bool,
        flatten_mode: str,
    ):
        super().__init__()
        self.window_len      = window_len
        self.short_units     = short_units
        self.long_units      = long_units
        self.use_conv        = use_conv
        self.use_tcn         = use_tcn
        self.use_short_lstm  = use_short_lstm
        self.use_transformer = use_transformer
        self.use_long_lstm   = use_long_lstm
        self.use_delta       = use_delta

        # 0) Conv1d + BatchNorm1d or Identity
        if use_conv:
            conv_k        = params.hparams["CONV_K"]
            conv_dilation = params.hparams["CONV_DILATION"]
            padding       = (conv_k // 2) * conv_dilation
            self.conv = nn.Conv1d(n_feats, n_feats,
                                  kernel_size=conv_k,
                                  dilation=conv_dilation,
                                  padding=padding)
            self.bn   = nn.BatchNorm1d(n_feats)
        else:
            self.conv = nn.Identity()
            self.bn   = nn.Identity()
        self.relu = nn.ReLU()

        # 1) TCN or Identity
        if use_tcn:
            tcn_layers = params.hparams["TCN_LAYERS"]
            tcn_kernel = params.hparams["TCN_KERNEL"]
            blocks, in_ch = [], n_feats
            for i in range(tcn_layers):
                dilation = 2 ** i
                padding  = (tcn_kernel // 2) * dilation
                blocks += [
                    nn.Conv1d(in_ch, n_feats, tcn_kernel,
                              dilation=dilation, padding=padding),
                    nn.BatchNorm1d(n_feats),
                    nn.ReLU()
                ]
                in_ch = n_feats
            self.tcn = nn.Sequential(*blocks)
        else:
            self.tcn = nn.Identity()

        # 2) Short Bi-LSTM or Identity
        if use_short_lstm:
            assert short_units % 2 == 0
            self.short_lstm = nn.LSTM(
                input_size=n_feats,
                hidden_size=short_units // 2,
                batch_first=True,
                bidirectional=True
            )
            self.ln_short = nn.LayerNorm(short_units)
            self.do_short = nn.Dropout(dropout_short)
        else:
            self.short_lstm = None
            self.ln_short   = nn.Identity()
            self.do_short   = nn.Identity()
        self.h_short = None
        self.c_short = None

        # 3) Transformer w/ PositionalEncoding + optional feature projection
        if use_transformer:
            d_model = short_units
            heads   = params.hparams["TRANSFORMER_HEADS"]
            layers  = params.hparams["TRANSFORMER_LAYERS"]
            ff_mult = params.hparams["TRANSFORMER_FF_MULT"]
            ff_dim  = d_model * ff_mult

            in_proj_dim = short_units if use_short_lstm else n_feats
            self.feature_proj = nn.Linear(in_proj_dim, d_model)

            self.pos_enc    = PositionalEncoding(d_model, dropout_short, window_len)
            encoder_layer   = nn.TransformerEncoderLayer(
                d_model         = d_model,
                nhead           = heads,
                dim_feedforward = ff_dim,
                dropout         = dropout_short,
                # batch_first     = True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers
            )
        else:
            self.feature_proj = nn.Identity()
            self.pos_enc      = nn.Identity()
            self.transformer  = nn.Identity()

        # 4) Projection → LayerNorm → Dropout
        proj_in     = short_units if use_short_lstm else n_feats
        self.short2long = nn.Linear(proj_in, long_units)
        self.ln_proj    = nn.LayerNorm(long_units)
        self.do_proj    = nn.Dropout(dropout_long)

        # 5) Long Bi-LSTM or Identity
        if use_long_lstm:
            assert long_units % 2 == 0
            self.long_lstm = nn.LSTM(
                input_size=long_units,
                hidden_size=long_units // 2,
                batch_first=True,
                bidirectional=True
            )
            self.ln_long = nn.LayerNorm(long_units)
            self.do_long = nn.Dropout(dropout_long)
        else:
            self.long_lstm = None
            self.ln_long   = nn.Identity()
            self.do_long   = nn.Identity()
        self.h_long = None
        self.c_long = None

        # 6) Flatten/Last/Pool + plain MLP head
        assert flatten_mode in ("flatten", "last", "pool")
        self.flatten_mode = flatten_mode
        flat_dim = (window_len * long_units
                    if flatten_mode == "flatten"
                    else long_units)

        self.ln_flat   = nn.LayerNorm(flat_dim)

        # 7) Delta baseline vs features predictions head
        self.head_flat = nn.Sequential(
            weight_norm(nn.Linear(flat_dim, pred_hidden)),
            nn.ReLU(),
            weight_norm(nn.Linear(pred_hidden, 1))
        )
        self.baseline_lin = nn.Linear(n_feats, 1)
        if self.use_delta:
            self.delta_head = weight_norm(nn.Linear(flat_dim, 1))
        else:
            self.delta_head = nn.Identity()

    def reset_short(self):
        if self.h_short is not None:
            bsz, dev = self.h_short.size(1), self.h_short.device
            self.h_short, self.c_short = _allocate_lstm_states(
                bsz, self.short_units // 2, True, dev
            )

    def reset_long(self):
        if self.h_long is not None:
            bsz, dev = self.h_long.size(1), self.h_long.device
            self.h_long, self.c_long = _allocate_lstm_states(
                bsz, self.long_units // 2, True, dev
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collapse extra leading dims to (batch, time_steps, features)
        if x.dim() > 3:
            *lead, T, F = x.shape
            x = x.view(-1, T, F)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, time_steps, _ = x.shape

        # 0) Conv + BN + ReLU
        xc = x.transpose(1, 2)
        xc = self.conv(xc); xc = self.bn(xc); xc = self.relu(xc)
        x  = xc.transpose(1, 2)

        # 1) TCN
        tcn_out = self.tcn(x.transpose(1, 2)).transpose(1, 2)

        # 2) Short LSTM
        if self.use_short_lstm:
            if self.h_short is None or self.h_short.size(1) != batch_size:
                self.h_short, self.c_short = _allocate_lstm_states(
                    batch_size, self.short_units // 2, True, x.device
                )
            out_s, (h_s, c_s) = self.short_lstm(
                tcn_out, (self.h_short, self.c_short)
            )
            self.h_short, self.c_short = h_s.detach(), c_s.detach()
            out_s = self.ln_short(out_s); out_s = self.do_short(out_s)
        else:
            out_s = tcn_out

        # 3) Transformer
        tr_in  = self.feature_proj(out_s)
        tr_in  = self.pos_enc(tr_in)
        tr_out = self.transformer(tr_in.transpose(0, 1).contiguous())
        out_t  = tr_out.transpose(0, 1).contiguous()

        # 4) Projection
        out_p = self.short2long(out_t)
        out_p = self.ln_proj(out_p); out_p = self.do_proj(out_p)

        # 5) Long LSTM
        if self.use_long_lstm:
            if self.h_long is None or self.h_long.size(1) != batch_size:
                self.h_long, self.c_long = _allocate_lstm_states(
                    batch_size, self.long_units // 2, True, out_p.device
                )
            out_l, (h_l, c_l) = self.long_lstm(
                out_p, (self.h_long, self.c_long)
            )
            self.h_long, self.c_long = h_l.detach(), c_l.detach()
            out_l = self.ln_long(out_l); out_l = self.do_long(out_l)
        else:
            out_l = out_p

        # 6) Flatten/Last/Pool + MLP head
        if self.flatten_mode == "flatten":
            flat = out_l.reshape(batch_size, -1)
        elif self.flatten_mode == "last":
            flat = out_l[:, -1, :]
        else:  # "pool"
            flat = out_l.mean(dim=1)

        norm_flat = self.ln_flat(flat)
        main_out = self.head_flat(norm_flat)
        assert main_out.ndim == 2 and main_out.shape[1] == 1, "head_flat must return (B,1)"

        # 7) Delta baseline vs features predictions head
        base = main_out.squeeze(-1)                           # baseline from MLP on norm_flat (B,)
        if self.use_delta:
            delta = self.delta_head(norm_flat).squeeze(-1)    # correction from same features (B,)
        else:
            delta = torch.zeros(base.shape, device=base.device, dtype=base.dtype)
        final = base + delta                                  # (B,)
        
        # return final, delta, base each as (B,1)
        assert final.ndim == 1 and delta.ndim == 1 and base.ndim == 1, "expected flat vectors before unsqueeze"
        return final.unsqueeze(-1), delta.unsqueeze(-1), base.unsqueeze(-1)


######################################################################################################


class SmoothMSELoss(nn.Module):
    """
    Combined level + slope MSE: standard MSE mean + α*MSE of one‐step diffs.
    """
    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        # final‐step head
        assert preds.shape == targs.shape, "preds and targs must have identical shapes"
        L1 = torch.nn.functional.mse_loss(preds, targs, reduction="mean")
        if self.alpha <= 0:
            return L1
        dp = preds[1:] - preds[:-1]
        dt =  targs[1:] - targs[:-1]
        L2 = torch.nn.functional.mse_loss(dp, dt, reduction="mean")
        return L1 + self.alpha * L2

############################


def compute_baselines(loader) -> tuple[float, float]:
    """
    Compute two static, one-step-per-window baselines over all windows in `loader`:
      1) mean_rmse       – RMSE if you predict the per-window mean mu = mean(y_sig[:L]) for the target (last element)
      2) persistence_rmse – RMSE if you predict the last-seen value y_{t-1} as the next value y_t
    """
    mean_errors = []
    last_deltas = []
    
    for x_pad, y_sig, _y_bin, _y_ret, _y_ter, _rc, wd, ts_list, lengths in loader:
        for i, L in enumerate(lengths):
            if L < 1:
                continue
            arr = y_sig[i, :L].view(-1).cpu().numpy().astype(float)
            target = arr[-1]
            mu = arr.mean()
            mean_errors.append((target - mu) ** 2)
            if L > 1:
                last_deltas.append((target - arr[-2]) ** 2)
    
    if mean_errors:
        mean_rmse = float(np.sqrt(np.mean(mean_errors)))
    else:
        mean_rmse = float("nan")
    
    if last_deltas:
        persistence_rmse = float(np.sqrt(np.mean(last_deltas)))
    else:
        persistence_rmse = float("nan")
        
    return mean_rmse, persistence_rmse

###############


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
    day = int(wd_i.item())

    # daily reset if the day changed
    if prev_day is None or day != prev_day:
        model.reset_short()

        # weekly reset if we've wrapped past the end of the week
        if prev_day is not None and day < prev_day:
            model.reset_long()

    return day


###################################################################################################### 

def _prepare_windows_and_targets_batch(x_batch: torch.Tensor,
                                       y_signal: torch.Tensor,
                                       seq_lengths: torch.Tensor,
                                       wd_batch: torch.Tensor,
                                       reset_state_fn,
                                       model,
                                       prev_day=None):
    """
    Build windows_tensor and per-window scalar targets_tensor for one batch.

    Returns (windows_tensor, targets_tensor, prev_day) where:
      - windows_tensor: torch.Tensor shape (num_windows, T_i, F) on x_batch.device
      - targets_tensor: torch.Tensor shape (num_windows,) on x_batch.device (one scalar last-step target per window)
      - prev_day: updated prev_day from reset_state_fn calls

    Caller must handle the (None, None, prev_day) case when no windows in batch.
    """
    device = x_batch.device
    batch_size = x_batch.size(0)

    windows_list = []
    targets_list = []

    for example_idx in range(batch_size):
        # call reset_state_fn exactly as your loops do; preserve prev_day update
        try:
            prev_day = reset_state_fn(model, wd_batch[example_idx], prev_day)
        except TypeError:
            # fallback if reset_state_fn uses a different signature wrapper
            prev_day = reset_state_fn(model, example_idx, prev_day)

        L = int(seq_lengths[example_idx])
        if L == 0:
            continue

        windows_list.append(x_batch[example_idx, :L])
        # # collect last-step scalar target and preserve shape (1,) for torch.cat
        # targets_list.append(y_signal[example_idx, :L][-1].reshape(1))
        # collecting one target per window
        targets_list.append(y_signal[example_idx, :L].reshape(-1))

    if not windows_list:
        return None, None, prev_day

    windows_tensor = torch.cat(windows_list, dim=0).to(device)
    targets_tensor = torch.cat(targets_list, dim=0).to(device)

    return windows_tensor, targets_tensor, prev_day


##############################################


def eval_on_loader(loader, model: nn.Module, clamp_preds: bool = True) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a stateful windowed model on a DataLoader and return metrics plus CPU arrays.

    Purpose
    - Run a deterministic evaluation pass over `loader` and produce aligned per-window
      predictions, baseline predictions, and scalar targets for metric calculation.

    Behavior and guarantees
    - Puts model on the device of its parameters and sets model to eval mode with
      torch.no_grad. Resets stateful hidden handles by setting model.h_short and
      model.h_long to None and preserves `prev_day` semantics across batches.
    - Uses the shared helper `_prepare_windows_and_targets_batch` to assemble a
      windows tensor and a per-window 1-D targets tensor; if that helper returns
      (None, None, prev_day) the batch is skipped.
    - Calls model(windows_tensor) and accepts outputs that include at least the
      first three tensors (final, delta, base) in their expected shapes; the
      implementation extracts the first three returned values and asserts that
      `final` has shape (W, 1).
    - Aggregates per-window final predictions, per-window baseline predictions,
      and per-window scalar targets in the same order into CPU numpy arrays.
    - Stores the last-eval tensors on the model as float tensors named
      `last_val_preds`, `last_val_targs`, and `last_val_base` for easy inspection.
    - Returns a 4-tuple: `(metrics_dict, preds_np, base_preds_np, targets_np)` where
      `metrics_dict` is the result of `_compute_metrics(preds_np, targets_np)`.
    """
    device = next(model.parameters()).device
    model.to(device).eval()
    model.h_short = model.h_long = None
    prev_day = None
    all_preds, all_targs, all_base = [], [], []

    with torch.no_grad():
        for x_batch, y_signal, y_bin, y_ret, y_ter, rc, wd, ts_list, seq_lengths in \
                tqdm(loader, desc="eval", leave=False):

            x_batch = x_batch.to(device, non_blocking=True)
            y_signal = y_signal.to(device, non_blocking=True)
            wd = wd.to(device, non_blocking=True)

            # prepare windows and per-window scalar targets (single helper)
            windows_tensor, targets_tensor, prev_day = _prepare_windows_and_targets_batch(
                x_batch, y_signal, seq_lengths, wd, _reset_states, model, prev_day
            )
            if windows_tensor is None:
                continue
   
            final_tensor, delta_tensor, base_tensor = model(windows_tensor)[:3]
            assert final_tensor.dim() == 2 and final_tensor.size(1) == 1
            
            # preds and base as CPU 1D lists
            preds_batch = final_tensor.reshape(final_tensor.size(0), -1)[:, 0].detach().cpu()
            base_batch  = base_tensor.reshape(base_tensor.size(0), -1)[:, 0].detach().cpu()
            
            all_preds.extend(preds_batch.tolist())
            all_targs.extend(targets_tensor.cpu().tolist())
            all_base.extend(base_batch.tolist())

    preds = np.array(all_preds, dtype=float)
    targs = np.array(all_targs, dtype=float)
    base_preds = np.array(all_base, dtype=float)

    model.last_val_preds = torch.from_numpy(preds).float()
    model.last_val_targs = torch.from_numpy(targs).float()
    model.last_val_base  = torch.from_numpy(base_preds).float()

    return _compute_metrics(preds, targs), preds, base_preds, targs


###################################################################################################### 


def model_training_loop(
    model:                nn.Module,
    optimizer:            torch.optim.Optimizer,
    scheduler:            torch.optim.lr_scheduler.OneCycleLR,
    scaler:               torch.amp.GradScaler,
    train_loader,
    val_loader,
    *,
    max_epochs:          int,
    early_stop_patience: int,
    clipnorm:            float,
    alpha_smooth:        float,
    lambda_delta:        float,
    top_k:               int
) -> float:
    """
    Train a stateful, windowed regression model using per-window batching and AMP.

    Key responsibilities
    - Iterates epochs and batches, preserving model state across windows via `prev_day`
      and calling the shared helper `_prepare_windows_and_targets_batch` to build
      the per-window input tensor and matching per-window scalar targets.
    - Uses automatic mixed precision (torch.amp.autocast) and GradScaler for safe AMP
      training and handles the case where scaler may skip optimizer.step (guarded scheduler.step).
    - Computes three losses per forward:
        * main_loss  — supervision on the model's baseline head (L2-ish via SmoothMSELoss).
        * aux_loss   — L2 teaching signal for the delta head (delta_pred vs targets - baseline).
        * final_loss — MSE-like loss on the model's final prediction (true objective for reporting).
      Combined backprop uses loss = final_loss + lambda_delta * aux_loss.
    - Performs gradient clipping (clip_grad_norm_) on parameters with gradients,
      steps the optimizer through the scaler (scaler.step/scaler.update), and steps the scheduler
      only when an optimizer update actually occurred (wraps optimizer.step).
    - Records lightweight diagnostic snapshot on first batch to verify shapes/grad presence.
    - Aggregates epoch-level metrics: average losses, RMSE/MAE/R2 (via _compute_metrics),
      and runs validation with `eval_on_loader` which returns aligned per-window preds/base/targets.
    - Calls models_core.log_epoch_summary and models_core.maybe_save_chkpt; if best model found,
      stores best_state and saves final checkpoint at the end.
    - Cleans a few ephemeral model attributes before returning.
    """
    device = next(model.parameters()).device
    model.to(device)

    expected_total = len(train_loader) * max_epochs
    if hasattr(scheduler, "_total_steps") and scheduler._total_steps != expected_total:
        raise RuntimeError(f"Scheduler total_steps mismatch: scheduler={scheduler._total_steps} expected={expected_total}")

    base_tr_mean, base_tr_pers = compute_baselines(train_loader)
    base_vl_mean, base_vl_pers = compute_baselines(val_loader)

    smooth_loss = SmoothMSELoss(alpha_smooth)
    live_plot = plots.LiveRMSEPlot()
    best_val, best_state, patience = float("inf"), None, 0

    models_dir = Path(params.models_folder)
    first_snapshot_captured = False

    for epoch in range(1, max_epochs + 1):

        model.train()
        model.h_short = model.h_long = None

        train_preds, train_targs = [], []
        prev_day = None

        epoch_loss_sum = 0.0
        epoch_main_loss_sum = 0.0
        epoch_aux_loss_sum = 0.0
        epoch_final_loss_sum = 0.0 
        epoch_loss_count = 0
        
        epoch_start = datetime.utcnow().timestamp()
        epoch_samples = 0

        for x_batch, y_signal, y_bin, y_ret, y_ter, rc, wd, ts_list, seq_lengths in \
                tqdm(train_loader, desc=f"Epoch {epoch} ▶ Train", leave=False):

            x_batch = x_batch.to(device, non_blocking=True)
            y_signal = y_signal.to(device, non_blocking=True)
            wd = wd.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # prepare windows and per-window scalar targets (single helper)
            windows_tensor, targets_tensor, prev_day = _prepare_windows_and_targets_batch(
                x_batch, y_signal, seq_lengths, wd, _reset_states, model, prev_day
            )
            if windows_tensor is None:
                continue

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                final_tensor, delta_tensor, base_tensor = model(windows_tensor) # each shaped (B,1)

                # flatten to vectors (B,)
                preds      = final_tensor.reshape(final_tensor.size(0), -1)[:, 0]
                delta_pred  = delta_tensor.reshape(delta_tensor.size(0), -1)[:, 0]
                base_batch  = base_tensor.reshape(base_tensor.size(0), -1)[:, 0]
                assert preds.shape[0] == targets_tensor.shape[0], "mismatch between model outputs and assembled targets"
            
                # MAIN: baseline supervised (baseline_head output)
                main_loss = smooth_loss(base_batch, targets_tensor)
                # AUX: teach delta to match residual (targets - baseline)
                delta_target = targets_tensor - base_batch
                aux_loss = (delta_pred - delta_target).pow(2).mean()
                # final metric for reporting (true objective)
                final_loss = smooth_loss(preds, targets_tensor)
                # Combined objective used for backprop
                loss = final_loss + lambda_delta * aux_loss
        
            # Backward (AMP) with event timing and a single host sync
            if torch.cuda.is_available():
                be0 = torch.cuda.Event(enable_timing=True); be1 = torch.cuda.Event(enable_timing=True)
                be0.record()
                scaler.scale(loss).backward()
                be1.record()
                torch.cuda.synchronize()    # single explicit sync to complete events
                backward_ms = be0.elapsed_time(be1)
            else:
                t0b = perf_counter()
                scaler.scale(loss).backward()
                backward_ms = (perf_counter() - t0b) * 1000.0
            model._last_backward_ms = backward_ms

            scaler.unscale_(optimizer)

            # lightweight first-batch snapshot (non-blocking, cheap)
            if (not first_snapshot_captured) and (not hasattr(model, "_first_batch_snapshot")):
                try:
                    raw_shape = tuple(final_tensor.shape) if isinstance(final_tensor, torch.Tensor) else None
            
                    group_nonzero_counts = [
                        sum(1 for p in g.get("params", []) if p.grad is not None)
                        for g in optimizer.param_groups
                    ]
                    backbone_has = any(p.grad is not None for n, p in model.named_parameters() if n.startswith("short") or n.startswith("long"))
                    head_has = any(p.grad is not None for n, p in model.named_parameters() if n.startswith("pred") or "pred" in n)
            
                    model._first_batch_snapshot = {
                        "raw_reg_shape": raw_shape,
                        "group_nonzero_counts": group_nonzero_counts,
                        "grads": {"backbone": bool(backbone_has), "head": bool(head_has)},
                    }
                except Exception:
                    pass
                first_snapshot_captured = True


            # Clip and step (operate on pre-built params_with_grad list to avoid repeated generator overhead)
            params_with_grad = [p for p in model.parameters() if p.grad is not None]
            if params_with_grad:
                torch.nn.utils.clip_grad_norm_(params_with_grad, clipnorm)

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
            epoch_loss_sum += float(loss.detach().cpu())
            epoch_main_loss_sum += float(main_loss.detach().cpu())
            epoch_aux_loss_sum += float(aux_loss.detach().cpu()) 
            epoch_final_loss_sum += float(final_loss.detach().cpu())
            epoch_loss_count += 1

            train_preds.extend(preds.detach().cpu().tolist())
            train_targs.extend(targets_tensor.cpu().tolist())
            epoch_samples += int(targets_tensor.size(0))

        # Metrics & validation
        tr_metrics = _compute_metrics(
            np.array(train_preds, dtype=float),
            np.array(train_targs, dtype=float),
        )
        vl_metrics, vl_preds, vl_base_preds, vl_targs = eval_on_loader(val_loader, model)

        avg_loss = epoch_loss_sum / max(1, epoch_loss_count)
        avg_main_loss = epoch_main_loss_sum / max(1, epoch_loss_count)
        avg_aux_loss  = epoch_aux_loss_sum  / max(1, epoch_loss_count) if params.hparams['USE_DELTA'] else 0.0

        epoch_elapsed = datetime.utcnow().timestamp() - epoch_start
        model._last_epoch_elapsed = float(epoch_elapsed)
        model._last_epoch_samples = int(epoch_samples)

        tr_rmse, tr_mae, tr_r2 = tr_metrics["rmse"], tr_metrics["mae"], tr_metrics["r2"]
        vl_rmse, vl_mae, vl_r2 = vl_metrics["rmse"], vl_metrics["mae"], vl_metrics["r2"]

        models_core.log_epoch_summary(
            epoch,
            model,
            optimizer,
            train_metrics=tr_metrics,
            val_metrics=vl_metrics,
            val_preds=vl_preds,
            val_base_preds=vl_base_preds,
            val_targets=vl_targs,
            base_tr_mean=base_tr_mean,
            base_tr_pers=base_tr_pers,
            base_vl_mean=base_vl_mean,
            base_vl_pers=base_vl_pers,
            avg_main_loss=avg_main_loss,
            avg_aux_loss=avg_aux_loss,
            log_file=params.log_file,
            top_k=top_k,
            hparams=params.hparams,
        )
        live_plot.update(tr_rmse, vl_rmse)

        best_val, improved, *_ = models_core.maybe_save_chkpt(
            models_dir, model, vl_rmse, best_val,
            {"rmse": tr_rmse}, {"rmse": vl_rmse},
            live_plot, params
        )

        model._last_epoch_checkpoint = bool(improved)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}  "
            f"TRAIN→ RMSE={tr_rmse:.5f}, R²={tr_r2:.3f} |  "
            f"VALID→ RMSE={vl_rmse:.5f}, R²={vl_r2:.3f} |  "
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
            if patience >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        models_core.save_final_chkpt(
            models_dir, best_state, best_val, params,
            tr_metrics, vl_metrics, live_plot, suffix="_fin"
        )

    for attr in ("_last_epoch_elapsed","_last_epoch_samples","_last_epoch_checkpoint","_first_batch_snapshot"):
        try:
            delattr(model, attr)
        except Exception:
            pass

    return best_val