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
    Multi-backbone sequence model with optional short/long LSTMs, transformer, and a baseline+delta head.
    
    Key behavior
    - Input: (batch, time, features) windows.
    - Options: conv, TCN, short/long Bi-LSTM, transformer; flatten modes: flatten/last/pool.
    - head_flat: MLP producing baseline prediction (B,1).
    - delta_head (optional): single-layer linear correction on same flattened features (B,1).
    - forward returns (base, delta) each shaped (B,1); final prediction = base + delta.
    - Statefulness: maintains detached LSTM states h_short/h_long across batches; reset helpers provided.
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
        if self.use_delta:
            self.delta_head = weight_norm(nn.Linear(flat_dim, 1))
        else:
            self.delta_head = None # nn.Identity()

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
        base_out = self.head_flat(norm_flat)
        assert base_out.ndim == 2 and base_out.shape[1] == 1, "head_flat must return (B,1)"

        # 7) Delta baseline vs features predictions head
        base = base_out.squeeze(-1)                           # baseline from MLP on norm_flat (B,)
        if self.use_delta:
            delta = self.delta_head(norm_flat).squeeze(-1)    # correction from same features (B,)
        else:
            delta = torch.zeros(base.shape, device=base.device, dtype=base.dtype)

        return base.unsqueeze(-1), delta.unsqueeze(-1)


######################################################################################################



def compute_baselines(loader) -> tuple[float, float]:
    """
    Compute mean_rmse and persistence_rmse assuming loader yields flattened batches.

    Expects each batch from loader to be:
      (x_flat, ysig_flat, ybin_flat, yret_flat, yter_flat, rc_flat, wd_expanded, ts_list, lengths)
    where:
      - ysig_flat: Tensor (N,) and lengths is a list of per-day counts summing to N

    For each day we extract the day's arr = ysig_flat[offset:offset+L] and compute:
      mean error: (target - mean(arr))^2  where target = arr[-1]
      persistence error: (target - arr[-2])^2 when L > 1

    Returns:
      (mean_rmse, persistence_rmse)
    """
    mean_errors = []
    last_deltas = []

    for batch in loader:
        # Unpack strictly according to the flattened-collate contract
        x_flat, y_sig, _ybin, _yret, _yter, _rc, wd_expanded, ts_list, lengths = batch

        # Sanity checks
        if not (isinstance(y_sig, torch.Tensor) and y_sig.dim() == 1):
            raise RuntimeError(f"compute_baselines expects flattened y_sig (N,), got shape {getattr(y_sig,'shape',None)}")
        if not (isinstance(lengths, (list, tuple)) and sum(lengths) == y_sig.size(0)):
            raise RuntimeError(f"compute_baselines expects lengths summing to y_sig.size(0); got lengths={lengths} y_sig.shape={tuple(y_sig.shape)}")

        start = 0
        for L in lengths:
            if L <= 0:
                start += L
                continue
            end = start + L
            arr = y_sig[start:end].view(-1).cpu().numpy().astype(float)
            if arr.size == 0:
                start = end
                continue
            target = arr[-1]
            mu = arr.mean()
            mean_errors.append((target - mu) ** 2)
            if arr.size > 1:
                last_deltas.append((target - arr[-2]) ** 2)
            start = end

    mean_rmse = float(np.sqrt(np.mean(mean_errors))) if mean_errors else float("nan")
    persistence_rmse = float(np.sqrt(np.mean(last_deltas))) if last_deltas else float("nan")
    return mean_rmse, persistence_rmse


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
        model.reset_short()
        model._reset_log.append(("short", day))

        # weekly reset if we've wrapped past the end of the week
        if prev_day is not None and day < prev_day:
            model.reset_long()
            model._reset_log.append(("long", day))


    return day


############################ 


class CustomMSELoss(nn.Module):
    """
    Combined level + slope MSE: standard MSE mean + α*MSE of one‐step diffs.
    alpha: slope-penalty weight; ↑smoothness, ↓spike fidelity
    """
    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        # final‐step head
        assert preds.shape == targs.shape, "preds and targs must have identical shapes"
        L1 = torch.nn.functional.mse_loss(preds, targs, reduction="mean")
        if self.alpha <= 0 or preds.numel() < 2:
            return L1
        dp = preds[1:] - preds[:-1]
        dt =  targs[1:] - targs[:-1]
        L2 = torch.nn.functional.mse_loss(dp, dt, reduction="mean")
        return L1 + self.alpha * L2


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


def _prepare_windows_and_targets_batch(x_batch: torch.Tensor,
                                       y_signal: torch.Tensor,
                                       seq_lengths: list,
                                       wd_batch: torch.Tensor,
                                       reset_state_fn,
                                       model,
                                       prev_day=None):
    """
    Build (windows_tensor, targets_tensor, prev_day) from a flattened collate.

    Assumes:
      - x_batch: Tensor shape (N, T, F)  where N = sum(seq_lengths)
      - y_signal: Tensor shape (N,)
      - seq_lengths: list/tuple of per-day window counts [W0, W1, ...] summing to N
      - wd_batch: per-day weekday tensor (len == B)

    Behaviour:
      - Calls reset_state_fn once per day (using wd_batch[day_idx]) before that day's windows.
      - Returns windows_tensor: (N, T, F), targets_tensor: (N,)
      - If no windows present returns (None, None, prev_day)
    """
    device = x_batch.device

    # Quick sanity
    if not (isinstance(seq_lengths, (list, tuple)) and sum(seq_lengths) == x_batch.size(0)):
        raise RuntimeError(f"_prepare_windows_and_targets_batch expects seq_lengths summing to x_batch.size(0); got seq_lengths={seq_lengths} x_batch.shape={tuple(x_batch.shape)}")

    windows_list = []
    targets_list = []
    start = 0

    for day_idx, L in enumerate(seq_lengths):
        prev_day = reset_state_fn(model, wd_batch[day_idx], prev_day)  

        if L <= 0:
            start += L
            continue

        end = start + L
        day_windows = x_batch[start:end]     # (L, T, F)
        day_targets = y_signal[start:end]    # (L,)

        windows_list.append(day_windows)
        targets_list.append(day_targets)
        start = end

    if not windows_list:
        return None, None, prev_day

    windows_tensor = torch.cat(windows_list, dim=0).to(device).float()   # (N, T, F)
    targets_tensor = torch.cat(targets_list, dim=0).to(device).float()   # (N,)
    return windows_tensor, targets_tensor, prev_day


########################


def eval_on_loader(loader, model: nn.Module, clamp_preds: bool = True) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model validation over a loader and return metrics and predictions.
    
    Returns: (metrics_dict, preds_array, base_preds_array, targets_array)
    - Moves model to current parameter device and runs in eval mode with no grad.
    - Preserves model LSTM state across windows using prev_day and _prepare_windows_and_targets_batch.
    - Calls model(windows) -> (base, delta) and composes total = base + delta.
    - Collects CPU numpy arrays: tot_preds, base_preds, and targets for metrics.
    - Stores last_val_tot_preds, last_val_targs, last_val_base_preds on the model for logging-
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.h_short = model.h_long = None
    prev_day = None
    all_basepreds, all_totpreds, all_targs = [], [], []

    with torch.no_grad():
        for x_batch, y_signal, y_bin, y_ret, y_ter, rc, wd, ts_list, seq_lengths in \
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
            all_basepreds.extend(base_tensor.reshape(total_tensor.size(0), -1)[:, 0].detach().cpu().tolist())
            all_totpreds.extend(total_tensor.reshape(total_tensor.size(0), -1)[:, 0].detach().cpu().tolist())
            all_targs.extend(targets_tensor.cpu().tolist())
            
    val_base_preds = np.array(all_basepreds, dtype=float)
    val_tot_preds = np.array(all_totpreds, dtype=float)
    val_targs = np.array(all_targs, dtype=float)

    # model.last_val_base_preds = torch.from_numpy(base_preds).float()
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
    *,
    max_epochs:          int,
    early_stop_patience: int,
    clipnorm:            float,
    alpha_smooth:        float,
    lambda_delta:        float,
) -> float:
    """
    Train the model with a baseline head and a detached residual (delta) head using AMP.
    
    Behavior summary
    - model(windows) -> (base, delta); prediction total = base + delta.
    - base is supervised by base_loss = CustomMSELoss(base, targets) smoothed.
    - delta is trained to match detached residual delta_target = (targets - base).detach(); delta_loss = CustomMSELoss(delta, delta_target) not smoothed.
    - Combined objective: total_loss = base_loss + lambda_delta * delta_loss.
    - Uses torch.amp.autocast + GradScaler; unscale -> clip_grad_norm -> scaler.step/update; scheduler.step guarded to only run when optimizer.step executed.
    - Aggregates epoch metrics on total predictions and runs validation via eval_on_loader.
    
    Returns best_val (best validation RMSE).
    """
    # zero-init delta head bias so delta starts neutral
    if getattr(model, "delta_head", None) is not None:
        torch.nn.init.zeros_(model.delta_head.bias)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    live_plot = plots.LiveRMSEPlot()
    best_val, best_state, patience = float("inf"), None, 0
    models_dir = Path(params.models_folder)

    MSE_base = CustomMSELoss(alpha=alpha_smooth) 
    MSE_delta = CustomMSELoss(alpha=0) # alpha default 0.0

    # compute baselines for logging
    bl_tr_mean, bl_tr_pers = compute_baselines(train_loader)
    bl_val_mean, bl_val_pers = compute_baselines(val_loader)

    for epoch in range(1, max_epochs + 1):

        model.train()
        model.h_short = model.h_long = None

        tr_base_preds, tr_tot_preds, tr_targs, tr_delta_preds, tr_delta_targs = [], [], [], [], []            
        prev_day = None

        epoch_total_loss_sum = 0.0
        epoch_base_loss_sum = 0.0
        epoch_delta_loss_sum = 0.0
        epoch_loss_count = 0
        
        epoch_start = datetime.utcnow().timestamp()
        epoch_samples = 0

        for x_batch, y_signal, y_bin, y_ret, y_ter, rc, wd, ts_list, seq_lengths in \
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
                base_loss = MSE_base(base_tensor, targets_tensor)
                
                # teach delta to predict the detached residual
                if model.use_delta and lambda_delta > 0:
                    delta_target = (targets_tensor - base_tensor).detach()
                    delta_loss = MSE_delta(delta_tensor, delta_target)
                    tr_delta_preds.extend(delta_tensor.detach().cpu().tolist())
                    tr_delta_targs.extend(delta_target.cpu().tolist())
                else:
                    delta_loss = torch.tensor(0.0, device=base_tensor.device)

                tr_base_preds.extend(base_tensor.detach().cpu().tolist())
                tr_tot_preds.extend(total_tensor.detach().cpu().tolist())
                tr_targs.extend(targets_tensor.cpu().tolist())
                epoch_samples += int(targets_tensor.size(0))

                # combined training objective (final scalar used for backward)
                total_loss = base_loss + lambda_delta * delta_loss

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
            epoch_total_loss_sum += float(total_loss.detach().cpu())
            epoch_base_loss_sum += float(base_loss.detach().cpu())
            epoch_delta_loss_sum += float(delta_loss.detach().cpu())
            epoch_loss_count += 1

        # Metrics & validation
        tr_tot_metrics = _compute_metrics(np.array(tr_tot_preds, dtype=float), np.array(tr_targs, dtype=float))
        tr_base_metrics = _compute_metrics(np.array(tr_base_preds, dtype=float), np.array(tr_targs, dtype=float))  
        if len(tr_delta_preds) > 0:
            tr_delta_metrics = _compute_metrics(np.array(tr_delta_preds, dtype=float), np.array(tr_delta_targs, dtype=float))
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
            bl_tr_mean=bl_tr_mean,
            bl_tr_pers=bl_tr_pers,
            bl_val_mean=bl_val_mean,
            bl_val_pers=bl_val_pers,
            avg_base_loss=avg_base_loss,
            avg_delta_loss=avg_delta_loss,
            log_file=params.log_file,
            hparams=params.hparams,
        )
        live_plot.update(tr_tot_rmse, val_tot_rmse)

        best_val, improved, *_ = models_core.maybe_save_chkpt(
            models_dir, model, val_tot_rmse, best_val,
            {"rmse": tr_tot_rmse}, {"rmse": val_tot_rmse},
            live_plot, params
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
            if patience >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        models_core.save_final_chkpt(
            models_dir, best_state, best_val, params,
            tr_tot_metrics, val_tot_metrics, live_plot, suffix="_fin"
        )

    for attr in ("_last_epoch_elapsed","_last_epoch_samples","_last_epoch_checkpoint","_first_batch_snapshot"):
        try:
            delattr(model, attr)
        except Exception:
            pass

    return best_val