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
    Modular time-series regressor. Layers can be toggled independently:
      0) Conv1d + BatchNorm
      1) Temporal ConvNet (TCN)
      2) Short Bi-LSTM
      3) TransformerEncoder w/ optional feature projection
      4) Linear projection → LayerNorm → Dropout
      5) Long Bi-LSTM
      6) Flatten/Last/Pool → MLP head (no skip branch)
    Stateful LSTM buffers reset via _reset_states() per day/week.
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
        self.aux_head = weight_norm(nn.Linear(long_units, 1))  # a tiny auxiliary head that predicts every timestep


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
        self.head_flat = nn.Sequential(
            weight_norm(nn.Linear(flat_dim, pred_hidden)),
            nn.ReLU(),
            weight_norm(nn.Linear(pred_hidden, 1))
        )

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

        aux_out = self.aux_head(out_l) # shape (B, T, 1)


        # 6) Flatten/Last/Pool + MLP head
        if self.flatten_mode == "flatten":
            flat = out_l.reshape(batch_size, -1)
        elif self.flatten_mode == "last":
            flat = out_l[:, -1, :]
        else:  # "pool"
            flat = out_l.mean(dim=1)

        norm_flat = self.ln_flat(flat)
        main_out  = self.head_flat(norm_flat)  # (B, 1)
        return main_out.unsqueeze(-1), aux_out   # → ((B,1,1), (B,T,1))
        
######################################################################################################


class SmoothMSELoss(nn.Module):
    """
    Combined level + slope MSE, with special handling for 2D aux-loss.

    - 1D inputs (B,): standard MSE mean + α*MSE of one‐step diffs.
    - 2D inputs (B, T): flattens to (B*T,) with reduction='sum' for level loss,
      plus mean‐reduction slope penalty across time steps.
    """
    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        if preds.dim() == 1:
            # final‐step head
            L1 = torch.nn.functional.mse_loss(preds, targs, reduction="mean")
            if self.alpha <= 0:
                return L1
            dp = preds[1:] - preds[:-1]
            dt =  targs[1:] - targs[:-1]
            L2 = torch.nn.functional.mse_loss(dp, dt, reduction="mean")
            return L1 + self.alpha * L2

        elif preds.dim() == 2:
            # aux‐head over T steps
            # level loss: sum so each step sends full gradient
            flat_p = preds.reshape(-1)
            flat_t = targs.reshape(-1)
            L1 = torch.nn.functional.mse_loss(flat_p, flat_t, reduction="sum") / preds.size(1)

            if self.alpha <= 0:
                return L1

            # slope penalty across time axis
            dp = preds[:, 1:] - preds[:, :-1]
            dt =  targs[:, 1:] - targs[:, :-1]
            L2 = torch.nn.functional.mse_loss(dp, dt, reduction="mean")
            return L1 + self.alpha * L2

        else:
            raise ValueError(f"Unsupported preds.dim()={preds.dim()}")


######################################################################################################

def compute_baselines(loader) -> tuple[float, float]:
    """
    Compute two static, one‐step‐per‐window baselines over all windows in `loader`:
      1) mean_rmse       – RMSE if you always predict μ = average y_sig per window
      2) persistence_rmse – RMSE if you always predict last‐seen (yₜ = yₜ₋₁)
    """
    nexts, last_deltas = [], []
    
    for x_pad, y_sig, _y_bin, _y_ret, _y_ter, _rc, wd, ts_list, lengths in loader:
        for i, L in enumerate(lengths):
            if L < 1:
                continue
            arr = y_sig[i, :L].view(-1).cpu().numpy()
            nexts.append(arr[-1])
            if L > 1:
                last_deltas.append(arr[-1] - arr[-2])
    
    nexts = np.array(nexts, dtype=float)
    mean_rmse = float(np.sqrt(((nexts - nexts.mean()) ** 2).mean()))
    
    if last_deltas:
        last_deltas = np.array(last_deltas, dtype=float)
        persistence_rmse = float(np.sqrt((last_deltas ** 2).mean()))
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

############### 

 
def eval_on_loader(loader, model: nn.Module, clamp_preds: bool = True) -> tuple[dict, np.ndarray]:
    """
    Evaluate `model` on `loader` and return metrics plus CPU numpy predictions.

    Behavior
    - Runs model in eval mode with torch.no_grad.
    - Accepts model outputs as raw_reg 
    - Supports raw_reg with shape (W, T, 1) or (W, T) and uses model.pred when needed.
    - Aggregates per-segment final-step predictions (seq_reg[:, -1]) and targets,
      converts them to CPU numpy arrays for metric computation.
    - Stores last_val_preds and last_val_targs on the model as torch.FloatTensors
      for downstream diagnostics (slope RMSE etc).
    - This function performs host copies (preds.cpu()) which is acceptable for eval
      since it is off the training hot path.    """
    device = next(model.parameters()).device
    model.to(device).eval()
    model.h_short = model.h_long = None
    prev_day = None
    all_preds, all_targs = [], []

    with torch.no_grad():
        for x_batch, y_signal, y_bin, y_ret, y_ter, rc, wd, ts_list, seq_lengths in \
                tqdm(loader, desc="eval", leave=False):

            x_batch = x_batch.to(device, non_blocking=True)
            y_signal = y_signal.to(device, non_blocking=True)
            wd = wd.to(device, non_blocking=True)

            batch_size = x_batch.size(0)
            windows_list, targets_list = [], []
            for example_idx in range(batch_size):
                prev_day = _reset_states(model, wd[example_idx], prev_day)
                L = int(seq_lengths[example_idx])
                if L == 0:
                    continue
                windows_list.append(x_batch[example_idx, :L])
                targets_list.append(y_signal[example_idx, :L].reshape(-1))

            if not windows_list:
                continue

            windows_tensor = torch.cat(windows_list, dim=0)
            targets_tensor = torch.cat(targets_list, dim=0)

            # if not windows_tensor.is_contiguous():
            #     windows_tensor = windows_tensor.contiguous()

            raw_out = model(windows_tensor)
            raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

            if raw_reg.dim() == 3:
                if raw_reg.size(-1) == 1:
                    seq_reg = raw_reg.view(raw_reg.size(0), raw_reg.size(1))
                else:
                    raw_reg_pred = model.pred(raw_reg)
                    if raw_reg_pred.dim() != 3 or raw_reg_pred.size(-1) != 1:
                        raise ValueError("eval_on_loader: model.pred must return shape (W, T, 1) when passed 3D raw_reg")
                    seq_reg = raw_reg_pred.view(raw_reg_pred.size(0), raw_reg_pred.size(1))
            elif raw_reg.dim() == 2:
                seq_reg = raw_reg
            else:
                raise ValueError(f"eval_on_loader: unexpected raw_reg.dim()={raw_reg.dim()}")

            preds = seq_reg[:, -1]

            all_preds.extend(preds.cpu().tolist())
            all_targs.extend(targets_tensor.cpu().tolist())

    preds = np.array(all_preds, dtype=float)
    targs = np.array(all_targs, dtype=float)

    model.last_val_preds = torch.from_numpy(preds).float()
    model.last_val_targs = torch.from_numpy(targs).float()

    return _compute_metrics(preds, targs), preds



############### 


# def model_training_loop(
#     model:                nn.Module,
#     optimizer:            torch.optim.Optimizer,
#     scheduler:            torch.optim.lr_scheduler.OneCycleLR,
#     scaler:               torch.amp.GradScaler,
#     train_loader,
#     val_loader,
#     *,
#     max_epochs:          int,
#     early_stop_patience: int,
#     clipnorm:            float,
#     alpha_smooth:        float,
# ) -> float:
#     """
#     Training loop with remaining GPU->CPU syncs removed, cuDNN warmup enabled, aux outputs unused,
#     and all per-batch host copies deferred until after optimizer.step.

#     Behavior:
#       - model(...) still returns (raw_reg, aux_out) for API parity; aux_out is not inspected in the hot path.
#       - No .cpu()/.item()/.tolist() calls occur before scaler.step for any training batch.
#       - torch.backends.cudnn.benchmark is enabled once at startup to avoid repeated cudnn kernel autotune stalls.
#       - First-batch snapshot is a non-blocking summary (shapes and boolean grad presence only).
#       - Uses torch.amp.autocast('cuda') and torch.amp.GradScaler for AMP path (scaler passed in).
#     """
#     import torch as _torch
#     _torch.backends.cudnn.benchmark = True

#     device = next(model.parameters()).device
#     model.to(device)

#     # Scheduler sanity check (best-effort)
#     try:
#         expected_total = len(train_loader) * max_epochs
#         if hasattr(scheduler, "_total_steps") and scheduler._total_steps != expected_total:
#             raise RuntimeError(
#                 f"Scheduler total_steps mismatch: scheduler={scheduler._total_steps} expected={expected_total}"
#             )
#     except Exception:
#         pass

#     base_tr_mean, base_tr_pers = compute_baselines(train_loader)
#     base_vl_mean, base_vl_pers = compute_baselines(val_loader)

#     smooth_loss = SmoothMSELoss(alpha_smooth)
#     live_plot = plots.LiveRMSEPlot()
#     best_val, best_state, patience = float("inf"), None, 0

#     if hasattr(model, "_first_batch_snapshot"):
#         try:
#             delattr = getattr(model, "__delattr__", None)
#             if callable(delattr):
#                 model.__delattr__("_first_batch_snapshot")
#         except Exception:
#             pass

#     models_dir = Path(params.models_folder)
#     first_snapshot_captured = False

#     for epoch in range(1, max_epochs + 1):
#         freeze_till = params.hparams["FREEZE_TILL"]
#         for p in model.head_flat.parameters():
#             p.requires_grad = False if epoch <= freeze_till else True

#         model.train()
#         model.h_short = model.h_long = None

#         train_preds, train_targs = [], []
#         prev_day = None

#         epoch_loss_main_sum = 0.0
#         epoch_loss_aux_sum = 0.0
#         epoch_loss_count = 0
#         epoch_start = datetime.utcnow().timestamp()
#         epoch_samples = 0

#         for x_batch, y_signal, y_bin, y_ret, y_ter, rc, wd, ts_list, seq_lengths in \
#                 tqdm(train_loader, desc=f"Epoch {epoch} ▶ Train", leave=False):

#             x_batch = x_batch.to(device, non_blocking=True)
#             y_signal = y_signal.to(device, non_blocking=True)
#             wd = wd.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)

#             # gather windows & targets (avoid many small Python ops if possible)
#             batch_size = x_batch.size(0)
#             windows_list, targets_list = [], []
#             for example_idx in range(batch_size):
#                 prev_day = _reset_states(model, wd[example_idx], prev_day)
#                 L = int(seq_lengths[example_idx])
#                 if L == 0:
#                     continue
#                 windows_list.append(x_batch[example_idx, :L])
#                 targets_list.append(y_signal[example_idx, :L].reshape(-1))

#             if not windows_list:
#                 continue

#             windows_tensor = torch.cat(windows_list, dim=0)
#             targets_tensor = torch.cat(targets_list, dim=0)

#             # Ensure contiguous input to help cudnn reuse
#             if not windows_tensor.is_contiguous():
#                 windows_tensor = windows_tensor.contiguous()

#             # Forward + loss under AMP autocast (no host-syncs)
#             with _torch.amp.autocast(device_type="cuda", enabled=True):
#                 raw_out = model(windows_tensor)
#                 raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

#                 if raw_reg.dim() == 3:
#                     if raw_reg.size(-1) == 1:
#                         seq_reg = raw_reg.view(raw_reg.size(0), raw_reg.size(1))
#                     else:
#                         raw_reg_pred = model.pred(raw_reg)
#                         if raw_reg_pred.dim() != 3 or raw_reg_pred.size(-1) != 1:
#                             raise ValueError("train: model.pred must return shape (W, T, 1) when passed 3D raw_reg")
#                         seq_reg = raw_reg_pred.view(raw_reg_pred.size(0), raw_reg_pred.size(1))
#                 elif raw_reg.dim() == 2:
#                     seq_reg = raw_reg
#                 else:
#                     raise ValueError(f"train: unexpected raw_reg.dim()={raw_reg.dim()}")

#                 preds = seq_reg[:, -1]
#                 loss_main = smooth_loss(preds, targets_tensor)
#                 loss = loss_main

#             # Backward (AMP) with no host-syncs before step
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)

#             # non-blocking first-batch snapshot: shapes and grad-presence only (guarded, sanitized)
#             if (not first_snapshot_captured) and (not hasattr(model, "_first_batch_snapshot")):
#                 try:
#                     # basic shape info (same as before)
#                     raw_shape = tuple(raw_reg.shape) if isinstance(raw_reg, torch.Tensor) else None
#                     aux_shape = None

#                     # gradient presence summary (same as before)
#                     per_param_has_grad = {
#                         (getattr(p, "name", None) or id(p)): (p.grad is not None)
#                         for g in optimizer.param_groups
#                         for p in g.get("params", [])
#                     }
#                     group_nonzero_counts = [
#                         sum(1 for p in g.get("params", []) if p.grad is not None)
#                         for g in optimizer.param_groups
#                     ]
#                     backbone_has = any((p.grad is not None) for n, p in model.named_parameters() if n.startswith("short") or n.startswith("long"))
#                     head_has = any((p.grad is not None) for n, p in model.named_parameters() if n.startswith("pred") or "pred" in n)
#                     aux_has = any((p.grad is not None) for n, p in model.named_parameters() if n.startswith("aux") or "aux" in n)

#                     # --- Safe, deterministic loss extraction and sanitization ---
#                     # We intentionally do NOT perform extra forward/backward work here.
#                     # If loss tensor objects exist in scope (loss_main), sanitize them; otherwise store None.
#                     import math, traceback
#                     try:
#                         # If loss_main was computed above as a tensor, reduce it safely to a scalar in FP32.
#                         lm_val = None
#                         try:
#                             if 'loss_main' in locals() and isinstance(loss_main, torch.Tensor):
#                                 lm_cpu = loss_main.detach().cpu().float()
#                                 if lm_cpu.numel() == 0:
#                                     lm_val = None
#                                 else:
#                                     finite_mask = torch.isfinite(lm_cpu)
#                                     if finite_mask.any():
#                                         lm_val = float(lm_cpu[finite_mask].mean().item())
#                                     else:
#                                         lm_val = float("nan")
#                             else:
#                                 # not present in locals (normal) -> store None
#                                 lm_val = None
#                         except Exception:
#                             lm_val = float("nan")
#                     except Exception:
#                         lm_val = float("nan")

#                     # Aux loss sanitization (if aux loss was computed in scope)
#                     try:
#                         la_val = None
#                         if 'loss_aux' in locals() and isinstance(loss_aux, torch.Tensor):
#                             la_cpu = loss_aux.detach().cpu().float()
#                             if la_cpu.numel() == 0:
#                                 la_val = None
#                             else:
#                                 finite_mask = torch.isfinite(la_cpu)
#                                 if finite_mask.any():
#                                     la_val = float(la_cpu[finite_mask].mean().item())
#                                 else:
#                                     la_val = float("nan")
#                         else:
#                             la_val = None
#                     except Exception:
#                         la_val = float("nan")

#                     # Aux weight extraction (safe)
#                     try:
#                         aux_w_val = float(params.hparams.get("AUX_LOSS_WEIGHT", 0.0)) if params is not None and hasattr(params, "hparams") else 0.0
#                     except Exception:
#                         aux_w_val = 0.0

#                     # Weighted aux contribution (safe arithmetic only when numeric)
#                     try:
#                         if la_val is None:
#                             waux_val = None
#                         elif isinstance(la_val, float) and math.isfinite(la_val):
#                             waux_val = float(aux_w_val) * float(la_val)
#                         else:
#                             waux_val = float("nan")
#                     except Exception:
#                         waux_val = float("nan")

#                     # Finalize snapshot dict: ensure plain Python floats or None
#                     model._first_batch_snapshot = {
#                         "raw_reg_shape": raw_shape,
#                         "aux_out_shape": aux_shape,
#                         "loss_main": (None if lm_val is None else float(lm_val)),
#                         "loss_aux": (None if la_val is None else float(la_val)),
#                         "aux_w": float(aux_w_val),
#                         "group_nonzero_counts": group_nonzero_counts,
#                         "grads": {"backbone": bool(backbone_has), "head": bool(head_has), "aux": bool(aux_has)},
#                     }
#                 except Exception:
#                     # keep failure silent to avoid interfering with training startup
#                     pass
#                 first_snapshot_captured = True

                    
#             # Clip and step (operate on pre-built params_with_grad list to avoid repeated generator overhead)
#             params_with_grad = [p for p in model.parameters() if p.grad is not None]
#             if params_with_grad:
#                 _torch.nn.utils.clip_grad_norm_(params_with_grad, clipnorm)

#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()

#             # After step: minimal host work (convert scalars/lists once)
#             epoch_loss_main_sum += float(loss_main.detach().cpu())
#             epoch_loss_count += 1

#             train_preds.extend(preds.detach().cpu().tolist())
#             train_targs.extend(targets_tensor.cpu().tolist())

#             epoch_samples += int(targets_tensor.size(0))

#         # Metrics & validation
#         tr_metrics = _compute_metrics(
#             np.array(train_preds, dtype=float),
#             np.array(train_targs, dtype=float),
#         )
#         vl_metrics, _ = eval_on_loader(val_loader, model)

#         if epoch_loss_count > 0:
#             avg_loss_main = epoch_loss_main_sum / epoch_loss_count
#         else:
#             avg_loss_main = float("nan")
#         avg_loss_aux = 0.0

#         epoch_elapsed = datetime.utcnow().timestamp() - epoch_start
#         model._last_epoch_elapsed = float(epoch_elapsed)
#         model._last_epoch_samples = int(epoch_samples)

#         tr_rmse, tr_mae, tr_r2 = tr_metrics["rmse"], tr_metrics["mae"], tr_metrics["r2"]
#         vl_rmse, vl_mae, vl_r2 = vl_metrics["rmse"], vl_metrics["mae"], vl_metrics["r2"]

#         models_core.log_epoch_summary(
#             epoch,
#             model,
#             optimizer,
#             train_metrics=tr_metrics,
#             val_metrics=vl_metrics,
#             base_tr_mean=base_tr_mean,
#             base_tr_pers=base_tr_pers,
#             base_vl_mean=base_vl_mean,
#             base_vl_pers=base_vl_pers,
#             slip_thresh=1e-6,
#             log_file=params.log_file,
#             top_k=999,
#             hparams=params.hparams,
#         )
#         live_plot.update(tr_rmse, vl_rmse)

#         best_val, improved, *_ = models_core.maybe_save_chkpt(
#             models_dir, model, vl_rmse, best_val,
#             {"rmse": tr_rmse}, {"rmse": vl_rmse},
#             live_plot, params
#         )

#         model._last_epoch_checkpoint = bool(improved)

#         current_lr = optimizer.param_groups[0]["lr"]
#         print(
#             f"Epoch {epoch:02d}  "
#             f"TRAIN→ RMSE={tr_rmse:.5f}, R²={tr_r2:.3f} |  "
#             f"VALID→ RMSE={vl_rmse:.5f}, R²={vl_r2:.3f} |  "
#             f"lr={current_lr:.2e} |  "
#             f"improved={bool(improved)} |  "
#             f"loss_main={avg_loss_main:.5e}, loss_aux={avg_loss_aux:.5e}"
#         )

#         if improved:
#             best_state, patience = {
#                 k: v.cpu() for k, v in model.state_dict().items()
#             }, 0
#         else:
#             patience += 1
#             if patience >= early_stop_patience:
#                 break

#     if best_state is not None:
#         model.load_state_dict(best_state)
#         models_core.save_final_chkpt(
#             models_dir, best_state, best_val, params,
#             tr_metrics, vl_metrics, live_plot, suffix="_fin"
#         )

#     for attr in ("_last_epoch_elapsed", "_last_epoch_samples", "_last_epoch_checkpoint", "_first_batch_snapshot"):
#         if hasattr(model, attr):
#             try:
#                 delattr = getattr(model, "__delattr__", None)
#                 if callable(delattr):
#                     model.__delattr__(attr)
#             except Exception:
#                 pass

#     return best_val


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
) -> float:
    """
    Train `model` with AMP and one-cycle LR while minimizing GPU->CPU syncs.
    
    - AMP hot path: forward/backward under torch.amp.autocast and torch.amp.GradScaler;
      host work (detach/to-list) deferred until after scaler.step to avoid blocking.
    - Captures a cheap first-batch snapshot (shapes and boolean grad-presence) without
      forcing GPU->CPU copies at startup.
    - Performs gradient clipping, scaler.step/optimizer.step and scheduler.step every batch.
    - Logs epoch metrics, checkpointing and lightweight diagnostics; AUX paths are disabled
      in the hot path.
    
    Parameters: model, optimizer, scheduler, scaler, train_loader, val_loader,
    max_epochs, early_stop_patience, clipnorm, alpha_smooth.
    
    Returns: best_val (best validation metric observed).
    """
    import torch as _torch
    _torch.backends.cudnn.benchmark = True

    device = next(model.parameters()).device
    model.to(device)

    # Scheduler sanity check (best-effort)
    try:
        expected_total = len(train_loader) * max_epochs
        if hasattr(scheduler, "_total_steps") and scheduler._total_steps != expected_total:
            raise RuntimeError(
                f"Scheduler total_steps mismatch: scheduler={scheduler._total_steps} expected={expected_total}"
            )
    except Exception:
        pass

    base_tr_mean, base_tr_pers = compute_baselines(train_loader)
    base_vl_mean, base_vl_pers = compute_baselines(val_loader)

    smooth_loss = SmoothMSELoss(alpha_smooth)
    live_plot = plots.LiveRMSEPlot()
    best_val, best_state, patience = float("inf"), None, 0

    models_dir = Path(params.models_folder)
    first_snapshot_captured = False

    for epoch in range(1, max_epochs + 1):
        freeze_till = params.hparams["FREEZE_TILL"]
        for p in model.head_flat.parameters():
            p.requires_grad = False if epoch <= freeze_till else True

        model.train()
        model.h_short = model.h_long = None

        train_preds, train_targs = [], []
        prev_day = None

        epoch_loss_main_sum = 0.0
        epoch_loss_count = 0
        epoch_start = datetime.utcnow().timestamp()
        epoch_samples = 0

        for x_batch, y_signal, y_bin, y_ret, y_ter, rc, wd, ts_list, seq_lengths in \
                tqdm(train_loader, desc=f"Epoch {epoch} ▶ Train", leave=False):

            x_batch = x_batch.to(device, non_blocking=True)
            y_signal = y_signal.to(device, non_blocking=True)
            wd = wd.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # gather windows & targets (avoid many small Python ops if possible)
            batch_size = x_batch.size(0)
            windows_list, targets_list = [], []
            for example_idx in range(batch_size):
                prev_day = _reset_states(model, wd[example_idx], prev_day)
                L = int(seq_lengths[example_idx])
                if L == 0:
                    continue
                windows_list.append(x_batch[example_idx, :L])
                targets_list.append(y_signal[example_idx, :L].reshape(-1))

            if not windows_list:
                continue

            windows_tensor = torch.cat(windows_list, dim=0)
            targets_tensor = torch.cat(targets_list, dim=0)

            # # Ensure contiguous input to help cudnn reuse
            # if not windows_tensor.is_contiguous():
            #     windows_tensor = windows_tensor.contiguous()

            # Forward + loss under AMP autocast (no host-syncs)
            with _torch.amp.autocast(device_type="cuda", enabled=True):
                raw_out = model(windows_tensor)
                raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

                if raw_reg.dim() == 3:
                    if raw_reg.size(-1) == 1:
                        seq_reg = raw_reg.view(raw_reg.size(0), raw_reg.size(1))
                    else:
                        raw_reg_pred = model.pred(raw_reg)
                        if raw_reg_pred.dim() != 3 or raw_reg_pred.size(-1) != 1:
                            raise ValueError("train: model.pred must return shape (W, T, 1) when passed 3D raw_reg")
                        seq_reg = raw_reg_pred.view(raw_reg_pred.size(0), raw_reg_pred.size(1))
                elif raw_reg.dim() == 2:
                    seq_reg = raw_reg
                else:
                    raise ValueError(f"train: unexpected raw_reg.dim()={raw_reg.dim()}")

                preds = seq_reg[:, -1]
                loss_main = smooth_loss(preds, targets_tensor)
                loss = loss_main

            # Backward (AMP) with no host-syncs before step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # lightweight first-batch snapshot (non-blocking, cheap)
            if (not first_snapshot_captured) and (not hasattr(model, "_first_batch_snapshot")):
                try:
                    raw_shape = tuple(raw_reg.shape) if isinstance(raw_reg, torch.Tensor) else None
            
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
                _torch.nn.utils.clip_grad_norm_(params_with_grad, clipnorm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # After step: minimal host work (convert scalars/lists once)
            epoch_loss_main_sum += float(loss_main.detach().cpu())
            epoch_loss_count += 1

            train_preds.extend(preds.detach().cpu().tolist())
            train_targs.extend(targets_tensor.cpu().tolist())

            epoch_samples += int(targets_tensor.size(0))

        # Metrics & validation
        tr_metrics = _compute_metrics(
            np.array(train_preds, dtype=float),
            np.array(train_targs, dtype=float),
        )
        vl_metrics, _ = eval_on_loader(val_loader, model)

        if epoch_loss_count > 0:
            avg_loss_main = epoch_loss_main_sum / epoch_loss_count
        else:
            avg_loss_main = float("nan")

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
            base_tr_mean=base_tr_mean,
            base_tr_pers=base_tr_pers,
            base_vl_mean=base_vl_mean,
            base_vl_pers=base_vl_pers,
            slip_thresh=1e-6,
            log_file=params.log_file,
            top_k=999,
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
            f"improved={bool(improved)} |  "
            f"loss_main={avg_loss_main:.5e}"
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
