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
import time

import datetime as dt
from datetime import datetime
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
    Minimal Configurable Backbone for Time-Series Prediction.
    
    Functionality:
    - Routes a 3D tensor (Batch, Time, Features) through a configurable pipeline.
    - Conditionally executes Convolutional, Temporal ConvNet (TCN), Bi-LSTM, 
      and Transformer blocks based on provided hyperparameters.
    - Bypasses overhead operations (like ReLUs, transposes) for disabled blocks.
    - Optimized with batch_first=True for sequential layers.
    - Provides flexible pooling/flattening options (flatten, last, pool, attn).
    - Outputs a base target prediction and an optional delta (residual) prediction.
    """
    def layer_projection(self, in_dim: int, out_dim: int) -> nn.Module:
        # Identity mapping if dimensions match, else linear projection
        return nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def __init__(
        self, n_feats: int, short_units: int, long_units: int, transformer_d_model: int,
        transformer_layers: int, transformer_heads: int, transformer_ff_mult: int,
        dropout_short: float, dropout_long: float, dropout_trans: float,
        pred_hidden: int, look_back: int, use_conv: bool, use_tcn: bool, use_short_lstm: bool,
        use_transformer: bool, use_long_lstm: bool, use_delta: bool, flatten_mode: str,
    ):
        super().__init__()
        
        # Store configuration flags and shapes
        self.look_back = look_back
        self.short_units = short_units
        self.long_units = long_units
        self.use_conv = use_conv
        self.use_tcn = use_tcn
        self.use_short_lstm = use_short_lstm
        self.use_transformer = use_transformer
        self.use_long_lstm = use_long_lstm
        self.use_delta = use_delta

        # Ensure Bi-LSTM units are divisible by 2 for bidirectional splits
        assert short_units % 2 == 0 and long_units % 2 == 0

        # 0) Conv Block
        # Safely packaged to prevent stray ReLUs on raw data
        CONV_CHANNELS = params.hparams["CONV_CHANNELS"]
        if self.use_conv:
            conv_k, conv_dilation = params.hparams["CONV_K"], params.hparams["CONV_DILATION"]
            padding = (conv_k // 2) * conv_dilation
            self.conv_block = nn.Sequential(
                nn.Conv1d(n_feats, CONV_CHANNELS, kernel_size=conv_k, dilation=conv_dilation, padding=padding),
                nn.GroupNorm(8, CONV_CHANNELS),
                nn.ReLU()
            )
        else:
            self.conv_block = None

        # 1) TCN Block
        TCN_CHANNELS = params.hparams["TCN_CHANNELS"]
        if self.use_tcn:
            layers, k = params.hparams["TCN_LAYERS"], params.hparams["TCN_KERNEL"]
            blocks = []
            in_ch = CONV_CHANNELS if self.use_conv else n_feats
            for i in range(layers):
                d = 2 ** i
                pad = (k // 2) * d
                blocks += [
                    nn.Conv1d(in_ch, TCN_CHANNELS, k, dilation=d, padding=pad), 
                    nn.GroupNorm(8, TCN_CHANNELS), 
                    nn.ReLU()
                ]
                in_ch = TCN_CHANNELS
            self.tcn = nn.Sequential(*blocks)
        else:
            self.tcn = None

        # 2) Short LSTM Block
        short_in = (TCN_CHANNELS if self.use_tcn else (CONV_CHANNELS if self.use_conv else n_feats))
        if self.use_short_lstm:
            self.short_lstm = nn.LSTM(input_size=short_in, hidden_size=short_units // 2, batch_first=True, bidirectional=True)
            self.ln_short = nn.LayerNorm(short_units)
            self.do_short = nn.Dropout(dropout_short)
        else:
            self.short_lstm = None
            
        self.h_short = self.c_short = None
        upstream_dim = short_units if self.use_short_lstm else short_in

        # Canonical input projection
        self.input_proj = nn.Linear(n_feats, upstream_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        if getattr(self.input_proj, "bias", None) is not None:
            nn.init.zeros_(self.input_proj.bias)

        # 3) Transformer Block
        if self.use_transformer:
            d_model = transformer_d_model
            # Use arguments instead of global params.hparams
            heads = transformer_heads
            ff_dim = d_model * transformer_ff_mult
            
            self.feature_proj = self.layer_projection(upstream_dim, d_model)
            self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout_trans, max_len=look_back)
            
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=ff_dim, 
                dropout=dropout_trans, batch_first=True 
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        else:
            self.feature_proj = nn.Linear(upstream_dim, upstream_dim)
            if upstream_dim == upstream_dim: nn.init.eye_(self.feature_proj.weight) 
            if getattr(self.feature_proj, "bias", None) is not None: nn.init.zeros_(self.feature_proj.bias)
            self.pos_enc = nn.Identity()
            self.transformer = nn.Identity()

        # 4) Bridge Projection
        proj_in = transformer_d_model if self.use_transformer else upstream_dim
        self.short2long = self.layer_projection(proj_in, long_units)
        self.ln_proj = nn.LayerNorm(long_units)
        self.do_proj = nn.Dropout(dropout_long)

        # 5) Long LSTM Block
        if self.use_long_lstm:
            self.long_lstm = nn.LSTM(input_size=long_units, hidden_size=long_units // 2, batch_first=True, bidirectional=True)
            self.ln_long = nn.LayerNorm(long_units)
            self.do_long = nn.Dropout(dropout_long)
        else:
            self.long_lstm = None
            
        self.h_long = self.c_long = None

        # 6) Output Head
        assert flatten_mode in ("flatten", "last", "pool", "attn")
        self.flatten_mode = flatten_mode
        flat_dim = (look_back * long_units if flatten_mode == "flatten" else long_units)
        
        self.ln_flat = nn.LayerNorm(flat_dim)
        self.head_flat = nn.Sequential(
            weight_norm(nn.Linear(flat_dim, pred_hidden)), 
            nn.ReLU(), 
            weight_norm(nn.Linear(pred_hidden, 1))
        )
        nn.init.zeros_(self.head_flat[-1].bias)
        self.attn_pool = nn.Linear(long_units, 1)
        nn.init.zeros_(self.attn_pool.bias)

        # 7) Delta Head
        if self.use_delta:
            _d = nn.Linear(flat_dim, 1)
            nn.init.zeros_(_d.weight)
            nn.init.zeros_(_d.bias)
            self.delta_head = _d

    def forward(self, x: torch.Tensor):
        # Format input dimensions
        if x.dim() > 3:
            *lead, T, F = x.shape; x = x.view(-1, T, F)
        if x.dim() == 2: 
            x = x.unsqueeze(0)
            
        B, T, _ = x.shape

        # OPTIMIZATION: Only transpose if the layer is actually enabled
        if self.conv_block is not None:
            x = self.conv_block(x.transpose(1, 2)).transpose(1, 2)
            
        if self.tcn is not None:
            x = self.tcn(x.transpose(1, 2)).transpose(1, 2)

        # Execute Short LSTM
        if self.use_short_lstm:
            if self.h_short is None or self.h_short.size(1) != B:
                self.h_short, self.c_short = _allocate_lstm_states(B, self.short_units // 2, True, x.device)
            out_s, (h_s, c_s) = self.short_lstm(x, (self.h_short, self.c_short))
            self.h_short, self.c_short = h_s.detach(), c_s.detach()
            out_s = self.do_short(self.ln_short(out_s))
        else:
            out_s = x

        # Execute Input Projection
        if hasattr(self, "input_proj"):
            in_dim_expected = self.input_proj.weight.shape[1]
            pre_tr = self.input_proj(out_s) if out_s.size(-1) == in_dim_expected else out_s
        else:
            pre_tr = out_s

        # Execute Transformer
        tr_in = self.pos_enc(self.feature_proj(pre_tr))
        out_t = self.transformer(tr_in)
        out_p = self.do_proj(self.ln_proj(self.short2long(out_t)))

        # Execute Long LSTM
        if self.use_long_lstm:
            if self.h_long is None or self.h_long.size(1) != B:
                self.h_long, self.c_long = _allocate_lstm_states(B, self.long_units // 2, True, out_p.device)
            out_l, (h_l, c_l) = self.long_lstm(out_p, (self.h_long, self.c_long))
            self.h_long, self.c_long = h_l.detach(), c_l.detach()
            out_l = self.do_long(self.ln_long(out_l))
        else:
            out_l = out_p

        # Execute Flattening / Pooling strategy
        if self.flatten_mode == "flatten": 
            flat = out_l.reshape(B, -1)
        elif self.flatten_mode == "last": 
            flat = out_l[:, -1, :]
        elif self.flatten_mode == "attn":
            weights = torch.softmax(self.attn_pool(out_l).squeeze(-1), dim=1)
            flat = (out_l * weights.unsqueeze(-1)).sum(dim=1)
        else: 
            flat = out_l.mean(dim=1)

        # Output Heads calculation
        norm_flat = self.ln_flat(flat)
        base = self.head_flat(norm_flat).squeeze(-1)
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
    wd_list:  list,
    prev_day: int | None
) -> int:
    """
    Manages model state resets (short-term and long-term) for sequential data.
    
    Functionality:
    1. Detects day changes to reset short-term memory (h_short/c_short).
    2. Detects weekly wrap-around or non-sequential days to reset long-term memory.
    3. Operates entirely on CPU using Python lists to avoid expensive GPU-CPU syncs.
    """
    if not hasattr(model, "_reset_log"): 
        model._reset_log = []
    
    # 1. Boundary Check (Pure CPU path is fastest for small batches)
    first_day = wd_list[0]
    last_day = wd_list[-1]
    
    # 2. Daily Reset Trigger
    if prev_day is None or first_day != prev_day:
        for attr in ["h_short", "c_short"]:
            if hasattr(model, attr): setattr(model, attr, None)
        model._reset_log.append(("short", first_day))

    # 3. Weekly Reset Trigger (Checks for internal wrap-around)
    has_wrap_internal = any(wd_list[i] < wd_list[i-1] for i in range(1, len(wd_list)))
    
    if prev_day is not None and (first_day < prev_day or has_wrap_internal):
        for attr in ["h_long", "c_long"]:
            if hasattr(model, attr): setattr(model, attr, None)
        model._reset_log.append(("long", first_day))

    return last_day


############################ 


class CustomMSELoss(nn.Module):
    """
    Combined Level and Slope (Derivative) Loss for Time-Series.

    Functionality:
    - Calculates the standard distance (Level) between predictions and targets.
    - Calculates the distance between first-order differences (Slope/Velocity).
    - Vectorized: Uses slicing and indexing to calculate batch-wide slopes in one pass.
    - Sequence Aware: Uses seq_lengths to avoid calculating slopes across 
      discontinuous day boundaries (e.g., Friday's last tick to Monday's first).
    - Warmup: Gradually introduces the slope penalty (alpha) to allow the model 
      to learn basic levels before focusing on movement shapes.
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
        """ Calculates the current weight of the slope penalty based on training step. """
        if self.alpha <= 0.0:
            return 0.0
        # If no step or warmup provided, use full alpha
        if self.warmup_steps <= 0 or step is None:
            return self.alpha
        # Linear ramp from 0 to alpha
        t = min(1.0, float(step) / float(self.warmup_steps))
        return self.alpha * t

    def forward(
        self,
        preds: torch.Tensor,
        targs: torch.Tensor,
        seq_lengths: List[int],
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Main loss computation for Level and Slope.
        """
        # Ensure flat shapes for regression
        assert preds.shape == targs.shape, "Shape mismatch between preds and targets"

        # 1. PRIMARY LOSS: Level alignment (MSE or Huber)
        if self.use_huber:
            L1 = torch.nn.functional.huber_loss(preds, targs, reduction="mean", delta=self.huber_delta)
        else:
            L1 = torch.nn.functional.mse_loss(preds, targs, reduction="mean")

        # 2. SLOPE PREPARATION: Determine current alpha
        eff_alpha = self._effective_alpha(step)
        if eff_alpha <= 0.0:
            return L1

        # 3. SECONDARY LOSS: Slope (Derivative) alignment
        # To calculate slope, we subtract each value from the next (i+1 - i)
        if len(seq_lengths) > 1:
            device = preds.device
            # Compute indices where one day ends and another begins
            # We must NOT calculate a slope between these points
            boundaries = torch.as_tensor(seq_lengths, device=device).cumsum(dim=0)[:-1]
            
            # Create a boolean mask for valid 'next-step' transitions
            valid_indices = torch.ones(preds.size(0) - 1, dtype=torch.bool, device=device)
            # Mark the jumps between days as False
            valid_indices[boundaries - 1] = False
            
            # Slice and subtract only the valid sequential pairs
            diff_p = preds[1:][valid_indices] - preds[:-1][valid_indices]
            diff_t = targs[1:][valid_indices] - targs[:-1][valid_indices]
        else:
            # Single sequence batch: just do simple vector subtraction
            diff_p = preds[1:] - preds[:-1]
            diff_t = targs[1:] - targs[:-1]

        # 4. AGGREGATE: Level Loss + Scaled Slope Loss
        if diff_p.numel() == 0:
            return L1

        L2 = torch.nn.functional.mse_loss(diff_p, diff_t, reduction="mean")
        return L1 + (eff_alpha * L2)

        
#############################


def _compute_metrics(preds: np.ndarray, targs: np.ndarray) -> dict:
    """
    Calculates standard regression performance metrics using NumPy.

    Functionality:
    - Calculates RMSE (Root Mean Squared Error) for error magnitude.
    - Calculates MAE (Mean Absolute Error) for linear error.
    - Calculates R² (Coefficient of Determination) for model explanatory power.
    - Safety: Handles empty arrays and zero-variance targets to prevent NaNs/crashes.
    - Performance: Uses vectorized variance-based R² for high-speed evaluation.
    """
    # 1. Edge Case: Empty data
    if preds.size == 0 or targs.size == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    # 2. Vectorized Error Calculation
    # Faster to compute once than calling mean_squared_error repeatedly
    diff = preds - targs
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)

    # 3. Explanatory Power (R²)
    # R² = 1 - (Unexplained Variance / Total Variance)
    targs_var = np.var(targs)
    
    # If there is no variance in the target (e.g., a single sample or flat line), 
    # R² is mathematically undefined. We use a small epsilon check.
    if preds.size < 2 or targs_var < 1e-9:
        r2 = np.nan
    else:
        # Optimized R² formula for bulk NumPy arrays
        r2 = 1.0 - (mse / targs_var)

    return {
        "rmse": float(rmse), 
        "mae": float(mae), 
        "r2": float(r2)
    } 

    
#################### 


def _prepare_windows_and_targets_batch(
    x_batch: torch.Tensor, y_signal: torch.Tensor, seq_lengths: list,
    wd_list: list, reset_state_fn, model, prev_day=None
):
    """
    Optimized: Collapses padded batches and orchestrates state resets.
    
    Functionality:
    1. Triggers state resets (CPU-bound) before GPU processing.
    2. Vectorized Boolean Masking: Removes zero-padding from the collated batch.
    3. Memory Efficiency: Uses cached arithmetic to generate the mask, 
       reducing GPU memory fragmentation.
    """
    B = len(seq_lengths)
    if B == 0: 
        return None, None, prev_day
    
    # x_batch shape is (B * W_max, T, F)
    W_max = x_batch.size(0) // B
    device = x_batch.device
    
    # 1. CPU-bound state management
    prev_day = reset_state_fn(model, wd_list, prev_day)

    total_valid = sum(seq_lengths)
    if total_valid == 0: 
        return None, None, prev_day
        
    # 2. Shortcut: If batch is full, return immediately
    if total_valid == B * W_max:
        return x_batch.float(), y_signal.float(), prev_day

    # 3. Optimized Masking
    # We create a (B, W_max) grid of indices and compare to lengths
    # This specific pattern is faster for PyTorch to jit-compile
    lens_tensor = torch.as_tensor(seq_lengths, device=device).view(B, 1)
    
    # torch.arange is slightly faster when kept in a local scope
    indices = torch.arange(W_max, device=device).view(1, W_max)
    mask = (indices < lens_tensor).reshape(-1)
            
    return x_batch[mask].float(), y_signal[mask].float(), prev_day

    
########################


def eval_on_loader(loader, model: nn.Module) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates the model on a provided dataloader to compute metrics and predictions.

    Functionality:
    - Runs in eval mode with no_grad to minimize VRAM usage.
    - Accumulates predictions as GPU tensors to avoid PCIe sync bottlenecks during the loop.
    - Moves data to CPU in bulk at the end for final metric calculation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.h_short = model.h_long = None
    prev_day = None
    
    # Track results on GPU to maximize throughput
    val_base_preds_tensors = []
    val_tot_preds_tensors = []
    val_targs_tensors = []
    val_lengths = []

    # with torch.no_grad():
    with torch.inference_mode():
        # leave=False keeps the notebook clean; smoothing=0 shows true global speed
        for x_batch, y_signal, rc, wd, ts_list, seq_lengths in tqdm(loader, desc="eval", leave=False, smoothing=0):

            x_batch = x_batch.to(device, non_blocking=True)
            y_signal = y_signal.to(device, non_blocking=True)

            windows_tensor, targets_tensor, prev_day = _prepare_windows_and_targets_batch(
                x_batch, y_signal, seq_lengths, wd, _reset_states, model, prev_day
            )
            if windows_tensor is None:
                continue
   
            # Forward pass only
            base_tensor, delta_tensor = model(windows_tensor)
            total_tensor = base_tensor + delta_tensor
            
            # Detach and store
            val_base_preds_tensors.append(base_tensor.view(-1).detach())
            val_tot_preds_tensors.append(total_tensor.view(-1).detach())
            val_targs_tensors.append(targets_tensor.view(-1).detach())
            val_lengths.extend([L for L in seq_lengths if L > 0])
            
    # Batch move to CPU
    val_base_preds = torch.cat(val_base_preds_tensors).cpu().numpy().astype(np.float64) if val_base_preds_tensors else np.array([], dtype=np.float64)
    val_tot_preds = torch.cat(val_tot_preds_tensors).cpu().numpy().astype(np.float64) if val_tot_preds_tensors else np.array([], dtype=np.float64)
    val_targs = torch.cat(val_targs_tensors).cpu().numpy().astype(np.float64) if val_targs_tensors else np.array([], dtype=np.float64)

    model.bl_val_mean, model.bl_val_pers = compute_baselines(val_targs, val_lengths)
    model.last_val_tot_preds = torch.from_numpy(val_tot_preds).float()
    model.last_val_targs = torch.from_numpy(val_targs).float()
   
    return _compute_metrics(val_tot_preds, val_targs), _compute_metrics(val_base_preds, val_targs), val_tot_preds, val_base_preds, val_targs


####################################################################################################### 


def model_training_loop(
    model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.OneCycleLR,
    scaler: torch.amp.GradScaler, train_loader, val_loader
) -> float:
    """
    Executes an optimized training loop with minimized CPU-GPU synchronization.

    Functionality:
    - Moves loss accumulation to the GPU to avoid per-batch .item() sync points.
    - Replaces dynamic optimizer patching with standard GradScaler state checks.
    - Ensures states (h_short, c_short) are reset correctly at epoch start.
    - Maintains all logging, plotting, and checkpointing logic as originally designed.
    """
    if getattr(model, "delta_head", None) is not None:
        torch.nn.init.zeros_(model.delta_head.bias)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_feats = params.features_cols_tick
    model_hparams = params.hparams

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model.to(device)

    live_plot = plots.LiveRMSEPlot()
    best_val, best_state, patience = float("inf"), None, 0
    models_dir = Path(params.models_folder)

    MSE_base = CustomMSELoss(
        alpha=float(model_hparams["ALPHA_SMOOTH"]), warmup_steps=int(model_hparams["WARMUP_STEPS"]),
        use_huber=bool(model_hparams["USE_HUBER"]), huber_delta=float(model_hparams["HUBER_DELTA"]),
    )
    MSE_delta = CustomMSELoss(alpha=0.0, warmup_steps=0, use_huber=False)

    for epoch in range(1, model_hparams["MAX_EPOCHS"] + 1):
        model.train()
        model.h_short = model.h_long = None
        prev_day = None

        tr_base_preds_tensors, tr_tot_preds_tensors, tr_targs_tensors = [], [], []
        tr_delta_preds_tensors, tr_delta_targs_tensors, tr_lengths = [], [], []
        
        # OPTIMIZATION: Accumulate loss on GPU as tensors to avoid .item() syncs in the loop
        loss_total_acc = torch.tensor(0.0, device=device)
        loss_base_acc = torch.tensor(0.0, device=device)
        loss_delta_acc = torch.tensor(0.0, device=device)
        epoch_loss_count = epoch_samples = 0
        
        # --- TRAINING PHASE ---
        train_start = time.time()

        for x_batch, y_signal, rc, wd, ts_list, seq_lengths in tqdm(train_loader, desc=f"Epoch {epoch} ▶ Train", leave=False, smoothing=0):
            x_batch = x_batch.to(device, non_blocking=True)
            y_signal = y_signal.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            windows_tensor, targets_tensor, prev_day = _prepare_windows_and_targets_batch(
                x_batch, y_signal, seq_lengths, wd, _reset_states, model, prev_day
            )
            if windows_tensor is None: continue

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                base_tensor, delta_tensor = model(windows_tensor)  
                total_tensor = base_tensor + delta_tensor 
                
                base_tensor, delta_tensor, total_tensor = base_tensor.view(-1), delta_tensor.view(-1), total_tensor.view(-1)
                base_loss = MSE_base(base_tensor, targets_tensor, seq_lengths)
                
                if model.use_delta and model_hparams["LAMBDA_DELTA"] > 0:
                    delta_target = (targets_tensor - base_tensor).detach()
                    delta_loss = MSE_delta(delta_tensor, delta_target, seq_lengths)
                    tr_delta_preds_tensors.append(delta_tensor.detach())
                    tr_delta_targs_tensors.append(delta_target)
                else:
                    delta_loss = torch.tensor(0.0, device=device)

                tr_base_preds_tensors.append(base_tensor.detach())
                tr_tot_preds_tensors.append(total_tensor.detach())
                tr_targs_tensors.append(targets_tensor.detach())
                
                tr_lengths += [L for L in seq_lengths if L > 0]
                epoch_samples += int(targets_tensor.size(0))
                total_loss = base_loss + model_hparams["LAMBDA_DELTA"] * delta_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)

            params_with_grad = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            if params_with_grad: 
                torch.nn.utils.clip_grad_norm_(params_with_grad, model_hparams["CLIPNORM"])

            # OPTIMIZATION: Standard GradScaler state check instead of lambda monkey-patching
            before_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            
            # If the scale didn't decrease, the optimizer.step was successfully called
            if scaler.get_scale() >= before_scale:
                scheduler.step()
  
            # Accumulate values on GPU (detaching to avoid graph retention)
            loss_total_acc += total_loss.detach()
            loss_base_acc += base_loss.detach()
            loss_delta_acc += delta_loss.detach()
            epoch_loss_count += 1

        train_stop = time.time()
        val_start_time = time.time()

        # --- DATA PROCESSING & VALIDATION ---
        # Moving to CPU in one bulk move at the end of the epoch
        tr_tot_preds = torch.cat(tr_tot_preds_tensors).cpu().numpy().astype(np.float64) if tr_tot_preds_tensors else np.array([], dtype=np.float64)
        tr_targs     = torch.cat(tr_targs_tensors).cpu().numpy().astype(np.float64) if tr_targs_tensors else np.array([], dtype=np.float64)

        model.bl_tr_mean, model.bl_tr_pers = compute_baselines(tr_targs, tr_lengths)
        tr_tot_metrics = _compute_metrics(tr_tot_preds, tr_targs)
        
        val_res, val_base_res, val_tot_preds, val_base_preds, val_targs = eval_on_loader(val_loader, model)
        val_stop_time = time.time()

        # Timing and Speeds
        train_elapsed = train_stop - train_start
        val_elapsed = val_stop_time - val_start_time
        train_speed = epoch_loss_count / max(0.1, train_elapsed)
        val_speed = len(val_loader) / max(0.1, val_elapsed)
        
        # Only pull results from GPU to CPU once per epoch
        avg_loss = loss_total_acc.item() / max(1, epoch_loss_count)
        avg_base_loss = loss_base_acc.item() / max(1, epoch_loss_count)
        avg_delta_loss = loss_delta_acc.item() / max(1, epoch_loss_count)

        # Logging
        models_core.log_epoch_summary(
            epoch, model, optimizer, tr_tot_metrics=tr_tot_metrics, tr_base_metrics=_compute_metrics(tr_tot_preds, tr_targs),
            tr_delta_metrics={"rmse":0, "mae":0, "r2":0}, val_tot_metrics=val_res, val_base_metrics=val_base_res,
            val_tot_preds=val_tot_preds, val_base_preds=val_base_preds, val_targs=val_targs,
            avg_base_loss=avg_base_loss, avg_delta_loss=avg_delta_loss, 
            log_file=params.log_file, hparams=model_hparams,
        )
        live_plot.update(tr_tot_metrics["rmse"], val_res["rmse"])

        best_val, improved, *_ = models_core.maybe_save_chkpt(
            models_dir, model, val_res["rmse"], best_val, model_feats, model_hparams,
            {"rmse": tr_tot_metrics["rmse"]}, {"rmse": val_res["rmse"]}, live_plot
        )

        # --- DETAILED SUMMARY PRINT ---
        print(f"Epoch {epoch:02d} | "
              f"TRAIN→ RMSE={tr_tot_metrics['rmse']:.5f}, R²={tr_tot_metrics['r2']:.3f}, {train_elapsed:.1f}s ({train_speed:.1f} batch/s) | "
              f"VALID→ RMSE={val_res['rmse']:.5f}, R²={val_res['r2']:.3f}, {val_elapsed:.1f}s ({val_speed:.1f} batch/s) | "
              f"loss={avg_loss:.5e} | improved={bool(improved)}")

        if improved:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= model_hparams["EARLY_STOP_PATIENCE"]:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        models_core.save_final_chkpt(models_dir, best_state, best_val, model_feats, model_hparams, tr_tot_metrics, val_res, live_plot, suffix="_fin")

    return best_val

    
######################################################################################################


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
    Stamps predictions onto the main DataFrame using high-speed vectorization.
    
    Functionality:
    - Validates prediction lengths against timestamps.
    - Adds synthetic bid/ask spreads.
    - Uses Pandas .loc vectorization for instant data mapping (No tqdm needed).
    - Splits the final results into train+val and test sets.
    """
    start_ts = time.time()
    
    # 1. Length checks
    print("📋 Validating prediction lengths...")
    for name, preds, times in (
        ("train", train_preds, end_times_tr),
        ("val",   val_preds,   end_times_val),
        ("test",  test_preds,  end_times_te),
    ):
        if len(np.asarray(preds).ravel()) != len(np.asarray(times)):
            raise ValueError(f"{name} preds length != times length: {len(preds)} != {len(times)}")

    df = df.copy()
    df["pred_signal"] = np.nan
    
    # 2. Add synthetic bid/ask spread
    df["ask"] = df["close_raw"] * (1 + params.bidask_spread_pct / 100.0)
    df["bid"] = df["close_raw"] * (1 - params.bidask_spread_pct / 100.0)

    # 3. Helper to build validated Pandas Series
    def _series(preds, times, name):
        idx = pd.to_datetime(times)
        if pd.Index(idx).has_duplicates:
            dup = pd.Index(idx)[pd.Index(idx).duplicated(keep=False)][:5]
            raise ValueError(f"{name} timestamps contain duplicates, e.g. {dup.tolist()}")
        return pd.Series(np.asarray(preds).ravel(), index=idx)

    s_tr  = _series(train_preds, end_times_tr, "train")
    s_val = _series(val_preds,   end_times_val, "val")
    s_te  = _series(test_preds,  end_times_te,  "test")

    # 4. VECTORIZED STAMPING (Rocket Speed)
    print(f"🚀 Vector-stamping {len(s_tr)+len(s_val)+len(s_te):,} predictions onto DataFrame...")
    for desc, s in (("train", s_tr), ("val", s_val), ("test", s_te)):
        missing = s.index.difference(df.index)
        if not missing.empty:
            raise KeyError(f"{desc} timestamps not in df.index; e.g. {missing[:5].tolist()}")
        
        # This is the line that makes tqdm impossible because it's too fast
        df.loc[s.index, "pred_signal"] = s.values

    # 5. Split outputs
    print("✂️ Splitting DataFrames...")
    idx_trval = s_tr.index.union(s_val.index)
    idx_te    = s_te.index
    
    df_trainval = df.loc[idx_trval].dropna(subset=["pred_signal"])
    df_test     = df.loc[idx_te].dropna(subset=["pred_signal"])

    print(f"✨ Stamping complete in {time.time() - start_ts:.2f}s")
    return df_trainval, df_test