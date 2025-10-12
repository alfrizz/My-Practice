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
        conv_k:          int,
        conv_dilation:   int,
        pred_hidden:     int
    ):
        super().__init__()

        # store sizes
        self.n_feats     = n_feats
        self.short_units = short_units
        # long_units now defines projection/feature dim that pred expects
        self.long_units  = long_units
        self.pred_hidden = pred_hidden

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

        # 3) Regression head (time-distributed) and optional small head MLP to shift capacity upstream
        self.pred = nn.Sequential(
                    nn.Linear(self.long_units, self.pred_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.pred_hidden, 1)
                )

        # 4) Lazy hidden states (only short LSTM now)
        self.h_short = self.c_short = None
        # keep long buffers but unused (to avoid breaking external code)
        self.h_long  = self.c_long  = None

    def _init_states(self, B: int):
        device = next(self.parameters()).device
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
            self._init_states(B)

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
            self._init_states(B)

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

class SmoothMSELoss(nn.Module):
    """
    Combines pointwise MSE with a penalty on 1-step differences
    to force model predictions to match both level and slope of the target.
    """
    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.level_loss = nn.MSELoss()
        self.alpha      = alpha

    def forward(self, preds: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        # preds/targs shape: (B,), or (B, W) if batched windows
        L1 = self.level_loss(preds, targs)

        # Compute one-step differences along time dimension
        if preds.dim() == 1:
            dp, dt = preds[1:] - preds[:-1], targs[1:] - targs[:-1]
        else:
            dp, dt = preds[:,1:] - preds[:,:-1], targs[:,1:] - targs[:,:-1]

        L2 = self.level_loss(dp, dt)
        return L1 + self.alpha * L2
        
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
    
    # print(f"###### BASELINE CHECK on y_sig: nexts mean±std: "f"{nexts.mean():.5f} ± {nexts.std():.5f}") ################################
    
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

def _reset_states(model: nn.Module, wd_i: torch.Tensor, prev_day: int | None) -> int:
    """
    Reset the model’s short‐term state at each new calendar day.
    """
    day = int(wd_i.item())
    if prev_day is None or day != prev_day:
        model.reset_short()
    return day

############### 

def eval_on_loader(loader, model: nn.Module, clamp_preds: bool = True) -> tuple[dict,np.ndarray]:
    """
    One‐step‐per‐window evaluation.

    For each sliding window:
      1. Reset LSTM short‐term state on day rollover.
      2. Skip zero‐length windows.
      3. Run model(x_windows) → raw_out.
      4. Unpack & head‐apply → raw_reg of shape (W, look_back, 1).
      5. Squeeze → (W, look_back).
      6. Take preds = raw_reg[:, -1]; targs = y_sig[i, :W].
      7. Accumulate all preds & targs.

    Returns:
      metrics: dict with keys "rmse", "mae", "r2" computed once over
               the flat arrays of all predictions vs. all targets.
    """
    device = next(model.parameters()).device
    model.to(device).eval()
    model.h_short = model.h_long = None
    prev_day = None

    all_preds, all_targs = [], []
    with torch.no_grad():
        for x_pad, y_sig, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths in tqdm(loader, desc="eval", leave=False):

            # print(f"###### EVAL LOOP y_sig "f"mean±std: {y_sig.mean():.5f} ± {y_sig.std():.5f}") ############################

            x_pad = x_pad.to(device)
            y_sig = y_sig.to(device)
            wd    = wd.to(device)

            B = x_pad.size(0)
            for i in range(B):
                prev_day = _reset_states(model, wd[i], prev_day)
                W = int(lengths[i])
                if W == 0:
                    continue

                seqs    = x_pad[i, :W]                              # (W, look_back, F)
                raw_out = model(seqs)
                raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

                # force through regression head if needed
                if raw_reg.dim() == 3 and raw_reg.size(-1) != 1:
                    raw_reg = model.pred(raw_reg)
                elif raw_reg.dim() == 2:
                    raw_reg = model.pred(raw_reg.unsqueeze(0)).squeeze(0)

                raw_reg   = raw_reg.squeeze(-1)                  # → (W, look_back)
                preds_win = raw_reg[:, -1]                       # (W,)
                targs_win = y_sig[i, :W].view(-1)                # (W,)

                all_preds.extend(preds_win.cpu().tolist())
                all_targs.extend(targs_win.cpu().tolist())

    preds = np.array(all_preds, dtype=float)
    targs = np.array(all_targs, dtype=float)
    
    # if clamp_preds and preds.size:
    #     preds = np.clip(preds, 0.0, 1.0)

    # attach Torch tensors to model for logger
    model.last_val_preds  = torch.from_numpy(preds).float()
    model.last_val_targs  = torch.from_numpy(targs).float()

    return _compute_metrics(preds, targs), preds


############### 


def model_training_loop(
    model:           nn.Module,
    optimizer:       torch.optim.Optimizer,
    scheduler:       torch.optim.lr_scheduler.OneCycleLR,
    scaler:          GradScaler,
    train_loader, val_loader,
    *, max_epochs:         int,
       early_stop_patience: int,
       clipnorm:           float,
       alpha_smooth:       int,
) -> float:
    """
    Stateful training + per‐epoch validation + baseline logging,
    with linear warmup followed by cosine restarts, and console printing
    of TRAIN/VALID RMSE, R², and current LR each epoch.

    1. Precompute four static baselines via compute_baselines().
    2. For each epoch:
       • Linear warmup of LR over the first LR_EPOCHS_WARMUP epochs.
       • After warmup, step the CosineAnnealingWarmRestarts scheduler.
       • TRAIN loop: iterate days, reset day‐state, batch all windows,
         accumulate per‐window MSELoss and collect preds/targs.
       • Compute train_metrics via _compute_metrics.
       • Compute val_metrics via eval_on_loader.
       • Call log_epoch_summary with all metrics and baselines.
       • Print a one‐line summary: TRAIN RMSE/R², VALID RMSE/R², LR.
       • Checkpoint, update early‐stop, and optionally break.
    Returns:
      best_val_rmse: float
    """
    device = next(model.parameters()).device
    model.to(device)

    # 1) Precompute static baselines
    base_tr_mean, base_tr_pers = compute_baselines(train_loader)
    base_vl_mean, base_vl_pers = compute_baselines(val_loader)

    mse_loss  = nn.MSELoss()
    live_plot = plots.LiveRMSEPlot()
    best_val, best_state, patience = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):

        # --- TRAIN PHASE ---
        model.train()
        # model.h_short = model.h_long = None
        train_preds, train_targs = [], []
        prev_day = None

        for x_pad, y_sig, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths in tqdm(     
            train_loader, desc=f"Epoch {epoch} ▶ Train", leave=False
        ):
            x_pad, y_sig, wd = x_pad.to(device), y_sig.to(device), wd.to(device)
            optimizer.zero_grad(set_to_none=True)

            batch_loss, windows = 0.0, 0
            B = x_pad.size(0)

            # Iterate each day in the batch
            for i in range(B):
                prev_day = _reset_states(model, wd[i], prev_day)
                W = int(lengths[i])
                if W == 0:
                    continue

                # Forward all windows of day i
                x_win   = x_pad[i, :W]                        
                raw_out = model(x_win)
                raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

                # Ensure regression head applied
                if raw_reg.dim() == 3 and raw_reg.size(-1) != 1:
                    raw_reg = model.pred(raw_reg)
                elif raw_reg.dim() == 2:
                    raw_reg = model.pred(raw_reg.unsqueeze(0)).squeeze(0)

                seq_preds = raw_reg.squeeze(-1)      # (W,)
                preds_win = seq_preds[:, -1]         # final window-end preds
                targs_win = y_sig[i, :W].view(-1)    # true window-end targets

                # Accumulate loss & collect preds/targs
                smooth_loss = SmoothMSELoss(alpha_smooth)  
                loss = smooth_loss(preds_win, targs_win)
                batch_loss += loss
                windows   += 1

                train_preds.extend(preds_win.detach().cpu().tolist())
                train_targs.extend(targs_win.detach().cpu().tolist())

            # Backprop once per batch
            batch_loss = batch_loss / max(1, windows)
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # --- METRICS & VALIDATION ---
        train_metrics = _compute_metrics(
            np.array(train_preds, dtype=float),
            np.array(train_targs, dtype=float),
        )
        val_metrics, _ = eval_on_loader(val_loader, model)

        # --- LOG & CHECKPOINT ---
        tr_rmse, tr_mae, tr_r2 = (
            train_metrics["rmse"],
            train_metrics["mae"],
            train_metrics["r2"],
        )
        vl_rmse, vl_mae, vl_r2 = (
            val_metrics["rmse"],
            val_metrics["mae"],
            val_metrics["r2"],
        )
        lr = optimizer.param_groups[0]["lr"]

        # Append to log file and update live plot
        models_core.log_epoch_summary(
            epoch,
            model,
            optimizer,
            train_metrics   = train_metrics,
            val_metrics     = val_metrics,
            base_tr_mean    = base_tr_mean,
            base_tr_pers    = base_tr_pers,
            base_vl_mean    = base_vl_mean,
            base_vl_pers    = base_vl_pers,
            slip_thresh     = 1e-6,
            log_file        = params.log_file,
            top_k           = 999,
            hparams         = params.hparams,
        )
        live_plot.update(tr_rmse, vl_rmse)

        # Console summary: TRAIN/VALID RMSE, R², and LR
        print(
            f"Epoch {epoch:02d}  "
            f"TRAIN → RMSE={tr_rmse:.5f}, R²={tr_r2:.3f} |  "
            f"VALID → RMSE={vl_rmse:.5f}, R²={vl_r2:.3f} |  "
            f"lr={lr:.2e}"
        )

        # Checkpointing & early stopping
        models_dir = Path(params.models_folder)
        best_val, improved, *_ = models_core.maybe_save_chkpt(
            models_dir, model, vl_rmse, best_val,
            {"rmse": tr_rmse}, {"rmse": vl_rmse},
            live_plot, params
        )

        if improved:
            best_state, patience = {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # --- FINAL CHECKPOINT RESTORE ---
    if best_state is not None:
        model.load_state_dict(best_state)
        models_core.save_final_chkpt(
            models_dir, best_state, best_val, params,
            {}, {}, live_plot, suffix="_fin"
        )

    return best_val
