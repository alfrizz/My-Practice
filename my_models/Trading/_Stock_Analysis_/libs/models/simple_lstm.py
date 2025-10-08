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


def compute_baselines(loader) -> tuple[float, float]:
    """
    Compute two static one‐step‐per‐window baselines over all windows in `loader`:
      1) mean_rmse       – RMSE if you always predict μ = average next‐price per window
      2) persistence_rmse – RMSE if you always predict last‐seen (yₜ = yₜ₋₁) at window end

    Returns:
      mean_rmse, persistence_rmse
    """
    # Collect exactly one “next‐price” and one “persistence‐error” per window
    nexts, last_deltas = [], []
    for xb, y_r, *_ignore, wd, ts_list, lengths in loader:
        for i, L in enumerate(lengths):
            if L < 1:
                continue
            y = y_r[i, :L].view(-1).cpu().numpy()
            # “next‐price” target at window end
            nexts.append(y[-1])

            # For persistence baseline, predict y[-1] as y[-2]
            if L > 1:
                last_deltas.append(y[-1] - y[-2])

    # Baseline #1: always predict global μ of the per-window next‐prices
    nexts = np.array(nexts, dtype=float)
    mean_rmse = float(np.sqrt(((nexts - nexts.mean()) ** 2).mean()))

    # Baseline #2: always predict no-change at end of each window
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
      6. Take preds = raw_reg[:, -1]; targs = y_reg[i, :W].
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
        for xb, y_reg, *_ignored, wd, ts_list, lengths in tqdm(loader, desc="eval", leave=False):
            xb, y_reg, wd = xb.to(device), y_reg.to(device), wd.to(device)
            B = xb.size(0)

            for i in range(B):
                prev_day = _reset_states(model, wd[i], prev_day)
                W = int(lengths[i])
                if W == 0:
                    continue

                seqs    = xb[i, :W]                              # (W, look_back, F)
                raw_out = model(seqs)
                raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

                # force through regression head if needed
                if raw_reg.dim() == 3 and raw_reg.size(-1) != 1:
                    raw_reg = model.pred(raw_reg)
                elif raw_reg.dim() == 2:
                    raw_reg = model.pred(raw_reg.unsqueeze(0)).squeeze(0)

                raw_reg   = raw_reg.squeeze(-1)                  # → (W, look_back)
                preds_win = raw_reg[:, -1]                       # (W,)
                targs_win = y_reg[i, :W].view(-1)                # (W,)

                all_preds.extend(preds_win.cpu().tolist())
                all_targs.extend(targs_win.cpu().tolist())

    preds = np.array(all_preds, dtype=float)
    targs = np.array(all_targs, dtype=float)
    # if clamp_preds and preds.size:
    #     preds = np.clip(preds, 0.0, 1.0)

    return _compute_metrics(preds, targs), preds


############### 


def model_training_loop(
    model:           nn.Module,
    optimizer:       torch.optim.Optimizer,
    cosine_sched:    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    scaler:          GradScaler,
    train_loader, val_loader,
    *, max_epochs:         int,
       early_stop_patience: int,
       clipnorm:           float
) -> float:
    """
    Stateful training + per‐epoch validation + baseline logging.

    1. Precompute four static baselines via compute_baselines().
    2. For each epoch:
       • Train on each window → collect train_preds/train_targs.
       • Compute train_metrics = _compute_metrics(train_preds, train_targs).
       • Step LR.
       • val_metrics = eval_on_loader(val_loader, model).
       • Call log_epoch_summary(epoch, model, optimizer,
                                train_metrics, val_metrics,
                                base_tr_mean, base_tr_pers,
                                base_vl_mean, base_vl_pers,
                                slip_thresh, log_file, top_k, hparams).
       • Checkpoint & early stop.
    Returns:
      best_val_rmse: float
    """
    device = next(model.parameters()).device
    model.to(device)

    # Precompute mean & persistence baselines
    base_tr_mean, base_tr_pers = compute_baselines(train_loader)
    base_vl_mean, base_vl_pers = compute_baselines(val_loader)

    mse_loss  = nn.MSELoss()
    live_plot = plots.LiveRMSEPlot()
    best_val, best_state, patience = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        models_core.linear_warmup(
            optimizer,
            epoch,
            params.hparams["LR_EPOCHS_WARMUP"],
            params.hparams["INITIAL_LR"],
        )
        model.train()
        model.h_short = model.h_long = None
        train_preds, train_targs = [], []
        prev_day = None

        # — TRAIN LOOP —
        for xb, y_reg, *_ignore, wd, ts_list, lengths in tqdm(
            train_loader, desc=f"Epoch {epoch} ▶ Train", leave=False
        ):
            xb, y_reg, wd = xb.to(device), y_reg.to(device), wd.to(device)
            optimizer.zero_grad(set_to_none=True)

            batch_loss, windows = 0.0, 0
            B = xb.size(0)

            for i in range(B):
                prev_day = _reset_states(model, wd[i], prev_day)
                W = int(lengths[i])
                if W == 0:
                    continue

                x_win    = xb[i, :W]
                raw_out  = model(x_win)
                raw_reg  = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

                if raw_reg.dim() == 3 and raw_reg.size(-1) != 1:
                    raw_reg = model.pred(raw_reg)
                elif raw_reg.dim() == 2:
                    raw_reg = model.pred(raw_reg.unsqueeze(0)).squeeze(0)

                seq_preds = raw_reg.squeeze(-1)      # → (W, look_back)
                preds_win = seq_preds[:, -1]         # → (W,)
                targs_win = y_reg[i, :W].view(-1)    # → (W,)

                loss = mse_loss(preds_win, targs_win)
                batch_loss += loss
                windows    += 1

                p_np = preds_win.detach().cpu().numpy()
                t_np = targs_win.detach().cpu().numpy()
                train_preds .extend(p_np.tolist())
                train_targs .extend(t_np.tolist())

            batch_loss = batch_loss / max(1, windows)
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()

        # Compute training metrics once
        train_preds_arr = np.array(train_preds, dtype=float)
        train_targs_arr = np.array(train_targs, dtype=float)
        train_metrics   = _compute_metrics(train_preds_arr, train_targs_arr)

        # Compute validation metrics once
        val_metrics, _ = eval_on_loader(val_loader, model)

        # Then update scheduler for next epoch:
        cosine_sched.step(epoch)

        # Extract scalars for logging, plotting, checkpointing
        tr_rmse = train_metrics["rmse"]
        vl_rmse = val_metrics["rmse"]
        
        # Log & checkpoint
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

        print(f"Epoch {epoch:02d}  TRAIN RMSE={tr_rmse:.5f}  VALID RMSE={vl_rmse:.5f}")

    # Restore best model & final checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        models_core.save_final_chkpt(
            models_dir, best_state, best_val, params,
            {}, {}, live_plot, suffix="_fin"
        )

    return best_val

