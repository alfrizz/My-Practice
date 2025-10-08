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


def _compute_metrics(preds: np.ndarray, targs: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, and R² on flat NumPy arrays.
    """
    if preds.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    diff = preds - targs
    mse  = float((diff**2).mean())
    mae  = float(np.abs(diff).mean())
    rmse = float(np.sqrt(mse))
    if preds.size < 2 or np.isclose(targs.var(), 0.0):
        r2 = float("nan")
    else:
        tss = float(((targs - targs.mean())**2).sum())
        rss = float((diff**2).sum())
        r2  = float("nan") if tss == 0.0 else 1.0 - (rss / tss)
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ###############


def _reset_states(model: nn.Module, wd_i: torch.Tensor, prev_day: int | None) -> int:
    day = int(wd_i.item())
    if prev_day is None or day != prev_day:
        model.reset_short()
    return day

############

def eval_on_loader(loader, model: nn.Module, clamp_preds: bool = True):
    """
    One‐step‐per‐window evaluation.  For each sliding window in `loader`,
    emit exactly one prediction (the final timestep) plus its target & timestamp.
    """
    device = next(model.parameters()).device
    model.to(device).eval()
    model.h_short = model.h_long = None
    prev_day = None

    all_preds, all_targs, all_times = [], [], []

    with torch.no_grad():
        for xb, y_reg, *_ignored, wd, ts_list, lengths in tqdm(loader, desc="eval", leave=False):
            xb, y_reg, wd = (
                xb.to(device, non_blocking=True),
                y_reg.to(device, non_blocking=True),
                wd.to(device, non_blocking=True),
            )

            B = xb.size(0)
            for i in range(B):
                prev_day = _reset_states(model, wd[i], prev_day)

                W = int(lengths[i])   # number of windows in this batch‐item
                if W == 0:
                    continue

                # x_windows: shape (W, look_back, n_feats)
                x_windows = xb[i, :W]
                raw_out   = model(x_windows)

                # unpack tuple if needed
                raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

                # if the last dim ≠1 or dims==2, force through your head:
                if raw_reg.dim() == 3 and raw_reg.size(-1) != 1:
                    raw_reg = model.pred(raw_reg)             # → (W, look_back, 1)
                elif raw_reg.dim() == 2:
                    raw_reg = model.pred(raw_reg.unsqueeze(0))# → (1, W, 1)
                    raw_reg = raw_reg.squeeze(0)              # → (W, 1, 1)

                # raw_reg now (W, look_back, 1)
                raw_reg = raw_reg.squeeze(-1)  # → (W, look_back)

                # final‐timestep predictions: shape (W,)
                preds_win = raw_reg[:, -1]
                targs_win = y_reg[i, :W].view(-1)  # also (W,)
                times_win = ts_list[:W]            # list of length W

                all_preds.extend(preds_win.cpu().tolist())
                all_targs.extend(targs_win.cpu().tolist())
                all_times.extend(times_win)

    preds = np.array(all_preds, dtype=float)
    targs = np.array(all_targs, dtype=float)
    times = np.array(all_times, dtype=object)

    if clamp_preds and preds.size:
        preds = np.clip(preds, 0.0, 1.0)

    metrics = _compute_metrics(preds, targs)
    return metrics, preds, targs, times


def model_training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cosine_sched: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    scaler: GradScaler,
    train_loader, val_loader,
    *, max_epochs: int, early_stop_patience: int, clipnorm: float
):
    """
    Stateful training: for each window, predict only the last timestep, do MSE against y_reg.
    Then per‐epoch, run eval_on_loader for validation, log, checkpoint, early stop.
    """
    device = next(model.parameters()).device
    model.to(device)
    mse_loss = nn.MSELoss()
    live_plot = plots.LiveRMSEPlot()

    best_val, best_state, patience = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        model.h_short = model.h_long = None
        train_se, train_count = 0.0, 0
        train_preds, train_targs = [], []
        prev_day = None

        for xb, y_reg, *_ignored, wd, ts_list, lengths in tqdm(
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

                x_windows = xb[i, :W]       # (W, look_back, n_feats)
                raw_out   = model(x_windows)
                raw_reg   = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

                # ensure we have (W, look_back, 1)
                if raw_reg.dim() == 3 and raw_reg.size(-1) != 1:
                    raw_reg = model.pred(raw_reg)
                elif raw_reg.dim() == 2:
                    raw_reg = model.pred(raw_reg.unsqueeze(0)).squeeze(0)

                raw_reg = raw_reg.squeeze(-1)    # → (W, look_back)

                # pick only the last‐timestep per window
                preds_win = raw_reg[:, -1]       # (W,)
                targs_win = y_reg[i, :W].view(-1)

                # compute MSE over these W scalars
                loss = mse_loss(preds_win, targs_win)
                batch_loss += loss
                windows    += 1

                # accumulate stats & lists
                p_np = preds_win.detach().cpu().numpy()
                t_np = targs_win.detach().cpu().numpy()
                train_se    += float(((p_np - t_np)**2).sum())
                train_count += W
                train_preds .extend(p_np.tolist())
                train_targs .extend(t_np.tolist())

            # gradient step
            batch_loss = batch_loss / max(1, windows)
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()

        # end of train epoch: compute metrics & scheduler
        train_arr = np.array(train_preds)
        targ_arr  = np.array(train_targs)
        train_res = _compute_metrics(train_arr, targ_arr)
        cosine_sched.step(epoch)

        # validate
        val_res, val_preds, val_targs, _ = eval_on_loader(val_loader, model)
        train_rmse, val_rmse = train_res["rmse"], val_res["rmse"]

        # log + checkpoint
        models_core.log_epoch_summary(
            epoch, model, optimizer,
            train_preds  = train_arr,  train_targs = targ_arr,
            val_preds    = val_preds,  val_targs   = val_targs,
            batch_mse    = train_se / max(1, train_count),
            base_tr_rmse = float(np.std(train_targs)),
            base_vl_rmse = float(np.std(val_targs)),
            slip_thresh  = 1e-6,
            log_file     = params.log_file,
            top_k        = 999,
            hparams      = params.hparams,
        )
        live_plot.update(train_rmse, val_rmse)

        models_dir = Path(params.models_folder)
        best_val, improved, *_ = models_core.maybe_save_chkpt(
            models_dir, model, val_rmse, best_val,
            {"rmse": train_rmse}, {"rmse": val_rmse},
            live_plot, params
        )

        if improved:
            best_state, patience = {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch:02d}  TRAIN RMSE={train_rmse:.5f}  VALID RMSE={val_rmse:.5f}")

    # restore best model & final checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        models_core.save_final_chkpt(
            models_dir, best_state, best_val, params,
            {}, {}, live_plot, suffix="_fin"
        )

    return best_val


############### ---------------------- $$$$$$$$$$$$$$$$$$$ &&&&&&&&&&&&&& 333333333333333333333 




########### 


def compute_baseline(loader):
    arr = []
    for xb, y_r, *_, lengths in loader:
        for i, L in enumerate(lengths):
            if L > 0:
                arr.append(float(y_r[i, :L].view(-1)[-1].item()))
    return np.std(arr)  # ddof=0