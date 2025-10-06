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
    Return a dict of torchmetrics Metric objects on `device` for regression.

    Metrics:
      - rmse: Root Mean Squared Error (MeanSquaredError(squared=False))
      - mae : Mean Absolute Error
      - r2  : R2 score

    Note:
      - These metric objects remain available for callers that want torchmetrics
        stateful objects, but the training/eval loops compute deterministic
        epoch-level numeric metrics from flattened arrays to avoid torchmetrics'
        compute fragility in small-sample cases.
    """
    return {
        "rmse": torchmetrics.MeanSquaredError(squared=False).to(device),
        "mae": torchmetrics.MeanAbsoluteError().to(device),
        "r2": torchmetrics.R2Score().to(device),
    }


# ####################### 

def update_metrics(
    metrics: dict[str, torchmetrics.Metric],
    pr_seq: torch.Tensor,
    t_r: torch.Tensor,
):
    """
    Minimal updater for regression metrics.

    Expectations:
      - pr_seq and t_r are 1-D torch Tensors on the same device and dtype.
      - Only regression metrics present in `metrics` are updated.
      - Function purpose is focused and deterministic; it does not attempt
        classification updates or defensive try/except checks.
    """
    if pr_seq is None or t_r is None:
        raise ValueError("pr_seq and t_r (regression preds/targets) are required")

    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dict of torchmetrics objects")

    # Update only regression metrics if present
    if "rmse" in metrics:
        metrics["rmse"].update(pr_seq, t_r)
    if "mae" in metrics:
        metrics["mae"].update(pr_seq, t_r)
    if "r2" in metrics:
        metrics["r2"].update(pr_seq, t_r)


####################### 


# def eval_on_loader(
#     loader,
#     model: torch.nn.Module,
#     device: torch.device,
#     metrics: dict | None = None,
#     collect_preds: bool = False,
#     disable_tqdm: bool = False,
#     clamp_preds: bool = True,
#     debug_sample: bool = False
# ):
#     """
#     Evaluate model on loader and compute regression metrics deterministically.

#     Behavior:
#       - Treats the regression head as linear (no sigmoid) to match training.
#       - Accepts model.forward returning either:
#           * raw_reg (Tensor) or
#           * (raw_reg, raw_cls, raw_ter) tuple/list.
#       - Collects regression predictions and targets across windows in chronological order.
#       - By default clamps predictions into [0,1] before metric computation (clamp_preds=True).
#         Disable if your targets are unbounded and you prefer raw outputs.
#       - Computes numeric RMSE, MAE, R2 at epoch end using numpy (no per-batch torchmetrics.compute).
#       - If `metrics` is provided (torchmetrics dict), updates regression metric objects once at epoch end.
#       - If collect_preds=True returns (results_dict, preds_array) else (results_dict, None).

#     Useful toggles:
#       - clamp_preds: clamp eval predictions before metric computation (default True).
#       - debug_sample: print first window's pred/targ pairs for quick alignment checks.
#     """
#     import numpy as np

#     preds_list = []
#     targs_list = []
#     expected_total_preds = 0
#     full_end_times = getattr(loader.dataset, "end_times", None)

#     model.eval()
#     model.h_short = model.h_long = None
#     prev_day = None

#     with torch.no_grad():
#         loop = tqdm(loader, desc="eval", unit="batch", disable=disable_tqdm)
#         for batch in loop:
#             # Unpack DayWindowDataset output
#             xb, y_reg, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths = batch

#             xb    = xb.to(device, non_blocking=True)
#             y_reg = y_reg.to(device, non_blocking=True)
#             wd    = wd.to(device, non_blocking=True)

#             B = xb.size(0)
#             for i in range(B):
#                 W_day  = int(lengths[i])
#                 if W_day == 0:
#                     continue

#                 day_id = int(wd[i].item())

#                 # stateful resets
#                 model.reset_short()
#                 if prev_day is not None and day_id < prev_day:
#                     model.reset_long()
#                 prev_day = day_id

#                 x_seq = xb[i, :W_day]
#                 out = model(x_seq)
#                 raw_reg = out[0] if isinstance(out, (tuple, list)) else out

#                 if raw_reg is None:
#                     raise RuntimeError("Model returned None for regression output")

#                 # Index final timestep logits exactly as training: use dim>=3 branch
#                 if raw_reg.dim() >= 3:
#                     pr_tensor = raw_reg[..., -1, 0]
#                 else:
#                     pr_tensor = raw_reg[..., 0]

#                 # Convert to explicit 1-D numpy arrays (detached)
#                 pr_seq = pr_tensor.detach().cpu().numpy().reshape(-1)
#                 t_r = y_reg[i, :W_day].view(-1).cpu().numpy().reshape(-1)

#                 # Optional debug print for first window processed
#                 if debug_sample and len(preds_list) == 0:
#                     print("EVAL DEBUG sample pred[:10], targ[:10]:", pr_seq[:10], t_r[:10])

#                 preds_list.append(pr_seq)
#                 targs_list.append(t_r)
#                 expected_total_preds += pr_seq.shape[0]

#     # Flatten collected arrays
#     if preds_list:
#         preds_flat = np.concatenate([np.asarray(p).reshape(-1) for p in preds_list])
#         targs_flat = np.concatenate([np.asarray(t).reshape(-1) for t in targs_list])
#     else:
#         preds_flat = np.array([], dtype=float)
#         targs_flat = np.array([], dtype=float)

#     # Optional clamp for bounded targets
#     if clamp_preds and preds_flat.size > 0:
#         preds_flat = np.clip(preds_flat, 0.0, 1.0)

#     # Sanity check if collection requested
#     if collect_preds:
#         if full_end_times is not None:
#             total_from_dataset = len(full_end_times)
#             if preds_flat.size != total_from_dataset:
#                 raise AssertionError(
#                     f"Collected preds ({preds_flat.size}) != dataset end_times length ({total_from_dataset})."
#                 )
#         else:
#             if preds_flat.size != expected_total_preds:
#                 raise AssertionError(
#                     f"Final collected preds ({preds_flat.size}) != expected_total_preds ({expected_total_preds})."
#                 )

#     # Compute numeric metrics deterministically
#     results = {}
#     if preds_flat.size == 0:
#         results["rmse"] = float("nan")
#         results["mae"] = float("nan")
#         results["r2"]  = float("nan")
#     else:
#         diff = preds_flat - targs_flat
#         mse = float((diff ** 2).mean())
#         mae = float(np.abs(diff).mean())
#         rmse = float(np.sqrt(mse))
#         if preds_flat.size < 2:
#             r2 = float("nan")
#         else:
#             tss = float(((targs_flat - targs_flat.mean()) ** 2).sum())
#             rss = float((diff ** 2).sum())
#             r2 = float("nan") if tss == 0.0 else 1.0 - (rss / tss)
#         results["rmse"] = rmse
#         results["mae"]  = mae
#         results["r2"]   = r2

#     # Sync torchmetrics (if provided) with epoch-level tensors once
#     if metrics is not None and preds_flat.size > 0:
#         preds_torch = torch.from_numpy(preds_flat).to(device)
#         targs_torch = torch.from_numpy(targs_flat).to(device)
#         if "rmse" in metrics:
#             metrics["rmse"].update(preds_torch, targs_torch)
#         if "mae" in metrics:
#             metrics["mae"].update(preds_torch, targs_torch)
#         if "r2" in metrics:
#             metrics["r2"].update(preds_torch, targs_torch)

#     preds_arr = preds_flat if collect_preds else None
#     return results, preds_arr

def eval_on_loader(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    metrics: dict | None = None,
    collect_preds: bool = False,
    disable_tqdm: bool = False,
    clamp_preds: bool = True,
    debug_sample: bool = False,
    return_targs: bool = False,  # new explicit opt-in flag; default False keeps old signature
):
    """
    Evaluate model on `loader` and compute regression metrics deterministically.

    Backwards-compatible return values:
      - If collect_preds is False: returns (results_dict, None)
      - If collect_preds is True and return_targs is False: returns (results_dict, preds_array)
      - If collect_preds is True and return_targs is True: returns (results_dict, preds_array, targs_array)

    Behavior and details:
      - Assumes the model's regression head is linear (no final sigmoid) to match training.
      - Accepts model(x) returning either raw_reg Tensor or (raw_reg, raw_cls, raw_ter).
      - Indexing parity with training: when raw_reg.dim() >= 3 we use raw_reg[..., -1, 0].
      - Collects per-window predictions and targets in chronological order, flattens
        them at epoch end and computes RMSE, MAE, R2 with numpy (no per-batch torchmetrics.compute).
      - By default clamps predictions into [0,1] before metric computation (clamp_preds=True).
      - If `metrics` (torchmetrics dict) is provided, does a single epoch-level .update()
        for regression metrics using the flattened tensors (keeps compatibility).
      - debug_sample prints the first processed window (pred[:10], targ[:10]) for quick sanity checks.
    """
    import numpy as np

    preds_list = []
    targs_list = []
    expected_total_preds = 0
    full_end_times = getattr(loader.dataset, "end_times", None)

    model.eval()
    model.h_short = model.h_long = None
    prev_day = None

    with torch.no_grad():
        loop = tqdm(loader, desc="eval", unit="batch", disable=disable_tqdm)
        for batch in loop:
            # DayWindowDataset unpacking (keeps original shape contract)
            xb, y_reg, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths = batch

            xb    = xb.to(device, non_blocking=True)
            y_reg = y_reg.to(device, non_blocking=True)
            wd    = wd.to(device, non_blocking=True)

            B = xb.size(0)
            for i in range(B):
                W_day = int(lengths[i])
                if W_day == 0:
                    continue

                day_id = int(wd[i].item())

                # stateful resets consistent with training
                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                x_seq = xb[i, :W_day]
                out = model(x_seq)
                raw_reg = out[0] if isinstance(out, (tuple, list)) else out

                if raw_reg is None:
                    raise RuntimeError("Model returned None for regression output")

                # Index final timestep logits exactly as training uses
                if raw_reg.dim() >= 3:
                    pr_tensor = raw_reg[..., -1, 0]
                else:
                    pr_tensor = raw_reg[..., 0]

                # convert to explicit 1-D numpy arrays
                pr_seq = pr_tensor.detach().cpu().numpy().reshape(-1)
                t_r = y_reg[i, :W_day].view(-1).cpu().numpy().reshape(-1)

                if debug_sample and len(preds_list) == 0:
                    print("EVAL DEBUG sample pred[:10], targ[:10]:", pr_seq[:10], t_r[:10])

                preds_list.append(pr_seq)
                targs_list.append(t_r)
                expected_total_preds += pr_seq.shape[0]

    # Flatten collected arrays deterministically
    if preds_list:
        preds_flat = np.concatenate([np.asarray(p).reshape(-1) for p in preds_list])
        targs_flat = np.concatenate([np.asarray(t).reshape(-1) for t in targs_list])
    else:
        preds_flat = np.array([], dtype=float)
        targs_flat = np.array([], dtype=float)

    # Optional clamp for bounded targets
    if clamp_preds and preds_flat.size > 0:
        preds_flat = np.clip(preds_flat, 0.0, 1.0)

    # Sanity check when collect_preds requested
    if collect_preds:
        if full_end_times is not None:
            total_from_dataset = len(full_end_times)
            if preds_flat.size != total_from_dataset:
                raise AssertionError(
                    f"Collected preds ({preds_flat.size}) != dataset end_times length ({total_from_dataset})."
                )
        else:
            if preds_flat.size != expected_total_preds:
                raise AssertionError(
                    f"Final collected preds ({preds_flat.size}) != expected_total_preds ({expected_total_preds})."
                )

    # Deterministic numeric metrics (numpy)
    results = {}
    if preds_flat.size == 0:
        results["rmse"] = float("nan")
        results["mae"] = float("nan")
        results["r2"] = float("nan")
    else:
        diff = preds_flat - targs_flat
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        rmse = float(np.sqrt(mse))
        if preds_flat.size < 2:
            r2 = float("nan")
        else:
            tss = float(((targs_flat - targs_flat.mean()) ** 2).sum())
            rss = float((diff ** 2).sum())
            r2 = float("nan") if tss == 0.0 else 1.0 - (rss / tss)
        results["rmse"] = rmse
        results["mae"] = mae
        results["r2"] = r2

    # Optionally sync torchmetrics objects once with flattened tensors
    if metrics is not None and preds_flat.size > 0:
        preds_torch = torch.from_numpy(preds_flat).to(device)
        targs_torch = torch.from_numpy(targs_flat).to(device)
        if "rmse" in metrics:
            metrics["rmse"].update(preds_torch, targs_torch)
        if "mae" in metrics:
            metrics["mae"].update(preds_torch, targs_torch)
        if "r2" in metrics:
            metrics["r2"].update(preds_torch, targs_torch)

    # Backwards-compatible returns:
    # - old callers expect (results, preds_arr) so preserve that when collect_preds=True
    # - if caller also passed return_targs=True, return (results, preds_arr, targs_arr)
    if collect_preds:
        if return_targs:
            return results, preds_flat, targs_flat
        return results, preds_flat
    return results, None



###################### 


# def model_training_loop(
#     model: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     cosine_sched: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
#     plateau_sched: torch.optim.lr_scheduler.ReduceLROnPlateau,
#     scaler: torch.cuda.amp.GradScaler,
#     train_loader,
#     val_loader,
#     *,
#     max_epochs: int,
#     early_stop_patience: int,
#     clipnorm: float,
#     device: torch.device,
#     mode: str = "train",
# ):
#     """
#     Minimal stateful LSTM loop with full logging.

#     - Mixed‐precision, grad clipping, Cosine per epoch + Plateau on val RMSE.
#     - Per‐window final‐step scalar target: pred = model(x_seq)[…,-1], targ = y_r[i,-1].
#     - Computes train/val RMSE each epoch.
#     - Calls models_core.log_gradient_norms and log_loss_components for your diagnostics.
#     - Keeps maybe_save_chkpt / save_final_chkpt & live_plot updates.
#     """
#     model.to(device)
#     mse = torch.nn.MSELoss()

#     if mode != "train":
#         return eval_on_loader(val_loader, model, device,
#                               get_metrics(device), collect_preds=True)

#     best_val, best_state = float("inf"), None
#     patience = 0
#     live_plot = plots.LiveRMSEPlot()

#     all_tr_targs = []
#     for xb, y_r, *_, lengths in train_loader:
#         for i, L in enumerate(lengths):
#             if L > 0:
#                 all_tr_targs.append(
#                     float(y_r[i, :L].view(-1)[-1].item())
#                 )
#     base_tr_rmse = float(np.std(all_tr_targs))
#     base_tr_r2   = 0.0

#     all_vl_targs = []
#     for xb, y_r, *_, lengths in val_loader:
#         for i, L in enumerate(lengths):
#             if L > 0:
#                 all_vl_targs.append(
#                     float(y_r[i, :L].view(-1)[-1].item())
#                 )
#     base_vl_rmse = float(np.std(all_vl_targs))
#     base_vl_r2   = 0.0

#     for epoch in range(1, max_epochs + 1):        
#         # — TRAIN —
#         model.train()
#         train_preds, train_targs = [], []
#         for xb, y_r, *_, lengths in tqdm(train_loader,
#                                          desc=f"Epoch {epoch} ▶ Train",
#                                          leave=False):
#             xb, y_r = xb.to(device), y_r.to(device)
#             optimizer.zero_grad(set_to_none=True)
    
#             batch_loss = torch.tensor(0.0, device=device)
#             num_windows = 0
    
#             for i, L in enumerate(lengths):
#                 if L == 0:
#                     continue
#                 seq = xb[i, :L]
#                 out = model(seq)
#                 pred = (out[..., -1, 0] if out.dim() >= 3 else out.view(-1))[-1]
#                 targ = y_r[i, :L].view(-1)[-1]
    
#                 l = mse(pred, targ)
#                 batch_loss += l
#                 num_windows += 1
    
#                 train_preds.append(pred.item())
#                 train_targs.append(targ.item())
    
#             batch_loss = batch_loss / num_windows
#             scaler.scale(batch_loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
#             scaler.step(optimizer)
#             scaler.update()
    
#         # 1) TRAIN metrics & epoch‐level MSE
#         train_rmse = float(batch_loss.sqrt().item())
#         tp = np.array(train_preds)
#         tt = np.array(train_targs)
#         train_r2 = (
#             1.0
#             - ((tp - tt)**2).sum()
#               / ((tt - tt.mean())**2).sum()
#         ) if tt.size > 1 and not np.isclose(tt.var(), 0.0) else float("nan")
#         epoch_mse = train_rmse ** 2
    
#         cosine_sched.step(epoch)
    
#         # — VALIDATION —
#         model.eval()
#         val_preds, val_targs = [], []
#         with torch.no_grad():
#             for xb, y_r, *_, lengths in tqdm(val_loader,
#                                              desc=f"Epoch {epoch} ▶ Valid",
#                                              leave=False):
#                 xb, y_r = xb.to(device), y_r.to(device)
#                 for i, L in enumerate(lengths):
#                     if L == 0:
#                         continue
#                     seq = xb[i, :L]
#                     out = model(seq)
#                     pred = (out[..., -1, 0] if out.dim() >= 3 else out.view(-1))[-1]
#                     targ = y_r[i, :L].view(-1)[-1]
#                     val_preds.append(pred.item())
#                     val_targs.append(targ.item())
    
#         # 2) VAL metrics & val_loss
#         vp = np.array(val_preds)
#         vt = np.array(val_targs)
#         val_rmse       = float(np.sqrt(((vp - vt) ** 2).mean()))
#         val_batch_loss = val_rmse ** 2
#         vl_r2 = (
#             1.0
#             - ((vp - vt)**2).sum()
#               / ((vt - vt.mean())**2).sum()
#         ) if vt.size > 1 and not np.isclose(vt.var(), 0.0) else float("nan")
    
#         plateau_sched.step(val_rmse)
    
#         # — LOGGING —
#         models_core.model = model
#         models_core.log_gradient_norms(epoch, 0, model, params.log_file)
    
#         rep_targ  = torch.tensor([vt[-1], vt[-1]], device=device)
#         rep_hub   = torch.zeros_like(rep_targ)
#         lr_val    = optimizer.param_groups[0]["lr"]
#         lr_tensor = torch.tensor([lr_val, lr_val], device=device)
    
#         models_core.log_loss_components(
#             epoch=epoch,
#             batch_idx=0,
#             lr_vals=lr_tensor,
#             y_r=rep_targ,
#             b_logits=None,
#             hub=rep_hub,
#             prev_lr=None,
#             prev_prev_lr=None,
#             mse_loss=mse,
#             bce_loss=None,
#             cls_loss_weight=0.0,
#             smooth_beta=0.0,
#             diff1_weight=0.0,
#             diff2_weight=0.0,
#             model=model,
#             log_file=params.log_file,
#             optimizer=optimizer,
#             batch_loss=epoch_mse,
#             epoch_train_metrics={
#                 "rmse":      train_rmse,
#                 "r2":        train_r2,
#                 "base_rmse": base_tr_rmse,
#                 "base_r2":   base_tr_r2,
#             },
#             epoch_val_metrics={
#                 "rmse":      val_rmse,
#                 "r2":        vl_r2,
#                 "base_rmse": base_vl_rmse,
#                 "base_r2":   base_vl_r2,
#                 "val_loss":  val_batch_loss,
#             },
#         )


#         # live‐plot + checkpoint + early stop
#         live_plot.update(train_rmse, val_rmse)
#         models_dir = Path(params.models_folder)
#         best_val, improved, _, _, tmp = models_core.maybe_save_chkpt(
#             models_dir, model, val_rmse, best_val,
#             {"rmse": train_rmse}, {"rmse": val_rmse}, live_plot, params
#         )
#         if improved:
#             best_state = {k: v.cpu() for k, v in model.state_dict().items()}
#             patience = 0
#         else:
#             patience += 1
#             if patience >= early_stop_patience:
#                 print(f"Early stopping at epoch {epoch}")
#                 break

#         print(f"Epoch {epoch:02d}  TRAIN RMSE={train_rmse:.4f}  VALID RMSE={val_rmse:.4f}")

#     # restore best and save final
#     if best_state is not None:
#         model.load_state_dict(best_state)
#         models_core.save_final_chkpt(
#             models_dir, best_state, best_val, params, {}, {}, live_plot, suffix="_fin"
#         )

#     return best_val

def model_training_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cosine_sched: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    plateau_sched: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.cuda.amp.GradScaler,
    train_loader,
    val_loader,
    *,
    max_epochs: int,
    early_stop_patience: int,
    clipnorm: float,
    device: torch.device,
    mode: str = "train",
):
    """
    Minimal stateful LSTM loop with full logging.

    - Mixed-precision training, per-batch grad clipping.
    - CosineAnnealingWarmRestarts per epoch + ReduceLROnPlateau on val RMSE.
    - Static baselines (std of true targets, trivial R²=0) computed once.
    - Per-window final-step scalar target: 
        pred = model(x_seq)[…,-1,0], targ = y_r[i,-1].
    - Computes epoch-level TRAIN and VAL RMSE, MSE, R².
    - Logs gradient norms and loss components once per epoch.
    - Live RMSE plot, checkpointing, early stopping.
    """

    import math
    import numpy as np

    model.to(device)
    mse_loss_fn = torch.nn.MSELoss()

    # If not training, run pure evaluation
    if mode != "train":
        return eval_on_loader(val_loader, model, device,
                              get_metrics(device), collect_preds=True)

    # Prepare checkpointing and live-plot
    best_val = float("inf")
    patience = 0
    live_plot = plots.LiveRMSEPlot()

    # — STATIC BASELINES (compute once, before epochs) —
    all_tr_targs = []
    for xb, y_r, *_, lengths in train_loader:
        for i, L in enumerate(lengths):
            if L > 0:
                all_tr_targs.append(
                    float(y_r[i, :L].view(-1)[-1].item())
                )
    base_tr_rmse = float(np.std(all_tr_targs))
    base_tr_r2   = 0.0

    all_vl_targs = []
    for xb, y_r, *_, lengths in val_loader:
        for i, L in enumerate(lengths):
            if L > 0:
                all_vl_targs.append(
                    float(y_r[i, :L].view(-1)[-1].item())
                )
    base_vl_rmse = float(np.std(all_vl_targs))
    base_vl_r2   = 0.0

    # — EPOCH LOOP —
    for epoch in range(1, max_epochs + 1):

        # — TRAINING PASS —  
        model.train()
        train_preds, train_targs = [], []
        train_se, train_count = 0.0, 0

        for xb, y_r, *_, lengths in tqdm(
                train_loader, desc=f"Epoch {epoch} ▶ Train", leave=False):
            xb, y_r = xb.to(device), y_r.to(device)
            optimizer.zero_grad(set_to_none=True)

            # accumulate per-batch loss for backprop
            batch_loss = torch.tensor(0.0, device=device)
            num_windows = 0

            for i, L in enumerate(lengths):
                if L == 0:
                    continue
                seq = xb[i, :L]
                out = model(seq)
                # extract the final scalar prediction
                pred = (out[..., -1, 0]
                        if out.dim() >= 3 else out.view(-1))[-1]
                targ = y_r[i, :L].view(-1)[-1]

                # MSE for this window
                l = mse_loss_fn(pred, targ)
                batch_loss += l
                num_windows += 1

                # accumulate epoch‐level squared error
                err = (pred - targ).item()
                train_se   += err * err
                train_count += 1

                # store for train R²
                train_preds.append(pred.item())
                train_targs.append(targ.item())

            # normalize batch loss, backward, clip, step
            batch_loss = batch_loss / num_windows
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()

        # — TRAIN METRICS (epoch‐level) —  
        epoch_mse   = train_se / train_count
        train_rmse  = float(math.sqrt(epoch_mse))
        tp_arr      = np.array(train_preds)
        tt_arr      = np.array(train_targs)
        train_r2 = (
            1.0
            - ((tp_arr - tt_arr)**2).sum()
              / ((tt_arr - tt_arr.mean())**2).sum()
        ) if tt_arr.size > 1 and not np.isclose(tt_arr.var(), 0.0) else float("nan")

        # step cosine scheduler once per epoch
        cosine_sched.step(epoch)

        # — VALIDATION PASS —  
        model.eval()
        val_preds, val_targs = [], []
        with torch.no_grad():
            for xb, y_r, *_, lengths in tqdm(
                    val_loader, desc=f"Epoch {epoch} ▶ Valid", leave=False):
                xb, y_r = xb.to(device), y_r.to(device)
                for i, L in enumerate(lengths):
                    if L == 0:
                        continue
                    seq = xb[i, :L]
                    out = model(seq)
                    pred = (out[..., -1, 0]
                            if out.dim() >= 3 else out.view(-1))[-1]
                    targ = y_r[i, :L].view(-1)[-1]
                    val_preds.append(pred.item())
                    val_targs.append(targ.item())

        # — VAL METRICS (epoch‐level) —
        vp_arr = np.array(val_preds)
        vt_arr = np.array(val_targs)
        val_rmse       = float(np.sqrt(((vp_arr - vt_arr)**2).mean()))
        val_batch_loss = val_rmse ** 2
        vl_r2 = (
            1.0
            - ((vp_arr - vt_arr)**2).sum()
              / ((vt_arr - vt_arr.mean())**2).sum()
        ) if vt_arr.size > 1 and not np.isclose(vt_arr.var(), 0.0) else float("nan")

        # step plateau scheduler on val_rmse
        plateau_sched.step(val_rmse)

        # — LOGGING —  
        models_core.model = model
        models_core.log_gradient_norms(epoch, 0, model, params.log_file)

        # replicate a minimal y_r + hub for loss‐component logging
        rep_targ  = torch.tensor([vt_arr[-1], vt_arr[-1]], device=device)
        rep_hub   = torch.zeros_like(rep_targ)
        lr_val    = optimizer.param_groups[0]["lr"]
        lr_tensor = torch.tensor([lr_val, lr_val], device=device)

        models_core.log_loss_components(
            epoch=epoch,
            batch_idx=0,
            lr_vals=lr_tensor,
            y_r=rep_targ,
            b_logits=None,
            hub=rep_hub,
            prev_lr=None,
            prev_prev_lr=None,
            mse_loss=mse_loss_fn,
            bce_loss=None,
            cls_loss_weight=0.0,
            smooth_beta=0.0,
            diff1_weight=0.0,
            diff2_weight=0.0,
            model=model,
            log_file=params.log_file,
            optimizer=optimizer,
            batch_loss=epoch_mse,
            epoch_train_metrics={
                "rmse":      train_rmse,
                "r2":        train_r2,
                "base_rmse": base_tr_rmse,
                "base_r2":   base_tr_r2,
            },
            epoch_val_metrics={
                "rmse":      val_rmse,
                "r2":        vl_r2,
                "base_rmse": base_vl_rmse,
                "base_r2":   base_vl_r2,
                "val_loss":  val_batch_loss,
            },
        )


        # live‐plot + checkpoint + early stop
        live_plot.update(train_rmse, val_rmse)
        models_dir = Path(params.models_folder)
        best_val, improved, _, _, tmp = models_core.maybe_save_chkpt(
            models_dir, model, val_rmse, best_val,
            {"rmse": train_rmse}, {"rmse": val_rmse}, live_plot, params
        )
        if improved:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch:02d}  TRAIN RMSE={train_rmse:.4f}  VALID RMSE={val_rmse:.4f}")

    # restore best and save final
    if best_state is not None:
        model.load_state_dict(best_state)
        models_core.save_final_chkpt(
            models_dir, best_state, best_val, params, {}, {}, live_plot, suffix="_fin"
        )

    return best_val