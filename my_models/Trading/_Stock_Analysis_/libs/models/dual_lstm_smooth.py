from libs import plots, params, models

from typing import Sequence, List, Tuple, Optional, Union
import gc 
import os
import io
import tempfile
import copy
import re

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

class DualMemoryLSTM(nn.Module):
    """
    Dual-Memory LSTM for smoothed continuous‐signal regression + binary‐flag prediction.

    Architecture steps:
      0) Conv1d + BatchNorm1d → ReLU
         • Encode local temporal patterns within each look-back window.
      1) Stateful Bidirectional “short” LSTM over each window
      2) Window-level self-attention + residual over short-LSTM outputs
      3) LayerNorm → Dropout on daily embedding (pre-norm style)
      4) Stateful Bidirectional “long” LSTM across windows
      5) Residual skip (projected short → long) + LayerNorm → Dropout
      6) Time-distributed heads in parallel:
         – regression head (1 real value per step)
         – binary head     (1 logit per step)
         – ternary head    (3 logits per step; reserved for future use)
      7) Automatic reset of hidden states at day/week boundaries
    """

    def __init__(
        self,
        n_feats:      int,
        short_units:  int,
        long_units:   int,
        dropout_short: float,
        dropout_long:  float,
        att_heads:     int,
        att_drop:      float,
        conv_k:        int = 3,
        conv_dilation: int = 1
    ):
        super().__init__()
        self.n_feats     = n_feats
        self.short_units = short_units
        self.long_units  = long_units

        # 0) Conv1d encoder + batch‐norm
        pad = (conv_k // 2) * conv_dilation
        self.conv = nn.Conv1d(n_feats, 
                              n_feats,
                              kernel_size=conv_k,
                              dilation=conv_dilation,
                              padding=pad)
        self.bn   = nn.BatchNorm1d(n_feats)

        # 1) Short‐term daily Bi‐LSTM
        assert short_units % 2 == 0
        self.short_lstm = nn.LSTM(
            input_size   = n_feats,
            hidden_size  = short_units // 2,
            batch_first  = True,
            bidirectional= True
        )

        # 2) Window‐level self‐attention
        self.attn = nn.MultiheadAttention(
            embed_dim   = short_units,
            num_heads   = att_heads,
            dropout     = att_drop,
            batch_first = True
        )

        # 3) Pre‐norm LayerNorm → Dropout on short embedding
        self.ln_short = nn.LayerNorm(short_units)
        self.do_short = nn.Dropout(dropout_short)

        # 4) Long‐term weekly Bi‐LSTM
        assert long_units % 2 == 0
        self.long_lstm = nn.LSTM(
            input_size   = short_units,
            hidden_size  = long_units // 2,
            batch_first  = True,
            bidirectional= True
        )

        # 5a) Project short → long for residual connection
        self.short2long = nn.Linear(short_units, long_units)

        # 5b) LayerNorm → Dropout on long embedding
        self.ln_long = nn.LayerNorm(long_units)
        self.do_long = nn.Dropout(dropout_long)

        # 6) Time‐distributed heads
        self.pred     = nn.Linear(long_units, 1)  # regression
        self.cls_head = nn.Linear(long_units, 1)  # binary
        self.cls_ter  = nn.Linear(long_units, 3)  # ternary (future)

        # 7) Lazy‐init hidden/cell states
        self.h_short = None
        self.c_short = None
        self.h_long  = None
        self.c_long  = None

    def _init_states(self, B: int, device: torch.device):
        # 2 directions × 1 layer
        self.h_short = torch.zeros(2, B, self.short_units // 2, device=device)
        self.c_short = torch.zeros(2, B, self.short_units // 2, device=device)
        self.h_long  = torch.zeros(2, B, self.long_units  // 2, device=device)
        self.c_long  = torch.zeros(2, B, self.long_units  // 2, device=device)

    def reset_short(self):
        if self.h_short is not None:
            B, dev = self.h_short.size(1), self.h_short.device
            self._init_states(B, dev)

    def reset_long(self):
        if self.h_long is not None:
            B, dev = self.h_long.size(1), self.h_long.device
            hs, cs = self.h_short, self.c_short
            self._init_states(B, dev)
            self.h_short, self.c_short = hs.to(dev), cs.to(dev)

    def forward(self, x: torch.Tensor):
        # reshape  extra dims → (W, S, F)
        if x.dim() > 3:
            *lead, S, F = x.shape
            x = x.view(-1, S, F)

        # ensure (W, S, n_feats)
        if x.dim() == 3 and x.size(-1) != self.n_feats:
            x = x.transpose(1, 2).contiguous()

        B, S, _ = x.size()
        dev      = x.device

        # 0) Conv1d → BatchNorm1d → ReLU
        x_conv = x.transpose(1, 2)      # (W, F, S)
        x_conv = self.conv(x_conv)
        x_conv = self.bn(x_conv)
        x_conv = Funct.relu(x_conv)
        x      = x_conv.transpose(1, 2) # back to (W, S, F)

        # lazy init/reset hidden states
        if self.h_short is None or self.h_short.size(1) != B:
            self._init_states(B, dev)

        # 1) Short‐term Bi‐LSTM
        out_short_raw, (h_s, c_s) = self.short_lstm(
            x, (self.h_short, self.c_short)
        )
        h_s.detach_(); c_s.detach_()
        self.h_short, self.c_short = h_s, c_s

        # 2) Self‐attention + residual
        attn_out, _ = self.attn(out_short_raw, out_short_raw, out_short_raw)
        out_short   = out_short_raw + attn_out

        # 3) Pre‐norm LayerNorm → Dropout
        out_short = self.ln_short(out_short)
        out_short = self.do_short(out_short)

        # 4) Long‐term Bi‐LSTM
        out_long_raw, (h_l, c_l) = self.long_lstm(
            out_short, (self.h_long, self.c_long)
        )
        h_l.detach_(); c_l.detach_()
        self.h_long, self.c_long = h_l, c_l

        # 5) Residual skip via projection + LayerNorm → Dropout
        skip     = self.short2long(out_short)
        out_long = skip + out_long_raw
        out_long = self.ln_long(out_long)
        out_long = self.do_long(out_long)

        # 6) Time‐distributed heads
        raw_reg = self.pred(out_long)     # (W, S, 1)
        raw_cls = self.cls_head(out_long) # (W, S, 1)
        raw_ter = self.cls_ter(out_long)  # (W, S, 3)

        return raw_reg, raw_cls, raw_ter



######################################################################################################


def lstm_training_loop(
    model:         torch.nn.Module,
    optimizer:     torch.optim.Optimizer,
    cosine_sched:  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    plateau_sched: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler:        GradScaler,
    train_loader:  torch.utils.data.DataLoader,
    val_loader:    torch.utils.data.DataLoader,
    *,
    max_epochs:          int,
    early_stop_patience: int,
    clipnorm:            float,
    device:              torch.device = torch.device("cpu"),
) -> float:
    """
    Train a DualMemoryLSTM with:
      - Regression head (MSE) + binary and ternary classification heads
      - Exponential‐baseline smoothing applied to the *final* regression outputs
        via a one‐sided Huber penalty on downward slips
      - Mixed‐precision (amp) and LR scheduling (cosine warm restarts + plateau)

    Functionality:
      1. Device & model initialization
      2. Standard multi‐task losses (MSE + BCE)
      3. Exponential‐Baseline Tracking across windows:
           ewma_t = α·lr_t + (1−α)·ewma_{t−1}
      4. One‐Sided Huber penalty on downward slips: 
           slip = max(ewma_t − lr_t, 0)
      5. Metrics collection and checkpointing
    """
    # 1) Device & model setup
    model.to(device)
    torch.backends.cudnn.benchmark = True

    # 2) Losses & hyper‐weights
    mse_loss   = nn.MSELoss()
    bce_loss   = nn.BCEWithLogitsLoss()
    alpha_cls  = params.hparams["CLS_LOSS_WEIGHT"]

    # smoothing hyperparams from global hparams
    α       = params.hparams["SMOOTH_ALPHA"]   # e.g. = 1 - exp(-1/60)
    β       = params.hparams["SMOOTH_BETA"]    # smooth penalty weight
    δ       = params.hparams["SMOOTH_DELTA"]   # Huber threshold

    save_pat = re.compile(rf"{re.escape(params.ticker)}_(\d+\.\d+)_(?:chp|fin)\.pth")
    live_plot = plots.LiveRMSEPlot()

    # classification metrics
    thr         = 0.5
    train_rmse  = torchmetrics.MeanSquaredError(squared=False).to(device)
    train_mae   = torchmetrics.MeanAbsoluteError().to(device)
    train_r2    = torchmetrics.R2Score().to(device)
    train_acc   = torchmetrics.classification.BinaryAccuracy(threshold=thr).to(device)
    train_prec  = torchmetrics.classification.BinaryPrecision(threshold=thr).to(device)
    train_rec   = torchmetrics.classification.BinaryRecall(threshold=thr).to(device)
    train_f1    = torchmetrics.classification.BinaryF1Score(threshold=thr).to(device)
    train_auc   = torchmetrics.classification.BinaryAUROC().to(device)
    
    val_rmse    = torchmetrics.MeanSquaredError(squared=False).to(device)
    val_mae     = torchmetrics.MeanAbsoluteError().to(device)
    val_r2      = torchmetrics.R2Score().to(device)
    val_acc     = torchmetrics.classification.BinaryAccuracy(threshold=thr).to(device)
    val_prec    = torchmetrics.classification.BinaryPrecision(threshold=thr).to(device)
    val_rec     = torchmetrics.classification.BinaryRecall(threshold=thr).to(device)
    val_f1      = torchmetrics.classification.BinaryF1Score(threshold=thr).to(device)
    val_auc     = torchmetrics.classification.BinaryAUROC().to(device)

    best_val_rmse = float("inf")
    patience_ctr  = 0

    # 3) Epoch loop
    for epoch in range(1, max_epochs + 1):
        gc.collect()
        model.train()
        model.h_short = model.h_long = None
    
        # reset train metrics
        for m in (train_rmse, train_mae, train_r2,
                  train_acc, train_prec, train_rec,
                  train_f1, train_auc):
            m.reset()
    
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            (xb_days, y_sig_days, y_sig_cls_days,
             ret_days, y_ret_ter_days,
             wd_days, ts_list, lengths) = batch
    
            xb    = xb_days.to(device, non_blocking=True)
            y_sig = y_sig_days.to(device, non_blocking=True)
            y_cls = y_sig_cls_days.to(device, non_blocking=True)
            wd    = wd_days.to(device, non_blocking=True)
    
            optimizer.zero_grad(set_to_none=True)
            prev_day = None
            ewma     = None   # start fresh baseline at beginning of each batch
    
            # loop each window in the batch
            for di in range(xb.size(0)):
                W      = lengths[di]
                day_id = int(wd[di].item())
    
                x_seq   = xb[di, :W]
                sig_seq = y_sig[di, :W]
                cls_seq = y_cls[di, :W].view(-1)
    
                # reset hidden states per day & week 
                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                    # NOTE: we no longer clear ewma here—
                    # the EWMA persists across weeks for continuous smoothing
                prev_day = day_id
    
                pr, pc, _ = model(x_seq)
    
                with autocast(device_type=device.type):
                    lr_logits  = pr[..., -1, 0]   # shape = (W,)
                    cls_logits = pc[..., -1, 0]   # shape = (W,)
                    lr         = torch.sigmoid(lr_logits)
            
                    # build matching target vectors
                    targ_r = sig_seq              # shape = (W,)
                    targ_c = cls_seq              # shape = (W,)
            
                    # 1) base loss over all windows
                    loss_reg = mse_loss(lr,      targ_r)
                    loss_cls = bce_loss(cls_logits, targ_c)
                    loss     = loss_reg + alpha_cls * loss_cls
            
                    # 2) EWMA smoothing penalty (unchanged)
                    if ewma is None:
                        ewma = lr.detach()
                    else:
                        ewma = α * lr.detach() + (1 - α) * ewma
            
                    slip = torch.relu(ewma - lr)
                    hub  = torch.where(
                        slip <= δ,
                        0.5 * slip**2,
                        δ * (slip - 0.5 * δ)
                    )
                    loss += β * hub.mean()
            
                scaler.scale(loss).backward()
            
                # — UPDATE TRAIN METRICS WITH VECTORS INSTEAD OF SCALARS —
                train_rmse.update(lr,      targ_r)
                train_mae .update(lr,      targ_r)
                train_r2  .update(lr,      targ_r)
            
                probs = torch.sigmoid(cls_logits)
                train_acc .update(probs, targ_c)
                train_prec.update(probs, targ_c)
                train_rec .update(probs, targ_c)
                train_f1  .update(probs, targ_c)
                train_auc .update(probs, targ_c)


                # detach hidden states
                for h in (model.h_short, model.c_short, model.h_long, model.c_long):
                    h.detach_()

            # optimizer step & schedulers
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()

            frac = epoch - 1 + batch_idx / len(train_loader)
            cosine_sched.step(frac)

            pbar.set_postfix(
                train_rmse=train_rmse.compute().item(),
                lr=optimizer.param_groups[0]['lr'],
                refresh=False
            )

        # collect train metrics
        tr = {
            "rmse":  train_rmse.compute().item(),
            "mae":   train_mae.compute().item(),
            "r2":    train_r2.compute().item(),
            "acc":   train_acc.compute().item(),
            "prec":  train_prec.compute().item(),
            "rec":   train_rec.compute().item(),
            "f1":    train_f1.compute().item(),
            "auroc": train_auc.compute().item(),
        }

        # b) Validation (same as before, final‐step only)
        model.eval()
        model.h_short = model.h_long = None
        for m in (val_rmse, val_mae, val_r2,
                  val_acc, val_prec, val_rec,
                  val_f1, val_auc):
            m.reset()

        with torch.no_grad():
            prev_day = None
            for batch in val_loader:
                (xb_day, y_sig_day, y_sig_cls_day,
                 ret_day, y_ret_ter_day,
                 wd, ts_list, lengths) = batch

                W      = lengths[0]
                day_id = int(wd.item())
                x_seq   = xb_day[0, :W].to(device)
                sig_seq = y_sig_day[0, :W].to(device)
                cls_seq = y_sig_cls_day[0, :W].view(-1).to(device)

                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                pr, pc, _ = model(x_seq)
                # predictions for every window
                lr_logits  = pr[..., -1, 0]        # shape (W,)
                cls_logits = pc[..., -1, 0]        # shape (W,)
                lr         = torch.sigmoid(lr_logits)
                
                # full-target vectors
                targ_r = sig_seq                 # shape (W,)
                targ_c = cls_seq                 # shape (W,)
                
                # update regression metrics on full vectors
                val_rmse.update(lr,      targ_r)
                val_mae .update(lr,      targ_r)
                val_r2  .update(lr,      targ_r)
                
                # update classification metrics on full vectors
                probs = torch.sigmoid(cls_logits)
                val_acc .update(probs, targ_c)
                val_prec.update(probs, targ_c)
                val_rec .update(probs, targ_c)
                val_f1  .update(probs, targ_c)
                val_auc .update(probs, targ_c)

        # collect val metrics
        vl = {
            "rmse":  val_rmse.compute().item(),
            "mae":   val_mae.compute().item(),
            "r2":    val_r2.compute().item(),
            "acc":   val_acc.compute().item(),
            "prec":  val_prec.compute().item(),
            "rec":   val_rec.compute().item(),
            "f1":    val_f1.compute().item(),
            "auroc": val_auc.compute().item(),
        }

        # c) Live plot & logging
        live_plot.update(tr["rmse"], vl["rmse"])
        print(f"Epoch {epoch:03d}")
        print(
            f'TRAIN→ '
            f'RMSE={tr["rmse"]:.4f} MAE={tr["mae"]:.4f} R2={tr["r2"]:.4f} | '
            f'Acc={tr["acc"]:.4f} Prec={tr["prec"]:.4f} Rec={tr["rec"]:.4f} '
            f'F1={tr["f1"]:.4f} AUROC={tr["auroc"]:.4f}'
        )
        print(
            f'VALID→ '
            f'RMSE={vl["rmse"]:.4f} MAE={vl["mae"]:.4f} R2={vl["r2"]:.4f} | '
            f'Acc={vl["acc"]:.4f} Prec={vl["prec"]:.4f} Rec={vl["rec"]:.4f} '
            f'F1={vl["f1"]:.4f} AUROC={vl["auroc"]:.4f}'
        )

        # d) Plateau scheduler after warmup
        if epoch > params.hparams["LR_EPOCHS_WARMUP"]:
            plateau_sched.step(vl["rmse"])

        # e) Save checkpoints (unchanged)
        models_dir = Path(params.models_folder)
        models_dir.mkdir(exist_ok=True)
        existing_rmses = [
            float(m.group(1))
            for f in models_dir.glob("*.pth")
            if (m := save_pat.match(f.name))
        ]
        best_existing = min(existing_rmses, default=float("inf"))

        if vl["rmse"] < best_val_rmse:
            best_val_rmse, best_state = vl["rmse"], model.state_dict()
            best_tr, best_vl           = tr.copy(), vl.copy()
            patience_ctr               = 0

            buf = io.BytesIO()
            live_plot.fig.savefig(buf, format="png")
            buf.seek(0)
            best_plot = buf.read()

            if best_val_rmse < best_existing:
                ckpt = {
                    "model_state_dict": best_state,
                    "hparams":          params.hparams,
                    "train_metrics":    best_tr,
                    "val_metrics":      best_vl,
                    "train_plot_png":   best_plot,
                }
                name = f"{params.ticker}_{best_val_rmse:.5f}_chp.pth"
                torch.save(ckpt, models_dir / name)
                print(f"🔖 Saved folder‐best checkpoint (_chp): {name}")
                best_existing = best_val_rmse
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print("Early stopping at epoch", epoch)
                break

    # f) Final checkpoint (unchanged)
    buf = io.BytesIO()
    live_plot.fig.savefig(buf, format="png")
    buf.seek(0)
    final_plot = buf.read()
    final_ckpt = {
        "model_state_dict": best_state,
        "hparams":          params.hparams,
        "train_metrics":    best_tr,
        "val_metrics":      best_vl,
        "train_plot_png":   final_plot,
    }
    fin_name = f"{params.ticker}_{best_val_rmse:.5f}_fin.pth"
    torch.save(final_ckpt, models_dir / fin_name)
    print(f"✅ Final best model (_fin) saved: {fin_name}")

    return best_val_rmse
