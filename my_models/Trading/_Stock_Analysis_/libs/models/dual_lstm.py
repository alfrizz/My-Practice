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


#########################################################################################################


class DualMemoryLSTM(nn.Module):
    """
    Stateful CNN→BiLSTM→Attention→BiLSTM network with three time-distributed heads:
      • regression head      → one real value per time-step
      • binary-signal head   → one logit per time-step (signal > buy_threshold)
      • ternary-return head  → three logits per time-step (down/flat/up on bar return)
      
      0) 1D convolution capturing local temporal patterns within each window/day
      1) Bidirectional short-term (daily) LSTM
      2) Window-level self-attention over the daily Bi-LSTM output
      3) Variational Dropout + LayerNorm on attended daily features
      4) Bidirectional long-term (weekly) LSTM
      5) Variational Dropout + LayerNorm on weekly features
      6) Automatic resets of hidden states at day/week boundaries

    """

    def __init__(
        self,
        n_feats: int,
        short_units: int,
        long_units: int,
        dropout_short: float,
        dropout_long: float,
        att_heads: int,
        att_drop: float
    ):
        super().__init__()
        self.n_feats     = n_feats
        self.short_units = short_units
        self.long_units  = long_units

        # 0) 1D conv encoder over time
        self.conv = nn.Conv1d(
            in_channels = n_feats,
            out_channels= n_feats,
            kernel_size = 3,
            padding     = 1
        )

        # 1) Short-term daily Bi-LSTM (stateful across windows)
        assert short_units % 2 == 0
        self.short_lstm = nn.LSTM(
            input_size   = n_feats,
            hidden_size  = short_units // 2,
            batch_first  = True,
            bidirectional= True
        )

        # 2) Self-attention over daily LSTM output
        self.attn = nn.MultiheadAttention(
            embed_dim   = short_units,
            num_heads   = att_heads,
            dropout     = att_drop,
            batch_first = True
        )

        # 3) Dropout + LayerNorm on daily features
        self.do_short = nn.Dropout(dropout_short)
        self.ln_short = nn.LayerNorm(short_units)

        # 4) Long-term weekly Bi-LSTM (stateful across days)
        assert long_units % 2 == 0
        self.long_lstm = nn.LSTM(
            input_size   = short_units,
            hidden_size  = long_units // 2,
            batch_first  = True,
            bidirectional= True
        )
        self.do_long = nn.Dropout(dropout_long)
        self.ln_long = nn.LayerNorm(long_units)

        # 5) Three time-distributed heads
        self.pred       = nn.Linear(long_units, 1)   # regression
        self.cls_head   = nn.Linear(long_units, 1)   # binary
        self.cls_ter    = nn.Linear(long_units, 3)   # ternary

        # 6) Hidden/cell states (initialized lazily)
        self.h_short = None
        self.c_short = None
        self.h_long  = None
        self.c_long  = None

    def _init_states(self, B: int, device: torch.device):
        # 2 directions × 1 layer = 2
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
        # reshape if input has extra dims
        if x.dim() > 3:
            *lead, S, F = x.shape # lead = [1, N_windows]; S = look_back
            x = x.view(-1, S, F)  # becomes (batch=N_windows, seq_len=look_back, features=F)

        # ensure last dim is features
        if x.dim() == 3 and x.size(-1) != self.n_feats:
            x = x.transpose(1, 2).contiguous()

        B, S, _ = x.size()
        dev      = x.device

        # 0) conv over time
        x_conv = x.transpose(1, 2)              # (W, F, look_back)
        x_conv = Funct.relu(self.conv(x_conv))  # still (W, F, look_back)
        x      = x_conv.transpose(1, 2)         # back to (W, look_back, F)

        # init or resize states
        if self.h_short is None or self.h_short.size(1) != B:
            self._init_states(B, dev)

        # 1) daily Bi-LSTM
        out_short_raw, (h_s, c_s) = self.short_lstm(x, (self.h_short, self.c_short))
        h_s.detach_(); c_s.detach_()
        self.h_short, self.c_short = h_s, c_s

        # 2) self-attention + residual
        attn_out, _ = self.attn(out_short_raw, out_short_raw, out_short_raw)
        out_short   = out_short_raw + attn_out

        # 3) dropout + layernorm daily
        out_short = self.do_short(out_short)
        out_short = self.ln_short(out_short)

        # 4) weekly Bi-LSTM
        out_long, (h_l, c_l) = self.long_lstm(out_short, (self.h_long, self.c_long))
        h_l.detach_(); c_l.detach_()
        self.h_long, self.c_long = h_l, c_l

        # 5) dropout + layernorm weekly
        out_long = self.do_long(out_long)
        out_long = self.ln_long(out_long)

        # 6) three heads
        raw_reg = self.pred(out_long)     # (B, S, 1)
        raw_cls = self.cls_head(out_long) # (B, S, 1)
        raw_ter = self.cls_ter(out_long)  # (B, S, 3)

        return raw_reg, raw_cls, raw_ter


#########################################################################################################


def lstm_training_loop(
    model:         torch.nn.Module,
    optimizer:     torch.optim.Optimizer,
    cosine_sched:  CosineAnnealingWarmRestarts,
    plateau_sched: ReduceLROnPlateau,
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
    Train and validate a stateful CNN→BiLSTM→Attention→BiLSTM model with two heads:
      • regression head → continuous (smoothed) signal  
      • binary head     → buy/sell threshold indicator  

    Functionality:
      1) Move model to device, enable CuDNN benchmark for speed.
      2) Define Huber loss for regression and BCEWithLogits for the binary head.
      3) Instantiate regression metrics (RMSE, MAE, R²) and binary metrics
         (Accuracy, Precision, Recall, F1, AUROC).
      4) For each epoch:
         a) Reset model’s LSTM states and all training metrics.
         b) Loop over train_loader (with tqdm progress bar):
            – Unpack a batch of sequences (multiple “days”) and their targets.
            – Zero gradients, track a single prev_day to reset “long” LSTM on day rollover.
            – For each sequence in the batch:
               • Slice the true signal and binary target up to its valid length.
               • Reset or carry the LSTM short/long states on day rollover.
               • Forward pass → get regression logits and binary logits.
               • Sigmoid‐activate regression logits into [0,1].
               • Compute Huber(regression) + α·BCE(binary) loss.
               • Backward (mixed precision), clip grads, step optimizer, update cosine schedule.
               • Detach hidden states to prevent backprop through time.
               • Update regression and binary metrics on that sequence.
         c) At epoch end, collect train‐metric summaries into a dict.
      5) Validation loop and checkpointing.
    """

    # 1) Device setup
    model.to(device)
    torch.backends.cudnn.benchmark = True
    
    # 2) Losses & weights
    beta_huber = params.hparams["HUBER_BETA"]
    huber_loss = nn.SmoothL1Loss(beta=beta_huber)      # still defined, but unused
    bce_loss   = nn.BCEWithLogitsLoss()                # still defined, but unused
    alpha_cls  = params.hparams["CLS_LOSS_WEIGHT"]     # still defined, but unused
    
    mse_loss   = nn.MSELoss()
    
    save_pat  = re.compile(rf"{re.escape(params.ticker)}_(\d+\.\d+)\.pth")
    live_plot = plots.LiveRMSEPlot()
    
    # 3) Metrics (regression + binary classification)
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

    # 4) Epochs
    for epoch in range(1, max_epochs + 1):
        gc.collect()
    
        # a) Training pass
        model.train()
        model.h_short = model.h_long = None
        for m in (train_rmse, train_mae, train_r2,
                  train_acc, train_prec, train_rec,
                  train_f1, train_auc):
            m.reset()
    
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            xb_days, y_sig_days, y_sig_cls_days, ret_days, y_ret_ter_days, wd_days, ts_list, lengths = batch
    
            xb    = xb_days.to(device, non_blocking=True)
            y_sig = y_sig_days.to(device, non_blocking=True)
            y_cls = y_sig_cls_days.to(device, non_blocking=True)
            wd    = wd_days.to(device, non_blocking=True)
    
            optimizer.zero_grad(set_to_none=True)
            prev_day = None
    
            for di in range(xb.size(0)):
                W      = lengths[di]
                day_id = int(wd[di].item())
    
                x_seq   = xb[di, :W]
                sig_seq = y_sig[di, :W]
                cls_seq = y_cls[di, :W].view(-1)
    
                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id
    
                # Forward
                pr, pc, _ = model(x_seq)
    
                with autocast(device_type=device.type):
                    # Regression logits → probability
                    lr = torch.sigmoid(pr[..., -1, 0])   # (W,)
    
                    # ONLY optimize MSE
                    loss = mse_loss(lr, sig_seq)
    
                scaler.scale(loss).backward()
    
                # Update regression metrics
                train_rmse.update(lr, sig_seq)
                train_mae .update(lr, sig_seq)
                train_r2  .update(lr, sig_seq)
    
                # Update binary metrics (still tracked but NOT in loss)
                probs = torch.sigmoid(pc[..., -1, 0])
                train_acc .update(probs, cls_seq)
                train_prec.update(probs, cls_seq)
                train_rec .update(probs, cls_seq)
                train_f1  .update(probs, cls_seq)
                train_auc .update(probs, cls_seq)
    
                model.h_short.detach_(); model.c_short.detach_()
                model.h_long .detach_(); model.c_long .detach_()
    
            # Clip & step
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
    
        # ——— collect training metrics ———
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

        # b) VALIDATION
        model.eval()
        model.h_short = model.h_long = None
        for m in (
            val_rmse, val_mae, val_r2,
            val_acc,  val_prec, val_rec,
            val_f1,   val_auc
        ):
            m.reset()

        with torch.no_grad():
            prev_day = None
            for batch in val_loader:
                xb_day, y_sig_day, y_sig_cls_day, ret_day, y_ret_ter_day, wd, ts_list, lengths = batch

                W      = lengths[0]
                day_id = int(wd.item())

                x_seq   = xb_day[0, :W].to(device)
                sig_seq = y_sig_day[0, :W].to(device)
                cls_seq = y_sig_cls_day[0, :W].view(-1).to(device)

                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                # Forward pass (drop ternary head)
                pr, pc, pt = model(x_seq)

                # sigmoid regression output
                lr = torch.sigmoid(pr[..., -1, 0])

                # binary logits
                lc = pc[..., -1, 0]

                # update regression metrics
                val_rmse.update(lr,      sig_seq)
                val_mae .update(lr,      sig_seq)
                val_r2  .update(lr,      sig_seq)

                # update binary‐classification metrics
                probs = torch.sigmoid(lc)
                val_acc .update(probs, cls_seq)
                val_prec.update(probs, cls_seq)
                val_rec .update(probs, cls_seq)
                val_f1  .update(probs, cls_seq)
                val_auc .update(probs, cls_seq)

        # ——— collect validation metrics ———
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

        # c) live plot & print
        live_plot.update(tr["rmse"], vl["rmse"])
        print(f"Epoch {epoch:03d}")
        print(
            f'TRAIN→ '
            f'"R": RMSE={tr["rmse"]:.4f} MAE={tr["mae"]:.4f} R2={tr["r2"]:.4f} | '
            f'"B": Acc={tr["acc"]:.4f} Prec={tr["prec"]:.4f} Rec={tr["rec"]:.4f} '
            f'F1={tr["f1"]:.4f} AUROC={tr["auroc"]:.4f}'
        )
        print(
            f'VALID→ '
            f'"R": RMSE={vl["rmse"]:.4f} MAE={vl["mae"]:.4f} R2={vl["r2"]:.4f} | '
            f'"B": Acc={vl["acc"]:.4f} Prec={vl["prec"]:.4f} Rec={vl["rec"]:.4f} '
            f'F1={vl["f1"]:.4f} AUROC={vl["auroc"]:.4f}'
        )

        # d) adjust LR scheduler after warmup
        if epoch > params.hparams["LR_EPOCHS_WARMUP"]:
            plateau_sched.step(vl["rmse"])
            
        # e) folder‐best checkpoint (→ `_chp`)
        save_pat = re.compile(rf"{params.ticker}_(\d+\.\d+)_chp\.pth")
        # list all existing checkpoint files
        models_dir = Path(params.models_folder)
        models_dir.mkdir(exist_ok=True)

        existing_rmses = [
            float(m.group(1))
            for f in models_dir.glob(f"{params.ticker}_*_chp.pth")
            for m in (save_pat.match(f.name),) if m
        ]
        best_existing = min(existing_rmses) if existing_rmses else float("inf")

        # f) early‐stop + update run‐best
        if vl["rmse"] < best_val_rmse:
            best_val_rmse = vl["rmse"]
            best_state    = model.state_dict()
            best_tr       = tr.copy()
            best_vl       = vl.copy()
            patience_ctr  = 0

            # cache the run‐best plot
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
                chp_name = f"{params.ticker}_{best_val_rmse:.5f}_chp.pth"
                torch.save(ckpt, models_dir / chp_name)
                print(f"🔖 Saved folder‐best checkpoint (_chp): {chp_name}")
                
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print("Early stopping at epoch", epoch)
                break

    # ── after the epoch loop ends: always write final‐run best (_fin) ──
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
