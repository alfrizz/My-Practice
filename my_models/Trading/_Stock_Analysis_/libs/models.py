from libs import plots, params

from typing import Sequence, List, Tuple, Optional, Union
import gc 
import os
import io
import shutil
import tempfile
import atexit
import copy
import re

import pandas as pd
import numpy  as np
import math

import datetime as dt
from datetime import datetime, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as Funct
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.optim import AdamW

import torchmetrics

torch.backends.cuda.matmul.allow_tf32   = True
torch.backends.cudnn.allow_tf32         = True
torch.backends.cudnn.benchmark          = True

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from tqdm.auto import tqdm


#########################################################################################################


def build_lstm_tensors(
    df: pd.DataFrame,
    *,
    look_back:  int = params.look_back_tick, # number of past bars in each window
    tmpdir:     str = None,             # where to write memmaps (auto‐created if None)
    device:     torch.device = torch.device("cpu"),
    sess_start: time                    # only include windows ending at/after this time
) -> tuple[
    torch.Tensor,  # X         shape=(N, look_back, F)
    torch.Tensor,  # y_sig     shape=(N,)
    torch.Tensor,  # y_ret     shape=(N,)
    torch.Tensor,  # raw_close shape=(N,)
    torch.Tensor,  # raw_bid   shape=(N,)
    torch.Tensor,  # raw_ask   shape=(N,)
    np.ndarray     # end_times shape=(N,) dtype=datetime64[ns]
]:
    """
    Build sliding‐window tensors for LSTM from a full‐day DataFrame.

    1) Automatically select ALL columns except {label_col, 'bid','ask'} as features.
    2) Split data by calendar‐day; count how many windows end ≥ sess_start each day.
    3) Allocate on‐disk numpy memmaps for:
         X       = feature windows         (N, look_back, F)
         y_sig   = next‐bar smoothed signal (N,)
         y_ret   = next‐bar log‐return      (N,)
         raw_*   = bid/mid/ask at window end (N,)
         end_times = timestamp of each window end (N,)
    4) For each day:
       a) extract feature array and target signal
       b) reconstruct mid‐price = (bid + ask)/2 and compute bar‐to‐bar log‐returns
       c) form sliding windows of features (drop last to align labels)
       d) align next‐bar labels y_sig, y_ret, raw_* and timestamps
       e) mask out windows ending before sess_start and write slices to memmaps
    5) Wrap each memmap with torch.from_numpy, free CPU memory, and return tensors + end_times.
    """
    # copy df so we don’t modify user’s DataFrame
    df = df.copy()

    # 1) Automatically derive features_cols
    exclude = {params.label_col, "bid", "ask"}
    features_cols = [c for c in df.columns if c not in exclude]
    F = len(features_cols)

    # group by calendar day
    day_groups = df.groupby(df.index.normalize(), sort=False)

    # 2) Count total valid windows across all days
    N = 0
    for _, day_df in tqdm(day_groups, desc="Counting windows", leave=False):
        T = len(day_df)
        if T <= look_back:
            continue
        ends = day_df.index[look_back:]
        mask = np.array([ts.time() >= sess_start for ts in ends])
        N += int(mask.sum())

    # 3) Allocate numpy memmaps on disk
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="lstm_memmap_")
    else:
        os.makedirs(tmpdir, exist_ok=True)

    X_mm = np.lib.format.open_memmap(
        os.path.join(tmpdir, "X.npy"), mode="w+",
        dtype=np.float32, shape=(N, look_back, F)
    )
    y_mm = np.lib.format.open_memmap(
        os.path.join(tmpdir, "y_sig.npy"), mode="w+",
        dtype=np.float32, shape=(N,)
    )
    r_mm = np.lib.format.open_memmap(
        os.path.join(tmpdir, "y_ret.npy"), mode="w+",
        dtype=np.float32, shape=(N,)
    )
    c_mm = np.lib.format.open_memmap(
        os.path.join(tmpdir, "c.npy"), mode="w+",
        dtype=np.float32, shape=(N,)
    )
    b_mm = np.lib.format.open_memmap(
        os.path.join(tmpdir, "b.npy"), mode="w+",
        dtype=np.float32, shape=(N,)
    )
    a_mm = np.lib.format.open_memmap(
        os.path.join(tmpdir, "a.npy"), mode="w+",
        dtype=np.float32, shape=(N,)
    )
    t_mm = np.lib.format.open_memmap(
        os.path.join(tmpdir, "t.npy"), mode="w+",
        dtype="datetime64[ns]", shape=(N,)
    )

    # 4) Fill memmaps
    idx = 0
    for _, day_df in tqdm(day_groups, desc="Writing memmaps", leave=False):
        day_df = day_df.sort_index()
        T = len(day_df)
        if T <= look_back:
            continue

        # a) extract features and signal
        feats_np = day_df[features_cols].to_numpy(np.float32)
        sig_np   = day_df[params.label_col]    .to_numpy(np.float32)

        # b) reconstruct mid-price and compute log-returns
        bid_np   = day_df["bid"]        .to_numpy(np.float32)
        ask_np   = day_df["ask"]        .to_numpy(np.float32)
        mid_np   = ((bid_np + ask_np)/2).astype(np.float32)
        ret_full = np.zeros_like(mid_np, dtype=np.float32)
        ret_full[1:] = np.log(mid_np[1:] / mid_np[:-1])

        # c) build sliding windows of features
        wins = np.lib.stride_tricks.sliding_window_view(
            feats_np, window_shape=(look_back, F)
        ).reshape(T - look_back + 1, look_back, F)
        wins = wins[:-1]  # drop last window to align next-step labels

        # d) align next-bar labels & raw prices
        lab_sig = sig_np[look_back:]
        lab_ret = ret_full[look_back:]
        c_pts   = mid_np[look_back:]
        b_pts   = bid_np[look_back:]
        a_pts   = ask_np[look_back:]
        times   = day_df.index.to_numpy()[look_back:]

        # e) mask by session start and write
        mask = np.array([pd.Timestamp(ts).time() >= sess_start for ts in times])
        if not mask.any():
            continue

        m = mask.sum()
        X_mm [idx:idx+m] = wins[mask]
        y_mm [idx:idx+m] = lab_sig[mask]
        r_mm [idx:idx+m] = lab_ret[mask]
        c_mm [idx:idx+m] = c_pts[mask]
        b_mm [idx:idx+m] = b_pts[mask]
        a_mm [idx:idx+m] = a_pts[mask]
        t_mm [idx:idx+m] = times[mask]
        idx += m

    # 5) Wrap memmaps as torch Tensors
    X         = torch.from_numpy(X_mm).to(device, non_blocking=True)
    y_sig     = torch.from_numpy(y_mm).to(device, non_blocking=True)
    y_ret     = torch.from_numpy(r_mm).to(device, non_blocking=True)
    raw_close = torch.from_numpy(c_mm).to(device, non_blocking=True)
    raw_bid   = torch.from_numpy(b_mm).to(device, non_blocking=True)
    raw_ask   = torch.from_numpy(a_mm).to(device, non_blocking=True)
    end_times = t_mm.copy()  # numpy datetime64 array

    # cleanup temporary buffers
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return X, y_sig, y_ret, raw_close, raw_bid, raw_ask, end_times

    
#########################################################################################################


def chronological_split(
    X:           torch.Tensor,
    y_sig:       torch.Tensor,
    y_ret:       torch.Tensor,
    raw_close:   torch.Tensor,
    raw_bid:     torch.Tensor,
    raw_ask:     torch.Tensor,
    end_times:   np.ndarray,      # (N,), dtype datetime64[ns]
    *,
    train_prop:  float,
    val_prop:    float,
    train_batch: int,
    device = torch.device("cpu")
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],          # (X_tr, y_sig_tr, y_ret_tr)
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],          # (X_val, y_sig_val, y_ret_val)
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  
                                                              # (X_te, y_sig_te, y_ret_te, raw_close_te, raw_bid_te, raw_ask_te)
    list,                                                     # samples_per_day
    torch.Tensor, torch.Tensor, torch.Tensor                  # day_id_tr, day_id_val, day_id_te
]:
    """
    Chronologically split tensors by calendar-day into train/val/test:
      1) Count windows per normalized date → samples_per_day.
      2) Determine how many days go to train/val/test (by proportions,
         rounding train_days up to full batches of train_batch).
      3) Cum‐sum the daily counts → slice X, y_sig, y_ret, raw_*, end_times.
      4) Build per-window day_id tags for each split.
    """
    # 1) Count windows per day
    dt_idx    = pd.to_datetime(end_times)
    normed    = dt_idx.normalize()
    days, counts = np.unique(normed.values, return_counts=True)
    samples_per_day = counts.tolist()

    # sanity
    total = sum(samples_per_day)
    if total != X.size(0):
        raise ValueError(f"Window count mismatch {total} vs {X.size(0)}")

    # 2) determine day splits
    D = len(samples_per_day)
    orig_tr_days   = int(D * train_prop)
    full_batches   = (orig_tr_days + train_batch - 1) // train_batch
    tr_days        = min(D, full_batches * train_batch)
    cut_train      = tr_days - 1
    cut_val        = int(D * (train_prop + val_prop))

    # 3) slice indices by window count
    cumsum = np.concatenate([[0], np.cumsum(counts)])
    i_tr   = int(cumsum[tr_days])
    i_val  = int(cumsum[cut_val + 1])

    X_tr, y_sig_tr, y_ret_tr = X[:i_tr],       y_sig[:i_tr],       y_ret[:i_tr]
    X_val, y_sig_val, y_ret_val = X[i_tr:i_val], y_sig[i_tr:i_val], y_ret[i_tr:i_val]
    X_te,  y_sig_te,  y_ret_te  = X[i_val:],    y_sig[i_val:],      y_ret[i_val:]
    close_te = raw_close[i_val:]; bid_te = raw_bid[i_val:]; ask_te = raw_ask[i_val:]

    # 4) day_id tags
    def make_day_ids(s, e):
        cnts = samples_per_day[s : e+1]
        days = torch.arange(s, e+1, device=device)
        return days.repeat_interleave(torch.tensor(cnts, device=device))

    day_id_tr  = make_day_ids(0,          cut_train)
    day_id_val = make_day_ids(cut_train+1, cut_val)
    day_id_te  = make_day_ids(cut_val+1,  D-1)

    return (
        (X_tr,  y_sig_tr,  y_ret_tr),
        (X_val, y_sig_val, y_ret_val),
        (X_te,  y_sig_te,  y_ret_te,  close_te, bid_te, ask_te),
        samples_per_day,
        day_id_tr, day_id_val, day_id_te
    )


#########################################################################################################


class DayWindowDataset(Dataset):
    """
    Group sliding windows by calendar day and return per-day batches of:
      • x_day     : input features, shape (1, W, look_back, F)
      • y_day     : regression target (your precomputed signal), shape (1, W)
      • y_sig_cls : binary label = 1 if signal > signal_thresh else 0
      • ret_day   : true bar-to-bar returns, shape (1, W)
      • y_ret_ter : ternary label ∈ {0,1,2} for (down, flat, up) based
                    on ret_day and return_thresh
      • [rc, rb, ra]  : optional raw price tensors, each shape (W,)
      • weekday   : integer day-of-week
      • end_ts    : timestamp of the last bar in the window
    """
    def __init__(
        self,
        X:              torch.Tensor,   # (N_windows, look_back, F)
        y_signal:       torch.Tensor,   # (N_windows,)
        y_return:       torch.Tensor,   # (N_windows,)
        raw_close:      torch.Tensor,   # or None
        raw_bid:        torch.Tensor,   # or None
        raw_ask:        torch.Tensor,   # or None
        end_times:      np.ndarray,     # (N_windows,), dtype datetime64[ns]
        sess_start_time: time,          # cutoff for trading session
        signal_thresh:  float,          # buy_threshold for y_sig_cls
        return_thresh:  float           # dead-zone for up/down/flat
    ):
        self.signal_thresh = signal_thresh
        self.return_thresh = return_thresh
        self.has_raw   = raw_close is not None

        # Filter windows by trading‐session start
        valid = [
            i for i, ts in enumerate(end_times)
            if pd.Timestamp(ts).time() >= sess_start_time
        ]
        self.X          = X[valid]
        self.y_signal   = y_signal[valid]
        self.y_return   = y_return[valid]
        self.end_times  = [pd.Timestamp(end_times[i]) for i in valid]

        if self.has_raw:
            self.raw_close = raw_close[valid]
            self.raw_bid   = raw_bid[valid]
            self.raw_ask   = raw_ask[valid]

        # Build day‐boundaries for grouping windows by calendar date
        dates        = pd.to_datetime(self.end_times).normalize()
        days, counts = np.unique(dates.values, return_counts=True)
        boundaries   = np.concatenate(([0], np.cumsum(counts)))
        self.start   = torch.tensor(boundaries[:-1], dtype=torch.long)
        self.end     = torch.tensor(boundaries[1:],  dtype=torch.long)
        self.weekday = torch.tensor(
            [d.dayofweek for d in pd.to_datetime(days)],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.start)

    def __getitem__(self, idx: int):
        s, e    = self.start[idx].item(), self.end[idx].item()

        # 1) inputs and regression target
        x_day   = self.X[s:e].unsqueeze(0)            # (1, W, look_back, F)
        y_day   = self.y_signal[s:e].unsqueeze(0)     # (1, W)

        # 2) binary signal‐threshold label
        y_sig_cls = (y_day > self.signal_thresh).float()

        # 3) true returns + ternary return label
        ret_day    = self.y_return[s:e].unsqueeze(0)  # (1, W)
        # start all as “flat”=1
        y_ret_ter  = torch.ones_like(ret_day, dtype=torch.long)
        y_ret_ter[ret_day >  self.return_thresh] = 2  # “up”
        y_ret_ter[ret_day < -self.return_thresh] = 0  # “down”

        wd     = int(self.weekday[idx].item())
        end_ts = self.end_times[e - 1]

        if self.has_raw:
            rc = self.raw_close[s:e]
            rb = self.raw_bid[s:e]
            ra = self.raw_ask[s:e]
            return (
                x_day, y_day, y_sig_cls, ret_day, y_ret_ter,
                rc, rb, ra, wd, end_ts
            )

        return x_day, y_day, y_sig_cls, ret_day, y_ret_ter, wd, end_ts

        
#########################################################################################################


def pad_collate(batch):
    """
    Pad a batch of per-day windows into fixed tensors and collect lengths.

    Batch items are either train/val:
      (x_day, y_day, y_sig_cls, ret_day, y_ret_ter, weekday, end_ts)
    or test (has raw prices):
      (x_day, y_day, y_sig_cls, ret_day, y_ret_ter,
       rc, rb, ra, weekday, end_ts)

    Returns:
      x_pad      Tensor (B, max_W, look_back, F)
      y_pad      Tensor (B, max_W)
      y_sig_pad  Tensor (B, max_W)
      ret_pad    Tensor (B, max_W)
      y_ter_pad  LongTensor (B, max_W)
      [rc_pad, rb_pad, ra_pad]  # only if has_raw
      wd_tensor  LongTensor (B,)
      ts_list    list of end_ts for each day
      lengths    list[int] true window counts per day
    """
    has_raw = len(batch[0]) == 10

    if has_raw:
        (x_list, y_list, ysig_list, ret_list, yter_list,
         rc_list, rb_list, ra_list, wd_list, ts_list) = zip(*batch)
    else:
        x_list, y_list, ysig_list, ret_list, yter_list, wd_list, ts_list = zip(*batch)

    # strip leading batch dim and collect sequences
    xs    = [x.squeeze(0) for x in x_list]     # (W_i, look_back, F)
    ys    = [y.squeeze(0) for y in y_list]     # (W_i,)
    ysig  = [yc.squeeze(0) for yc in ysig_list]
    rets  = [r.squeeze(0) for r in ret_list]
    yter  = [t.squeeze(0) for t in yter_list]

    lengths = [x.size(0) for x in xs]          # true W_i per day

    # pad along time-axis
    x_pad    = pad_sequence(xs,   batch_first=True)
    y_pad    = pad_sequence(ys,   batch_first=True)
    ysig_pad = pad_sequence(ysig, batch_first=True)
    ret_pad  = pad_sequence(rets, batch_first=True)
    yter_pad = pad_sequence(yter, batch_first=True)

    wd_tensor = torch.tensor(wd_list, dtype=torch.long)

    if has_raw:
        rc_pad = pad_sequence(rc_list, batch_first=True)
        rb_pad = pad_sequence(rb_list, batch_first=True)
        ra_pad = pad_sequence(ra_list, batch_first=True)
        return (
            x_pad, y_pad, ysig_pad, ret_pad, yter_pad,
            rc_pad, rb_pad, ra_pad, wd_tensor, list(ts_list), lengths
        )

    return x_pad, y_pad, ysig_pad, ret_pad, yter_pad, wd_tensor, list(ts_list), lengths


###################


def split_to_day_datasets(
    # train split tensors + times
    X_tr,   y_sig_tr,  y_ret_tr,  end_times_tr,
    # val split
    X_val,  y_sig_val, y_ret_val, end_times_val,
    # test split + raw-price arrays
    X_te,   y_sig_te,  y_ret_te,  end_times_te,
    raw_close_te, raw_bid_te, raw_ask_te,
    *,
    sess_start_time: time,    # session cutoff
    signal_thresh:  float,    # threshold for binary signal head
    return_thresh:  float,    # dead-zone threshold for ternary return head
    train_batch:           int = 32,
    train_workers:         int = 0,
    train_prefetch_factor: int = 1
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build three DataLoaders that yield *per‐day* batches of your LSTM windows:
    
      1) Instantiate DayWindowDataset for train, val, test.  
         - train/val: raw_close/raw_bid/raw_ask = None  
         - test:    raw_close/raw_bid/raw_ask = real price arrays  
         Each dataset filters windows before sess_start_time and computes:
           • y_sig_cls (binary from y_signal > signal_thresh)  
           • y_ret_ter (ternary from y_return vs ±return_thresh)  

      2) Wrap each dataset in a DataLoader:
         - train_loader: batch_size=train_batch, shuffle=False, pad_collate,
           num_workers=train_workers, pin_memory=True,
           persistent_workers=(train_workers>0), prefetch_factor=...
         - val_loader:   batch_size=1, shuffle=False, pad_collate
         - test_loader:  batch_size=1, shuffle=False, pad_collate

    Returns:
      (train_loader, val_loader, test_loader)
    """
    # 1) Build the three DayWindowDatasets with a brief progress bar
    splits = [
        ("train", X_tr, y_sig_tr, y_ret_tr, end_times_tr, None, None, None),
        ("val",   X_val, y_sig_val, y_ret_val, end_times_val,   None, None, None),
        ("test",  X_te,  y_sig_te,  y_ret_te,  end_times_te,  raw_close_te, raw_bid_te, raw_ask_te)
    ]

    datasets = {}
    for name, Xd, ys, yr, et, rc, rb, ra in tqdm(
        splits, desc="Creating DayWindowDatasets", unit="split"
    ):
        datasets[name] = DayWindowDataset(
            X=Xd,
            y_signal=ys,
            y_return=yr,
            raw_close=rc,
            raw_bid=rb,
            raw_ask=ra,
            end_times=et,
            sess_start_time=sess_start_time,
            signal_thresh=signal_thresh,
            return_thresh=return_thresh
        )

    # 2) Wrap in DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=train_batch,
        shuffle=False,
        drop_last=False,
        collate_fn=pad_collate,
        num_workers=train_workers,
        pin_memory=True,
        persistent_workers=(train_workers > 0),
        prefetch_factor=(train_prefetch_factor if train_workers > 0 else None)
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=1,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=1,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

#########################################################################################################


def naive_rmse(data_loader):
    """
    Zero‐forecast baseline RMSE for any DayDataset loader:
      – always predicts 0
      – works for val_loader (xb, yb, wd)
      – and test_loader (xb, yb, raw_close, raw_bid, raw_ask, wd)
    """
    total_se = 0.0
    total_n  = 0

    for batch in data_loader:
        # batch[1] is always y_day regardless of extra fields
        y_day = batch[1]  
        # y_day: shape (1, W) → squeeze→ (W,)
        y = y_day.squeeze(0).view(-1)

        # accumulate (0 - y)^2 = y^2
        total_se += float((y ** 2).sum().item())
        total_n  += y.numel()

    return math.sqrt(total_se / total_n)
    

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
            *lead, S, F = x.shape
            x = x.view(-1, S, F)

        # ensure last dim is features
        if x.dim() == 3 and x.size(-1) != self.n_feats:
            x = x.transpose(1, 2).contiguous()

        B, S, _ = x.size()
        dev      = x.device

        # 0) conv over time
        x_conv = x.transpose(1, 2)
        x_conv = Funct.relu(self.conv(x_conv))
        x      = x_conv.transpose(1, 2)

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


def make_optimizer_and_scheduler(
    model: nn.Module,
    initial_lr: float,
    weight_decay: float,
    clipnorm: float
):
    """
    1) AdamW optimizer with decoupled weight decay
    2) ReduceLROnPlateau for val‐RMSE stagnation
    3) CosineAnnealingWarmRestarts at batch‐level
    4) GradScaler for mixed‐precision safety
    """
    optimizer = AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay
    )

    plateau_sched = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=params.hparams['PLATEAU_FACTOR'],
        patience=params.hparams['PLATEAU_PATIENCE'],
        min_lr=params.hparams['MIN_LR'],
    )

    cosine_sched = CosineAnnealingWarmRestarts(
        optimizer,
        T_0   = params.hparams['T_0'],
        T_mult= params.hparams['T_MULT'],
        eta_min=params.hparams['ETA_MIN']
    )

    scaler = GradScaler()

    return optimizer, plateau_sched, cosine_sched, scaler, clipnorm


#########################################################################################################


def custom_stateful_training_loop(
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
    baseline_val_rmse:   float,
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
      5) (Validation loop and checkpointing follow unchanged.)
    """
    # 1) Device setup
    model.to(device)
    torch.backends.cudnn.benchmark = True

    # 2) Losses & weights
    beta_huber = params.hparams["HUBER_BETA"]
    huber_loss = nn.SmoothL1Loss(beta=beta_huber)      # mean reduction
    bce_loss   = nn.BCEWithLogitsLoss()
    alpha_cls  = params.hparams["CLS_LOSS_WEIGHT"]
    # trinary‐head loss and spike/derivative weighting removed

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
    # ternary‐head metrics removed

    val_rmse  = torchmetrics.MeanSquaredError(squared=False).to(device)
    val_mae   = torchmetrics.MeanAbsoluteError().to(device)
    val_r2    = torchmetrics.R2Score().to(device)
    val_acc   = torchmetrics.classification.BinaryAccuracy(threshold=thr).to(device)
    val_prec  = torchmetrics.classification.BinaryPrecision(threshold=thr).to(device)
    val_rec   = torchmetrics.classification.BinaryRecall(threshold=thr).to(device)
    val_f1    = torchmetrics.classification.BinaryF1Score(threshold=thr).to(device)
    val_auc   = torchmetrics.classification.BinaryAUROC().to(device)
    # ternary‐head metrics removed

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
            # Unpack: drop ternary‐head targets but still load them
            xb_days, y_sig_days, y_sig_cls_days, ret_days, y_ret_ter_days, wd_days, ts_list, lengths = batch

            xb    = xb_days.to(device, non_blocking=True)
            y_sig = y_sig_days.to(device, non_blocking=True)
            y_cls = y_sig_cls_days.to(device, non_blocking=True)
            wd    = wd_days.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prev_day = None

            # Process each sequence (“day”) in the batch
            for di in range(xb.size(0)):
                W      = lengths[di]
                day_id = int(wd[di].item())

                x_seq   = xb[di, :W]
                sig_seq = y_sig[di, :W]
                cls_seq = y_cls[di, :W].view(-1)

                # Reset or carry LSTM states on day rollover
                model.reset_short()
                if prev_day is not None and day_id < prev_day:
                    model.reset_long()
                prev_day = day_id

                # 1) Full-precision forward
                pr, pc, _ = model(x_seq)
                
                # 2) Inside autocast, build only the loss & backward
                with autocast(device_type=device.type):
                    # Sigmoid/reg logits
                    lr = torch.sigmoid(pr[..., -1, 0])   # (W,)
                    lc = pc[...,    -1, 0]               # (W,)
                
                    # Loss = Huber(regression) + α·BCE(binary)
                    loss_r = huber_loss(lr, sig_seq)
                    loss_b = bce_loss(lc, cls_seq)
                    loss   = loss_r + alpha_cls * loss_b
                
                scaler.scale(loss).backward()
                

                # Update regression metrics
                train_rmse.update(lr, sig_seq)
                train_mae .update(lr, sig_seq)
                train_r2  .update(lr, sig_seq)

                # Update binary‐classification metrics
                probs = torch.sigmoid(lc)
                train_acc .update(probs, cls_seq)
                train_prec.update(probs, cls_seq)
                train_rec .update(probs, cls_seq)
                train_f1  .update(probs, cls_seq)
                train_auc .update(probs, cls_seq)

                # Truncate backprop through states
                model.h_short.detach_(); model.c_short.detach_()
                model.h_long .detach_(); model.c_long .detach_()

            # Gradient clipping & optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer); scaler.update()

            # Cosine learning‐rate update
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

        # plateau & checkpoint (unchanged) …
        if epoch > params.hparams["LR_EPOCHS_WARMUP"]:
            plateau_sched.step(vl["rmse"])
        if vl["rmse"] < best_val_rmse:
            best_val_rmse = vl["rmse"]
            best_state    = model.state_dict()
            patience_ctr  = 0
            model.load_state_dict(best_state)
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print("Early stopping at epoch", epoch)
                break

    # 5) Final save if improved
    existing = [
        float(m.group(1))
        for f in params.save_path.glob(f"{params.ticker}_*.pth")
        for m in (save_pat.match(f.name),) if m
    ]
    best_existing = min(existing) if existing else float("inf")
    if best_val_rmse < best_existing:
        buf = io.BytesIO()
        live_plot.fig.savefig(buf, format="png")
        buf.seek(0)
        torch.save({
            "model_obj":        model,
            "model_state_dict": best_state,
            "hparams":          params.hparams,
            "train_plot_png":   buf.read(),
            "train_metrics":      tr,
            "val_metrics":        vl,
        }, params.save_path / f"{params.ticker}_{best_val_rmse:.4f}.pth")

    return best_val_rmse

#########################################################################################################


def feature_engineering(df: pd.DataFrame,
                        features_cols: list,
                        label_col: str) -> pd.DataFrame:
    """
    Compute only the features in features_cols + ['bid','ask',label_col].
    High-impact features are calculated first, then the rest in list order.
    """
    # ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # VWAP deviation
    if "vwap_dev" in features_cols:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        cum_vp = (tp * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        vwap = cum_vp / cum_vol
        df["vwap_dev"] = (df["close"] - vwap) / vwap

    # Intraday log‐returns
    if "r_1" in features_cols:
        df["r_1"] = np.log(df["close"] / df["close"].shift(1))
    if "r_5" in features_cols:
        df["r_5"] = np.log(df["close"] / df["close"].shift(5))
    if "r_15" in features_cols:
        df["r_15"] = np.log(df["close"] / df["close"].shift(15))

    # Average True Range
    if "atr_14" in features_cols:
        hl = df["high"] - df["low"]
        hp = (df["high"] - df["close"].shift()).abs()
        lp = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([hl, hp, lp], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

    # Rolling 15-period volatility
    if "vol_15" in features_cols:
        # ensure r_1 exists
        if "r_1" not in df.columns: # and "r_1" in features_cols:
            df["r_1"] = np.log(df["close"] / df["close"].shift(1))
        df["vol_15"] = df["r_1"].rolling(15).std()

    # Volume spike over 15
    if "volume_spike" in features_cols:
        df["volume_spike"] = df["volume"] / df["volume"].rolling(15).mean()

    # RSI (14)
    if "rsi_14" in features_cols:
        d = df["close"].diff()
        gain = d.clip(lower=0)
        loss = -d.clip(upper=0)
        ag = gain.rolling(14).mean()
        al = loss.rolling(14).mean()
        rs = ag / al
        df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Band width (20, ±2σ)
    if "bb_width_20" in features_cols:
        m20 = df["close"].rolling(20).mean()
        s20 = df["close"].rolling(20).std()
        upper = m20 + 2 * s20
        lower = m20 - 2 * s20
        df["bb_width_20"] = (upper - lower) / m20

    # Stochastic oscillator
    if "stoch_k_14" in features_cols or "stoch_d_3" in features_cols:
        lo14 = df["low"].rolling(14).min()
        hi14 = df["high"].rolling(14).max()
        k = 100 * (df["close"] - lo14) / (hi14 - lo14)
        if "stoch_k_14" in features_cols:
            df["stoch_k_14"] = k
        if "stoch_d_3" in features_cols:
            df["stoch_d_3"] = k.rolling(3).mean()

    # Moving averages & cross
    if "ma_5" in features_cols:
        df["ma_5"] = df["close"].rolling(5).mean()
    if "ma_20" in features_cols:
        df["ma_20"] = df["close"].rolling(20).mean()
    if "ma_diff" in features_cols:
        if {"ma_5", "ma_20"}.issubset(df.columns):
            df["ma_diff"] = df["ma_5"] - df["ma_20"]

    # MACD & signal
    if {"macd_12_26", "macd_signal_9"} & set(features_cols):
        e12 = df["close"].ewm(span=12, adjust=False).mean()
        e26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = e12 - e26
        sig  = macd.ewm(span=9, adjust=False).mean()
        if "macd_12_26" in features_cols:
            df["macd_12_26"] = macd
        if "macd_signal_9" in features_cols:
            df["macd_signal_9"] = sig

    # On-Balance Volume
    if "obv" in features_cols:
        dir_ = np.sign(df["close"].diff()).fillna(0)
        df["obv"] = (dir_ * df["volume"]).cumsum()

    # Calendar flags
    if "hour" in features_cols:
        df["hour"] = df.index.hour
    if "day_of_week" in features_cols:
        df["day_of_week"] = df.index.dayofweek
    if "month" in features_cols:
        df["month"] = df.index.month

    # Order-book imbalance
    if "order_imbalance" in features_cols:
        bv = df.get("bid_volume", 0)
        av = df.get("ask_volume", 0)
        df["order_imbalance"] = (bv - av) / (bv + av).replace(0, np.nan)

    if "in_trading" in features_cols:
        df["in_trading"] = (
            (df.index.time >= params.sess_start) &
            (df.index.time < params.sess_end) &
            (df.index.dayofweek < 5)           # Monday=0 … Friday=4
        ).astype(int)

    # Final filter
    cols = [c for c in features_cols if c in df.columns] + ["bid", "ask", label_col]

    df = df.loc[:, cols].dropna()

    return df


#########################################################################################################

def scale_with_splits(
    df: pd.DataFrame,
    features_cols: list[str],
    label_col: str,
    train_prop: float = 0.70,
    val_prop: float   = 0.15
) -> pd.DataFrame:
    """
    1) Copy and chronologically split df into train/val/test by row index.
    2) Encode 'hour', 'day_of_week', 'month' as cyclic sin/cos features.
    3) Define feature groups:
       - price_feats   : open, high, low, close, volume, ATR, moving averages, etc.
       - ratio_feats   : returns, vol_15, vwap_dev, indicators, etc.
       - binary_feats  : in_trading flag
       - cyclic_feats  : hour, day_of_week, month (post‐PCA)
    4) Fit StandardScaler on TRAIN’s ratio_feats.
    5) **Per-day expanding robust scaling** on price_feats:
       for each calendar day, use only data ≤ current bar to compute 
       median & IQR, then scale that bar.
    6) Apply ratio and price scalers to train/val/test splits.
    7) Fit PCA(1) on train’s sin/cos pairs and transform all splits back to single
       'hour', 'day_of_week', 'month' columns.
    8) Reattach label_col, concat splits, select and return final columns.
    """
    df = df.copy()

    # 1) Split into train/val/test
    n       = len(df)
    n_train = int(n * train_prop)
    n_val   = int(n * val_prop)
    if n_train + n_val >= n:
        raise ValueError("train_prop + val_prop must sum to < 1.0")

    df_tr = df.iloc[:n_train].copy()
    df_v  = df.iloc[n_train : n_train + n_val].copy()
    df_te = df.iloc[n_train + n_val :].copy()

    # 2) Generate cyclic sin/cos for time features
    for sub in (df_tr, df_v, df_te):
        h = sub["hour"]
        sub["hour_sin"],     sub["hour_cos"]     = np.sin(2*np.pi*h/24),     np.cos(2*np.pi*h/24)
        d = sub["day_of_week"]
        sub["day_of_week_sin"], sub["day_of_week_cos"] = np.sin(2*np.pi*d/7),  np.cos(2*np.pi*d/7)
        m = sub["month"]
        sub["month_sin"],    sub["month_cos"]    = np.sin(2*np.pi*m/12),    np.cos(2*np.pi*m/12)

    # 3) Define feature groups
    all_price = ["open","high","low","close","volume","atr_14",
                 "ma_5","ma_20","ma_diff","macd_12_26","macd_signal_9","obv"]
    all_ratio = ["r_1","r_5","r_15","vol_15","volume_spike",
                 "vwap_dev","rsi_14","bb_width_20","stoch_k_14","stoch_d_3"]

    price_feats  = [f for f in all_price  if f in features_cols]
    ratio_feats  = [f for f in all_ratio  if f in features_cols]
    binary_feats = [f for f in ["in_trading"] if f in features_cols]
    cyclic_feats = ["hour","day_of_week","month"]

    # 4) Fit StandardScaler on TRAIN ratio_feats
    ratio_scaler = StandardScaler()
    if ratio_feats:
        ratio_scaler.fit(df_tr[ratio_feats])

    # 5) Per-day expanding robust scaling on price_feats (no leakage)
    def scale_price_per_day(sub: pd.DataFrame, desc: str) -> pd.DataFrame:
        out = sub.copy()
        days = out.index.normalize().unique()
        for day in tqdm(days, desc=f"Scaling price per day ({desc})", unit="day"):
            mask  = out.index.normalize() == day
            block = out.loc[mask, price_feats]

            # expanding median & IQR up to each bar
            med = block.expanding().median()
            q75 = block.expanding().quantile(0.75)
            q25 = block.expanding().quantile(0.25)
            iqr = (q75 - q25).replace(0, 1e-6)

            # apply element-wise: bar i uses only bars ≤ i
            out.loc[mask, price_feats] = (block - med) / iqr
        return out

    # 6) Transform splits
    def transform(sub: pd.DataFrame, split_name: str) -> pd.DataFrame:
        out = sub.copy()
        if price_feats:
            out = scale_price_per_day(out, split_name)
        if ratio_feats:
            out[ratio_feats] = ratio_scaler.transform(sub[ratio_feats])
        return out

    df_tr_s  = transform(df_tr,  "train")
    df_val_s = transform(df_v,   "val")
    df_te_s  = transform(df_te,  "test")

    # 7) PCA(1) on sin/cos pairs, fit on TRAIN → transform all
    pca_hour = PCA(n_components=1).fit(df_tr_s[["hour_sin","hour_cos"]])
    pca_dow  = PCA(n_components=1).fit(df_tr_s[["day_of_week_sin","day_of_week_cos"]])
    pca_mo   = PCA(n_components=1).fit(df_tr_s[["month_sin","month_cos"]])

    def apply_cyclic_pca(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.copy()
        out["hour"]        = np.round(pca_hour.transform(out[["hour_sin","hour_cos"]])[:,0], 3)
        out["day_of_week"] = np.round(pca_dow.transform(out[["day_of_week_sin","day_of_week_cos"]])[:,0], 3)
        out["month"]       = np.round(pca_mo.transform(out[["month_sin","month_cos"]])[:,0], 3)
        out.drop([
            "hour_sin","hour_cos",
            "day_of_week_sin","day_of_week_cos",
            "month_sin","month_cos"
        ], axis=1, inplace=True)
        return out

    df_tr_s  = apply_cyclic_pca(df_tr_s)
    df_val_s = apply_cyclic_pca(df_val_s)
    df_te_s  = apply_cyclic_pca(df_te_s)

    # 8) Reattach label and recombine
    for part in (df_tr_s, df_val_s, df_te_s):
        part[label_col] = df.loc[part.index, label_col]

    df_final = pd.concat([df_tr_s, df_val_s, df_te_s]).sort_index()

    # 9) Return only final ordered columns
    final_cols = (
        price_feats +
        ratio_feats +
        binary_feats +
        cyclic_feats +
        ["bid", "ask"] +
        [label_col]
    )
    return df_final[final_cols]
