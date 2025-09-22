from libs import plots, params

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

import torch
import torch.nn as nn
import torch.nn.functional as Funct
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.optim import AdamW

from tqdm.auto import tqdm


#########################################################################################################


def build_tensors(
    df: pd.DataFrame,
    *,
    look_back:  int                   = params.look_back_tick,
    tmpdir:     str                   = None,
    device:     torch.device          = torch.device("cpu"),
    sess_start: time                  = None
) -> tuple[
    torch.Tensor,  # X         shape=(N, look_back, F)
    torch.Tensor,  # y_sig     shape=(N,)
    torch.Tensor,  # y_ret     shape=(N,)
    torch.Tensor,  # raw_close shape=(N,)
    np.ndarray     # end_times shape=(N,) dtype=datetime64[ns]
]:
    """
    Build sliding‐window tensors for an LSTM trading model from full‐day data.

    1) Select all columns except {label_col, 'close_raw'} as features.
    2) Group by calendar day and count windows ending ≥ sess_start.
    3) Allocate on‐disk memmaps for inputs, targets, raw close, and timestamps.
    4) For each day:
       a) extract feature array and label signal;
       b) compute log‐returns on raw close price;
       c) build sliding windows of features;
       d) align next‐bar labels, raw close points, and times;
       e) mask windows before sess_start and write to memmaps.
    5) Wrap memmaps as torch tensors, free CPU memory, return tensors + end_times.
    """
    # copy to avoid mutating user’s DataFrame
    df = df.copy()
    
    # 1) derive feature columns
    exclude = {params.label_col, "close_raw"}
    features_cols = [c for c in df.columns if c not in exclude]
    print("Inside build_tensors, features:", features_cols)
    F = len(features_cols)
    
    # group by calendar day
    day_groups = df.groupby(df.index.normalize(), sort=False)
    
    # 2) count valid windows
    N = 0
    for _, day_df in tqdm(day_groups, desc="Counting windows", leave=False):
        T = len(day_df)
        if T <= look_back:
            continue
        ends = day_df.index[look_back:]
        mask = np.array([ts.time() >= sess_start for ts in ends])
        N += int(mask.sum())
    
    # 3) allocate memmaps
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
        os.path.join(tmpdir, "close.npy"), mode="w+",
        dtype=np.float32, shape=(N,)
    )
    t_mm = np.lib.format.open_memmap(
        os.path.join(tmpdir, "t.npy"), mode="w+",
        dtype="datetime64[ns]", shape=(N,)
    )
    
    # 4) fill memmaps
    idx = 0
    for _, day_df in tqdm(day_groups, desc="Writing memmaps", leave=False):
        day_df = day_df.sort_index()
        T = len(day_df)
        if T <= look_back:
            continue
    
        # a) features and signal
        feats_np = day_df[features_cols].to_numpy(np.float32)
        sig_np   = day_df[params.label_col].to_numpy(np.float32)
    
        # b) raw close and log‐returns
        close_np = day_df["close_raw"].to_numpy(np.float32)
        ret_full = np.zeros_like(close_np, dtype=np.float32)
        ret_full[1:] = np.log(close_np[1:] / close_np[:-1])
    
        # c) sliding windows of features
        wins = np.lib.stride_tricks.sliding_window_view(
            feats_np, window_shape=(look_back, F)
        ).reshape(T - look_back + 1, look_back, F)
        wins = wins[:-1]  # drop last to align next‐bar labels
    
        # d) align labels and raw close
        lab_sig = sig_np[look_back:]
        lab_ret = ret_full[look_back:]
        c_pts   = close_np[look_back:]
        times   = day_df.index.to_numpy()[look_back:]
    
        # e) mask by session start
        mask = np.array([pd.Timestamp(ts).time() >= sess_start for ts in times])
        if not mask.any():
            continue
    
        m = mask.sum()
        X_mm[idx:idx+m] = wins[mask]
        y_mm[idx:idx+m] = lab_sig[mask]
        r_mm[idx:idx+m] = lab_ret[mask]
        c_mm[idx:idx+m] = c_pts[mask]
        t_mm[idx:idx+m] = times[mask]
        idx += m
    
    # 5) wrap as torch tensors
    X         = torch.from_numpy(X_mm).to(device, non_blocking=True)
    y_sig     = torch.from_numpy(y_mm).to(device, non_blocking=True)
    y_ret     = torch.from_numpy(r_mm).to(device, non_blocking=True)
    raw_close = torch.from_numpy(c_mm).to(device, non_blocking=True)
    end_times = t_mm.copy()  # numpy datetime64 array
    
    # cleanup
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return X, y_sig, y_ret, raw_close, end_times

    
#########################################################################################################


def chronological_split(
    X:           torch.Tensor,
    y_sig:       torch.Tensor,
    y_ret:       torch.Tensor,
    raw_close:   torch.Tensor,
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
                                                              # (X_te, y_sig_te, y_ret_te, raw_close_te)
    list,                                                     # samples_per_day
    torch.Tensor, torch.Tensor, torch.Tensor                  # day_id_tr, day_id_val, day_id_te
]:
    """
    Split time‐series windows into train/val/test by calendar day.

    1) Count how many windows fall on each normalized date.
    2) Decide how many days go to train/val/test by proportions
       (training days rounded up to full batches of train_batch).
    3) Compute cumulative sums of daily counts and slice X, y_sig, y_ret,
       plus raw_close for the test set.
    4) Build per‐window day_id tags for each split as CPU tensors
       (so they can be moved to GPU per‐batch in a DataLoader).
    """
    # 1) Count windows per normalized day
    dt_idx          = pd.to_datetime(end_times)
    normed          = dt_idx.normalize()
    days, counts    = np.unique(normed.values, return_counts=True)
    samples_per_day = counts.tolist()

    # sanity check: total windows equals first dim of X
    total = sum(samples_per_day)
    if total != X.size(0):
        raise ValueError(f"Window count mismatch {total} vs {X.size(0)}")

    # 2) Determine cut‐points in days
    D             = len(samples_per_day)
    orig_tr_days  = int(D * train_prop)
    full_batches  = (orig_tr_days + train_batch - 1) // train_batch
    tr_days       = min(D, full_batches * train_batch)
    cut_train     = tr_days - 1
    cut_val       = int(D * (train_prop + val_prop))

    # 3) Slice by window counts
    cumsum        = np.concatenate([[0], np.cumsum(counts)])
    i_tr          = int(cumsum[tr_days])
    i_val         = int(cumsum[cut_val + 1])

    X_tr,  y_sig_tr,  y_ret_tr  = X[:i_tr],       y_sig[:i_tr],       y_ret[:i_tr]
    X_val, y_sig_val, y_ret_val = X[i_tr:i_val],  y_sig[i_tr:i_val],  y_ret[i_tr:i_val]
    X_te,  y_sig_te,  y_ret_te   = X[i_val:],     y_sig[i_val:],      y_ret[i_val:]
    close_te = raw_close[i_val:]

    # 4) Build day‐ID vectors on CPU
    def make_day_ids(start_day: int, end_day: int) -> torch.Tensor:
        # counts for days start_day…end_day inclusive
        cnts = samples_per_day[start_day : end_day + 1]
        # day indices  start_day, start_day+1, …, end_day
        days_idx = torch.arange(start_day, end_day + 1, dtype=torch.long)
        # repeat each day index by its count
        return days_idx.repeat_interleave(torch.tensor(cnts, dtype=torch.long))

    day_id_tr  = make_day_ids(0,          cut_train)
    day_id_val = make_day_ids(cut_train+1, cut_val)
    day_id_te  = make_day_ids(cut_val+1,  D - 1)

    return (
        (X_tr,  y_sig_tr,  y_ret_tr),
        (X_val, y_sig_val, y_ret_val),
        (X_te,  y_sig_te,  y_ret_te,  close_te),
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
      • y_ret_ter : ternary label ∈ {0,1,2} for (down, flat, up) based on ret_day and return_thresh
      • rc        : optional raw close tensor, each shape (W,)
      • weekday   : integer day-of-week
      • end_ts    : timestamp of the last bar in the window
    """
    def __init__(
        self,
        X:              torch.Tensor,   # (N_windows, look_back, F)
        y_signal:       torch.Tensor,   # (N_windows,)
        y_return:       torch.Tensor,   # (N_windows,)
        raw_close:      torch.Tensor,   # or None
        end_times:      np.ndarray,     # (N_windows,), dtype datetime64[ns]
        sess_start_time: time,          # cutoff for trading session
        signal_thresh:  float,          # buy_threshold for y_sig_cls
        return_thresh:  float           # dead-zone for up/down/flat
    ):
        self.signal_thresh = signal_thresh
        self.return_thresh = return_thresh
        self.has_raw       = raw_close is not None

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
            return (
                x_day, y_day, y_sig_cls, ret_day, y_ret_ter, rc, wd, end_ts
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
       rc, weekday, end_ts)

    Returns:
      x_pad      Tensor (B, max_W, look_back, F)
      y_pad      Tensor (B, max_W)
      y_sig_pad  Tensor (B, max_W)
      ret_pad    Tensor (B, max_W)
      y_ter_pad  LongTensor (B, max_W)
      rc_pad  # only if has_raw
      wd_tensor  LongTensor (B,)
      ts_list    list of end_ts for each day
      lengths    list[int] true window counts per day
    """
    has_raw = len(batch[0]) == 8

    if has_raw:
        (x_list, y_list, ysig_list, ret_list, yter_list, rc_list, wd_list, ts_list) = zip(*batch)
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
        return (
            x_pad, y_pad, ysig_pad, ret_pad, yter_pad, rc_pad, wd_tensor, list(ts_list), lengths
        )

    return x_pad, y_pad, ysig_pad, ret_pad, yter_pad, wd_tensor, list(ts_list), lengths


###################


def split_to_day_datasets(
    # train split tensors + times
    X_tr,   y_sig_tr,  y_ret_tr,  end_times_tr,
    # val split
    X_val,  y_sig_val, y_ret_val, end_times_val,
    # test split + raw-price arrays
    X_te,   y_sig_te,  y_ret_te,  end_times_te,  raw_close_te,
    *,
    sess_start_time: time,    # session cutoff
    signal_thresh:  float,    # threshold for binary signal head
    return_thresh:  float,    # dead-zone threshold for ternary return head
    train_batch:           int = 32,
    train_workers:         int = 0,
    train_prefetch_factor: int = 1
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build three DataLoaders that yield *per‐day* batches of the LSTM windows:
    
      1) Instantiate DayWindowDataset for train, val, test.  
         - train/val: raw_close = None  
         - test:    raw_close = real price arrays  
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
        ("train", X_tr, y_sig_tr, y_ret_tr, end_times_tr, None),
        ("val",   X_val, y_sig_val, y_ret_val, end_times_val,   None),
        ("test",  X_te,  y_sig_te,  y_ret_te,  end_times_te,  raw_close_te)
    ]

    datasets = {}
    for name, Xd, ys, yr, et, rc in tqdm(
        splits, desc="Creating DayWindowDatasets", unit="split"
    ):
        datasets[name] = DayWindowDataset(
            X=Xd,
            y_signal=ys,
            y_return=yr,
            raw_close=rc,
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


