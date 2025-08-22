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
from datetime import datetime
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

torch.backends.cuda.matmul.allow_tf32    = True
torch.backends.cudnn.allow_tf32         = True
torch.backends.cudnn.benchmark          = True

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tqdm.auto import tqdm


#########################################################################################################

def build_lstm_tensors(
    df: pd.DataFrame,
    *,
    look_back: int,
    features_cols: Sequence[str],
    label_col: str,
    tmpdir: str = None,
    device: torch.device = torch.device("cpu"),
    sess_start = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build disk-backed memmaps for LSTM windows, then wrap in PyTorch Tensors.

    1) First pass: count total valid windows (N) to size memmaps
    2) Allocate .npy memmaps: X_mm, y_mm, c_mm, b_mm, a_mm
    3) Second pass: for each day
       a) load raw arrays (features, labels, prices) as float32
       b) build sliding windows via np.lib.stride_tricks
       c) align targets (drop last window)
       d) filter by sess_start (RTH mask)
       e) write slices into memmaps at offset idx
    4) Wrap memmaps in torch.from_numpy and move to device
    5) Return five tensors: X, y, raw_close, raw_bid, raw_ask

    Auto‐cleanup: the contents of tmpdir persist after this function returns.
    We register an atexit handler so that when Python exits, tmpdir is
    automatically removed. 
    """
    if not sess_start: # if we want the predictions not to start from sess_start, but from sess_start_pred
        sess_start = dt.time(*divmod((params.sess_start.hour * 60 + params.sess_start.minute) - look_back, 60))
    else:
        sess_start = params.sess_start

    # ── Create / verify tmpdir ────────────────────────────────────────────
    # 0) temp directory
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="lstm_memmap_")
    else:
        os.makedirs(tmpdir, exist_ok=True)

    # ── 1) Count total number of valid windows across all days ────────────
    N = 0                     # total windows after RTH filtering
    F = len(features_cols)    # number of features per time step
    for _, day in df.groupby(df.index.normalize(), sort=False):
        T = len(day)  # minutes in this day
        # windows before filter: T – look_back
        if T <= look_back:
            continue
        # boolean mask: which end‐times ≥ sess_start
        mask = np.array(day.index.time[look_back:]) >= sess_start
        if mask.any():
            N += int(mask.sum())

    # ── 2) Allocate on‐disk memmaps for X, y, and raw prices ─────────────
    #   X_mm: shape (N, look_back, F)
    X_mm = np.lib.format.open_memmap(os.path.join(tmpdir, "X.npy"), mode="w+", dtype=np.float32, shape=(N, look_back, F))
    # y_mm and price memmaps: shape (N,)
    y_mm = np.lib.format.open_memmap(os.path.join(tmpdir, "y.npy"), mode="w+", dtype=np.float32, shape=(N,))
    c_mm = np.lib.format.open_memmap(os.path.join(tmpdir, "c.npy"), mode="w+", dtype=np.float32, shape=(N,))
    b_mm = np.lib.format.open_memmap(os.path.join(tmpdir, "b.npy"), mode="w+", dtype=np.float32, shape=(N,))
    a_mm = np.lib.format.open_memmap(os.path.join(tmpdir, "a.npy"), mode="w+", dtype=np.float32, shape=(N,))

    # ── 3) Fill memmaps day by day ────────────────────────────────────────
    idx = 0  # write offset into memmaps
    for _, day in df.groupby(df.index.normalize(), sort=False):
        day = day.sort_index()
        T   = len(day)
        if T <= look_back:
            continue  # not enough points to form one window

        # 3a) Extract raw arrays as NumPy float32
        feats_np  = day[features_cols].to_numpy(dtype=np.float32)  # (T, F)
        labels_np = day[label_col].to_numpy(dtype=np.float32)      # (T,)
        close_np  = day["close"].to_numpy(dtype=np.float32)       # (T,)
        bid_np    = day["bid"].to_numpy(dtype=np.float32)
        ask_np    = day["ask"].to_numpy(dtype=np.float32)

        # 3c) Build sliding windows via stride_tricks
        #    windows shape before drop: (T - look_back + 1, look_back, F)
        windows = np.lib.stride_tricks.sliding_window_view(feats_np, window_shape=(look_back, F))
        windows = windows.reshape(T - look_back + 1, look_back, F)

        # 3d) Align to next‐step label: drop the last window
        windows = windows[:-1]                # now (T - look_back, look_back, F)
        targets = labels_np[look_back:]       # (T - look_back,)
        c_pts   = close_np [look_back:]
        b_pts   = bid_np   [look_back:]
        a_pts   = ask_np   [look_back:]

        # 3e) RTH filter: end‐time ≥ sess_start
        mask = np.array(day.index.time[look_back:]) >= sess_start
        if not mask.any():
            continue  # skip days with no valid windows

        # 3f) Write valid slices into memmaps
        m = mask.sum()  # number of valid windows
        X_mm[idx:idx+m] = windows[mask]
        y_mm[idx:idx+m] = targets[mask]
        c_mm[idx:idx+m] = c_pts[mask]
        b_mm[idx:idx+m] = b_pts[mask]
        a_mm[idx:idx+m] = a_pts[mask]
        idx += m  # advance write index

    # ── 4) Wrap memmaps in PyTorch Tensors and move to device ────────────

    # COPY the memmaps into brand‐new NumPy arrays
    X_arr = np.array(X_mm, copy=True)
    y_arr = np.array(y_mm, copy=True)
    c_arr = np.array(c_mm, copy=True)
    b_arr = np.array(b_mm, copy=True)
    a_arr = np.array(a_mm, copy=True)

    # CLEAN UP the memmaps + folder
    del X_mm, y_mm, c_mm, b_mm, a_mm
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Convert to PyTorch Tensors (this is where the actual copy happens)
    X = torch.tensor(X_arr, device=device)
    y = torch.tensor(y_arr, device=device)
    raw_close = torch.tensor(c_arr, device=device)
    raw_bid   = torch.tensor(b_arr, device=device)
    raw_ask   = torch.tensor(a_arr, device=device)
    
    # Clean up intermediates & cache
    del X_arr, y_arr, c_arr, b_arr, a_arr
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return X, y, raw_close, raw_bid, raw_ask


#########################################################################################################


def chronological_split(
    X: torch.Tensor,
    y: torch.Tensor,
    raw_close: torch.Tensor,
    raw_bid: torch.Tensor,
    raw_ask: torch.Tensor,
    df: pd.DataFrame,
    *,
    look_back: int,
    train_prop: float,
    val_prop: float,
    train_batch: int,
    device = torch.device("cpu"),
    sess_start = True
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    List[int],
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Split the full sliding‐window dataset (X, y, raw_*) into train, validation,
    and test sets by calendar day, then apply Max‐Abs scaling fitted on
    the train split only.

    We first move all tensors to the chosen device. By grouping `df` on
    normalized dates and checking `look_back` vs. `sess_start`, we
    build `samples_per_day` and verify it matches X.size(0). We then
    decide how many days go to train (rounded up to `train_batch`),
    val, and test.

    Using a cumulative sum of daily counts, we slice X, y, and raw price
    tensors into their respective splits without boolean indexing.
    
    Finally build day‐ID tags via `repeat_interleave` so each window 
    can be traced back to its original calendar day.
    """
    if not sess_start: # if we want the predictions not to start from sess_start, but from sess_start_pred
        sess_start = dt.time(*divmod((params.sess_start.hour * 60 + params.sess_start.minute) - look_back, 60))
    else:
        sess_start = params.sess_start
        
    # Pick a real device & move the full dataset there
    device = device or (X.device if X.device is not None else torch.device("cpu"))
    X         = X.to(device)
    y         = y.to(device)
    raw_close = raw_close.to(device)
    raw_bid   = raw_bid.to(device)
    raw_ask   = raw_ask.to(device)

    # 1) Count how many windows come from each calendar day
    samples_per_day: List[int] = []
    all_days: List[pd.Timestamp] = []

    for day, day_df in df.groupby(df.index.normalize(), sort=False):
        all_days.append(day)
        end_times = day_df.index.time[look_back:]
        mask_rth  = np.array([t >= sess_start for t in end_times])
        samples_per_day.append(int(mask_rth.sum()))

    # 2) Sanity‐check total windows
    total = sum(samples_per_day)
    if total != X.size(0):
        raise ValueError(f"Window count mismatch: {total} vs {X.size(0)}")

    # 3) Decide how many days go to train/val/test
    D               = len(samples_per_day)
    train_days_orig = int(D * train_prop)
    batches_needed  = (train_days_orig + train_batch - 1) // train_batch
    train_days      = min(D, batches_needed * train_batch)
    cut_train       = train_days - 1
    cut_val         = int(D * (train_prop + val_prop))  # inclusive last val‐day index

    # 4) Build a cumulative‐sum array of sample counts
    #    cum[i] = total windows in days[0..i-1], so slices are views
    cum = np.concatenate([[0], np.cumsum(samples_per_day)])

    # 5) Compute slice indices
    end_train = int(cum[train_days])          # start of day train_days
    end_val   = int(cum[cut_val + 1])        # start of day cut_val+1

    # 6) Range‐slice the big tensors (these are views, not copies)
    X_tr       = X[:end_train]
    y_tr       = y[:end_train]

    X_val      = X[end_train:end_val]
    y_val      = y[end_train:end_val]

    X_te       = X[end_val:]
    y_te       = y[end_val:]
    close_te   = raw_close[end_val:]
    bid_te     = raw_bid[end_val:]
    ask_te     = raw_ask[end_val:]

    # 7) Build day‐ID tags via repeat_interleave per split
    #    (you can drop these if you don’t strictly need them)
    def make_day_ids(start_day: int, end_day: int) -> torch.Tensor:
        # day indices start_day .. end_day inclusive
        counts = samples_per_day[start_day : end_day + 1]
        days   = torch.arange(start_day, end_day + 1, device=device, dtype=torch.long)
        return torch.repeat_interleave(days, torch.tensor(counts, device=device))

    day_id_tr  = make_day_ids(0,          cut_train)
    day_id_val = make_day_ids(cut_train+1, cut_val)
    day_id_te  = make_day_ids(cut_val+1,  D-1)


    # Return splits + metadata
    return (
        (X_tr, y_tr),
        (X_val, y_val),
        (X_te, y_te, close_te, bid_te, ask_te),
        samples_per_day,
        day_id_tr, day_id_val, day_id_te
    )


#########################################################################################################


class DayWindowDataset(Dataset):
    """
    Dataset of calendar‐day windows. Each __getitem__(i) returns:
      - x_day: (1, W_i, look_back, F)
      - y_day: (1, W_i)
      - weekday: int
    Or, for test days:
      - x_day, y_day, raw_close, raw_bid, raw_ask, weekday
    """
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        day_id: torch.Tensor,
        weekday: torch.Tensor,
        raw_close: Optional[torch.Tensor] = None,
        raw_bid:   Optional[torch.Tensor] = None,
        raw_ask:   Optional[torch.Tensor] = None
    ):
        self.X    = X
        self.y    = y
        self.day_id = day_id
        self.weekday = weekday
        self.raw_close = raw_close
        self.raw_bid   = raw_bid
        self.raw_ask   = raw_ask
        # compute counts per day and build boundaries
        counts = torch.bincount(day_id)
        boundaries = torch.cat([
            torch.tensor([0], dtype=torch.long),
            torch.cumsum(counts, dim=0)
        ])
        self.start = boundaries[:-1]
        self.end   = boundaries[1:]
        self.has_raw = raw_close is not None

    def __len__(self):
        return len(self.start)

    def __getitem__(self, idx: int):
        s = self.start[idx].item()
        e = self.end[idx].item()
        x_day = self.X[s:e].unsqueeze(0)   # (1, W_i, look_back, F)
        y_day = self.y[s:e].unsqueeze(0)   # (1, W_i)
        wd    = int(self.weekday[s].item())
        if self.has_raw:
            rc = self.raw_close[s:e]
            rb = self.raw_bid[s:e]
            ra = self.raw_ask[s:e]
            return x_day, y_day, rc, rb, ra, wd
        return x_day, y_day, wd

        
#########################################################################################################

def pad_collate(
    batch: List[
        Union[
            Tuple[torch.Tensor, torch.Tensor, int],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
        ]
    ]
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    Pads a batch of per-day tensors to the maximum window length within the batch.
    
    Supports two element types:
      1) (x_day, y_day, weekday)
      2) (x_day, y_day, raw_close, raw_bid, raw_ask, weekday)
    
    Returns either:
      - (batch_x, batch_y, batch_wd)
      - (batch_x, batch_y, batch_rc, batch_rb, batch_ra, batch_wd)
    
    All outputs live on CPU. Shapes:
      batch_x: (B, W_max, look_back, F)
      batch_y: (B, W_max)
      batch_wd: (B,)
      batch_rc/rb/ra: (B, W_max) if present
    """
    
    # 1) Unzip batch into lists
    xs, ys, *rest = zip(*batch)
    has_raw = len(rest) > 1   # raw_close/bid/ask present if rest holds >1 list

    # 2) Strip the leading “1” dim: now x_i:(W_i,L,F), y_i:(W_i,)
    xs = [x.squeeze(0) for x in xs]
    ys = [y.squeeze(0) for y in ys]

    # 3) Vectorized pad for X: flatten last two dims → pad → reshape back
    B = len(xs)
    look_back, F = xs[0].shape[1], xs[0].shape[2]
    flat_xs   = [x.view(x.size(0), -1) for x in xs]                # (W_i, L*F)
    padded_fx = pad_sequence(flat_xs, batch_first=True)            # (B, W_max, L*F)
    batch_x   = padded_fx.view(B, padded_fx.size(1), look_back, F) # (B,W_max,L,F)

    # 4) Vectorized pad for Y: (B, W_max)
    batch_y = pad_sequence(ys, batch_first=True)

    # 5) Weekdays: always last element of each tuple
    batch_wd = torch.tensor([elem[-1] for elem in batch], dtype=torch.int64)

    # 6) If raw features exist, pad them too
    if has_raw:
        # rest is [(rc, rb, ra, wd), …]; drop wd and unzip
        raw_triplets = [t[:3] for t in rest]
        rc_list, rb_list, ra_list = zip(*raw_triplets)
        batch_rc = pad_sequence(rc_list, batch_first=True)
        batch_rb = pad_sequence(rb_list, batch_first=True)
        batch_ra = pad_sequence(ra_list, batch_first=True)
        return batch_x, batch_y, batch_rc, batch_rb, batch_ra, batch_wd

    return batch_x, batch_y, batch_wd


###################


def split_to_day_datasets(
    X_tr:         torch.Tensor,
    y_tr:         torch.Tensor,
    day_id_tr:    torch.Tensor,
    X_val:        torch.Tensor,
    y_val:        torch.Tensor,
    day_id_val:   torch.Tensor,
    X_te:         torch.Tensor,
    y_te:         torch.Tensor,
    day_id_te:    torch.Tensor,
    raw_close_te: torch.Tensor,
    raw_bid_te:   torch.Tensor,
    raw_ask_te:   torch.Tensor,
    *,
    df:           pd.DataFrame,   # full-minute DataFrame for weekday lookup
    train_batch:  int = 32,        # days per training batch
    train_workers:int = 0,        # number of background workers
    train_prefetch_factor:int = 1,# number of batches pulled by the worker ahead of time
    device = torch.device("cpu")
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build three DataLoaders over DayWindowDataset:
      - train_loader (batch_size=train_batch days, padded)
      - val_loader   (1 day per batch)
      - test_loader  (1 day per batch, includes raw prices)
    """

    print("▶️ Entered split_to_day_datasets")

    # 1) Build weekday‐code tensors for each split
    print("1) building weekday arrays")
    all_wd = df.index.dayofweek.to_numpy(np.int64)
    n_tr   = X_tr.size(0)
    n_val  = X_val.size(0)
    wd_tr  = torch.from_numpy(all_wd[:n_tr])
    wd_val = torch.from_numpy(all_wd[n_tr : n_tr + n_val])
    wd_te  = torch.from_numpy(
        all_wd[n_tr + n_val : n_tr + n_val + X_te.size(0)]
    )
    print(f"   Weekdays counts → tr={len(wd_tr)}, val={len(wd_val)}, te={len(wd_te)}")

    # 2) Move all splits to CPU for fast indexing in the Dataset
    print("2) moving all splits to CPU")
    X_tr, y_tr, day_id_tr = X_tr.cpu(), y_tr.cpu(), day_id_tr.cpu()
    X_val, y_val, day_id_val = X_val.cpu(), y_val.cpu(), day_id_val.cpu()
    X_te, y_te, day_id_te = X_te.cpu(), y_te.cpu(), day_id_te.cpu()
    rc_te, rb_te, ra_te   = raw_close_te.cpu(), raw_bid_te.cpu(), raw_ask_te.cpu()
    print("   CPU casts done")

    # 3) Zero-base the val/test day IDs so day indices start at 0
    #    This shrinks each DayWindowDataset down to exactly the number of days
    print("3) zero-bas­ing day_id for val & test")
    val_min  = int(day_id_val.min().item())
    test_min = int(day_id_te.min().item())
    day_id_val = day_id_val - val_min
    day_id_te  = day_id_te  - test_min
    print(f"   val_day_id ∈ [0..{int(day_id_val.max().item())}], total days={day_id_val.max().item()+1}")
    print(f"   te_day_id  ∈ [0..{int(day_id_te.max().item())}], total days={day_id_te.max().item()+1}")

    # 4) Instantiate your DayWindowDatasets exactly as before
    print("4) instantiating DayWindowDatasets")
    ds_tr = DayWindowDataset(X_tr, y_tr, day_id_tr, wd_tr)
    print("   ds_tr days:", len(ds_tr))
    ds_val = DayWindowDataset(X_val, y_val, day_id_val, wd_val)
    print("   ds_val days:", len(ds_val))
    ds_te = DayWindowDataset(
        X_te, y_te, day_id_te, wd_te,
        raw_close=rc_te,
        raw_bid=rb_te,
        raw_ask=ra_te
    )
    print("   ds_te days:", len(ds_te))

    # 5) Build DataLoaders with our pad_collate:
    print("5) building DataLoaders")

    # guard flags around persistent_workers and prefetch_factor
    use_persistent = train_workers > 0
    prefetch_factor = train_prefetch_factor if use_persistent else None

    train_loader = DataLoader(
        ds_tr,
        batch_size=train_batch,
        shuffle=False,
        drop_last=False, ######################
        collate_fn=pad_collate,

        # these three lines were already here,
        # but now only turn on persistent if workers>0
        num_workers       = train_workers,
        pin_memory        = True,
        persistent_workers= use_persistent,
        prefetch_factor   = prefetch_factor
    )
    print("   train_loader ready")

    # For val/test we can also pin_memory to speed up host→GPU
    val_loader = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print("   val_loader ready")

    test_loader = DataLoader(
        ds_te,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print("   test_loader ready")

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
    Two‐stage, stateful sequence model with:
      • One short‐term (daily) LSTM
      • One long‐term (weekly) LSTM
      • Window‐level self‐attention over each day's LSTM output
      • Variational Dropout + LayerNorm after every major block
      • Automatic resets of hidden states at day/week boundaries
      • Time‐distributed linear head producing one scalar per time‐step
    """

    def __init__(
        self,
        n_feats: int,
        short_units: int,
        long_units: int,
        dropout_short: float = 0.4,
        dropout_long: float  = 0.5,
        att_heads: int    = 4,
        att_drop: float   = 0.1,
    ):
        super().__init__()
        self.n_feats     = n_feats
        self.short_units = short_units
        self.long_units  = long_units

        # 1) Daily LSTM (stateful across windows)
        self.short_lstm = nn.LSTM(
            input_size  = n_feats,
            hidden_size = short_units,
            batch_first = True,
            num_layers  = 1,
            dropout     = 0.0
        )

        # 2) Self‐attention on each day's LSTM outputs
        self.attn = nn.MultiheadAttention(
            embed_dim   = short_units,
            num_heads   = att_heads,
            dropout     = att_drop,
            batch_first = True
        )

        # 3) Dropout + LayerNorm on attended daily features
        self.do_short = nn.Dropout(dropout_short)
        self.ln_short = nn.LayerNorm(short_units)

        # 4) Weekly LSTM (stateful across days)
        self.long_lstm = nn.LSTM(
            input_size  = short_units,
            hidden_size = long_units,
            batch_first = True,
            num_layers  = 1,
            dropout     = 0.0
        )
        self.do_long = nn.Dropout(dropout_long)
        self.ln_long = nn.LayerNorm(long_units)

        # 5) Time‐distributed linear head → one scalar per time‐step
        self.pred = nn.Linear(long_units, 1)

        # 6) Hidden/cell buffers, lazily initialized on first forward
        self.h_short = None
        self.c_short = None
        self.h_long  = None
        self.c_long  = None

    def _init_states(self, B: int, device: torch.device):
        """
        Allocate zero hidden+cell states for both LSTMs.
        Shapes:
          (1, B, short_units) and (1, B, long_units)
        """
        self.h_short = torch.zeros(1, B, self.short_units, device=device)
        self.c_short = torch.zeros(1, B, self.short_units, device=device)
        self.h_long  = torch.zeros(1, B, self.long_units,  device=device)
        self.c_long  = torch.zeros(1, B, self.long_units,  device=device)

    def reset_short(self):
        """
        Zero out daily‐LSTM state at each new window/day.
        """
        if self.h_short is not None:
            B, dev = self.h_short.size(1), self.h_short.device
            self._init_states(B, dev)

    def reset_long(self):
        """
        Zero out weekly‐LSTM state at each new week,
        preserving the daily‐LSTM state across the reset.
        """
        if self.h_long is not None:
            B, dev = self.h_long.size(1), self.h_long.device
            hs, cs = self.h_short, self.c_short
            self._init_states(B, dev)
            # restore only the daily‐LSTM state
            self.h_short, self.c_short = hs.to(dev), cs.to(dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1) daily LSTM → out_short_raw, h_s, c_s
        2) detach_() daily state in‐place
        3) self‐attention + residual → out_short
        4) dropout + layernorm on out_short
        5) weekly LSTM → out_long, h_l, c_l
        6) detach_() weekly state in‐place
        7) dropout + layernorm on out_long
        8) linear head → (B, S, 1)
        """
        # — reshape if extra dims
        if x.dim() > 3:
            *lead, S, F = x.shape
            x = x.view(-1, S, F)

        # — ensure last dim is features
        if x.dim() == 3 and x.size(-1) != self.n_feats:
            x = x.transpose(1, 2).contiguous()

        B, S, _ = x.size()
        dev      = x.device

        # Lazy init or batch‐size change
        if self.h_short is None or self.h_short.size(1) != B:
            self._init_states(B, dev)

        # 1) daily LSTM
        out_short_raw, (h_s, c_s) = self.short_lstm(
            x, (self.h_short, self.c_short)
        )
        # 2) in‐place detach hidden/cell → no new allocation
        h_s.detach_();  c_s.detach_()
        self.h_short, self.c_short = h_s, c_s

        # 3) self‐attention over the day's windows
        attn_out, _ = self.attn(out_short_raw,
                                out_short_raw,
                                out_short_raw)
        out_short = out_short_raw + attn_out

        # 4) dropout + layernorm
        out_short = self.do_short(out_short)
        out_short = self.ln_short(out_short)

        # 5) weekly LSTM
        out_long, (h_l, c_l) = self.long_lstm(
            out_short, (self.h_long, self.c_long)
        )
        # 6) in‐place detach weekly state
        h_l.detach_(); c_l.detach_()
        self.h_long, self.c_long = h_l, c_l

        # 7) dropout + layernorm
        out_long = self.do_long(out_long)
        out_long = self.ln_long(out_long)

        # 8) linear head → raw output
        raw = self.pred(out_long)        # shape: (B, S, 1)

        return raw



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



def train_step(
    model:     nn.Module,
    x_day:     torch.Tensor,    # (W, look_back, F), already on device
    y_day:     torch.Tensor,    # (W,), already on device
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    clipnorm:  float,
) -> float:
    """
    Single‐day step:
      1) zero grads
      2) mixed-precision forward + loss
      3) backward + unscale + clip → step → update scaler
      4) return loss.item()
    """
    optimizer.zero_grad(set_to_none=True)
    model.train()

    # 2) mixed precision forward
    device_type = x_day.device.type
    with autocast(device_type=device_type):
        out  = model(x_day)        # (W, seq_len, 1)
        last = out[:, -1, 0]       # extract final time‐step
        loss = mse_loss(last, y_day, reduction='mean')

    # 3) backward + clip + step
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

    
#########################################################################################################


# def custom_stateful_training_loop(
#     model:         torch.nn.Module,
#     optimizer:     torch.optim.Optimizer,
#     cosine_sched:  CosineAnnealingWarmRestarts,
#     plateau_sched: ReduceLROnPlateau,
#     scaler:        GradScaler,
#     train_loader:  torch.utils.data.DataLoader,
#     val_loader:    torch.utils.data.DataLoader,
#     *,
#     max_epochs:          int,
#     early_stop_patience: int,
#     baseline_val_rmse:   float,
#     clipnorm:            float,
#     device:              torch.device = torch.device("cpu"),
# ) -> float:
#     """
#     • Device and performance setup
#       – Sends the model to CPU/GPU
#       – Enables cuDNN benchmarking for dynamic kernel tuning
#     • Pre-binding of heavy objects
#       – Caches MSE-loss function
#       – Compiles regex once for checkpoint filename matching
#     • Epoch-wise Python-side garbage collection
#       – Frees unused objects at the start of each epoch to limit memory churn
#     • Dual-memory LSTM state management
#       – Resets short‐term LSTM state on every new slice
#       – Detects week boundaries via weekday index to reset long‐term memory
#     • Mixed precision training
#       – Wraps forward/backward in autocast
#       – Scales and unscales gradients, clips by norm, steps optimizer & scaler
#     • Cosine Annealing with Warm Restarts
#       – Steps the scheduler by fractional epoch to smoothly vary LR
#       – Logs every restart event with the new LR
#     • Plateau-based LR reduction
#       – After warmup epochs, steps ReduceLROnPlateau on validation RMSE
#       – Re-anchors the cosine scheduler whenever the plateau cuts LR
#     • Live, real-time plotting
#       – Streams train/val RMSE by epoch via an IPython or inline draw widget
#       – Retains full history for a final summary plot
#     • Global RMSE calculation
#       – Accumulates squared errors and counts across all validation windows
#       – Reports one RMSE that weights each window equally
#     • Early stopping
#       – Halts training if validation RMSE fails to improve for patience epochs
#     """

#     # 1) Send model to GPU/CPU and enable cudnn autotuner for speed
#     model.to(device)
#     torch.backends.cudnn.benchmark = True

#     # 2) Pre-bind often-used objects outside the epoch loop
#     loss_fn      = torch.nn.functional.mse_loss
#     save_pattern = re.compile(rf"{re.escape(params.ticker)}_(\d+\.\d+)\.pth")

#     best_val_rmse = float('inf')   # Best validation RMSE seen so far
#     best_state    = None           # To store model weights for best RMSE
#     patience_ctr  = 0              # Counter for early stopping
#     live_plot     = plots.LiveRMSEPlot()  # Real-time training/validation plot

#     # 3) Main epoch loop
#     for epoch in range(1, max_epochs + 1):
#         # a) Single garbage collection at epoch start to free Python objects
#         gc.collect()

#         model.train()                # Set dropout, batchnorm, etc. to train mode
#         model.h_short = model.h_long = None  # Reset hidden/cell to trigger lazy-init
#         train_losses = []            # Accumulate per-batch MSEs

#         prev_T_cur = cosine_sched.T_cur  # Track cosine restarts

#         # b) Iterate over training data batches
#         pbar = tqdm(
#             enumerate(train_loader),
#             total=len(train_loader),
#             desc=f"Epoch {epoch}",
#             unit="bundle",
#         )
#         for batch_idx, (xb_days, yb_days, wd_days) in pbar:

#             # Move data to device; non_blocking if DataLoader uses pinned memory
#             xb_days = xb_days.to(device, non_blocking=True)
#             yb_days = yb_days.to(device, non_blocking=True)
#             wd_days = wd_days.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)
#             prev_wd = None

#             # c) Inner loop over each “day” slice in the bundle
#             for di in range(xb_days.size(0)):
#                 wd = int(wd_days[di].item())

#                 # Reset short‐term LSTM state at each new slice
#                 model.reset_short()
#                 # If weekday decreases, it's a new week → reset long‐term state
#                 if prev_wd is not None and wd < prev_wd:
#                     model.reset_long()
#                 prev_wd = wd

#                 # Mixed precision forward/backward
#                 with autocast(device_type=device.type):
#                     out  = model(xb_days[di])         # (look_back, 1) prediction tensor
#                     last = out[..., -1, 0]             # take only the final time‐step
#                     loss = loss_fn(last, yb_days[di], reduction='mean')

#                 # Scale, backpropagate, and collect loss
#                 scaler.scale(loss).backward()
#                 train_losses.append(loss.item())

#                 # In‐place detach hidden/cell tensors to avoid new allocations
#                 model.h_short.detach_()
#                 model.c_short.detach_()
#                 model.h_long.detach_()
#                 model.c_long.detach_()

#             # d) After stepping through all days in this batch:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
#             scaler.step(optimizer)
#             scaler.update()

#             # e) Update cosine‐annealing schedule by fractional epoch
#             frac_epoch = epoch - 1 + batch_idx / len(train_loader)
#             cosine_sched.step(frac_epoch)

#             # Log if a cosine “restart” happened
#             if cosine_sched.T_cur < prev_T_cur:
#                 lr = optimizer.param_groups[0]['lr']
#                 print(f"  [Cosine restart] epoch {epoch}, batch {batch_idx}, lr={lr:.2e}")
#             prev_T_cur = cosine_sched.T_cur

#             # f) Refresh progress bar metrics
#             rmse = math.sqrt(sum(train_losses) / len(train_losses))
#             lr   = optimizer.param_groups[0]['lr']
#             pbar.set_postfix(train_rmse=rmse, lr=lr, refresh=False)

#         pbar.close()

#         # 4) Validation phase (no gradients)
#         model.eval()
#         model.h_short = model.h_long = None
#         total_sq_error = 0.0
#         total_windows  = 0
#         prev_wd        = None

#         with torch.no_grad():
#             for xb_day, yb_day, wd in val_loader:
#                 wd = int(wd.item())
#                 x  = xb_day[0].to(device, non_blocking=True)
#                 y  = yb_day.view(-1).to(device, non_blocking=True)

#                 model.reset_short()
#                 if prev_wd is not None and wd < prev_wd:
#                     model.reset_long()
#                 prev_wd = wd

#                 out      = model(x)
#                 last     = out[..., -1, 0]
#                 sq_error = (last - y).pow(2).sum().item()
#                 total_sq_error += sq_error
#                 total_windows  += y.numel()

#         val_rmse = math.sqrt(total_sq_error / total_windows)

#         # 5) Live plot update and logging
#         live_plot.update(rmse, val_rmse)
#         print(f"Epoch {epoch:03d} • train={rmse:.4f} • val={val_rmse:.4f}"
#               f" • lr={optimizer.param_groups[0]['lr']:.2e}")

#         # 6) Plateau scheduler after warm-up
#         pre_lr = optimizer.param_groups[0]['lr']
#         if epoch > params.hparams['PLAT_EPOCHS_WARMUP']:
#             plateau_sched.step(val_rmse)
#         post_lr = optimizer.param_groups[0]['lr']
#         if post_lr < pre_lr:
#             print(f"  [Plateau cut] LR {pre_lr:.1e} → {post_lr:.1e} at epoch {epoch}")
#             cosine_sched.base_lrs = [post_lr for _ in cosine_sched.base_lrs]
#             cosine_sched.last_epoch = epoch - 1

#         # 7) Early-stopping and checkpointing
#         if val_rmse >= best_val_rmse:
#             patience_ctr += 1
#             if patience_ctr >= early_stop_patience:
#                 print("Early stopping at epoch", epoch)
#                 break
#         else:
#             best_val_rmse = val_rmse
#             best_state    = copy.deepcopy(model.state_dict())
#             patience_ctr  = 0

#             # reload best weights before saving new checkpoint
#             model.load_state_dict(best_state)
            
#     # Final conditional save (once per run)
#     rmses = [
#         float(m.group(1))
#         for f in params.save_path.glob(f"{params.ticker}_*.pth")
#         for m in (save_pattern.match(f.name),)
#         if m
#     ]
#     # if no prior checkpoint or improvement, save
#     if not rmses or best_val_rmse < max(rmses):
#         buf = io.BytesIO()
#         live_plot.fig.savefig(buf, format="png")
#         buf.seek(0)
#         plot_png = buf.read()

#         ckpt = params.save_path / f"{params.ticker}_{best_val_rmse:.4f}.pth"
#         torch.save({
#             "model_obj":        model,
#             "model_state_dict": best_state,
#             "hparams":          params.hparams,
#             "train_plot_png":   plot_png,
#         }, ckpt)
#         print(f"Saved final best model and training plot: {ckpt.name}")

#     return best_val_rmse



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
    • Device and performance setup
      – Sends the model to CPU/GPU
      – Enables cuDNN benchmarking for dynamic kernel tuning
    • Pre-binding of heavy objects
      – Caches MSE-loss function
      – Compiles regex once for checkpoint filename matching
      – Initializes torchmetrics RMSE for train/val on the correct device
    • Epoch-wise Python-side garbage collection
      – Frees unused objects at the start of each epoch to limit memory churn
    • Dual-memory LSTM state management
      – Resets short‐term LSTM state on every new slice
      – Detects week boundaries via weekday index to reset long‐term memory
    • Mixed precision training
      – Wraps forward/backward in autocast
      – Scales and unscales gradients, clips by norm, steps optimizer & scaler
    • Cosine Annealing with Warm Restarts
      – Steps the scheduler by fractional epoch to smoothly vary LR
      – Logs every restart event with the new LR
    • Plateau-based LR reduction
      – After warmup epochs, steps ReduceLROnPlateau on validation RMSE
      – Re-anchors the cosine scheduler whenever the plateau cuts LR
    • Live, real-time plotting
      – Streams train/val RMSE by epoch via an IPython or inline draw widget
      – Retains full history for a final summary plot
    • Uniform RMSE via torchmetrics
      – Uses torchmetrics.MeanSquaredError(squared=False) for both train & val
      – Resets at epoch start, updates per window, and computes one RMSE
    • Early stopping
      – Halts training if validation RMSE fails to improve for patience epochs
    """

    # 1) Send model to GPU/CPU and enable cudnn autotuner for speed
    model.to(device)
    torch.backends.cudnn.benchmark = True

    # 2) Pre-bind often-used objects outside the epoch loop
    loss_fn      = torch.nn.functional.mse_loss
    save_pattern = re.compile(rf"{re.escape(params.ticker)}_(\d+\.\d+)\.pth")

    # initialize torchmetrics RMSE accumulators
    train_metric = torchmetrics.MeanSquaredError(squared=False).to(device)
    val_metric   = torchmetrics.MeanSquaredError(squared=False).to(device)

    best_val_rmse = float('inf')   # Best validation RMSE seen so far
    best_state    = None           # To store model weights for best RMSE
    patience_ctr  = 0              # Counter for early stopping
    live_plot     = plots.LiveRMSEPlot()  # Real-time training/validation plot

    # 3) Main epoch loop
    for epoch in range(1, max_epochs + 1):
        # a) Single garbage collection at epoch start to free Python objects
        gc.collect()

        # reset train-phase metric and switch to train mode
        train_metric.reset()
        model.train()
        model.h_short = model.h_long = None

        prev_T_cur = cosine_sched.T_cur  # Track cosine restarts

        # b) Iterate over training data batches
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            unit="bundle",
        )
        for batch_idx, (xb_days, yb_days, wd_days) in pbar:

            xb_days = xb_days.to(device, non_blocking=True)
            yb_days = yb_days.to(device, non_blocking=True)
            wd_days = wd_days.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prev_wd = None

            # c) Inner loop over each “day” slice in the bundle
            for di in range(xb_days.size(0)):
                wd = int(wd_days[di].item())

                model.reset_short()
                if prev_wd is not None and wd < prev_wd:
                    model.reset_long()
                prev_wd = wd

                with autocast(device_type=device.type):
                    out  = model(xb_days[di])         # (look_back, 1)
                    last = out[..., -1, 0]            # final time‐step
                    loss = loss_fn(last, yb_days[di], reduction='mean')

                scaler.scale(loss).backward()

                # update train RMSE uniformly
                train_metric.update(last, yb_days[di])

                # detach LSTM states
                model.h_short.detach_()
                model.c_short.detach_()
                model.h_long.detach_()
                model.c_long.detach_()

            # d) Optimizer step with gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()

            # e) Cosine‐annealing by fractional epoch
            frac_epoch = epoch - 1 + batch_idx / len(train_loader)
            cosine_sched.step(frac_epoch)
            if cosine_sched.T_cur < prev_T_cur:
                lr = optimizer.param_groups[0]['lr']
                print(f"  [Cosine restart] epoch {epoch}, batch {batch_idx}, lr={lr:.2e}")
            prev_T_cur = cosine_sched.T_cur

            # f) Refresh progress bar metrics using torchmetrics RMSE
            rmse = train_metric.compute().cpu().item()
            lr   = optimizer.param_groups[0]['lr']
            pbar.set_postfix(train_rmse=rmse, lr=lr, refresh=False)

        pbar.close()

        # 4) Validation phase (no gradients), uniform RMSE
        model.eval()
        model.h_short = model.h_long = None
        prev_wd = None
        val_metric.reset()

        with torch.no_grad():
            for xb_day, yb_day, wd in val_loader:
                wd = int(wd.item())
                x  = xb_day[0].to(device, non_blocking=True)
                y  = yb_day.view(-1).to(device, non_blocking=True)

                model.reset_short()
                if prev_wd is not None and wd < prev_wd:
                    model.reset_long()
                prev_wd = wd

                out  = model(x)
                last = out[..., -1, 0]

                # update validation RMSE
                val_metric.update(last, y)

        val_rmse = val_metric.compute().cpu().item()

        # 5) Live plot update and logging
        live_plot.update(rmse, val_rmse)
        print(f"Epoch {epoch:03d} • train={rmse:.4f} • val={val_rmse:.4f}"
              f" • lr={optimizer.param_groups[0]['lr']:.2e}")

        # 6) Plateau scheduler after warm-up
        pre_lr = optimizer.param_groups[0]['lr']
        if epoch > params.hparams['PLAT_EPOCHS_WARMUP']:
            plateau_sched.step(val_rmse)
        post_lr = optimizer.param_groups[0]['lr']
        if post_lr < pre_lr:
            print(f"  [Plateau cut] LR {pre_lr:.1e} → {post_lr:.1e} at epoch {epoch}")
            cosine_sched.base_lrs = [post_lr for _ in cosine_sched.base_lrs]
            cosine_sched.last_epoch = epoch - 1

        # 7) Early-stopping and checkpointing
        if val_rmse >= best_val_rmse:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print("Early stopping at epoch", epoch)
                break
        else:
            best_val_rmse = val_rmse
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
            model.load_state_dict(best_state)

    # Final conditional save (unchanged) …
    rmses = [
        float(m.group(1))
        for f in params.save_path.glob(f"{params.ticker}_*.pth")
        for m in (save_pattern.match(f.name),)
        if m
    ]
    if not rmses or best_val_rmse < max(rmses):
        buf = io.BytesIO()
        live_plot.fig.savefig(buf, format="png")
        buf.seek(0)
        plot_png = buf.read()

        ckpt = params.save_path / f"{params.ticker}_{best_val_rmse:.4f}.pth"
        torch.save({
            "model_obj":        model,
            "model_state_dict": best_state,
            "hparams":          params.hparams,
            "train_plot_png":   plot_png,
        }, ckpt)
        print(f"Saved final best model and training plot: {ckpt.name}")

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
     Split a time-indexed DataFrame into train/validation/test by row order.
     Encode hour, day_of_week, month as single cyclic PCA features.
     Scale feature groups WITHOUT LEAKAGE:
       - Price–volume features: per-day robust scaling
       - Ratio/indicator features: global StandardScaler on train only
       - Binary flags: passthrough {0,1}
       - Cyclical time features (hour, day_of_week, month): PCA→1D passthrough
     Return a single DataFrame, preserving original feature names and the label_col.
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

    # 2) Generate sin & cos for each cyclic feature
    for sub in (df_tr, df_v, df_te):
        # hour
        h = sub["hour"]
        sub["hour_sin"] = np.sin(2 * np.pi * h / 24)
        sub["hour_cos"] = np.cos(2 * np.pi * h / 24)
        # day_of_week
        d = sub["day_of_week"]
        sub["day_of_week_sin"] = np.sin(2 * np.pi * d / 7)
        sub["day_of_week_cos"] = np.cos(2 * np.pi * d / 7)
        # month
        m = sub["month"]
        sub["month_sin"] = np.sin(2 * np.pi * m / 12)
        sub["month_cos"] = np.cos(2 * np.pi * m / 12)

    # 3) Define feature groups
    price_feats = [
        "open", "high", "low", "close", "volume",
        "atr_14", "ma_5", "ma_20", "ma_diff",
        "macd_12_26", "macd_signal_9", "obv"
    ]
    ratio_feats = [
        "r_1", "r_5", "r_15",
        "vol_15", "volume_spike", "vwap_dev",
        "rsi_14", "bb_width_20", "stoch_k_14", "stoch_d_3"
    ]
    binary_feats = ["in_trading"]
    cyclic_feats  = ["hour", "day_of_week", "month"]

    price_feats  = [f for f in price_feats  if f in features_cols]
    ratio_feats  = [f for f in ratio_feats  if f in features_cols]
    binary_feats = [f for f in binary_feats if f in features_cols]
    # note: cyclic_feats will be re‐added post-PCA

    # 4) Fit global scaler on ratio/indicator features
    ratio_scaler = StandardScaler()
    if ratio_feats:
        ratio_scaler.fit(df_tr[ratio_feats])

    # 5) Per-day robust scaler for price features
    def scale_price_per_day(sub: pd.DataFrame, desc: str) -> pd.DataFrame:
        out = sub.copy()
        days = out.index.normalize().unique()
        for day in tqdm(days, desc=f"Scaling price per day ({desc})", unit="day"):
            mask  = out.index.normalize() == day
            block = out.loc[mask, price_feats]
            med   = block.median(axis=0)
            iqr   = block.quantile(0.75, axis=0) - block.quantile(0.25, axis=0)
            iqr[iqr == 0] = 1e-6
            out.loc[mask, price_feats] = (block - med) / iqr
        return out

    # 6) Transform splits (price + ratio)
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

    # 7) Fit PCA on train‐only sin/cos pairs
    pca_hour = PCA(n_components=1).fit(df_tr_s[["hour_sin","hour_cos"]])
    pca_dow  = PCA(n_components=1).fit(df_tr_s[["day_of_week_sin","day_of_week_cos"]])
    pca_mo   = PCA(n_components=1).fit(df_tr_s[["month_sin","month_cos"]])

    # 8) Apply PCA → 1D, round, drop sin/cos, reattach original names
    def apply_cyclic_pca(sub: pd.DataFrame) -> pd.DataFrame:
        # hour
        h_pc1 = pca_hour.transform(sub[["hour_sin","hour_cos"]])[:,0]
        sub["hour"] = np.round(h_pc1, 3)
        # day_of_week
        d_pc1 = pca_dow.transform(sub[["day_of_week_sin","day_of_week_cos"]])[:,0]
        sub["day_of_week"] = np.round(d_pc1, 3)
        # month
        m_pc1 = pca_mo.transform(sub[["month_sin","month_cos"]])[:,0]
        sub["month"] = np.round(m_pc1, 3)

        # drop the intermediate columns
        sub.drop([
            "hour_sin","hour_cos",
            "day_of_week_sin","day_of_week_cos",
            "month_sin","month_cos"
        ], axis=1, inplace=True)

        return sub

    df_tr_s  = apply_cyclic_pca(df_tr_s)
    df_val_s = apply_cyclic_pca(df_val_s)
    df_te_s  = apply_cyclic_pca(df_te_s)

    # 9) Reattach label and recombine
    for subset in (df_tr_s, df_val_s, df_te_s):
        subset[label_col] = df.loc[subset.index, label_col]

    df_final = pd.concat([df_tr_s, df_val_s, df_te_s]).sort_index()

    # 10) Select final columns
    final_cols = (
        price_feats
        + ratio_feats
        + binary_feats
        + cyclic_feats      # now holds the PCA‐1D values
        + ["bid", "ask"]
        + [label_col]
    )
    return df_final[final_cols]

