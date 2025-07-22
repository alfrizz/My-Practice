from libs import plots, params

from typing import Sequence, List, Tuple, Optional, Union
import gc 
import os
import shutil
import atexit
import copy

import pandas as pd
import numpy  as np
import math

import datetime as dt
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as Funct
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.optim import AdamW

from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm


#########################################################################################################

def build_lstm_tensors(
    df: pd.DataFrame,
    *,
    look_back: int,
    features_cols: Sequence[str],
    label_col: str,
    regular_start: dt.time,
    tmpdir: str = "/tmp/lstm_memmap",           # directory for .npy files
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build disk-backed memmaps for LSTM windows, then wrap in PyTorch Tensors.

    1) Ensure tmpdir exists
    2) First pass: count total valid windows (N) to size memmaps
    3) Allocate .npy memmaps: X_mm, y_mm, c_mm, b_mm, a_mm
    4) Second pass: for each day
       a) load raw arrays (features, labels, prices) as float32
       b) standardize features in-place (float32)
       c) build sliding windows via np.lib.stride_tricks
       d) align targets (drop last window)
       e) filter by regular_start (RTH mask)
       f) write slices into memmaps at offset idx
    5) Wrap memmaps in torch.from_numpy and move to device
    6) Return five tensors: X, y, raw_close, raw_bid, raw_ask

    Auto‐cleanup: the contents of tmpdir persist after this function returns.
    We register an atexit handler so that when Python exits, tmpdir is
    automatically removed. 
    """

    # ── Create / verify tmpdir ────────────────────────────────────────────
    os.makedirs(tmpdir, exist_ok=True)

    # ── Register cleanup at program exit (only once) ──────────────────────
    if not hasattr(build_lstm_tensors, "_cleanup_registered"):
        atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
        build_lstm_tensors._cleanup_registered = True

    # ── 1) Count total number of valid windows across all days ────────────
    N = 0                     # total windows after RTH filtering
    F = len(features_cols)    # number of features per time step
    for _, day in df.groupby(df.index.normalize(), sort=False):
        T = len(day)  # minutes in this day
        # windows before filter: T – look_back
        if T <= look_back:
            continue
        # boolean mask: which end‐times ≥ regular_start
        mask = np.array(day.index.time[look_back:]) >= regular_start
        if mask.any():
            N += (T - look_back)  # count all windows; we'll only write masked ones later

    # ── 2) Allocate on‐disk memmaps for X, y, and raw prices ─────────────
    #   X_mm: shape (N, look_back, F)
    X_mm = np.lib.format.open_memmap(
        os.path.join(tmpdir, "X.npy"), mode="w+",
        dtype=np.float32, shape=(N, look_back, F)
    )
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

        # 3b) Per‐day standardization in float32
        mu        = feats_np.mean(axis=0, keepdims=True)
        sd        = feats_np.std (axis=0, keepdims=True) + 1e-6
        feats_np  = (feats_np - mu) / sd

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

        # 3e) RTH filter: end‐time ≥ regular_start
        mask = np.array(day.index.time[look_back:]) >= regular_start
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
    # This is zero‐copy: binary pages remain on disk until accessed.
    X         = torch.from_numpy(X_mm) .to(device, non_blocking=True)
    y         = torch.from_numpy(y_mm) .to(device, non_blocking=True)
    raw_close = torch.from_numpy(c_mm) .to(device, non_blocking=True)
    raw_bid   = torch.from_numpy(b_mm) .to(device, non_blocking=True)
    raw_ask   = torch.from_numpy(a_mm) .to(device, non_blocking=True)

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
    regular_start: dt.time,
    train_prop: float,
    val_prop: float,
    train_batch: int,
    device = torch.device("cpu")
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    List[int],
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Split the big (N, look_back, F) dataset into train/val/test by calendar day
    using index‐range slicing (no giant boolean masks). All splits happen on `device`.

    Returns exactly:
      (X_tr, y_tr),
      (X_val, y_val),
      (X_te, y_te, close_te, bid_te, ask_te),
      samples_per_day,
      day_id_tr, day_id_val, day_id_te
    """

    # 0) pick a real device & move the full dataset there
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
        mask_rth  = np.array([t >= regular_start for t in end_times])
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

    # ────────────────────────────────────────────────────────────────────────────
    # SIDE‐EFFECT: dump the raw‐bar test‐period DF exactly as before
    test_days = [all_days[i] for i in range(D) if i > cut_val]
    df_test   = df.loc[df.index.normalize().isin(test_days)]
    df_test.to_csv(f"dfs training/{params.ticker}_test_DF.csv", index=True)

    # 8) Return splits + metadata
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
    batch: List[Union[
        Tuple[torch.Tensor, torch.Tensor, int],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ]]
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    Pads each-day examples to the maximum window-length in `batch`.

    Supports elements:
      (x_day, y_day, weekday)
    or
      (x_day, y_day, raw_close, raw_bid, raw_ask, weekday)

    Returns either:
      (batch_x, batch_y, batch_wd)
    or:
      (batch_x, batch_y, batch_rc, batch_rb, batch_ra, batch_wd)
    All on CPU.
    """
    # find maximum number of windows W_max
    W_max = max(elem[0].shape[1] for elem in batch)

    xs, ys, rcs, rbs, ras, wds = [], [], [], [], [], []
    for elem in batch:
        x_day, y_day, *rest = elem
        weekday = rest[-1]
        has_raw = len(rest) == 4

        W_i = x_day.shape[1]
        pad_amt = W_max - W_i

        # pad x_day: (1, W_i, look_back, F) → (1, W_max, look_back, F)
        x_p = Funct.pad(
            x_day,
            pad=(0, 0, 0, 0, 0, pad_amt, 0, 0),
            mode='constant', value=0.0
        )
        xs.append(x_p)

        # pad y_day: (1, W_i) → (1, W_max)
        y_p = Funct.pad(y_day, pad=(0, pad_amt), mode='constant', value=0.0)
        ys.append(y_p)

        if has_raw:
            rc, rb, ra = rest[:-1]
            # raw vectors: (W_i,) → (1, W_i) → pad → (1, W_max)
            rc_p = Funct.pad(rc.unsqueeze(0), pad=(0, pad_amt), mode='constant', value=0.0)
            rb_p = Funct.pad(rb.unsqueeze(0), pad=(0, pad_amt), mode='constant', value=0.0)
            ra_p = Funct.pad(ra.unsqueeze(0), pad=(0, pad_amt), mode='constant', value=0.0)
            rcs.append(rc_p)
            rbs.append(rb_p)
            ras.append(ra_p)

        wds.append(weekday)

    batch_x  = torch.cat(xs, dim=0)          # (B, W_max, look_back, F)
    batch_y  = torch.cat(ys, dim=0)          # (B, W_max)
    batch_wd = torch.tensor(wds, dtype=torch.int64)

    if rcs:
        batch_rc = torch.cat(rcs, dim=0)     # (B, W_max)
        batch_rb = torch.cat(rbs, dim=0)
        batch_ra = torch.cat(ras, dim=0)
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
    df:           pd.DataFrame,  # full-minute DataFrame for weekday lookup
    train_batch:  int = 8,       # days per training batch
    train_workers:int = 0,       # no workers default
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

    # 2) Move all splits to CPU for indexing
    print("2) moving all splits to CPU")
    X_tr, y_tr, day_id_tr = X_tr.cpu(), y_tr.cpu(), day_id_tr.cpu()
    X_val, y_val, day_id_val = X_val.cpu(), y_val.cpu(), day_id_val.cpu()
    X_te, y_te, day_id_te = X_te.cpu(), y_te.cpu(), day_id_te.cpu()
    rc_te, rb_te, ra_te   = raw_close_te.cpu(), raw_bid_te.cpu(), raw_ask_te.cpu()
    print("   CPU casts done")

    # 3) Zero-base the val/test day IDs so each split’s day indices start at 0
    #    This makes len(ds_val)==410, len(ds_te)==422 instead of thousands.
    print("3) zero-bas­ing day_id for val & test")
    val_min = int(day_id_val.min().item())
    test_min= int(day_id_te.min().item())
    day_id_val = (day_id_val - val_min)
    day_id_te  = (day_id_te  - test_min)
    print(f"   val_day_id ∈ [0..{int(day_id_val.max().item())}], total days={day_id_val.max().item()+1}")
    print(f"   te_day_id  ∈ [0..{int(day_id_te .max().item())}], total days={day_id_te .max().item()+1}")

    # 4) Instantiate your DayWindowDataset exactly as before
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
    train_loader = DataLoader(
        ds_tr,
        batch_size=train_batch,
        shuffle=False,
        drop_last=True,
        collate_fn=pad_collate,
        num_workers=train_workers, # how many background processes are preparing data
        pin_memory=True, # ets the GPU DMA engine pull data off page-locked buffers without blocking the CPU (faster)
        persistent_workers=False, # so they stay alive across epochs (faster)
        prefetch_factor=None # how many batches per worker to pre-load into the DataLoader queue
    )
    print("   train_loader ready")

    val_loader = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print("   val_loader ready")

    test_loader = DataLoader(
        ds_te,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
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
    Two-stage, stateful sequence model with:
      • One short-term (daily) LSTM
      • One long-term (weekly) LSTM
      • Window-level self-attention over each day's LSTM output
      • Variational Dropout + LayerNorm after every major block
      • Automatic resets of hidden states at day/week boundaries
      • Time-distributed linear head producing one scalar per time-step

    Note on attention state:
      – The MultiheadAttention module is stateless: it recomputes
        attention weights on each forward pass over that day's window.
      – Only the LSTM hidden states carry memory across windows.
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
        """
        Args:
          n_feats       – number of input features per time-step
          short_units   – hidden size of the daily LSTM
          long_units    – hidden size of the weekly LSTM
          dropout_short – dropout rate after attention on daily features
          dropout_long  – dropout rate after weekly LSTM outputs
          att_heads     – number of heads in self-attention
          att_drop      – dropout inside the attention module
        """
        super().__init__()
        self.n_feats     = n_feats
        self.short_units = short_units
        self.long_units  = long_units

        # 1) Daily LSTM (single layer, carries hidden state across batches)
        self.short_lstm = nn.LSTM(
            input_size  = n_feats,
            hidden_size = short_units,
            batch_first = True,
            num_layers  = 1,       # one layer only
            dropout     = 0.0      # no built-in inter-layer dropout
        )

        # 2) Self-attention over the day's short-LSTM outputs
        self.attn = nn.MultiheadAttention(
            embed_dim   = short_units,
            num_heads   = att_heads,
            dropout     = att_drop,
            batch_first = True     # expects (B, S, C) format
        )

        # 3) Variational dropout + layer norm on the attended short features
        self.do_short = nn.Dropout(dropout_short)
        self.ln_short = nn.LayerNorm(short_units)

        # 4) Weekly LSTM (single layer, carries hidden state across days)
        self.long_lstm = nn.LSTM(
            input_size  = short_units,
            hidden_size = long_units,
            batch_first = True,
            num_layers  = 1,
            dropout     = 0.0
        )
        self.do_long = nn.Dropout(dropout_long)
        self.ln_long = nn.LayerNorm(long_units)

        # 5) Time-distributed linear head: project each time-step to a scalar
        self.pred = nn.Linear(long_units, 1)

        # 6) Buffers for hidden & cell states; inited lazily on first forward
        self.h_short = None
        self.c_short = None
        self.h_long  = None
        self.c_long  = None

    def _init_states(self, B: int, device: torch.device):
        """
        Create zero initial hidden & cell states for both LSTMs.
        Shapes:
          h_short, c_short – (1, B, short_units)
          h_long,  c_long  – (1, B, long_units)
        """
        self.h_short = torch.zeros(1, B, self.short_units, device=device)
        self.c_short = torch.zeros(1, B, self.short_units, device=device)
        self.h_long  = torch.zeros(1, B, self.long_units,  device=device)
        self.c_long  = torch.zeros(1, B, self.long_units,  device=device)

    def reset_short(self):
        """
        Zero out daily LSTM states at each new day.
        """
        if self.h_short is not None:
            B, dev = self.h_short.size(1), self.h_short.device
            # Re-init both short and long to keep shapes in sync
            # (long state preserved by reset_long if you wish)
            self._init_states(B, dev)

    def reset_long(self):
        """
        Zero out weekly LSTM states at each new week,
        preserving the daily LSTM state across the reset.
        """
        if self.h_long is not None:
            B, dev = self.h_long.size(1), self.h_long.device
            # stash daily states
            hs, cs = self.h_short, self.c_short
            # re-init both
            self._init_states(B, dev)
            # restore daily only
            self.h_short, self.c_short = hs.to(dev), cs.to(dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying:
          1) daily LSTM → out_short_raw
          2) self-attention over the day window
          3) dropout + layernorm → out_short
          4) weekly LSTM → out_long
          5) dropout + layernorm → out_long
          6) linear head → (B, S, 1) predictions

        Input:
          x: (B, S, F) or extra dims → will be reshaped to (B, S, F)
        Returns:
          (B, S, 1) per-timestep scalar outputs
        """
        # Collapse extra dims so x is (B, S, F)
        if x.dim() > 3:
            *lead, S, F = x.shape
            x = x.view(-1, S, F)

        # Ensure feature-last ordering
        if x.dim() == 3 and x.size(-1) != self.n_feats:
            x = x.transpose(1, 2).contiguous()

        B, S, _ = x.size()
        dev      = x.device

        # Lazy init states on first forward or batch-size change
        if self.h_short is None or self.h_short.size(1) != B:
            self._init_states(B, dev)

        # 1) Daily LSTM pass
        out_short_raw, (h_s, c_s) = self.short_lstm(x, (self.h_short, self.c_short))
        # detach so no backprop through time across days
        self.h_short, self.c_short = h_s.detach(), c_s.detach()

        # 2) Self-attention on the day's outputs
        #    Query = Key = Value = out_short_raw
        attn_out, _ = self.attn(out_short_raw, out_short_raw, out_short_raw)
        # Residual connection
        out_short = out_short_raw + attn_out

        # 3) Dropout + LayerNorm on attended features
        out_short = self.do_short(out_short)
        out_short = self.ln_short(out_short)

        # 4) Weekly LSTM pass on short representations
        out_long, (h_l, c_l) = self.long_lstm(out_short, (self.h_long, self.c_long))
        self.h_long, self.c_long = h_l.detach(), c_l.detach()

        # 5) Dropout + LayerNorm on weekly outputs
        out_long = self.do_long(out_long)
        out_long = self.ln_long(out_long)

        # 6) Time-distributed linear prediction
        return self.pred(out_long)


#########################################################################################################
 
def make_optimizer_and_scheduler(
    model: nn.Module,
    initial_lr: float,       
    weight_decay: float,    
    clipnorm: float
):
    """
    1) AdamW with decoupled weight decay.
    2) ReduceLROnPlateau: reduces LR when val‐RMSE stops improving.
    3) CosineAnnealingWarmRestarts: per‐batch cosine schedule.
    4) GradScaler for mixed precision.
    """
    # AdamW optimizer with L2 regularization
    optimizer = AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay
    )

    # LR ↓ when validation RMSE plateaus
    plateau_sched = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=params.hparams['PLATEAU_FACTOR'],
        patience=params.hparams['PLATEAU_PATIENCE'],
        min_lr=params.hparams['MIN_LR'],
    )

    # Cosine warm‐restarts scheduler (batch-level stepping)
    cosine_sched = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=params.hparams['T_0'], 
        T_mult=params.hparams['T_MULT'], 
        eta_min=params.hparams['ETA_MIN']
    )

    # AMP scaler for fp16 stability
    scaler = GradScaler()

    return optimizer, plateau_sched, cosine_sched, scaler, clipnorm


def train_step(
    model:     nn.Module,
    x_day:     torch.Tensor,    # (W, look_back, F), on device already
    y_day:     torch.Tensor,    # (W,), on device already
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    clipnorm:  float,
) -> float:
    """
    Single‐day step:
      1) zero grads
      2) fp16 forward+loss
      3) backward with scaler → unscale → clip → step → update scaler
      4) return scalar loss
    """
    optimizer.zero_grad(set_to_none=True)
    model.train()

    device_type = x_day.device.type
    # Mixed‐precision forward
    with autocast(device_type=device_type):
        out  = model(x_day)         # → (W, seq_len, 1)
        last = out[:, -1, 0]        # → (W,)
        loss = Funct.mse_loss(last, y_day, reduction='mean')

    # Backward + clip + optimizer step
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)     # bring grads to fp32 for clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

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
    Full training loop:
      • CosineAnnealingWarmRestarts stepping per batch, with restart-print
      • ReduceLROnPlateau stepping per epoch after warmup, with reduction-print
      • Mixed precision + gradient clipping
      • Per-day & per-week LSTM state resets
      • Early stopping + best-model checkpoint
      • When plateau cuts LR, cosine scheduler is reset to continue from new LR
    """
    model.to(device)
    torch.backends.cudnn.benchmark = True

    best_val_rmse = float('inf')
    best_state    = None
    patience_ctr  = 0
    live_plot     = plots.LiveRMSEPlot()

    for epoch in range(1, max_epochs + 1):
        gc.collect()

        # ── TRAIN ─────────────────────────────────────────────────────────
        model.train()
        model.h_short = model.h_long = None
        train_losses = []
        
        # 1) Capture the scheduler’s cycle counter before batch 0
        prev_T_cur = cosine_sched.T_cur
        
        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch}",
                    unit="bundle")
        for batch_idx, (xb_days, yb_days, wd_days) in pbar:
            # ensure we’re in train‐mode (for cudnn RNN backward)
            model.train()
        
            xb_days, yb_days = xb_days.to(device), yb_days.to(device)
            wd_days          = wd_days.to(device)
        
            optimizer.zero_grad(set_to_none=True)
            prev_wd = None
        
            for di in range(xb_days.size(0)):
                wd = int(wd_days[di].item())
                model.reset_short()
                if prev_wd is not None and wd < prev_wd:
                    model.reset_long()
                prev_wd = wd
        
                # Mixed‐precision forward/backward
                with autocast(device_type=device.type):
                    out  = model(xb_days[di])
                    last = out[..., -1, 0]
                    loss = Funct.mse_loss(last, yb_days[di], reduction='mean')
                scaler.scale(loss).backward()
                train_losses.append(loss.item())
        
                # Detach hidden states
                if isinstance(model.h_short, tuple):
                    model.h_short = tuple(h.detach() for h in model.h_short)
                if isinstance(model.h_long, tuple):
                    model.h_long  = tuple(h.detach() for h in model.h_long)
                del out, last, loss
        
            # Unscale → clip → step → update scaler
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()
        
            # 2) Cosine‐scheduler step (fractional epoch)
            frac_epoch = epoch - 1 + batch_idx / len(train_loader)
            cosine_sched.step(frac_epoch)
        
            # 3) True restart detection via T_cur wrap
            new_T_cur = cosine_sched.T_cur
            if new_T_cur < prev_T_cur:
                lr = optimizer.param_groups[0]['lr']
                print(f"  [Cosine restart] at epoch {epoch}, batch {batch_idx}, lr={lr:.2e}")
            prev_T_cur = new_T_cur
        
            # Logging
            rmse = math.sqrt(sum(train_losses) / len(train_losses))
            lr   = optimizer.param_groups[0]['lr']
            pbar.set_postfix(train_rmse=rmse, lr=lr, refresh=False)
            pbar.update(0)
            gc.collect()
        
        pbar.close()


        # ── VALIDATE ───────────────────────────────────────────────────────
        model.eval()
        model.h_short = model.h_long = None
        val_losses = []
        prev_wd    = None
        with torch.no_grad():
            for xb_day, yb_day, wd in val_loader:
                wd = int(wd.item())
                x  = xb_day[0].to(device)
                y  = yb_day.view(-1).to(device)

                model.reset_short()
                if prev_wd is not None and wd < prev_wd:
                    model.reset_long()
                prev_wd = wd

                out  = model(x)
                last = out[..., -1, 0]
                val_losses.append(Funct.mse_loss(last, y, reduction='mean').item())
                del xb_day, yb_day, x, y, out, last

        val_rmse = math.sqrt(sum(val_losses) / len(val_losses))

        # Live plot & print
        live_plot.update(rmse, val_rmse)
        print(f"Epoch {epoch:03d} • train={rmse:.4f} • val={val_rmse:.4f}"
              f" • lr={optimizer.param_groups[0]['lr']:.2e}")

        # ── ReduceLROnPlateau (after warmup) + reduction print ──────────
        pre_lr = optimizer.param_groups[0]['lr']
        if epoch > params.hparams['PLAT_EPOCHS_WARMUP']:
            plateau_sched.step(val_rmse)
        post_lr = optimizer.param_groups[0]['lr']
        if post_lr < pre_lr:
            print(f"  [Plateau cut] LR {pre_lr:.1e} → {post_lr:.1e}"
                  f" at epoch {epoch}")
            # — update cosine scheduler to continue from new LR —
            cosine_sched.base_lrs = [post_lr for _ in cosine_sched.base_lrs]
            cosine_sched.last_epoch = epoch - 1

        # ── Early stopping ────────────────────────────────────────────────
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print("Early stopping at epoch", epoch)
                break

    # ── Save best model weights ──────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        
    ckpt_file = params.save_path / f"{params.ticker}_{best_val_rmse:.4f}.pth"

    # Save both full model and state_dict+hparams, if you like:
    torch.save({
        "model_obj":         model,               
        "model_state_dict":  model.state_dict(),
        "hparams":           params.hparams
    }, ckpt_file)
    
    print(f"Saved full model + hparams to {ckpt_file}")

    return best_val_rmse