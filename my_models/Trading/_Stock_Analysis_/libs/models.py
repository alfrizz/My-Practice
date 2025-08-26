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

torch.backends.cuda.matmul.allow_tf32   = True
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
    Build disk-backed memmaps for LSTM training windows, then wrap in PyTorch tensors
    without loading the entire dataset into RAM at once.

    Steps:
      1) Count total number of valid look-back windows across all days (N).
      2) Allocate on-disk .npy memmaps (X_mm, y_mm, c_mm, b_mm, a_mm) sized (N, ...).
      3) Iterate day by day:
         a) extract raw feature, label, and price arrays (float32),
         b) build look-back windows via sliding_window_view,
         c) drop the final window to align with next-step labels,
         d) filter windows by regular_start (RTH mask),
         e) write valid slices into memmaps at offset idx.
      4) Wrap each memmap with torch.from_numpy and move to `device` (zero-copy).
      5) Return five tensors: (X, y, raw_close, raw_bid, raw_ask).
    
    Cleanup:
      - Registers an atexit handler on first call to remove `tmpdir` on interpreter exit.
      - No intermediate full-array copies, so peak RAM usage is minimal.

    Returns:
      X         : torch.Tensor of shape (N, look_back, F)
      y         : torch.Tensor of shape (N,)
      raw_close : torch.Tensor of shape (N,)
      raw_bid   : torch.Tensor of shape (N,)
      raw_ask   : torch.Tensor of shape (N,)
    """
    if not sess_start: # if we want the predictions not to start from sess_start, but from sess_start_pred
        sess_start = params.sess_start_pred_tick
    else:
        sess_start = params.sess_start

    # ── Create / verify tmpdir ────────────────────────────────────────────
    # 0) temp directory
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="lstm_memmap_")
    else:
        os.makedirs(tmpdir, exist_ok=True)
        
    # ── 1) Count total number of valid windows across all days ────────────
    N = 0
    F = len(features_cols)
    day_groups = df.groupby(df.index.normalize(), sort=False)

    for _, day in tqdm(day_groups, desc="Counting valid windows", leave=False):
        T = len(day)
        if T <= look_back:
            continue
        mask = np.array(day.index.time[look_back:]) >= sess_start
        if mask.any():
            N += int(mask.sum())

    # ── 2) Allocate on-disk memmaps for X, y, and raw prices ─────────────
    X_mm = np.lib.format.open_memmap(
        os.path.join(tmpdir, "X.npy"), mode="w+", dtype=np.float32, shape=(N, look_back, F)
    )
    y_mm = np.lib.format.open_memmap(os.path.join(tmpdir, "y.npy"), mode="w+", dtype=np.float32, shape=(N,))
    c_mm = np.lib.format.open_memmap(os.path.join(tmpdir, "c.npy"), mode="w+", dtype=np.float32, shape=(N,))
    b_mm = np.lib.format.open_memmap(os.path.join(tmpdir, "b.npy"), mode="w+", dtype=np.float32, shape=(N,))
    a_mm = np.lib.format.open_memmap(os.path.join(tmpdir, "a.npy"), mode="w+", dtype=np.float32, shape=(N,))

    # ── 3) Fill memmaps day by day ────────────────────────────────────────
    idx = 0
    for _, day in tqdm(day_groups, desc="Writing memmaps", leave=False):
        day = day.sort_index()
        T = len(day)
        if T <= look_back:
            continue

        # 3a) Extract raw arrays
        feats_np  = day[features_cols].to_numpy(dtype=np.float32)
        labels_np = day[label_col].to_numpy(dtype=np.float32)
        close_np  = day["close"].to_numpy(dtype=np.float32)
        bid_np    = day["bid"].to_numpy(dtype=np.float32)
        ask_np    = day["ask"].to_numpy(dtype=np.float32)

        # 3b) Build sliding windows
        windows = np.lib.stride_tricks.sliding_window_view(feats_np, window_shape=(look_back, F))
        windows = windows.reshape(T - look_back + 1, look_back, F)

        # 3c) Align to next-step label
        windows = windows[:-1]
        targets = labels_np[look_back:]
        c_pts   = close_np[look_back:]
        b_pts   = bid_np[look_back:]
        a_pts   = ask_np[look_back:]

        # 3d) RTH filter
        mask = np.array(day.index.time[look_back:]) >= sess_start
        if not mask.any():
            continue

        # 3e) Write valid slices
        m = int(mask.sum())
        X_mm[idx:idx + m] = windows[mask]
        y_mm[idx:idx + m] = targets[mask]
        c_mm[idx:idx + m] = c_pts[mask]
        b_mm[idx:idx + m] = b_pts[mask]
        a_mm[idx:idx + m] = a_pts[mask]
        idx += m

    # ── 4) Wrap memmaps in PyTorch Tensors and move to device ────────────
    X         = torch.from_numpy(X_mm).to(device, non_blocking=True)
    y         = torch.from_numpy(y_mm).to(device, non_blocking=True)
    raw_close = torch.from_numpy(c_mm).to(device, non_blocking=True)
    raw_bid   = torch.from_numpy(b_mm).to(device, non_blocking=True)
    raw_ask   = torch.from_numpy(a_mm).to(device, non_blocking=True)

    # Optional: free any CUDA cache to keep memory tight
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
    sess_start: dt.time,
    device: torch.device = torch.device("cpu")
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, np.ndarray],
    Tuple[torch.Tensor, torch.Tensor, np.ndarray],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray],
    List[int],
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Split a sliding‐window dataset into train/val/test by calendar day, fit
    MaxAbsScaler on train only, and return all metadata.

    Steps:
      1) Move everything to `device`.
      2) For each calendar day in `df`, count how many windows survive
         the `look_back` + `session_start` filter.
      3) Verify total windows == X.size(0).
      4) Allocate days to train/val/test based on `train_prop`/`val_prop`
         (rounding train days up to multiples of `train_batch`).
      5) Slice X, y, raw_* into X_tr/val/te & y_tr/val/te & raw_*_te views.
      6) Build per‐window `day_id_*` via `repeat_interleave`.
      7) Build per‐window `idxs_* = np.arange(n_windows)`, so downstream
         loaders can track original indices.
    """

    # 1) push to device
    X         = X.to(device)
    y         = y.to(device)
    raw_close = raw_close.to(device)
    raw_bid   = raw_bid.to(device)
    raw_ask   = raw_ask.to(device)

    if not sess_start: # if we want the predictions not to start from sess_start, but from sess_start_pred
        sess_start = params.sess_start_pred_tick
    else:
        sess_start = params.sess_start

    # 2) count windows per calendar day
    samples_per_day: List[int] = []
    all_days: List[pd.Timestamp] = []
    for day, day_df in df.groupby(df.index.normalize(), sort=False):
        all_days.append(day)
        # skip the first `look_back` minutes, then enforce session start
        end_times = day_df.index.time[look_back:]
        mask_rth  = np.array([t >= sess_start for t in end_times])
        samples_per_day.append(int(mask_rth.sum()))

    # 3) sanity check
    total_windows = X.size(0)
    if sum(samples_per_day) != total_windows:
        raise ValueError(
            f"Window count mismatch: {sum(samples_per_day)} vs {total_windows}"
        )

    # 4) decide how many days go to train/val/test
    D               = len(samples_per_day)
    train_days_orig = int(D * train_prop)
    batches_needed  = (train_days_orig + train_batch - 1) // train_batch
    train_days      = min(D, batches_needed * train_batch)
    cut_train       = train_days - 1
    cut_val         = int(D * (train_prop + val_prop))  # inclusive last val‐day

    # 5) cumulative sum to slice tensors
    cum = np.concatenate([[0], np.cumsum(samples_per_day)])
    end_train = int(cum[train_days])
    end_val   = int(cum[cut_val + 1])

    X_tr  = X[:end_train]
    y_tr  = y[:end_train]

    X_val = X[end_train:end_val]
    y_val = y[end_train:end_val]

    X_te       = X[end_val:]
    y_te       = y[end_val:]
    close_te   = raw_close[end_val:]
    bid_te     = raw_bid[end_val:]
    ask_te     = raw_ask[end_val:]

    # 6) build day_id tensors
    def make_day_ids(start: int, end: int) -> torch.Tensor:
        counts = samples_per_day[start : end + 1]
        days   = torch.arange(start, end + 1, device=device, dtype=torch.long)
        return torch.repeat_interleave(days, torch.tensor(counts, device=device))

    day_id_tr  = make_day_ids(0,          cut_train)
    day_id_val = make_day_ids(cut_train+1, cut_val)
    day_id_te  = make_day_ids(cut_val+1,  D - 1)

    # 7) build idx arrays so downstream knows each window’s original index
    n_tr, n_val, n_te = X_tr.size(0), X_val.size(0), X_te.size(0)
    idxs_tr  = np.arange(n_tr,  dtype=np.int64)
    idxs_val = np.arange(n_val, dtype=np.int64)
    idxs_te  = np.arange(n_te,  dtype=np.int64)

    return (
        (X_tr,  y_tr,  idxs_tr),
        (X_val, y_val, idxs_val),
        (X_te,  y_te,  close_te, bid_te, ask_te, idxs_te),
        samples_per_day,
        day_id_tr, day_id_val, day_id_te
    )

#########################################################################################################


class DayWindowDataset(Dataset):
    """
    A Dataset where each item is one calendar‐day’s worth of look-back windows.

    On init you supply:
      • X         tensor of shape (N_windows, look_back, F)
      • y         tensor of shape (N_windows,)
      • day_id    tensor of shape (N_windows,) mapping each window → day index
      • weekday   tensor of shape (N_windows,) giving 0=Mon…6=Sun
      • idxs      1D array of length N_windows of the exact
                  window-end timestamps (ns since epoch)
      • raw_close/bid/ask (optional) of shape (N_windows,)

    __getitem__(i) returns for the ith calendar day:
      • x_day      Tensor (1, W_i, look_back, F)
      • y_day      Tensor (1, W_i)
      • y_day_cls  Tensor (1, W_i)  – binary labels thresholded
      • wd         int              – weekday code for that day
      • idxs_day   np.ndarray of length W_i  – timestamps
      • (optionally) rc, rb, ra each of shape (W_i,)
    """
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        day_id: torch.Tensor,
        weekday: torch.Tensor,
        idxs: Union[torch.Tensor, np.ndarray, pd.DatetimeIndex],
        raw_close:  Optional[torch.Tensor] = None,
        raw_bid:    Optional[torch.Tensor] = None,
        raw_ask:    Optional[torch.Tensor] = None,
        threshold:  float = params.best_optuna_params['buy_threshold']
    ):
        # store raw tensors
        self.X       = X
        self.y       = y
        self.day_id  = day_id
        self.weekday = weekday
        self.threshold = threshold

        # normalize idxs → numpy array of ns timestamps
        if isinstance(idxs, torch.Tensor):
            self.idxs = idxs.cpu().numpy()
        else:
            # handles np.ndarray or pd.DatetimeIndex
            self.idxs = np.asarray(idxs, dtype='datetime64[ns]').astype('datetime64[ns]').astype(np.int64)

        # optional raw price series
        self.raw_close = raw_close
        self.raw_bid   = raw_bid
        self.raw_ask   = raw_ask
        self.has_raw   = raw_close is not None

        # build pointers day-by-day
        counts = torch.bincount(self.day_id)
        boundaries = torch.cat([
            torch.tensor([0], dtype=torch.long),
            torch.cumsum(counts, dim=0)
        ])
        self.start = boundaries[:-1]  # inclusive
        self.end   = boundaries[1:]   # exclusive

    def __len__(self) -> int:
        # one entry per calendar day
        return len(self.start)

    def __getitem__(self, idx: int):
        # locate this day’s block of windows in the flattened arrays
        s = self.start[idx].item()
        e = self.end[idx].item()

        # slice feature + target windows, add batch dim
        x_day     = self.X[s:e].unsqueeze(0)   # shape (1, W_i, look_back, F)
        y_day     = self.y[s:e].unsqueeze(0)   # shape (1, W_i)
        y_day_cls = (y_day > self.threshold).float()
        wd        = int(self.weekday[s].item())

        # exact window-end timestamps for this day
        idxs_day  = self.idxs[s:e]             # np.ndarray length W_i

        if self.has_raw:
            # return raw price slices for test‐time plotting
            rc = self.raw_close[s:e]
            rb = self.raw_bid[s:e]
            ra = self.raw_ask[s:e]
            return x_day, y_day, y_day_cls, rc, rb, ra, wd, idxs_day

        # train/val return
        return x_day, y_day, y_day_cls, wd, idxs_day



#########################################################################################################


def pad_collate(batch):
    """
    Collate and pad a batch of day-windows to a common time length.
    
    Supports both:
      - 5-tuples: (x_day, y_day, y_day_cls, weekday, idxs_day)
      - 8-tuples: (x_day, y_day, y_day_cls, raw_close, raw_bid, raw_ask, weekday, idxs_day)
    
    Pads all x/y sequences along the time axis (dim=1) to the maximum W in `batch`,
    then stacks into tensors. Discards raw-price fields if present.
    
    Returns:
      batch_x     Tensor of shape (B, W_max, look_back, F)
      batch_y     Tensor of shape (B, W_max)
      batch_yc    Tensor of shape (B, W_max)
      batch_wd    Int64Tensor of shape (B,)         — weekday codes
      batch_idxs  List of length B, each a 1D array of original window-lengths
    """
    # 1) find the maximum time-axis length in this batch
    max_w = max(elem[0].size(1) for elem in batch)

    xs, ys, ycs, wds, idxs_list = [], [], [], [], []

    for elem in batch:
        # unpack differently depending on tuple-length
        if len(elem) == 5:
            x_day, y_day, yc_day, wd, idxs_day = elem
        else:
            # ignore raw_close, raw_bid, raw_ask
            x_day, y_day, yc_day, _, _, _, wd, idxs_day = elem

        pad_amt = max_w - x_day.size(1)

        # pad on the right of the time axis (dim=1)
        # x_day shape: (1, W_i, look_back, F)
        x_p  = Funct.pad(x_day,  (0,0, 0,0, 0, pad_amt))
        y_p  = Funct.pad(y_day,  (0, pad_amt))
        yc_p = Funct.pad(yc_day, (0, pad_amt))

        # drop singleton batch-dim and collect
        xs.append(   x_p.squeeze(0))    # → (W_max, look_back, F)
        ys.append(   y_p.squeeze(0))    # → (W_max,)
        ycs.append(  yc_p.squeeze(0))   # → (W_max,)
        wds.append(  wd)                # scalar
        idxs_list.append(idxs_day)      # array of length W_i

    # 2) stack into batch tensors
    batch_x     = torch.stack(xs,   dim=0)  
    batch_y     = torch.stack(ys,   dim=0)  
    batch_y_cls = torch.stack(ycs,  dim=0)  
    batch_wd    = torch.tensor(wds, dtype=torch.int64)

    # 3) return variable-length idxs as Python list
    batch_idxs  = idxs_list

    return batch_x, batch_y, batch_y_cls, batch_wd, batch_idxs

    

###################


def split_to_day_datasets(
    X_tr, y_tr, day_id_tr, idxs_tr,
    X_val, y_val, day_id_val, idxs_val,
    X_te,  y_te,  day_id_te,  idxs_te,
    df: pd.DataFrame,
    *,
    raw_close_te=None,
    raw_bid_te=None,
    raw_ask_te=None,
    train_batch=32,
    num_workers=0,
    prefetch_factor=1
):
    """
    Build DayWindowDatasets and DataLoaders for train/val/test.

    - Extract weekday codes (0=Mon…6=Sun) from df.index.
    - Instantiate DayWindowDataset for each split.
    - Filter out any empty day‐windows (length 0) in val/test.
    - Create train DataLoader with pad_collate & drop_last.
    - Create val/test DataLoaders (batch_size=1) keeping per-day indices.
    """
    # 1) weekday codes
    all_wd = df.index.dayofweek.to_numpy(np.int64)
    n_tr, n_val, n_te = X_tr.size(0), X_val.size(0), X_te.size(0)
    wd_tr  = torch.from_numpy(all_wd[:n_tr])
    wd_val = torch.from_numpy(all_wd[n_tr : n_tr + n_val])
    wd_te  = torch.from_numpy(all_wd[n_tr + n_val : n_tr + n_val + n_te])

    # 2) build raw datasets
    ds_tr = DayWindowDataset(X_tr.cpu(), y_tr.cpu(),
                             day_id_tr.cpu().long(), wd_tr, idxs_tr)
    ds_val = DayWindowDataset(X_val.cpu(), y_val.cpu(),
                              day_id_val.cpu().long(), wd_val, idxs_val)
    ds_te = DayWindowDataset(X_te.cpu(), y_te.cpu(),
                             day_id_te.cpu().long(), wd_te, idxs_te,
                             raw_close=raw_close_te.cpu() if raw_close_te is not None else None,
                             raw_bid=  raw_bid_te.cpu()   if raw_bid_te   is not None else None,
                             raw_ask=  raw_ask_te.cpu()   if raw_ask_te   is not None else None)

    # 3) drop any empty days in val/test
    ds_val = [e for e in ds_val if len(e[-1]) > 0]
    ds_te  = [e for e in ds_te  if len(e[-1]) > 0]

    # 4) train loader
    train_loader = DataLoader(
        ds_tr,
        batch_size=train_batch,
        shuffle=False,
        drop_last=True,
        collate_fn=pad_collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None)
    )

    # 5) val/test loaders: one “day” per batch
    val_loader = DataLoader(
        ds_val, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=pad_collate
    )
    test_loader = DataLoader(
        ds_te, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=pad_collate
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
    CNN-BiLSTM-Attention model with dual memory for stock prediction:
    
      0) 1D convolution capturing local temporal patterns within each window/day
      1) Bidirectional short-term (daily) LSTM
      2) Window-level self-attention over the daily Bi-LSTM output
      3) Variational Dropout + LayerNorm on attended daily features
      4) Bidirectional long-term (weekly) LSTM
      5) Variational Dropout + LayerNorm on weekly features
      6) Two time-distributed linear heads producing:
         • regression output (one scalar per time-step)
         • binary classification logit (one logit per time-step)
      7) Automatic resets of hidden states at day/week boundaries
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

        # 0) Convolutional encoder: 1D conv over time axis
        #    input: (B, S, F) → permute to (B, F, S) → conv → (B, F, S) → back to (B, S, F)
        self.conv = nn.Conv1d(
            in_channels = n_feats,
            out_channels= n_feats,
            kernel_size = 3,
            padding     = 1
        )

        # 1) Short-term Bidirectional LSTM (stateful across windows)
        #    We split short_units evenly per direction
        assert short_units % 2 == 0, "short_units must be divisible by 2"
        self.short_lstm = nn.LSTM(
            input_size   = n_feats,
            hidden_size  = short_units // 2,
            batch_first  = True,
            bidirectional= True,
            num_layers   = 1,
            dropout      = 0.0
        )

        # 2) Self-attention on each day's Bi-LSTM outputs
        self.attn = nn.MultiheadAttention(
            embed_dim   = short_units,
            num_heads   = att_heads,
            dropout     = att_drop,
            batch_first = True
        )

        # 3) Dropout + LayerNorm on attended daily features
        self.do_short = nn.Dropout(dropout_short)
        self.ln_short = nn.LayerNorm(short_units)

        # 4) Long-term Bidirectional LSTM (stateful across days)
        assert long_units % 2 == 0, "long_units must be divisible by 2"
        self.long_lstm = nn.LSTM(
            input_size   = short_units,
            hidden_size  = long_units // 2,
            batch_first  = True,
            bidirectional= True,
            num_layers   = 1,
            dropout      = 0.0
        )
        self.do_long = nn.Dropout(dropout_long)
        self.ln_long = nn.LayerNorm(long_units)

        # 5) Two time-distributed linear heads
        #    • Regression → one real value per time-step
        #    • Classification → one logit per time-step
        self.pred       = nn.Linear(long_units, 1)
        self.cls_head   = nn.Linear(long_units, 1)

        # 6) Hidden/cell buffers, lazily initialized on first forward
        self.h_short = None
        self.c_short = None
        self.h_long  = None
        self.c_long  = None

    def _init_states(self, B: int, device: torch.device):
        """
        Allocate zero hidden+cell states for both Bi-LSTMs.
        Shapes:
          (num_layers*2, B, short_units//2) → outputs (B, S, short_units)
          (num_layers*2, B, long_units//2)  → outputs (B, S, long_units)
        """
        # 2 directions × 1 layer = 2
        self.h_short = torch.zeros(2, B, self.short_units // 2, device=device)
        self.c_short = torch.zeros(2, B, self.short_units // 2, device=device)
        self.h_long  = torch.zeros(2, B, self.long_units  // 2, device=device)
        self.c_long  = torch.zeros(2, B, self.long_units  // 2, device=device)

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        0) Convolution → x_conv
        1) daily Bi-LSTM → out_short_raw, h_s, c_s
        2) detach_() daily state in-place
        3) self-attention + residual → out_short
        4) dropout + layernorm on out_short
        5) weekly Bi-LSTM → out_long, h_l, c_l
        6) detach_() weekly state in-place
        7) dropout + layernorm on out_long
        8) two heads:
           • pred → regression (B, S, 1)
           • cls_head → classification logits (B, S, 1)
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

        # 0) apply 1D convolution
        #    input (B, S, F) → (B, F, S) → conv → (B, F, S) → back to (B, S, F)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = Funct.relu(x_conv)
        x      = x_conv.transpose(1, 2)

        # Lazy init or batch‐size change
        if self.h_short is None or self.h_short.size(1) != B:
            self._init_states(B, dev)

        # 1) daily Bi-LSTM
        out_short_raw, (h_s, c_s) = self.short_lstm(
            x, (self.h_short, self.c_short)
        )
        # 2) detach daily state in-place
        h_s.detach_();  c_s.detach_()
        self.h_short, self.c_short = h_s, c_s

        # 3) self-attention over the day's windows
        attn_out, _ = self.attn(
            out_short_raw,
            out_short_raw,
            out_short_raw
        )
        out_short = out_short_raw + attn_out

        # 4) dropout + layernorm on daily features
        out_short = self.do_short(out_short)
        out_short = self.ln_short(out_short)

        # 5) weekly Bi-LSTM
        out_long, (h_l, c_l) = self.long_lstm(
            out_short, (self.h_long, self.c_long)
        )
        # 6) detach weekly state in-place
        h_l.detach_(); c_l.detach_()
        self.h_long, self.c_long = h_l, c_l

        # 7) dropout + layernorm on weekly features
        out_long = self.do_long(out_long)
        out_long = self.ln_long(out_long)

        # 8) two time-distributed heads
        raw_reg = self.pred(out_long)     # (B, S, 1)
        raw_cls = self.cls_head(out_long) # (B, S, 1)

        return raw_reg, raw_cls


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
    Stateful training + validation loop for a dual‐memory LSTM with two heads.

    • Device & performance setup (cudnn, autocast)  
    • Pre-binds Huber (regression) & BCEWithLogits (classification) losses  
    • Resets short‐term/day & long‐term/week LSTM states at the right boundaries  
    • Mixed‐precision training + gradient clipping + cosine & plateau schedulers  
    • Live RMSE plotting & early stopping based on val RMSE  
    • **Multi-task monitoring**: tracks both regression metrics (RMSE, MAE, R2)  
      and classification metrics (accuracy, precision, recall, F1, AUROC)  
    • Classification head in the loss (α·BCE) helps bias the shared backbone  
      to accentuate true spikes in the continuous regression output.
    """

    # 1) Send model to device & enable fast convs
    model.to(device)
    torch.backends.cudnn.benchmark = True

    # 2) Loss functions & hyper-params
    beta          = params.hparams["HUBER_BETA"]
    huber_loss_fn = nn.SmoothL1Loss(beta=beta).to(device)
    cls_loss_fn   = nn.BCEWithLogitsLoss().to(device)
    alpha         = params.hparams["CLS_LOSS_WEIGHT"]
    save_pattern  = re.compile(rf"{re.escape(params.ticker)}_(\d+\.\d+)\.pth")

    # 3) Metrics (regression + classification)
    train_rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
    train_mae  = torchmetrics.MeanAbsoluteError().to(device)
    train_r2   = torchmetrics.R2Score().to(device)
    train_acc       = torchmetrics.Accuracy(task="binary", threshold=0.5).to(device)
    train_precision = torchmetrics.Precision(task="binary", threshold=0.5).to(device)
    train_recall    = torchmetrics.Recall(task="binary", threshold=0.5).to(device)
    train_f1        = torchmetrics.F1Score(task="binary", threshold=0.5).to(device)
    train_auroc     = torchmetrics.AUROC(task="binary").to(device)

    val_rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
    val_mae  = torchmetrics.MeanAbsoluteError().to(device)
    val_r2   = torchmetrics.R2Score().to(device)
    val_acc       = torchmetrics.Accuracy(task="binary", threshold=0.5).to(device)
    val_precision = torchmetrics.Precision(task="binary", threshold=0.5).to(device)
    val_recall    = torchmetrics.Recall(task="binary", threshold=0.5).to(device)
    val_f1        = torchmetrics.F1Score(task="binary", threshold=0.5).to(device)
    val_auroc     = torchmetrics.AUROC(task="binary").to(device)

    # 4) Early‐stopping & checkpointing state
    best_val_rmse = float('inf')
    best_state    = None
    patience_ctr  = 0
    live_plot     = plots.LiveRMSEPlot()

    # 5) Epoch loop
    for epoch in range(1, max_epochs + 1):
        gc.collect()

        # 5a) Reset train metrics + set train mode + clear hidden states
        for m in (train_rmse, train_mae, train_r2,
                  train_acc, train_precision, train_recall, train_f1, train_auroc):
            m.reset()
        model.train()
        model.h_short = model.h_long = None
        prev_wd = None
        prev_T_cur = cosine_sched.T_cur

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            unit="bundle"
        )

        # 5b) Training over day‐bundles
        for batch_idx, (xb_days, yb_days, yb_cls_days, wd_days) in pbar:
            xb_days = xb_days.to(device,    non_blocking=True)
            yb_days = yb_days.to(device,    non_blocking=True)
            yb_cls_days = yb_cls_days.to(device, non_blocking=True)
            wd_days = wd_days.to(device,    non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prev_inner_wd = None

            for di in range(xb_days.size(0)):
                wd = int(wd_days[di].item())

                model.reset_short()
                if prev_inner_wd is not None and wd < prev_inner_wd:
                    model.reset_long()
                prev_inner_wd = wd

                with autocast(device_type=device.type):
                    pred_reg, pred_cls = model(xb_days[di])
                    last_reg = pred_reg[..., -1, 0]
                    last_cls = pred_cls[..., -1, 0]

                    loss_reg = huber_loss_fn(last_reg, yb_days[di])
                    loss_cls = cls_loss_fn(last_cls, yb_cls_days[di])
                    loss     = loss_reg + alpha * loss_cls

                scaler.scale(loss).backward()

                # detach hidden states
                model.h_short.detach_(); model.c_short.detach_()
                model.h_long .detach_(); model.c_long .detach_()

            # optimizer step, gradient clipping & scaler update
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()

            # cosine‐annealing update & restart log
            frac_epoch = epoch - 1 + batch_idx / len(train_loader)
            cosine_sched.step(frac_epoch)
            if cosine_sched.T_cur < prev_T_cur:
                lr = optimizer.param_groups[0]['lr']
                print(f"  [Cosine restart] epoch {epoch}, batch {batch_idx}, lr={lr:.2e}")
            prev_T_cur = cosine_sched.T_cur

            pbar.set_postfix(lr=optimizer.param_groups[0]['lr'], refresh=False)

        pbar.close()

        # ——————————————————————————
        # 5c) Single‐pass TRAIN eval for final‐model metrics
        model.eval()
        model.h_short = model.h_long = None
        prev_wd = None
        for m in (train_rmse, train_mae, train_r2,
                  train_acc, train_precision, train_recall, train_f1, train_auroc):
            m.reset()

        with torch.no_grad():
            for xb_day, yb_day, yb_cls_day, wd in train_loader:
                wd = int(wd.item())
                x  = xb_day[0].to(device, non_blocking=True)
                y  = yb_day.view(-1).to(device, non_blocking=True)
                yc = yb_cls_day.view(-1).to(device, non_blocking=True)

                model.reset_short()
                if prev_wd is not None and wd < prev_wd:
                    model.reset_long()
                prev_wd = wd

                pred_r, pred_c = model(x)
                last_r = pred_r[..., -1, 0]
                last_c = pred_c[..., -1, 0]

                train_rmse.update(last_r, y)
                train_mae .update(last_r, y)
                train_r2  .update(last_r, y)

                probs = torch.sigmoid(last_c)
                preds = (probs > 0.5).long()
                train_acc       .update(preds, yc.long())
                train_precision .update(preds, yc.long())
                train_recall    .update(preds, yc.long())
                train_f1        .update(preds, yc.long())
                train_auroc     .update(probs, yc.long())

        tr_rmse = train_rmse.compute().cpu().item()
        tr_acc  = train_acc.compute().cpu().item()
        print(f"Epoch {epoch}: train_rmse={tr_rmse:.4f}, train_acc={tr_acc:.4f}")

        # 5d) Validation phase
        model.eval()
        model.h_short = model.h_long = None
        prev_wd = None
        for m in (val_rmse, val_mae, val_r2,
                  val_acc, val_precision, val_recall, val_f1, val_auroc):
            m.reset()

        with torch.no_grad():
            for xb_day, yb_day, yb_cls_day, wd in val_loader:
                wd = int(wd.item())
                x  = xb_day[0].to(device, non_blocking=True)
                y  = yb_day.view(-1).to(device, non_blocking=True)
                yc = yb_cls_day.view(-1).to(device, non_blocking=True)

                model.reset_short()
                if prev_wd is not None and wd < prev_wd:
                    model.reset_long()
                prev_wd = wd

                pred_r, pred_c = model(x)
                last_r = pred_r[..., -1, 0]
                last_c = pred_c[..., -1, 0]

                val_rmse.update(last_r, y)
                val_mae .update(last_r, y)
                val_r2  .update(last_r, y)

                probs = torch.sigmoid(last_c)
                preds = (probs > 0.5).long()
                val_acc       .update(preds, yc.long())
                val_precision .update(preds, yc.long())
                val_recall    .update(preds, yc.long())
                val_f1        .update(preds, yc.long())
                val_auroc     .update(probs, yc.long())

        val_rmse_val = val_rmse.compute().cpu().item()
        val_acc_val  = val_acc.compute().cpu().item()

        live_plot.update(tr_rmse, val_rmse_val)
        print(
            f"Epoch {epoch:03d} • "
            f"train_rmse={tr_rmse:.4f} • train_acc={tr_acc:.4f} • "
            f"val_rmse={val_rmse_val:.4f} • val_acc={val_acc_val:.4f} • "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
    
        # 6) Plateau scheduler & LR re‐anchor
        pre_lr = optimizer.param_groups[0]['lr']
        if epoch > params.hparams['PLAT_EPOCHS_WARMUP']:
            plateau_sched.step(val_rmse_val)
        post_lr = optimizer.param_groups[0]['lr']
        if post_lr < pre_lr:
            print(f"  [Plateau cut] LR {pre_lr:.1e} → {post_lr:.1e} at epoch {epoch}")
            cosine_sched.base_lrs = [post_lr] * len(cosine_sched.base_lrs)
            cosine_sched.last_epoch = epoch - 1
    
        # 7) Early stopping & checkpointing 
        if val_rmse_val >= best_val_rmse:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print("Early stopping at epoch", epoch)
                break
        else:
            best_val_rmse = val_rmse_val
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
            model.load_state_dict(best_state)
    
    # final conditional save
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

