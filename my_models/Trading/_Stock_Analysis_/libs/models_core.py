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

from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy.lib.format import header_data_from_array_1_0, write_array_header_1_0
from threading import Lock

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

from numba import njit
from tqdm.auto import tqdm


#########################################################################################################


def build_tensors(
    df: pd.DataFrame,
    look_back    = None,
    sess_start   = None,
    *,
    tmpdir:      str                  = None,
    device:      torch.device         = torch.device("cpu"),
    in_memory:   bool                 = True
) -> tuple[
    torch.Tensor,  # X         shape=(N, look_back, F)
    torch.Tensor,  # y_sig     shape=(N,)
    torch.Tensor,  # y_ret     shape=(N,)
    torch.Tensor,  # raw_close shape=(N,)
    np.ndarray     # end_times shape=(N,) dtype=datetime64[ns]
]:
    """
    Build sliding‐window tensors for an LSTM trading model.

    Functionality:
      1) Copy input DataFrame and select feature columns.
      2) First pass over calendar days:
         a) extract feature, signal and close arrays;
         b) compute log‐returns;
         c) compute window‐end timestamps and boolean mask ≥ sess_start;
         d) accumulate count of valid windows and store each day's payload.
      3) Allocate data buffers: either in‐RAM numpy arrays or on‐disk memmaps.
      4) Second pass (parallel per day):
         a) build sliding windows via numpy stride_tricks;
         b) write masked windows into the shared buffers;
         c) update one tick per day in a tqdm bar.
      5) If using memmaps, flush to disk and call os.sync().
      6) Wrap final buffers as PyTorch tensors on the target device.
      7) Cleanup Python and CUDA caches.

    Returns:
      X          Tensor of shape (N, look_back, F)
      y_sig      Tensor of shape (N,)
      y_ret      Tensor of shape (N,)
      raw_close  Tensor of shape (N,)
      end_times  numpy array of shape (N,), dtype datetime64[ns]
    """
    # 1) Prepare DataFrame & feature list
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in (params.label_col, "close_raw")]
    print("Inside build_tensors, features:", feature_cols)
    F = len(feature_cols)

    # Normalize session start to seconds‐since‐midnight
    sess_time = sess_start.time() if hasattr(sess_start, "time") else sess_start
    cutoff_sec = sess_time.hour * 3600 + sess_time.minute * 60

    # 2) First pass: group by day, build payloads and count total windows
    day_groups = df.groupby(df.index.normalize(), sort=False)
    payloads   = []
    N_total    = 0

    for _, day_df in tqdm(day_groups, desc="Preparing days", leave=False):
        day_df = day_df.sort_index()
        T = len(day_df)
        if T <= look_back:
            continue

        # a) Extract raw NumPy arrays
        feats_np = day_df[feature_cols].to_numpy(np.float32)      # (T, F)
        sig_np   = day_df[params.label_col].to_numpy(np.float32)  # (T,)
        close_np = day_df["close_raw"].to_numpy(np.float32)       # (T,)

        # b) Compute log‐returns
        ret_full       = np.empty_like(close_np, np.float32)
        ret_full[0]    = 0.0
        ret_full[1:]   = np.log(close_np[1:] / close_np[:-1])

        # c) Compute window-end timestamps and boolean mask
        ends_np = day_df.index.to_numpy()[look_back:]             # (T-look_back,)
        secs    = (ends_np - ends_np.astype("datetime64[D]")) \
                    / np.timedelta64(1, "s")
        mask    = secs >= cutoff_sec                              # (T-look_back,)

        m = int(mask.sum())
        if m == 0:
            continue

        # d) Slice next-bar arrays and record payload
        sig_end   = sig_np[look_back:]
        ret_end   = ret_full[look_back:]
        close_end = close_np[look_back:]
        payloads.append(
            (feats_np, sig_end, ret_end, close_end, ends_np, mask, N_total)
        )
        N_total += m

    # 3) Allocate buffers: try RAM first, fallback to memmap on OOM
    use_memmap = not in_memory
    try:
        if not use_memmap:
            X_buf = np.empty((N_total, look_back, F),   np.float32)
            y_buf = np.empty((N_total,),                np.float32)
            r_buf = np.empty((N_total,),                np.float32)
            c_buf = np.empty((N_total,),                np.float32)
            t_buf = np.empty((N_total,),      "datetime64[ns]")
        else:
            raise MemoryError
    except MemoryError:
        use_memmap = True
        if tmpdir is None:
            tmpdir = tempfile.mkdtemp(prefix="lstm_memmap_")
        else:
            os.makedirs(tmpdir, exist_ok=True)

        def _open_memmap(name, shape, dtype):
            return np.lib.format.open_memmap(
                os.path.join(tmpdir, name), mode="w+",
                dtype=dtype, shape=shape
            )

        X_buf = _open_memmap("X.npy",     (N_total, look_back, F), np.float32)
        y_buf = _open_memmap("y_sig.npy", (N_total,),             np.float32)
        r_buf = _open_memmap("y_ret.npy", (N_total,),             np.float32)
        c_buf = _open_memmap("close.npy", (N_total,),             np.float32)
        t_buf = _open_memmap("t.npy",     (N_total,),     "datetime64[ns]")

    # 4) Second pass: build sliding windows and write into buffers
    pbar = tqdm(total=len(payloads), desc="Writing days")

    def _write_np(payload):
        feats_np, sig_end, ret_end, close_end, ends_np, mask, offset = payload

        # Build all sliding windows, then drop the last bar
        wins = np.lib.stride_tricks.sliding_window_view(
                   feats_np, window_shape=(look_back, F)
               ).reshape(feats_np.shape[0] - look_back + 1, look_back, F)[:-1]

        m = mask.sum()
        X_buf[offset : offset + m] = wins[mask]
        y_buf[offset : offset + m] = sig_end[mask]
        r_buf[offset : offset + m] = ret_end[mask]
        c_buf[offset : offset + m] = close_end[mask]
        t_buf[offset : offset + m] = ends_np[mask]

        pbar.update(1)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as exe:
        exe.map(_write_np, payloads)
    pbar.close()

    # 5) Flush memmaps to disk if used
    if use_memmap:
        for arr in (X_buf, y_buf, r_buf, c_buf, t_buf):
            arr.flush()
        os.sync()

    # 6) Wrap buffers as PyTorch tensors on `device`
    X         = torch.from_numpy(X_buf).to(device, non_blocking=True)
    y_sig     = torch.from_numpy(y_buf).to(device, non_blocking=True)
    y_ret     = torch.from_numpy(r_buf).to(device, non_blocking=True)
    raw_close = torch.from_numpy(c_buf).to(device, non_blocking=True)
    end_times = t_buf.copy()

    # 7) Cleanup Python and CUDA caches
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
    device       = torch.device("cpu")
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],          
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],          
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  
    list,                                                    
    torch.Tensor, torch.Tensor, torch.Tensor                  
]:
    """
    Split flattened windows into train/val/test sets by calendar day,
    returning raw_close for every split so downstream __getitem__
    always has a tensor.

    Functionality:
      1) Normalize end_times to per-day bins and count windows per calendar day.
      2) Compute how many days go to train/val/test (train rounded to full batches).
      3) Build cumulative window‐count array, then derive slice indices i_tr, i_val.
      4) Slice X, y_sig, y_ret, raw_close into:
         - train  quadruple (X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr)
         - val    quadruple (X_val, y_sig_val, y_ret_val, raw_close_val)
         - test   quadruple (X_te,  y_sig_te,  y_ret_te,  raw_close_te)
      5) Build per-window day‐id tensors for each split for GPU collation.
      6) Return the three splits, samples_per_day list, and day_id_tr/val/te.
    """
    # 1) Count windows per normalized calendar day
    dates_norm = pd.to_datetime(end_times).normalize().values
    days, counts = np.unique(dates_norm, return_counts=True)
    samples_per_day = counts.tolist()

    # Sanity check total windows matches tensor length
    total = sum(samples_per_day)
    if total != X.size(0):
        raise ValueError(f"Window count mismatch {total} vs {X.size(0)}")

    # 2) Determine day‐level cut points for train/val/test
    D             = len(samples_per_day)
    orig_tr_days  = int(D * train_prop)
    full_batches  = (orig_tr_days + train_batch - 1) // train_batch
    tr_days       = min(D, full_batches * train_batch)
    cut_train     = tr_days - 1
    cut_val       = int(D * (train_prop + val_prop))

    # 3) Build cumulative window counts and compute slice indices
    cumsum = np.concatenate([[0], np.cumsum(counts)])
    i_tr   = int(cumsum[tr_days])
    i_val  = int(cumsum[cut_val + 1])

    # 4) Slice into train/val/test (each gets raw_close slice)
    X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr  = (
        X[:i_tr],    y_sig[:i_tr],   y_ret[:i_tr],   raw_close[:i_tr]
    )
    X_val, y_sig_val, y_ret_val, raw_close_val = (
        X[i_tr:i_val], y_sig[i_tr:i_val], y_ret[i_tr:i_val], raw_close[i_tr:i_val]
    )
    X_te,  y_sig_te,  y_ret_te,  raw_close_te  = (
        X[i_val:],    y_sig[i_val:],   y_ret[i_val:],   raw_close[i_val:]
    )

    # 5) Build per-window day_id tensors for grouping
    def make_day_ids(start_day: int, end_day: int) -> torch.Tensor:
        # repeat each day index by its window count
        day_counts = samples_per_day[start_day : end_day + 1]
        day_idxs   = torch.arange(start_day, end_day + 1, dtype=torch.long)
        return day_idxs.repeat_interleave(torch.tensor(day_counts, dtype=torch.long))

    day_id_tr  = make_day_ids(0,          cut_train)
    day_id_val = make_day_ids(cut_train+1, cut_val)
    day_id_te  = make_day_ids(cut_val+1,  D - 1)

    # 6) Return splits as 4-tuples + metadata
    return (
        (X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr),
        (X_val, y_sig_val, y_ret_val, raw_close_val),
        (X_te,  y_sig_te,  y_ret_te,  raw_close_te),
        samples_per_day,
        day_id_tr, day_id_val, day_id_te
    )


#########################################################################################################


class DayWindowDataset(Dataset):
    """
    Wrap sliding windows into per-day groups for DataLoader.

    Functionality:
      1) Accepts X, y_signal, y_return, raw_close, and end_times (all pre-filtered).
      2) Groups windows by calendar date (numpy datetime64[D]).
      3) Computes start/end indices and weekday for each day.
      4) On __getitem__, returns an 8-tuple:
         - x_day     : Tensor (1, W, look_back, F)
         - y_day     : Tensor (1, W)
         - y_sig_cls : Tensor (1, W) binary labels from y_day > signal_thresh
         - ret_day   : Tensor (1, W) true returns
         - y_ret_ter : LongTensor (1, W) ternary labels from ret_day vs return_thresh
         - rc        : Tensor (W,) raw_close slice for this day
         - wd        : int weekday index
         - end_ts    : numpy.datetime64[ns] of last window’s timestamp
    """
    def __init__(
        self,
        X:               torch.Tensor,   # (N_windows, look_back, F)
        y_signal:        torch.Tensor,   # (N_windows,)
        y_return:        torch.Tensor,   # (N_windows,)
        raw_close:       torch.Tensor,   # (N_windows,)
        end_times:       np.ndarray,     # (N_windows,), datetime64[ns]
        sess_start_time: time,           # unused: already filtered upstream
        signal_thresh:   float,
        return_thresh:   float
    ):
        # Store thresholds
        self.signal_thresh = signal_thresh
        self.return_thresh = return_thresh

        # 1) Store all buffers (raw_close always provided now)
        self.X         = X
        self.y_signal  = y_signal
        self.y_return  = y_return
        self.raw_close = raw_close
        self.end_times = end_times   # numpy.datetime64[ns]

        # 2) Group windows by calendar day
        days64 = end_times.astype("datetime64[D]")       # e.g. 2025-09-25
        days, counts = np.unique(days64, return_counts=True)
        boundaries = np.concatenate(([0], np.cumsum(counts)))

        # 3) Build start/end indices and weekday tensor
        self.start   = torch.tensor(boundaries[:-1], dtype=torch.long)
        self.end     = torch.tensor(boundaries[1:],  dtype=torch.long)
        weekdays     = pd.to_datetime(days).dayofweek
        self.weekday = torch.tensor(weekdays, dtype=torch.long)

    def __len__(self):
        return len(self.start)

    def __getitem__(self, idx: int):
        # Determine slice indices for this day
        s = self.start[idx].item()
        e = self.end[idx].item()

        # 4) Slice out windows and add batch‐dim
        x_day = self.X[s:e].unsqueeze(0)          # (1, W, look_back, F)
        y_day = self.y_signal[s:e].unsqueeze(0)   # (1, W)

        # Binary label per window
        y_sig_cls = (y_day > self.signal_thresh).float()

        # True returns + ternary label
        ret_day   = self.y_return[s:e].unsqueeze(0)
        y_ret_ter = torch.ones_like(ret_day, dtype=torch.long)
        y_ret_ter[ret_day >  self.return_thresh] = 2
        y_ret_ter[ret_day < -self.return_thresh] = 0

        # Extract raw_close slice (length W, no leading dim)
        rc = self.raw_close[s:e]

        # Weekday index and last-window timestamp
        wd     = int(self.weekday[idx].item())
        end_ts = self.end_times[e - 1]  # numpy.datetime64[ns]

        # Return the fixed 8-tuple
        return x_day, y_day, y_sig_cls, ret_day, y_ret_ter, rc, wd, end_ts


######################


def pad_collate(batch):
    """
    Pad variable-length per-day sequences into fixed tensors.

    Batch items (always 8-tuple):
      (x_day, y_day, y_sig_cls, ret_day, y_ret_ter, rc, weekday, end_ts)

    Returns 9 items:
      x_pad    Tensor (B, max_W, look_back, F)
      y_pad    Tensor (B, max_W)
      ysig_pad Tensor (B, max_W)
      ret_pad  Tensor (B, max_W)
      yter_pad LongTensor (B, max_W)
      rc_pad   Tensor (B, max_W)
      wd       LongTensor (B,)
      ts_list  list of end_ts per element
      lengths  list of true window counts per day
    """
    # Unpack fixed 8-tuple structure
    x_list, y_list, ysig_list, ret_list, yter_list, rc_list, wd_list, ts_list = zip(*batch)

    # Remove leading batch dim from each day's tensors
    xs   = [x.squeeze(0) for x in x_list]
    ys   = [y.squeeze(0) for y in y_list]
    ysig = [yc.squeeze(0) for yc in ysig_list]
    rets = [r.squeeze(0) for r in ret_list]
    yter = [t.squeeze(0) for t in yter_list]
    lengths = [seq.size(0) for seq in xs]

    # 1) Pad along the time axis for each field
    x_pad    = pad_sequence(xs,   batch_first=True)
    y_pad    = pad_sequence(ys,   batch_first=True)
    ysig_pad = pad_sequence(ysig, batch_first=True)
    ret_pad  = pad_sequence(rets, batch_first=True)
    yter_pad = pad_sequence(yter, batch_first=True)

    # 2) Pad raw_close similarly
    rc_pad = pad_sequence(rc_list, batch_first=True)

    # 3) Weekday tensor and ts_list remain as-is
    wd_tensor = torch.tensor(wd_list, dtype=torch.long)

    # Return all padded tensors, timestamps, and sequence lengths
    return x_pad, y_pad, ysig_pad, ret_pad, yter_pad, rc_pad, wd_tensor, list(ts_list), lengths


    
###############


def split_to_day_datasets(
    X_tr, y_sig_tr, y_ret_tr, raw_close_tr, end_times_tr,
    X_val, y_sig_val, y_ret_val, raw_close_val, end_times_val,
    X_te,  y_sig_te,  y_ret_te,  raw_close_te,  end_times_te,
    *,
    sess_start_time:       time,
    signal_thresh:         float,
    return_thresh:         float,
    train_batch:           int = 32,
    train_workers:         int = 0,
    train_prefetch_factor: int = 1
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Instantiate DayWindowDataset for train/val/test and wrap into DataLoaders.

    Functionality:
      1) Build three DayWindowDataset objects, each receiving raw_close tensor:
         - train set gets raw_close_tr
         - val   set gets raw_close_val
         - test  set gets raw_close_te
      2) Wrap each dataset in a DataLoader using pad_collate:
         - train: batch_size=train_batch, num_workers=train_workers, prefetch_factor.
         - val & test: batch_size=1, num_workers=0.
      This ensures __getitem__ always sees a real raw_close tensor, never None.
    """
    splits = [
        ("train", X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr,  end_times_tr),
        ("val",   X_val, y_sig_val, y_ret_val, raw_close_val, end_times_val),
        ("test",  X_te,  y_sig_te,  y_ret_te,  raw_close_te,  end_times_te),
    ]

    datasets = {}
    for name, Xd, ys, yr, rc, et in tqdm(splits, desc="Creating DayWindowDatasets", unit="split"):
        datasets[name] = DayWindowDataset(
            X               = Xd,
            y_signal        = ys,
            y_return        = yr,
            raw_close       = rc,   # now always a tensor
            end_times       = et,
            sess_start_time = sess_start_time,
            signal_thresh   = signal_thresh,
            return_thresh   = return_thresh
        )

    # Train loader: padded multi-day batches
    train_loader = DataLoader(
        datasets["train"],
        batch_size           = train_batch,
        shuffle              = False,
        drop_last            = False,
        collate_fn           = pad_collate,
        num_workers          = train_workers,
        pin_memory           = True,
        persistent_workers   = (train_workers > 0),
        prefetch_factor      = (train_prefetch_factor if train_workers > 0 else None),
    )

    # Validation loader: single-day batches
    val_loader = DataLoader(
        datasets["val"],
        batch_size = 1,
        shuffle    = False,
        collate_fn = pad_collate,
        num_workers= 0,
        pin_memory = True,
    )

    # Test loader: single-day batches
    test_loader = DataLoader(
        datasets["test"],
        batch_size = 1,
        shuffle    = False,
        collate_fn = pad_collate,
        num_workers= 0,
        pin_memory = True,
    )

    return train_loader, val_loader, test_loader


#########################################################################################################


def model_core_pipeline(
    df,                          # feature‐enriched DataFrame
    look_back: int,              # how many ticks per window
    sess_start: time,            # session‐start cutoff for windows
    train_prop: float,           # fraction of days → train
    val_prop: float,             # fraction of days → val
    train_batch: int,            # batch size for training
    num_workers: int,            # DataLoader worker count
    prefetch_factor: int,        # DataLoader prefetch_factor
    signal_thresh: float,        # y_signal threshold
    return_thresh: float         # y_return threshold
) -> tuple:
    """
    Build DataLoaders end‐to‐end from raw df.

    Steps & parameters:
      1) build_tensors(df, look_back, sess_start)
      2) chronological_split(..., train_prop, val_prop, train_batch)
      3) carve end_times into train/val/test
      4) split_to_day_datasets(
           X_tr, y_sig_tr, y_ret_tr, raw_close_tr, end_times_tr,
           X_val, y_sig_val, y_ret_val, raw_close_val, end_times_val,
           X_te, y_sig_te, y_ret_te, raw_close_te, end_times_te,
           sess_start_time=sess_start,
           signal_thresh, return_thresh,
           train_batch, num_workers, prefetch_factor
         )
      cleans up all intermediate arrays before returning.
    """
    # 1) slide‐window tensorization
    X, y_sig, y_ret, raw_close, end_times = build_tensors(
        df=df,
        look_back=look_back,
        sess_start=sess_start
    )

    # 2) split into train/val/test by calendar day
    (train_split, val_split, test_split,
     samples_per_day,
     day_id_tr, day_id_val, day_id_te) = chronological_split(
         X, y_sig, y_ret, raw_close,
         end_times=end_times,
         train_prop=train_prop,
         val_prop=val_prop,
         train_batch=train_batch
    )
    X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr  = train_split
    X_val, y_sig_val, y_ret_val, raw_close_val = val_split
    X_te,  y_sig_te,  y_ret_te,  raw_close_te  = test_split

    # 3) carve end_times in the same proportions
    n_tr  = day_id_tr .shape[0]
    n_val = day_id_val.shape[0]
    i_tr, i_val = n_tr, n_tr + n_val

    end_times_tr  = end_times[:i_tr]
    end_times_val = end_times[i_tr:i_val]
    end_times_te  = end_times[i_val:]

    # 4) DataLoader construction
    train_loader, val_loader, test_loader = split_to_day_datasets(
        # train split
        X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr,  end_times_tr,
        # val split
        X_val, y_sig_val, y_ret_val, raw_close_val, end_times_val,
        # test split
        X_te,  y_sig_te,  y_ret_te,  raw_close_te,  end_times_te,

        sess_start_time       = sess_start,
        signal_thresh         = signal_thresh,
        return_thresh         = return_thresh,
        train_batch           = train_batch,
        train_workers         = num_workers,
        train_prefetch_factor = prefetch_factor
    )

    # clean up large intermediates to free memory
    del X, y_sig, y_ret, raw_close, end_times
    del X_tr, y_sig_tr, y_ret_tr, raw_close_tr
    del X_val, y_sig_val, y_ret_val, raw_close_val
    del X_te, y_sig_te, y_ret_te, raw_close_te

    return train_loader, val_loader, test_loader, end_times_tr, end_times_val, end_times_te


#############


def summarize_split(name, loader, times):
    ds      = loader.dataset
    X       = ds.X                      # (N_windows, look_back, n_features)
    L, F    = X.shape[1], X.shape[2]
    Nw      = X.shape[0]                # total windows
    # normalize to days and count windows/day
    daysD, counts = np.unique(times.astype("datetime64[D]"), return_counts=True)
    Nd      = len(daysD)
    dmin, dmax = daysD.min(), daysD.max()

    print(f"--- {name.upper()} ---")
    print(f" calendar days : {Nd:3d}  ({dmin} → {dmax})")
    print(f" windows       : {Nw:4d}  (per-day min={counts.min():3d}, max={counts.max():3d}, mean={counts.mean():.1f})")
    print(f" window shape  : look_back={L}, n_features={F}")
    print(f" dataloader    : batches={len(loader):3d}, batch_size={loader.batch_size}, workers={loader.num_workers}, pin_memory={loader.pin_memory}")
    print()


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


def maybe_save_chkpt(models_dir: Path, model: torch.nn.Module, vl_rmse: float,
                     cur_best: float, tr: dict, vl: dict, live_plot, params):
    """
    If vl_rmse improves cur_best, capture the model state and live_plot.
    Additionally write a folder-best checkpoint (_chp) if this rmse is better than files on disk.
    Returns (updated_best, best_state, best_tr, best_vl, best_existing).
    """
    # expects filenames like TICKER_0.12345_chp.pth or TICKER_0.12345_fin.pth
    save_pat = re.compile(rf"{re.escape(params.ticker)}_(\d+\.\d+)_(?:chp|fin)\.pth")

    models_dir.mkdir(exist_ok=True)
    existing_rmses = [
        float(m.group(1))
        for f in models_dir.glob("*.pth")
        if (m := save_pat.match(f.name))
    ]
    best_existing = min(existing_rmses, default=float("inf"))

    if vl_rmse < cur_best:
        best_state = model.state_dict()
        best_tr, best_vl = tr.copy(), vl.copy()
        updated_best = vl_rmse

        # capture live plot bytes
        buf = io.BytesIO()
        live_plot.fig.savefig(buf, format="png")
        buf.seek(0)
        plot_bytes = buf.read()

        # save folder-best iff strictly better than existing on disk
        if updated_best < best_existing:
            name = f"{params.ticker}_{updated_best:.5f}_chp.pth"
            ckpt = {
                "model_state_dict": best_state,
                "hparams": params.hparams,
                "train_metrics": best_tr,
                "val_metrics": best_vl,
                "train_plot_png": plot_bytes,
            }
            torch.save(ckpt, models_dir / name)
            print(f"🔖 Saved folder-best checkpoint (_chp): {name}")
            best_existing = updated_best

        return updated_best, best_state, best_tr, best_vl, best_existing

    return cur_best, None, {}, {}, best_existing

    
################ 


def save_final_chkpt(models_dir: Path, best_state, best_val_rmse: float,
                     params, best_tr: dict, best_vl: dict, live_plot, suffix: str = "_fin"):
    """Write the final best checkpoint (_fin) using best_state and embed the final plot."""
    buf = io.BytesIO()
    live_plot.fig.savefig(buf, format="png")
    buf.seek(0)
    final_plot = buf.read()

    ckpt = {
        "model_state_dict": best_state,
        "hparams": params.hparams,
        "train_metrics": best_tr,
        "val_metrics": best_vl,
        "train_plot_png": final_plot,
    }
    name = f"{params.ticker}_{best_val_rmse:.5f}{suffix}.pth"
    torch.save(ckpt, Path(params.models_folder) / name)
    print(f"✅ Final best model saved: {name}")
