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
import inspect, platform
from time import perf_counter

from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy.lib.format import header_data_from_array_1_0, write_array_header_1_0
from threading import Lock
import warnings

import datetime as dt
from datetime import datetime, time
from pathlib import Path
import threading

import torch
import torch.nn as nn
import torch.nn.functional as Funct
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

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
         - x         : Tensor (1, W, look_back, F)
         - y_sig     : Tensor (1, W)
         - y_cls_bin : Tensor (1, W) binary labels from y_sig > signal_thresh
         - y_ret     : Tensor (1, W) true returns
         - y_ret_ter : LongTensor (1, W) ternary labels from ret vs return_thresh
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
        x     = self.X[s:e].unsqueeze(0)          # (1, W, look_back, F)
        y_sig = self.y_signal[s:e].unsqueeze(0)   # (1, W)

        # Binary label per window
        y_cls_bin = (y_sig > self.signal_thresh).float()

        # True returns + ternary label
        y_ret     = self.y_return[s:e].unsqueeze(0)
        y_ret_ter = torch.ones_like(y_ret, dtype=torch.long)
        y_ret_ter[y_ret >  self.return_thresh] = 2
        y_ret_ter[y_ret < -self.return_thresh] = 0

        # Extract raw_close slice (length W, no leading dim)
        rc = self.raw_close[s:e]

        # Weekday index and last-window timestamp
        wd     = int(self.weekday[idx].item())
        end_ts = self.end_times[e - 1]  # numpy.datetime64[ns]

        # Return the fixed 8-tuple
        return x, y_sig, y_cls_bin, y_ret, y_ret_ter, rc, wd, end_ts


######################


def pad_collate(batch):
    """
    Pad variable-length per-day sequences into fixed tensors.

    Batch items (always 8-tuple):
      (x, y_sig, y_cls_bin, y_ret, y_ret_ter, rc, wd, end_ts)

    Returns 9 items:
      x_pad      Tensor (B, max_W, look_back, F)
      ysig_pad   Tensor (B, max_W)
      ybin_pad   Tensor (B, max_W)
      y_ret_pad  Tensor (B, max_W)
      yter_pad   LongTensor (B, max_W)
      rc_pad     Tensor (B, max_W)
      wd         LongTensor (B,)
      ts_list    list of end_ts per element
      lengths    list of true window counts per day
    """
    # Unpack fixed 8-tuple structure
    x_list, ysig_list, ybin_list, yret_list, yter_list, rc_list, wd_list, ts_list = zip(*batch)

    # Remove leading batch dim from each day's tensors
    xs      = [x.squeeze(0) for x in x_list]
    ysig    = [y.squeeze(0) for y in ysig_list]
    ybin    = [yc.squeeze(0) for yc in ybin_list]
    yrets   = [r.squeeze(0) for r in yret_list]
    yter    = [t.squeeze(0) for t in yter_list]
    lengths = [seq.size(0) for seq in xs]

    # Pad along the time axis for each field
    x_pad    = pad_sequence(xs,   batch_first=True)
    ysig_pad = pad_sequence(ysig,   batch_first=True)
    ybin_pad = pad_sequence(ybin, batch_first=True)
    yret_pad = pad_sequence(yrets, batch_first=True)
    yter_pad = pad_sequence(yter, batch_first=True)
    rc_pad   = pad_sequence(rc_list, batch_first=True)

    # Weekday tensor and ts_list 
    wd_tensor = torch.tensor(wd_list, dtype=torch.long)

    # Return all padded tensors, timestamps, and sequence lengths
    return x_pad, ysig_pad, ybin_pad, yret_pad, yter_pad, rc_pad, wd_tensor, list(ts_list), lengths


    
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
    train_workers: int,            # DataLoader worker count
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
           train_batch, train_workers, prefetch_factor
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
        train_workers         = train_workers,
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
    """
    Summary of loaders content and times
    """
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


def maybe_save_chkpt(
    models_dir: Path,
    model: torch.nn.Module,
    vl_rmse: float,
    cur_best: float,
    tr: dict,
    vl: dict,
    live_plot,
    params
) -> tuple[float, bool, dict, dict, dict | None]:
    """
    Compare `vl_rmse` (current validation RMSE) against the best RMSE so far
    (cur_best) and on-disk checkpoints. If it’s an improvement, capture
    the model’s weights, metrics, and plot for both folder-best and
    in-run checkpointing.

    Returns:
      updated_best_rmse : new best RMSE (float)
      improved          : True if vl_rmse < cur_best
      best_train_metrics: snapshot of train metrics at this best
      best_val_metrics  : snapshot of val   metrics at this best
      best_state_dict   : model.state_dict() if improved, else None

    Notes (logging/audit):
      - This function performs the save side-effect (torch.save) when a new
        folder-best is found. To improve traceability, it emits a single,
        compact audit line into the main log file (using params.log_file) at
        the moment the file is written, containing the validation RMSE and
        the checkpoint filename.
      - Emitting that audit line here ties the on-disk artifact deterministically
        to the log stream for reproducibility and postmortem analysis.
      - All other logging (per-epoch summaries, header) is handled elsewhere.
    """
    # Ensure output folder exists
    models_dir.mkdir(exist_ok=True)

    # 1) Gather on-disk RMSEs to know if we're beating existing files
    pattern = rf"{re.escape(params.ticker)}_(\d+\.\d+)_(?:chp|fin)\.pth"
    save_re = re.compile(pattern)
    existing_rmses = [
        float(m.group(1))
        for f in models_dir.glob("*.pth")
        if (m := save_re.match(f.name))
    ]
    best_on_disk = min(existing_rmses, default=float("inf"))

    # 2) Check for improvement in this run
    if vl_rmse < cur_best:
        improved = True
        updated_best = vl_rmse

        # Capture weights + metric snapshots
        best_state = model.state_dict()
        best_tr    = tr.copy()
        best_vl    = vl.copy()

        # Render the live RMSE plot to bytes
        buf = io.BytesIO()
        live_plot.fig.savefig(buf, format="png")
        buf.seek(0)
        plot_bytes = buf.read()

        # 3) If we also beat any on-disk model, write a folder-best checkpoint
        if updated_best < best_on_disk:
            fname = f"{params.ticker}_{updated_best:.5f}_chp.pth"
            ckpt = {
                "model_state_dict": best_state,
                "hparams":          params.hparams,
                "train_metrics":    best_tr,
                "val_metrics":      best_vl,
                "train_plot_png":   plot_bytes,
            }
            ckpt_path = models_dir / fname
            torch.save(ckpt, ckpt_path)

            # Minimal audit line written at the moment of the save so logs
            # deterministically record which file was written for which val RMSE.
            try:
                _append_log(f"CHKPT SAVED vl={updated_best:.3f} path={ckpt_path}", params.log_file)
            except Exception:
                # best-effort: do not fail the save on logging errors
                pass

            # Keep original user-visible print for immediate feedback
            print(f"🔖 Saved folder-best checkpoint (_chp): {fname}")

        return updated_best, improved, best_tr, best_vl, best_state

    # No improvement
    return cur_best, False, {}, {}, None

    
################ 


def save_final_chkpt(
    models_dir: Path,
    best_state: dict,
    best_val_rmse: float,
    params,
    best_tr: dict,
    best_vl: dict,
    live_plot,
    suffix: str = "_fin"
):
    """
    Write the final overall‐best checkpoint (_fin):

      • Uses the state_dict mapping in `best_state`
      • Embeds hyperparameters, final train/val metrics, and the plot PNG
      • Filename: <TICKER>_<best_val_rmse><suffix>.pth
    """
    # Render the live RMSE plot to PNG bytes
    buf = io.BytesIO()
    live_plot.fig.savefig(buf, format="png")
    buf.seek(0)
    final_plot = buf.read()

    # Assemble the checkpoint dict
    ckpt = {
        "model_state_dict": best_state,
        "hparams":          params.hparams,
        "train_metrics":    best_tr,
        "val_metrics":      best_vl,
        "train_plot_png":   final_plot,
    }

    # Write to disk
    fname = f"{params.ticker}_{best_val_rmse:.5f}{suffix}.pth"
    (models_dir / fname).parent.mkdir(exist_ok=True, parents=True)
    torch.save(ckpt, models_dir / fname)
    print(f"✅ Final‐best model saved: {fname}")


################


def select_checkpoint(
    models_folder: Path,
    ticker: str,
    sel_val_rmse: float | None = None,
    tol: float = 1e-6
) -> Path:
    """
    Return the Path to the checkpoint:
      • If sel_val_rmse is not None, pick the file whose parsed RMSE
        equals sel_val_rmse within tol
      • Otherwise, or if no exact match, pick the file with the
        smallest parsed RMSE in the folder

    In either case, among ties prefer:
       1) filenames ending with '_fin.pth'
       2) then '_chp.pth'
       3) then any other match

    Raises FileNotFoundError if no *.pth files exist for the ticker.
    """
    ckpts = list(Path(models_folder).glob(f"{ticker}_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {models_folder} for {ticker}")

    # helper to extract rmse float from 'TICKER_<rmse>_*.pth'
    def parse_rmse(path: Path) -> float:
        stem = path.stem  # e.g. "AAPL_0.12345_chp"
        parts = stem.split("_")
        try:
            return float(parts[1])
        except (IndexError, ValueError):
            return float("inf")

    # build a map of path → rmse
    rmse_map = {p: parse_rmse(p) for p in ckpts}

    # decide candidate set
    if sel_val_rmse is not None:
        # find any whose rmse matches sel_val_rmse within tol
        exact = [p for p, v in rmse_map.items() if abs(v - sel_val_rmse) <= tol]
        if exact:
            candidates = exact
        else:
            warnings.warn(
                f"No exact checkpoint for sel_val_rmse={sel_val_rmse:.5f}; "
                "falling back to minimum RMSE"
            )
            candidates = ckpts
    else:
        candidates = ckpts

    # among candidates, pick by (priority, rmse)
    def priority(path: Path) -> tuple[int, float]:
        name = path.name
        if name.endswith("_fin.pth"):
            prio = 0
        elif name.endswith("_chp.pth"):
            prio = 1
        else:
            prio = 2
        return prio, rmse_map[path]

    return min(candidates, key=priority)


#########################################################################################################


def _append_log(text: str, log_file: Path):
    """Append a line to log_file, creating parent dirs if needed.

    This function is intentionally tolerant to I/O errors (best‑effort logging).
    It always ensures a trailing newline is present.
    """
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            f.flush()
    except Exception:
        # Swallow logging errors to avoid crashing training; logging is diagnostic only.
        pass
        
        
#############################################################


def collect_or_run_forward_micro_snapshot(model, train_loader=None, params=None, optimizer=None, log_file=None, clipnorm=None, return_snapshot=True):
    """
    Run a safe, forward-only micro-snapshot collector for a single batch and attach the
    canonical result to model._micro_snapshot.

    Purpose
    - Run one forward pass over a compact set of per-sample windows derived from the first
      batch of train_loader, measure timing/memory/IO metrics, and produce a single
      deterministic dictionary (the "micro snapshot") that summarizes batch shape,
      per-segment statistics, GPU/CPU usage, and lightweight optimizer/grad diagnostics.
    - Attach the snapshot at model._micro_snapshot and mark emission with
      model._microdetail_emitted to make the collector the single authoritative source
      for init-time diagnostics.
    """

    t_entry = perf_counter()
    intentional_syncs = 0

    # Resolve missing args from globals then shallow caller frames
    g = globals()
    if model is None and "model" in g:
        model = g["model"]
    if train_loader is None and "train_loader" in g:
        train_loader = g["train_loader"]
    if params is None and "params" in g:
        params = g["params"]

    if model is None or train_loader is None or params is None:
        for frame_info in inspect.stack()[1:6]:
            f_locals = frame_info.frame.f_locals
            f_globals = frame_info.frame.f_globals
            if model is None and "model" in f_locals:
                model = f_locals["model"]
            if model is None and "model" in f_globals:
                model = f_globals["model"]
            if train_loader is None and "train_loader" in f_locals:
                train_loader = f_locals["train_loader"]
            if train_loader is None and "train_loader" in f_globals:
                train_loader = f_globals["train_loader"]
            if params is None and "params" in f_locals:
                params = f_locals["params"]
            if params is None and "params" in f_globals:
                params = f_globals["params"]
            if model is not None and train_loader is not None and params is not None:
                break

    # Skip if another successful collection already ran
    if getattr(model, "_microdetail_emitted", False):
        return getattr(model, "_micro_snapshot", None)
    if getattr(model, "_micro_snapshot_in_progress", False):
        return None
    model._micro_snapshot_in_progress = True
    model_was_training = model.training
    try:
        # Fetch one batch; measure dataloader_ms when possible
        device = next(model.parameters()).device
        it = iter(train_loader)
        t0_dl = perf_counter()
        selected = next(it)
        t1_dl = perf_counter()
        dataloader_ms = (t1_dl - t0_dl) * 1000.0

        # Normalize batch
        elems = list(selected) if hasattr(selected, "__iter__") and not isinstance(selected, (str, bytes, dict)) else [selected]

        x_batch = elems[0]
        seq_lengths = elems[6] if len(elems) > 6 else None

        # Move inputs to device
        x_batch = x_batch.to(device)
        if seq_lengths.device != device:
            seq_lengths = seq_lengths.to(device)

        # Derive dims
        B = x_batch.size(0) if x_batch.dim() >= 1 else 0
        G = x_batch.size(1) if x_batch.dim() >= 4 else 1
        seq_len_full = x_batch.size(2) if x_batch.dim() >= 3 else (x_batch.size(1) if x_batch.dim() >= 2 else None)
        feat_dim = x_batch.size(-1)
        
        # Build per-sample segments padded/truncated to seq_len_full
        segments = []
        seg_lens = []
        for i in range(B):
            sl = int(seq_lengths[i].item()) if torch.is_tensor(seq_lengths[i]) else int(seq_lengths[i])
            if sl <= 0:
                continue

            sample_tensor = x_batch[i]
            take = min(sl, seq_len_full)
            try:
                idx = torch.arange(take, device=sample_tensor.device)
                seg = sample_tensor.index_select(1, idx)
            except Exception:
                seg = sample_tensor[:, :take, :] if sample_tensor.dim() >= 3 else sample_tensor[:take, :]

            cur_len = seg.size(1) if seg.dim() >= 2 else seg.size(0)
            if cur_len < seq_len_full:
                pad_rows = seq_len_full - cur_len
                pad_tensor = torch.zeros((seg.size(0), pad_rows, feat_dim), device=seg.device, dtype=seg.dtype) if seg.dim() >= 2 else torch.zeros((pad_rows, feat_dim), device=seg.device, dtype=seg.dtype)
                seg = torch.cat([seg, pad_tensor], dim=1) if seg.dim() >= 2 else torch.cat([seg, pad_tensor], dim=0)

            segments.append(seg)
            seg_lens.append(take)

        # Stack and reshape to [N_segments, seq_len, feat]
        stacked = torch.stack(segments, dim=0)
        if stacked.dim() == 4:
            B_valid, G_actual, seq_len_target, feat_dim = stacked.size()
            windows_tensor = stacked.view(-1, seq_len_target, feat_dim).to(device)
        elif stacked.dim() == 3:
            B_valid, seq_len_target, feat_dim = stacked.size()
            windows_tensor = stacked.view(-1, seq_len_target, feat_dim).to(device)

        num_segments = int(windows_tensor.size(0))
        mean_seg_len = float(sum(seg_lens) / len(seg_lens)) if seg_lens else float(seq_len_target)

        # GPU tracking: reset peak only when CUDA is available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            before_alloc = torch.cuda.memory_allocated(device)
            before_reserved = torch.cuda.memory_reserved(device)
        else:
            before_alloc = before_reserved = 0

        # Forward-only timing (AMP), use CUDA events when available; else use perf_counter
        if torch.cuda.is_available():
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            with torch.amp.autocast("cuda", enabled=True):
                raw_out = model(windows_tensor)
            e1.record()
            # one explicit sync to ensure event completion and to read elapsed_time
            torch.cuda.synchronize()
            intentional_syncs += 1
            full_forward_ms = e0.elapsed_time(e1)
        else:
            t0 = perf_counter()
            with torch.amp.autocast("cuda", enabled=False):
                raw_out = model(windows_tensor)
            full_forward_ms = (perf_counter() - t0) * 1000.0

        raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

        # Activation and memory after forward
        if torch.cuda.is_available():
            after_alloc = torch.cuda.memory_allocated(device)
            after_reserved = torch.cuda.memory_reserved(device)
            activation_bytes = max(0, after_alloc - before_alloc)
            activation_mb = int(math.ceil(activation_bytes / (1024 ** 2)))
            peak = torch.cuda.max_memory_allocated(device)
            gpu_peak_mb = int(peak // (1024 ** 2))
            reserved = torch.cuda.max_memory_reserved(device)
            gpu_reserved_mb = int(reserved // (1024 ** 2))
            gpu_allocated_bytes = int(after_alloc)
            gpu_reserved_bytes = int(reserved)
        else:
            activation_mb = gpu_peak_mb = gpu_reserved_mb = gpu_allocated_bytes = gpu_reserved_bytes = None

        # CPU transfer timing and bytes; output metadata
        out_shape = tuple(raw_reg.shape)
        out_dtype = str(raw_reg.dtype)
        out_numel = int(raw_reg.numel())

        elem_w = getattr(windows_tensor, "element_size", None)
        windows_bytes = int(windows_tensor.numel()) * int(elem_w()) if callable(elem_w) else None
        
        elem_o = getattr(raw_reg, "element_size", None)
        out_bytes = int(out_numel) * int(elem_o()) if out_numel is not None and callable(elem_o) else None

        preds = raw_reg.view(raw_reg.size(0), -1)[:, -1] if raw_reg.dim() >= 2 else raw_reg
        cpu_copy_bytes = int(out_numel * (raw_reg.element_size() if callable(getattr(raw_reg, "element_size", None)) else 4))

        # per-segment sampling controlled by params.hparams["MICRO_SAMPLE_K"]
        per_segment_p50_ms = None
        per_segment_p90_ms = None
        sample_k = int(params.hparams.get("MICRO_SAMPLE_K", 0)) if params is not None and hasattr(params, "hparams") else 0
        sample_k = max(0, min(256, sample_k))
        
        if sample_k > 0 and num_segments > 0:
            k = min(num_segments, sample_k)
            if num_segments <= k:
                idxs = list(range(num_segments))
            else:
                stride = max(1, num_segments // k)
                idxs = list(range(0, num_segments, stride))[:k]
                
        seg_times = []
        seg_events = []  # store event pairs to read after single sync
        for ii in idxs:
            seg = windows_tensor[ii : ii + 1]
            if torch.cuda.is_available():
                seg_e0 = torch.cuda.Event(enable_timing=True); seg_e1 = torch.cuda.Event(enable_timing=True)
                seg_e0.record()
                with torch.amp.autocast("cuda", enabled=True):
                    _ = model(seg)
                seg_e1.record()
                seg_events.append((seg_e0, seg_e1))
            else:
                t0s = perf_counter(); _ = model(seg); t1s = perf_counter()
                seg_times.append((t1s - t0s) * 1000.0)
        
        # if we recorded CUDA events, do one synchronize and extract elapsed times
        if torch.cuda.is_available() and seg_events:
            torch.cuda.synchronize()
            intentional_syncs += 1
            for e0, e1 in seg_events:
                seg_times.append(e0.elapsed_time(e1))
  
            if seg_times:
                seg_times_sorted = sorted(seg_times)
                def _pct(arr, p):
                    n = len(arr)
                    if n == 0:
                        return None
                    rank = int(math.floor(p / 100.0 * (n - 1)))
                    return float(arr[rank])
                per_segment_p50_ms = _pct(seg_times_sorted, 50.0)
                per_segment_p90_ms = _pct(seg_times_sorted, 90.0)

        t_exit = perf_counter()
        collector_ms = (t_exit - t_entry) * 1000.0

        # Estimate extra post-forward milliseconds (ms) consumed by CPU-side work
        seg_times_sum = sum(seg_times) if 'seg_times' in locals() and seg_times else 0.0
        pred_extra_ms = max(0.0, float(collector_ms) - float(full_forward_ms) - float(seg_times_sum))
        
        param_bytes = int(sum(int(p.numel()) * int(getattr(p, "element_size", lambda: 4)()) for p in model.parameters()))

        # Environment metadata (cheap, helpful)
        env = {}
        env["python"] = platform.python_version()
        env["torch"] = getattr(torch, "__version__", None)
        env["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["device_name"] = torch.cuda.get_device_name(device)
        else:
            env["device_name"] = platform.node()  # lightweight fallback host id


        # prefer explicit optimizer arg
        opt = optimizer or getattr(model, "optimizer", None) or globals().get("optimizer", None)
        optimizer_groups = getattr(opt, "param_groups", None) or []
        group_nonzero_counts = [
            sum(1 for p in (g.get("params") or []) if getattr(p, "grad", None) is not None)
            for g in optimizer_groups
        ]
    
        # single-pass named_parameters scan for grad flags
        backbone_has = False
        head_has = False
        for n, p in list(model.named_parameters()):
            has_grad = getattr(p, "grad", None) is not None
            ln = n.lower()
            if "pred" in ln or "head" in ln:
                head_has = head_has or has_grad
            else:
                backbone_has = backbone_has or has_grad
        
        snapshot = {
            "B": int(B),
            "groups": int(G),
            "seq_len_full": int(seq_len_full),
            "feat_dim": int(feat_dim),
            "raw_reg_shape": out_shape,                 # mirrors historical raw_reg_shape
            "group_nonzero_counts": group_nonzero_counts,
            "grads": {"backbone": bool(backbone_has), "head": bool(head_has)},
            "full_forward_ms": float(full_forward_ms),
            "pred_extra_ms": pred_extra_ms,
            "num_segments": int(num_segments),
            "segments_per_sec": float(num_segments / (full_forward_ms / 1000.0)) if full_forward_ms and num_segments else None,
            "expected_segments": int(B * G),
            "sum_seg_lens": int(sum(seg_lens)) if seg_lens else 0,
            "mean_seg_len": float(mean_seg_len),
            "gpu_peak_mb": int(gpu_peak_mb) if gpu_peak_mb is not None else None,
            "gpu_reserved_mb": int(gpu_reserved_mb) if gpu_reserved_mb is not None else None,
            "gpu_allocated_bytes": int(gpu_allocated_bytes) if gpu_allocated_bytes is not None else None,
            "gpu_reserved_bytes": int(gpu_reserved_bytes) if gpu_reserved_bytes is not None else None,
            "cpu_copy_bytes": int(cpu_copy_bytes) if cpu_copy_bytes is not None else None,
            "device_syncs_count": int(intentional_syncs),
            "per_segment_p50_ms": float(per_segment_p50_ms) if per_segment_p50_ms is not None else None,
            "per_segment_p90_ms": float(per_segment_p90_ms) if per_segment_p90_ms is not None else None,
            "activation_mb": int(activation_mb) if activation_mb is not None else None,
            "out_shape": out_shape,
            "out_dtype": out_dtype,
            "out_numel": int(out_numel) if out_numel is not None else None,
            "out_bytes": int(out_bytes) if out_bytes is not None else None,
            "windows_bytes": int(windows_bytes) if windows_bytes is not None else None,
            "collector_ms": float(collector_ms),
            "dataloader_ms": float(dataloader_ms) if dataloader_ms is not None else None,
            "param_bytes": int(param_bytes) if param_bytes is not None else None,
            "env": env,
            "backward_ms": getattr(model, "_last_backward_ms", None),
        }

        # attach and mark done
        model._micro_snapshot = snapshot
        model._microdetail_emitted = True
        
        if hasattr(model, "_micro_snapshot_in_progress"):
            delattr(model, "_micro_snapshot_in_progress")
        
        return snapshot if return_snapshot else None

    finally:
        # always cleanup and restore
        if hasattr(model, "_micro_snapshot_in_progress"):
            try:
                delattr(model, "_micro_snapshot_in_progress")
            except Exception:
                pass
        if model_was_training:
            try:
                model.train()
            except Exception:
                pass

#########################################################################################################


_RUN_STARTED = False
_RUN_DEBUG_DONE = False
_RUN_LOCK = threading.Lock()

def init_log(
    log_file:  Path,
    hparams:   dict,
    baselines: dict,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
):
    """
    Emit run-start diagnostics and a single authoritative micro-snapshot to the log.

    Purpose
    - Emit a compact run header, baselines and hyperparameters once per process.
    - Attempt a one-shot micro-snapshot collection (via collect_or_run_forward_micro_snapshot)
      and print a single canonical BATCH_SHAPE + MICRODETAIL line derived from the snapshot.
    - Provide a deterministic fallback to legacy first-batch snapshot data if the collector
      is unavailable.
    """
    global _RUN_STARTED, _RUN_DEBUG_DONE

    with _RUN_LOCK:
        if not _RUN_STARTED:
            sep = "-" * 150
            _append_log("\n" + sep, log_file)
            _append_log(f"RUN START: {datetime.utcnow().isoformat()}Z", log_file)
            
            _append_log("\nSINGLE RUN DIAGNOSTIC FORMAT (explanatory)", log_file)
            _append_log("    BATCH_SHAPE B=32 groups=451 seq_len_full=60 feat=20 : canonical input geometry used to build forwarded windows (snapshot.B, snapshot.groups, snapshot.seq_len_full, snapshot.feat_dim)", log_file)
            _append_log("    MICRODETAIL ms: <k=v ...> : single-line deterministic dump of snapshot keys printed as key=value (sorted); human-readable units used where applicable", log_file)
            _append_log("      -> B: number of batch samples used to build windows (snapshot.B)", log_file)
            _append_log("      -> groups: number of logical groups used when flattening windows (snapshot.groups)", log_file)
            _append_log("      -> seq_len_full: nominal full sequence length used when padding/truncating windows (snapshot.seq_len_full)", log_file)
            _append_log("      -> feat / feat_dim: feature dimension used to build windows (snapshot.feat_dim)", log_file)
            _append_log("      -> activation_mb: estimated activation footprint in MB computed from allocation delta (snapshot.activation_mb)", log_file)
            _append_log("      -> backward_ms: backward pass ms placeholder (snapshot.backward_ms)", log_file)
            _append_log("      -> collector_ms: total wall-clock ms spent by the collector including sampling and CPU/GPU syncs (snapshot.collector_ms)", log_file)
            _append_log("      -> cpu_copy_bytes: bytes copied to host for predictions, shown human-readable in log (snapshot.cpu_copy_bytes)", log_file)
            _append_log("      -> dataloader_ms: ms spent fetching the sampled batch from the dataloader (snapshot.dataloader_ms)", log_file)
            _append_log("      -> device_syncs_count: number of explicit device synchronizations performed during snapshot (snapshot.device_syncs_count)", log_file)
            _append_log("      -> env: small dict with python/torch/cuda/device_name strings (snapshot.env)", log_file)
            _append_log("      -> expected_segments: nominal B * groups used to estimate workload (snapshot.expected_segments)", log_file)
            _append_log("      -> full_forward_ms: wall-clock ms for the sampled forward over prepared windows (snapshot.full_forward_ms)", log_file)
            _append_log("      -> pred_extra_ms: estimated CPU-side post-forward ms (collector_ms - full_forward_ms - per-seg-sum) (snapshot.pred_extra_ms)", log_file)
            _append_log("      -> gpu_allocated_bytes: raw GPU bytes allocated (snapshot.gpu_allocated_bytes)", log_file)
            _append_log("      -> gpu_peak_mb: peak GPU memory in MB (snapshot.gpu_peak_mb)", log_file)
            _append_log("      -> gpu_reserved_bytes / gpu_reserved_mb: reserved GPU bytes and MB (snapshot.gpu_reserved_bytes, snapshot.gpu_reserved_mb)", log_file)
            _append_log("      -> grads: dict {'backbone': bool, 'head': bool} indicating gradient presence by name-bucket (snapshot.grads)", log_file)
            _append_log("      -> group_nonzero_counts: per-optimizer-group counts of parameters with non-None .grad (snapshot.group_nonzero_counts)", log_file)
            _append_log("      -> mean_seg_len: average per-segment time-series length in timesteps (snapshot.mean_seg_len)", log_file)
            _append_log("      -> num_segments: actual number of flattened segments forwarded (snapshot.num_segments)", log_file)
            _append_log("      -> out_bytes / out_dtype / out_numel / out_shape: model output bytes, dtype string, element count, and tuple shape (snapshot.out_bytes, snapshot.out_dtype, snapshot.out_numel, snapshot.out_shape)", log_file)
            _append_log("      -> param_bytes: total parameter memory in bytes (human-readable in log; raw int in snapshot.param_bytes)", log_file)
            _append_log("      -> per_segment_p50_ms / per_segment_p90_ms: empirical per-segment forward-ms percentiles when sampling enabled (snapshot.per_segment_p50_ms, snapshot.per_segment_p90_ms)", log_file)
            _append_log("      -> raw_reg_shape: the raw detached regression output shape (snapshot.raw_reg_shape)", log_file)
            _append_log("      -> segments_per_sec: inferred throughput = num_segments / (full_forward_ms/1000.0) (snapshot.segments_per_sec)", log_file)
            _append_log("      -> step_block_ms: placeholder for step-block timing if measured (snapshot.step_block_ms; reported 0.0 when not measured)", log_file)
            _append_log("      -> sum_seg_lens: sum of segment lengths used to compute mean_seg_len (snapshot.sum_seg_lens)", log_file)
            _append_log("      -> windows_bytes: total bytes for the windows tensor (human-readable in log; raw int in snapshot.windows_bytes)", log_file)
            
            _append_log("\nPER-EPOCH LOG FORMAT (explanatory):", log_file)
            _append_log("  E{ep:02d}                : epoch number formatted with two digits", log_file)
            _append_log("  OPTS[{groups}:{lr_main}|cnts=[c1,c2,...]] : optimizer groups count; lr_main is the representative LR (first group); cnts lists per-group parameter counts", log_file)
            _append_log("  GN[name:val,...,TOT=val] : per-bucket gradient L2 norms printed as short_name:curr_norm; TOT is sqrt(sum squares over reported buckets)", log_file)
            _append_log("  GD[med,p90,max]         : gradient-norm distribution statistics printed as median, index-based 90th-percentile, and maximum", log_file)
            _append_log("  UR[med,max]             : update-ratio statistics (median,max) where update_ratio = lr * grad_norm / max(weight_norm,1e-8)", log_file)
            _append_log("  LR_MAIN={lr:.1e} | lr={lr:.1e} : representative main LR and explicit first-group lr printed in scientific notation", log_file)
            _append_log("  TR[rmse,mae,r2]         : training metrics (RMSE, R^2, MAE) reported for the epoch (train_metrics)", log_file)
            _append_log("  VL[rmse,mae,r2]         : validation metrics (RMSE, R^2, MAE) reported for the epoch (val_metrics)", log_file)
            _append_log("  SR={slope_rmse:.3f}     : slope RMSE computed on model.last_val_preds/model.last_val_targs (trend calibration)", log_file)
            _append_log("  SL={slip:.2f},HR={hub_max:.3f} : slip fraction and hub max indicators derived from model.last_hub (defaults to 0.00/0.000 when missing)", log_file)
            _append_log("  FMB={val:.4f}           : first-mini-batch diagnostic metric from the first-batch snapshot if set (snapshot.FMB); in these logs FMB is high so validate baseline alignment", log_file)
            _append_log("  T={elapsed:.1f}s,TP={throughput:.1f} : epoch elapsed seconds and throughput (segments/sec or windows/sec depending on implementation)", log_file)
            _append_log("  chk={val:.3f}           : checkpoint-score token printed as chk in the line (implementation-specific)", log_file)
            _append_log("  GPU={GiB:.2f}GiB         : optional high-water GPU memory in GiB when CUDA available (torch.cuda.max_memory_allocated)", log_file)
            _append_log("  *CHKPT                  : optional marker when model._last_epoch_checkpoint is truthy", log_file)
            _append_log("  LAYER_GN[...]           : optional small set of monitored layer norms printed as name_short:curr_norm/ratio", log_file)
            _append_log("  TOP_K(G/U)=name:grad_norm/update_ratio,... : top-k parameter entries by gradient norm with their update ratios (short names use last one or two name segments)", log_file)

            if isinstance(baselines, dict) and baselines:
                _append_log("\nBASELINES:", log_file)
                _append_log(f"  TRAIN mean RMSE        = {baselines['base_tr_mean']:.5f}", log_file)
                _append_log(f"  TRAIN persistence RMSE = {baselines['base_tr_pers']:.5f}", log_file)
                _append_log(f"  VAL   mean RMSE        = {baselines['base_vl_mean']:.5f}", log_file)
                _append_log(f"  VAL   persistence RMSE = {baselines['base_vl_pers']:.5f}", log_file)
                
            if isinstance(hparams, dict) and hparams:
                _append_log("\nHYPERPARAMS:", log_file)
                for k, v in hparams.items():
                    _append_log(f"  {k} = {v}", log_file)
                _append_log("", log_file)

            # --- compact runtime summary (single-line appearance) ---
            debug_opt_line = None
            if optimizer is not None:
                opt_groups = len(optimizer.param_groups)
                opt_lrs = [g.get("lr", 0.0) for g in optimizer.param_groups]
                opt_counts = [sum(1 for _ in g["params"]) for g in optimizer.param_groups]
                debug_opt_line = f"DEBUG_OPT GROUPS={opt_groups} LRS={[f'{x:.1e}' for x in opt_lrs]} COUNTS={opt_counts}"

            model_static_line = None
            param_bytes_line = None
            if model is not None:
                total_params = sum(int(p.numel()) for p in model.parameters())
                trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
                frozen_params = total_params - trainable_params
                model_static_line = f"MODEL_STATIC: total_params={total_params:,} trainable={trainable_params:,} frozen={frozen_params:,}"

                # prefer model._micro_snapshot param_bytes if available
                param_bytes = None
                micro_snap = getattr(model, "_micro_snapshot", None)
                if isinstance(micro_snap, dict):
                    param_bytes = micro_snap.get("param_bytes")
                if param_bytes is not None:
                    if param_bytes >= 1024 * 1024:
                        param_bytes_line = f"param_bytes={param_bytes//(1024*1024)}MB"
                    elif param_bytes >= 1024:
                        param_bytes_line = f"param_bytes={param_bytes//1024}KB"
                    else:
                        param_bytes_line = f"param_bytes={param_bytes}B"

            compact_parts = []
            if debug_opt_line: compact_parts.append(debug_opt_line)
            if model_static_line: compact_parts.append(model_static_line)
            if param_bytes_line: compact_parts.append(param_bytes_line)

            _RUN_STARTED = True

        # --- One-shot read-only runtime snapshot emitted at most once ---
        if not _RUN_DEBUG_DONE:
            
            # Attempt to run the micro-snapshot collector if available
            tloader = locals().get("train_loader", None) or globals().get("train_loader", None)
            prms = locals().get("params", None) or globals().get("params", None)
            
            if callable(globals().get("collect_or_run_forward_micro_snapshot", None)):
                collect_or_run_forward_micro_snapshot(
                    model=model,
                    train_loader=tloader,
                    params=prms,
                    optimizer=locals().get("optimizer", None) or globals().get("optimizer", None),
                    log_file=log_file,
                )

            micro_ms = getattr(model, "_micro_snapshot", None)
            
            emitted = False
            
            # Emit DEBUG_SHAPES, GROUP_NONZERO_COUNTS, DEBUG_GRADS from micro snapshot if present
            if isinstance(micro_ms, dict):
                ms = micro_ms
                # single authoritative BATCH_SHAPE + MICRODETAIL from micro snapshot
                _append_log(f"BATCH_SHAPE B={ms.get('B')} groups={ms.get('groups')} seq_len_full={ms.get('seq_len_full')} feat={ms.get('feat_dim')}", log_file)
            
                def _hb(v):
                    try:
                        v = int(v)
                    except Exception:
                        return str(v)
                    if v >= 1024*1024: return f"{v//(1024*1024)}MB"
                    if v >= 1024: return f"{v//1024}KB"
                    return f"{v}B"
            
                def _fmt(k, v):
                    if v is None: return f"{k}=None"
                    if "bytes" in k or "param" in k or "cpu_copy" in k or "windows" in k:
                        return f"{k}={_hb(v)}"
                    if "shape" in k and not isinstance(v, str):
                        return f"{k}={tuple(v)}"
                    if isinstance(v, float):
                        return f"{k}={v:.2f}"
                    return f"{k}={v}"
            
                parts = [_fmt(k, ms.get(k)) for k in sorted(ms.keys())]
                _append_log("MICRODETAIL ms: " + " ".join(parts), log_file)
                emitted = True
            
            if emitted:
                _RUN_DEBUG_DONE = True

#################################################################################################################################


def log_epoch_summary(
    epoch:            int,
    model:            torch.nn.Module,
    optimizer:        torch.optim.Optimizer,
    train_metrics:    dict,
    val_metrics:      dict,
    val_preds:        float,
    val_base_preds:   float,
    val_targets:      float,
    base_tr_mean:     float,
    base_tr_pers:     float,
    base_vl_mean:     float,
    base_vl_pers:     float,
    avg_main_loss:    float,
    avg_aux_loss:     float,
    log_file:         Path,
    top_k:            int,
    hparams:          dict,
):
    """
    Emit a compact, human-readable per-epoch summary line and supporting diagnostics.

    Purpose
    - Produce a single-line epoch summary that combines optimizer structure, gradient
      diagnostics, training/validation metrics, scheduler progress, timing/throughput,
      GPU usage, checkpoint flag, and a small set of layer-wise gradient-ratio diagnostics.
    - Provide deterministic, single-pass computations of per-parameter gradient norms
      and update ratios to power the summary tokens.
    """

    # 1) Ensure header + run-static info (init_log handles guards and one-shot debug if available)
    init_log(
        log_file,
        hparams=hparams,
        baselines={
            "base_tr_mean": base_tr_mean,
            "base_tr_pers": base_tr_pers,
            "base_vl_mean": base_vl_mean,
            "base_vl_pers": base_vl_pers,
        },
        optimizer=optimizer,
        model=model,
    )

    # 2) detect top-level parameter groups ("heads") and their grad-norm totals — minimal
    recs = []; prefix_sq = {}; all_sq = 0.0
    lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
    
    for name, p in model.named_parameters():
        if p.grad is None:
            g = 0.0
        else:
            if getattr(p.grad, "is_sparse", False):
                vals = p.grad.data.coalesce().values()
                g = float(vals.norm().cpu()) if vals.numel() else 0.0
            else:
                g = float(p.grad.norm().cpu())
        w = float(p.detach().norm().cpu())
        u = (lr * g) / max(w, 1e-8)
        recs.append((name, g, u))
        sq = g * g; all_sq += sq
        pref = name.split('.', 1)[0]
        prefix_sq[pref] = prefix_sq.get(pref, 0.0) + sq

    g_vals = [g for _, g, _ in recs]; u_vals = [u for _, _, u in recs]

    # per-prefix GN and total GN
    prefix_gn = {p: math.sqrt(sq) for p, sq in prefix_sq.items()}
    GN_tot = math.sqrt(all_sq)
    
    # build a GN token listing all prefixes sorted by descending GN
    sorted_prefixes = [p for p, _ in sorted(prefix_gn.items(), key=lambda x: x[1], reverse=True)]
    gn_items = ",".join(f"{p}={prefix_gn[p]:.3f}" for p in sorted_prefixes) if sorted_prefixes else ""
    # gn_token = f"GN[{gn_items},TOT={GN_tot:.3f}] | "

    # 3) 4) safe percentiles for GD and UR
    if g_vals:
        g_sorted = sorted(g_vals)
        n = len(g_sorted)
        GD_med = g_sorted[n // 2]
        GD_p90 = g_sorted[max(0, min(n - 1, int(math.floor(0.9 * (n - 1)))))]
        GD_max = g_sorted[-1]
    else:
        GD_med = GD_p90 = GD_max = 0.0
    
    if u_vals:
        u_sorted = sorted(u_vals)
        n_u = len(u_sorted)
        UR_med = u_sorted[n_u // 2]
        UR_max = u_sorted[-1]
    else:
        UR_med = UR_max = 0.0


    # 5) Slip-rate & max hub (stateful LSTM diagnostic)
    slip_thresh=1e-6
    hub    = getattr(model, "last_hub", None)
    HR     = float(hub.max().cpu()) if hub is not None else 0.0
    SL     = float((hub > slip_thresh).float().mean().cpu()) if hub is not None else 0.0

    # 6) Slope-RMSE (SR) on last validation batch
    with torch.no_grad():
        pv, tv = getattr(model, "last_val_preds", None), getattr(model, "last_val_targs", None)
        SR = 0.0
        if pv is not None and tv is not None and pv.numel() >= 2 and tv.numel() >= 2:
            if pv.dim() == 1:
                dp, dt = pv[1:] - pv[:-1], tv[1:] - tv[:-1]
            else:
                if pv.size(1) >= 2 and tv.size(1) >= 2:
                    dp = pv[:, 1:] - pv[:, :-1]
                    dt = tv[:, 1:] - tv[:, :-1]
                else:
                    dp = dt = None
            if dp is not None and dt is not None and dp.numel() > 0:
                SR = float(torch.sqrt(((dp - dt) ** 2).mean()).item())

    # 6.1) Fraction model better than baseline on validation (per-sample)
    fraction_model_better = 0.0
    if val_preds is not None and val_base_preds is not None and val_targets is not None:
        # ensure numpy arrays (1D) and aligned length
        vp = np.asarray(val_preds, dtype=float).ravel()
        vb = np.asarray(val_base_preds, dtype=float).ravel()
        vt = np.asarray(val_targets, dtype=float).ravel()
        if vp.shape == vb.shape == vt.shape and vp.size > 0:
            fraction_model_better = float((np.abs(vp - vt) < np.abs(vb - vt)).mean())

    # 7) Pull train/val RMSE, R², MAE
    tr_rmse, tr_mae, tr_r2 = train_metrics["rmse"], train_metrics["mae"], train_metrics["r2"]
    vl_rmse, vl_mae, vl_r2 = val_metrics["rmse"],   val_metrics["mae"],   val_metrics["r2"]


    # prepare loss tokens if available
    loss_tokens = ""
    loss_tokens = f"MAIN_LOSS={avg_main_loss:.4e}"
    aux_ratio = avg_aux_loss / (avg_main_loss + 1e-12)
    loss_tokens += f",AUX_LOSS={avg_aux_loss:.4e},AUX_RATIO={aux_ratio:.3e}"

    # 8) # compact Top-K: dedupe by short name, show all non-zero, collapse tiny into one token
    def short_name(n):
        if "parametrizations" in n:
            return n.split('.')[-1]
        return ".".join(n.split('.')[-3:])

    
    seen = {}
    for name, g, u in recs:
        if "parametrizations" in name:
            continue
        s = short_name(name)
        if s not in seen or g > seen[s][0]:
            seen[s] = (g, u)
            
    items = sorted(seen.items(), key=lambda kv: kv[1][0], reverse=True)
    zero_thresh = 1e-6
    display = [f"{s}:{g:.3f}/{u:.1e}" for s, (g, u) in items if g > zero_thresh]
    zero_count = sum(1 for _, (g, _u) in items if g <= zero_thresh)
    if zero_count:
        display.append(f"zero:{zero_count}/0.000")
    topk_str = ", ".join(display)

    # 9) Assemble OPTS token (compact)
    try:
        opt_groups = len(optimizer.param_groups)
        opt_lrs_sh = ",".join(f"{g.get('lr',0.0):.1e}" for g in optimizer.param_groups[:3]) + (",..." if len(optimizer.param_groups) > 3 else "")
        opt_counts = [sum(1 for _ in g["params"]) for g in optimizer.param_groups]
        opt_token = f"OPTS[{opt_groups}:{opt_lrs_sh}|cnts={opt_counts}]"
    except Exception:
        opt_token = f"OPTS[1:{lr:.1e}]"

    # 10) Optional scheduler percent-complete token (read-only, best-effort)
    sched_pct_token = ""
    sched_obj = getattr(optimizer, "scheduler", None) or globals().get("scheduler", None)

    if sched_obj is not None and hasattr(sched_obj, "_total_steps"):
        total = int(getattr(sched_obj, "_total_steps"))
        # prefer a step counter if available, otherwise use last_epoch
        step_idx = getattr(sched_obj, "last_epoch", None)
        if step_idx is None:
            step_idx = getattr(sched_obj, "_step_count", None)
        if step_idx is not None and total > 0:
            pct = min(100.0, max(0.0, 100.0 * float(step_idx) / float(total)))
            sched_pct_token = f"SCHED_PCT={pct:.1f}%"

    # 11) Optional timing / throughput (if the training loop stored them on the model)
    elapsed = getattr(model, "_last_epoch_elapsed", 0)
    samples = getattr(model, "_last_epoch_samples", 0)
    tp = (samples / elapsed) if (elapsed and elapsed > 0) else 0.0

    # 12) Optional checkpoint marker (if training loop marked it on the model)
    chk = getattr(model, "_last_epoch_checkpoint", False)

    # # 13) Optional GPU memory high-water (low-noise, printed only if CUDA available)
    max_mem = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

    # 14) Primary LR scalar token (explicit, high-ROI and low-noise)
    try:
        primary_lr = optimizer.param_groups[0].get("lr", lr) if optimizer.param_groups else lr
        lr_token = f"LR_MAIN={primary_lr:.1e}"
    except Exception:
        lr_token = f"LR_MAIN={lr:.1e}"

    # 15) Layer-wise GN ratio diagnostics (small set of representative layers)
    layer_token = ""
    default_monitor = ["short_lstm.weight_ih_l0", "short2long.weight", "feature_proj.weight"]
    monitor_list = []
    if isinstance(hparams, dict) and hparams.get("MONITOR_LAYERS"):
        monitor_list = list(hparams.get("MONITOR_LAYERS"))
    else:
        monitor_list = default_monitor

    # Ensure a dict exists to store baseline GN observed first time
    if not hasattr(model, "_first_layer_gn"):
        setattr(model, "_first_layer_gn", {})

    pairs = []
    for name in monitor_list:
        # best-effort: find exact match; if not found, skip silently
        p = dict(model.named_parameters()).get(name)
        if p is None:
            continue
        if p.grad is None:
            curr_g = 0.0
        elif getattr(p.grad, "is_sparse", False):
            vals = p.grad.data.coalesce().values()
            curr_g = float(vals.norm().cpu()) if vals.numel() else 0.0
        else:
            curr_g = float(p.grad.norm().cpu())

        baseline = None
        try:
            baseline = getattr(model, "_first_layer_gn", {}).get(name)
        except Exception:
            baseline = None
        # if no baseline stored yet, store this epoch's value as baseline
        if baseline is None:
            try:
                model._first_layer_gn[name] = float(curr_g)
                ratio = 1.0
            except Exception:
                ratio = 1.0
        else:
            ratio = curr_g / max(baseline, 1e-12)
            label = ".".join(name.split('.')[-2:])
            pairs.append(f"{label}:{curr_g:.3e}/{ratio:.2f}")

    if pairs:
            layer_token = "LAYER_GN[" + ",".join(pairs) + "]"

    # 16) Final line assembly (compact, per-epoch changing values only)
    sched_field = f"{sched_pct_token}" if sched_pct_token else ""
    line = (
        f"\nE{epoch:02d} | "
        f"{opt_token} | "
        # f"GN[{GN_reg:.3f},{GN_cls:.3f},{GN_ter:.3f},{GN_tot:.3f}] | "
        f"GN[{gn_items},TOT={GN_tot:.3f}] | "
        f"GD[{GD_med:.1e},{GD_p90:.1e},{GD_max:.1e}] | "
        f"UR[{UR_med:.1e},{UR_max:.1e}] | "
        f"{lr_token} | "
        f"lr={lr:.1e} | "
        f"TR[{tr_rmse:.3f},{tr_mae:.3f},{tr_r2:.2f}] | "
        f"VL[{vl_rmse:.3f},{vl_mae:.3f},{vl_r2:.2f}] | "
        f"{loss_tokens} | "
        f"{sched_field} | "
        f"SR={SR:.3f} | "
        f"SL={SL:.2f},HR={HR:.3f} | "
        f"FMB={fraction_model_better:.4f} | "
        f"T={elapsed:.1f}s,TP={tp:.1f}seg/s | "
        f"chk={chk:.3f} | "
        f"GPU={max_mem:.2f}GiB | "
        f"{layer_token}"
        f"\nTOP_K(G/U)={topk_str}"
    )
    _append_log(line, log_file)
