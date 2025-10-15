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
    (cur_best) *and* on‐disk checkpoints. If it’s an improvement, capture
    the model’s weights, metrics, and plot for both folder‐best and
    in‐run checkpointing.

    Returns:
      updated_best_rmse : new best RMSE (float)
      improved          : True if vl_rmse < cur_best
      best_train_metrics: snapshot of train metrics at this best
      best_val_metrics  : snapshot of val   metrics at this best
      best_state_dict   : model.state_dict() if improved, else None
    """
    # Ensure output folder exists
    models_dir.mkdir(exist_ok=True)

    # 1) Gather on‐disk RMSEs to know if we're beating existing files
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

        # 3) If we also beat any on‐disk model, write a folder‐best checkpoint
        if updated_best < best_on_disk:
            fname = f"{params.ticker}_{updated_best:.5f}_chp.pth"
            ckpt = {
                "model_state_dict": best_state,
                "hparams":          params.hparams,
                "train_metrics":    best_tr,
                "val_metrics":      best_vl,
                "train_plot_png":   plot_bytes,
            }
            torch.save(ckpt, models_dir / fname)
            print(f"🔖 Saved folder‐best checkpoint (_chp): {fname}")

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


_RUN_STARTED = False
_RUN_LOCK    = threading.Lock()

def init_log(
    log_file:  Path,
    hparams:   dict | None = None,
    baselines: dict | None = None
):
    """
    Emit the run header exactly once. Prints:
      • A 150-char separator  
      • RUN START timestamp (UTC)  
      • Optional hyperparameters list  
      • Optional baseline RMSEs for train/val (mean & persistence)  
      • A sample “LOG FORMAT” comment explaining each field in the per-epoch summary.

    Subsequent calls do nothing.
    """
    global _RUN_STARTED
    with _RUN_LOCK:
        if _RUN_STARTED:
            return

        sep = "-" * 150
        _append_log("\n" + sep, log_file)
        _append_log(f"RUN START: {datetime.utcnow().isoformat()}Z", log_file)

        if isinstance(baselines, dict) and baselines:
            _append_log("\nBASELINES:", log_file)
            _append_log(f"  TRAIN mean RMSE       = {baselines['base_tr_mean']:.5f}", log_file)
            _append_log(f"  TRAIN persistence RMSE = {baselines['base_tr_pers']:.5f}", log_file)
            _append_log(f"  VAL   mean RMSE       = {baselines['base_vl_mean']:.5f}", log_file)
            _append_log(f"  VAL   persistence RMSE = {baselines['base_vl_pers']:.5f}", log_file)

        if isinstance(hparams, dict) and hparams:
            _append_log("\nHYPERPARAMS:", log_file)
            for k, v in hparams.items():
                _append_log(f"  {k} = {v}", log_file)

        # Sample comment: explain the per-epoch log line format
        _append_log(
            "\n# PER-EPOCH LOG FORMAT:\n"
            "#  E{ep:02d} | "
            "GN[reg,cls,ter,tot] | "
            "GD[med,p90,max] | "
            "UR[med,max] | "
            "lr={lr:.1e} | "
            "TR[rmse,r2,mae] | "
            "VL[rmse,r2,mae] | "
            "SR={slope_rmse:.3f} | "
            "SL={slip:.2f},HR={hub_max:.3f} | "
            "topK(g/u)=param:grad_norm/update_ratio,...",
            log_file
        )

        _RUN_STARTED = True


######################

def _append_log(text: str, log_file: Path):
    """Append a line to log_file, creating parent dirs if needed."""
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            f.flush()
    except Exception:
        pass


##############################################################


def log_epoch_summary(
    epoch:            int,
    model:            torch.nn.Module,
    optimizer:        torch.optim.Optimizer,
    train_metrics:    dict,
    val_metrics:      dict,
    base_tr_mean:     float,
    base_tr_pers:     float,
    base_vl_mean:     float,
    base_vl_pers:     float,
    slip_thresh:      float,
    log_file:         Path,
    top_k:            int         = 3,
    hparams:          dict | None = None,
):
    """
    One-line epoch summary with essential diagnostics. See init_log for format.

    Per-epoch fields:
      GN[...]   : ℓ2‐norm of gradients in [regression, classification, term, total]
      GD[...]   : gradient-norm distribution [median, 90th percentile, max]
      UR[...]   : update-ratio distribution [median, max]
      lr        : current learning rate
      TR[...]   : training metrics [rmse, r2, mae]
      VL[...]   : validation metrics [rmse, r2, mae]
      SR        : slope‐RMSE on last validation batch
      SL, HR    : slip‐rate & max hub (stateful LSTM state diagnostic)
      topK(g/u) : top‐K params by grad‐norm, showing grad_norm/update_ratio

    All values use consistent precision and engineering notation.
    """
    # 1) Header + baselines (only on first call)
    init_log(
        log_file,
        hparams=hparams,
        baselines={
            "base_tr_mean": base_tr_mean,
            "base_tr_pers": base_tr_pers,
            "base_vl_mean": base_vl_mean,
            "base_vl_pers": base_vl_pers,
        },
    )

    # 2) Collect per-parameter grad norms, update ratios, and block sums
    recs = []  # (name, grad_norm, update_ratio)
    reg_sq = cls_sq = ter_sq = all_sq = 0.0
    lr = optimizer.param_groups[0]["lr"]

    for name, p in model.named_parameters():
        g = float(p.grad.norm().cpu()) if p.grad is not None else 0.0
        w = float(p.detach().norm().cpu())
        u = (lr * g) / max(w, 1e-8)
        recs.append((name, g, u))

        sq = g * g
        all_sq += sq
        if   name.startswith("pred"):     reg_sq += sq
        elif name.startswith("cls_head"): cls_sq += sq
        elif name.startswith("cls_ter"):  ter_sq += sq

    # 3) Compute block gradient norms (GN) and gradient‐norm distribution (GD)
    GN_reg = math.sqrt(reg_sq)
    GN_cls = math.sqrt(cls_sq)
    GN_ter = math.sqrt(ter_sq)
    GN_tot = math.sqrt(all_sq)

    g_vals = [g for _, g, _ in recs]
    GD_med = sorted(g_vals)[len(g_vals)//2] if g_vals else 0.0
    GD_p90 = sorted(g_vals)[int(0.9*len(g_vals))] if g_vals else 0.0
    GD_max = max(g_vals)                           if g_vals else 0.0

    # 4) Update‐ratio distribution (UR)
    u_vals = [u for _, _, u in recs]
    UR_med = sorted(u_vals)[len(u_vals)//2] if u_vals else 0.0
    UR_max = max(u_vals)                         if u_vals else 0.0

    # 5) Slip‐rate & max hub (stateful LSTM diagnostic)
    hub    = getattr(model, "last_hub", None)
    HR     = float(hub.max().cpu()) if hub is not None else 0.0
    SL     = float((hub > slip_thresh).float().mean().cpu()) if hub is not None else 0.0

    # 6) Slope‐RMSE (SR) on last validation batch
    with torch.no_grad():
        pv, tv = model.last_val_preds, model.last_val_targs
        if pv is not None and tv is not None:
            if pv.dim() == 1:
                dp, dt = pv[1:]-pv[:-1], tv[1:]-tv[:-1]
            else:
                dp = pv[:,1:]-pv[:,:-1]
                dt = tv[:,1:]-tv[:,:-1]
            SR = torch.sqrt(((dp - dt)**2).mean()).item()
        else:
            SR = 0.0

    # 7) Pull train/val RMSE, R², MAE
    tr_rmse, tr_r2, tr_mae = train_metrics["rmse"], train_metrics["r2"], train_metrics["mae"]
    vl_rmse, vl_r2, vl_mae = val_metrics["rmse"],   val_metrics["r2"],   val_metrics["mae"]

    # 8) Top‐K parameter diagnostics: show only last two name segments + g/u
    topk = sorted(recs, key=lambda x: x[1], reverse=True)[:top_k]
    def short_name(n):
        parts = n.split('.')
        return ".".join(parts[-2:]) if len(parts) > 2 else n
    topk_str = ", ".join(f"{short_name(n)}:{g:.3f}/{u:.1e}" for n, g, u in topk)

    # 9) Assemble & append the log line
    line = (
        f"\nE{epoch:02d} | "
        f"GN[{GN_reg:.3f},{GN_cls:.3f},{GN_ter:.3f},{GN_tot:.3f}] | "
        f"GD[{GD_med:.1e},{GD_p90:.1e},{GD_max:.1e}] | "
        f"UR[{UR_med:.1e},{UR_max:.1e}] | "
        f"lr={lr:.1e} | "
        f"TR[{tr_rmse:.3f},{tr_r2:.2f},{tr_mae:.3f}] | "
        f"VL[{vl_rmse:.3f},{vl_r2:.2f},{vl_mae:.3f}] | "
        f"SR={SR:.3f} | "
        f"SL={SL:.2f},HR={HR:.3f} | "
        f"topK(g/u)={topk_str}"
    )
    _append_log(line, log_file)



##############################################################

