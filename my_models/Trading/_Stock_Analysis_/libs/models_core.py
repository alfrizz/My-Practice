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


# def build_tensors(
#     df: pd.DataFrame,
#     *,
#     look_back:   int                  = params.look_back_tick,
#     tmpdir:      str                  = None,
#     device:      torch.device         = torch.device("cpu"),
#     sess_start = None,
#     in_memory:   bool                 = True
# ) -> tuple[
#     torch.Tensor,  # X         shape=(N, look_back, F)
#     torch.Tensor,  # y_sig     shape=(N,)
#     torch.Tensor,  # y_ret     shape=(N,)
#     torch.Tensor,  # raw_close shape=(N,)
#     np.ndarray     # end_times shape=(N,) dtype=datetime64[ns]
# ]:
#     """
#     Build sliding‐window tensors for an LSTM trading model.

#     1) Copy DataFrame; select feature columns.
#     2) First pass per calendar day:
#        a) extract NumPy arrays (features, signal, raw_close);
#        b) compute log‐returns;
#        c) compute window‐end timestamps and boolean mask ≥ sess_start;
#        d) count valid windows, accumulate N, and store:
#           (feats_np, sig_end, ret_end, close_end, ends_np, mask, offset).
#     3) Allocate either in‐RAM numpy arrays or on‐disk memmaps of shape N:
#        X (N, look_back, F), y_sig, y_ret, raw_close, end_times.
#        On MemoryError, automatically fall back to memmaps.
#     4) Second pass (parallel, one job per stored day):
#        a) build sliding windows via numpy stride_tricks,
#        b) write masked slices into the shared buffers,
#        c) tick one per‐day tqdm bar.
#     5) If using memmaps, flush + os.sync() to ensure durability.
#     6) Wrap buffers as PyTorch tensors; cleanup; return.
#     """
#     # 1) Prepare DataFrame & feature list
#     df = df.copy()
#     exclude      = {params.label_col, "close_raw"}
#     feature_cols = [c for c in df.columns if c not in exclude]
#     print("Inside build_tensors, features:", feature_cols)
#     F = len(feature_cols)

#     # normalize sess_start to Python time
#     sess_time = sess_start.time() if hasattr(sess_start, "time") else sess_start
#     cutoff_sec = sess_time.hour * 3600 + sess_time.minute * 60

#     # 2) First pass: build payloads of pure NumPy + count N
#     day_groups = df.groupby(df.index.normalize(), sort=False)
#     payloads   = []
#     N_total    = 0

#     for _, day_df in tqdm(day_groups, desc="Preparing days", leave=False):
#         day_df = day_df.sort_index()
#         T = len(day_df)
#         if T <= look_back:
#             continue

#         # a) extract arrays
#         feats_np = day_df[feature_cols].to_numpy(np.float32)       # (T, F)
#         sig_np   = day_df[params.label_col].to_numpy(np.float32)  # (T,)
#         close_np = day_df["close_raw"].to_numpy(np.float32)       # (T,)

#         # b) compute log‐returns
#         ret_full       = np.empty_like(close_np, np.float32)
#         ret_full[0]    = 0.0
#         ret_full[1:]   = np.log(close_np[1:] / close_np[:-1])

#         # c) window‐end times & mask
#         ends_np = day_df.index.to_numpy()[look_back:]                      # (T-look_back,)
#         secs    = (ends_np - ends_np.astype("datetime64[D]")) \
#                     / np.timedelta64(1, "s")
#         mask    = secs >= cutoff_sec                                       # (T-look_back,)

#         m = int(mask.sum())
#         if m == 0:
#             continue

#         # slice next‐bar arrays once
#         sig_end   = sig_np[look_back:]     # (T-look_back,)
#         ret_end   = ret_full[look_back:]
#         close_end = close_np[look_back:]

#         payloads.append((feats_np, sig_end, ret_end, close_end, ends_np, mask, N_total))
#         N_total += m

#     # 3) Allocate buffers: in RAM or memmap
#     use_memmap = not in_memory
#     X_buf = y_buf = r_buf = c_buf = t_buf = None

#     if not use_memmap:
#         try:
#             X_buf = np.empty((N_total, look_back, F),   np.float32)
#             y_buf = np.empty((N_total,),                np.float32)
#             r_buf = np.empty((N_total,),                np.float32)
#             c_buf = np.empty((N_total,),                np.float32)
#             t_buf = np.empty((N_total,),      "datetime64[ns]")
#         except MemoryError:
#             print("Buffer allocation OOM, falling back to memmaps")
#             use_memmap = True

#     if use_memmap:
#         if tmpdir is None:
#             tmpdir = tempfile.mkdtemp(prefix="lstm_memmap_")
#         else:
#             os.makedirs(tmpdir, exist_ok=True)

#         def _open_memmap(name, shape, dtype):
#             return np.lib.format.open_memmap(
#                 os.path.join(tmpdir, name),
#                 mode="w+", dtype=dtype, shape=shape
#             )

#         X_buf = _open_memmap("X.npy",      (N_total, look_back, F), np.float32)
#         y_buf = _open_memmap("y_sig.npy",  (N_total,),             np.float32)
#         r_buf = _open_memmap("y_ret.npy",  (N_total,),             np.float32)
#         c_buf = _open_memmap("close.npy",  (N_total,),             np.float32)
#         t_buf = _open_memmap("t.npy",      (N_total,),     "datetime64[ns]")

#     # 4) Second pass: fill buffers in parallel
#     pbar = tqdm(total=len(payloads), desc="Writing days")

#     def _write_np(payload):
#         feats_np, sig_end, ret_end, close_end, ends_np, mask, offset = payload

#         # sliding windows + drop last
#         wins = np.lib.stride_tricks.sliding_window_view(
#                    feats_np, window_shape=(look_back, F)
#                ).reshape(feats_np.shape[0] - look_back + 1, look_back, F)[:-1]

#         m = mask.sum()
#         X_buf[offset:offset+m] = wins[mask]
#         y_buf[offset:offset+m] = sig_end[mask]
#         r_buf[offset:offset+m] = ret_end[mask]
#         c_buf[offset:offset+m] = close_end[mask]
#         t_buf[offset:offset+m] = ends_np[mask]

#         pbar.update(1)

#     with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as exe:
#         exe.map(_write_np, payloads)
#     pbar.close()

#     # 5) Flush memmaps if used
#     if use_memmap:
#         for arr in (X_buf, y_buf, r_buf, c_buf, t_buf):
#             arr.flush()
#         os.sync()

#     # 6) Wrap as PyTorch tensors
#     X         = torch.from_numpy(X_buf).to(device, non_blocking=True)
#     y_sig     = torch.from_numpy(y_buf).to(device, non_blocking=True)
#     y_ret     = torch.from_numpy(r_buf).to(device, non_blocking=True)
#     raw_close = torch.from_numpy(c_buf).to(device, non_blocking=True)
#     end_times = t_buf.copy()

#     # 7) Cleanup
#     gc.collect()
#     if device.type == "cuda":
#         torch.cuda.empty_cache()

#     return X, y_sig, y_ret, raw_close, end_times


def build_tensors(
    df: pd.DataFrame,
    *,
    look_back:   int                  = params.look_back_tick,
    tmpdir:      str                  = None,
    device:      torch.device         = torch.device("cpu"),
    sess_start   = None,
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
    exclude      = {params.label_col, "close_raw"}
    feature_cols = [c for c in df.columns if c not in exclude]
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


# def chronological_split(
#     X:           torch.Tensor,
#     y_sig:       torch.Tensor,
#     y_ret:       torch.Tensor,
#     raw_close:   torch.Tensor,
#     end_times:   np.ndarray,      # (N,), dtype datetime64[ns]
#     *,
#     train_prop:  float,
#     val_prop:    float,
#     train_batch: int,
#     device = torch.device("cpu")
# ) -> Tuple[
#     Tuple[torch.Tensor, torch.Tensor, torch.Tensor],          # (X_tr, y_sig_tr, y_ret_tr)
#     Tuple[torch.Tensor, torch.Tensor, torch.Tensor],          # (X_val, y_sig_val, y_ret_val)
#     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  
#                                                               # (X_te, y_sig_te, y_ret_te, raw_close_te)
#     list,                                                     # samples_per_day
#     torch.Tensor, torch.Tensor, torch.Tensor                  # day_id_tr, day_id_val, day_id_te
# ]:
#     """
#     Split time‐series windows into train/val/test by calendar day.

#     1) Count how many windows fall on each normalized date.
#     2) Decide how many days go to train/val/test by proportions
#        (training days rounded up to full batches of train_batch).
#     3) Compute cumulative sums of daily counts and slice X, y_sig, y_ret,
#        plus raw_close for the test set.
#     4) Build per‐window day_id tags for each split as CPU tensors
#        (so they can be moved to GPU per‐batch in a DataLoader).
#     """
#     # 1) Count windows per normalized day
#     dt_idx          = pd.to_datetime(end_times)
#     normed          = dt_idx.normalize()
#     days, counts    = np.unique(normed.values, return_counts=True)
#     samples_per_day = counts.tolist()

#     # sanity check: total windows equals first dim of X
#     total = sum(samples_per_day)
#     if total != X.size(0):
#         raise ValueError(f"Window count mismatch {total} vs {X.size(0)}")

#     # 2) Determine cut‐points in days
#     D             = len(samples_per_day)
#     orig_tr_days  = int(D * train_prop)
#     full_batches  = (orig_tr_days + train_batch - 1) // train_batch
#     tr_days       = min(D, full_batches * train_batch)
#     cut_train     = tr_days - 1
#     cut_val       = int(D * (train_prop + val_prop))

#     # 3) Slice by window counts
#     cumsum        = np.concatenate([[0], np.cumsum(counts)])
#     i_tr          = int(cumsum[tr_days])
#     i_val         = int(cumsum[cut_val + 1])

#     X_tr,  y_sig_tr,  y_ret_tr  = X[:i_tr],       y_sig[:i_tr],       y_ret[:i_tr]
#     X_val, y_sig_val, y_ret_val = X[i_tr:i_val],  y_sig[i_tr:i_val],  y_ret[i_tr:i_val]
#     X_te,  y_sig_te,  y_ret_te   = X[i_val:],     y_sig[i_val:],      y_ret[i_val:]
#     close_te = raw_close[i_val:]

#     # 4) Build day‐ID vectors on CPU
#     def make_day_ids(start_day: int, end_day: int) -> torch.Tensor:
#         # counts for days start_day…end_day inclusive
#         cnts = samples_per_day[start_day : end_day + 1]
#         # day indices  start_day, start_day+1, …, end_day
#         days_idx = torch.arange(start_day, end_day + 1, dtype=torch.long)
#         # repeat each day index by its count
#         return days_idx.repeat_interleave(torch.tensor(cnts, dtype=torch.long))

#     day_id_tr  = make_day_ids(0,          cut_train)
#     day_id_val = make_day_ids(cut_train+1, cut_val)
#     day_id_te  = make_day_ids(cut_val+1,  D - 1)

#     return (
#         (X_tr,  y_sig_tr,  y_ret_tr),
#         (X_val, y_sig_val, y_ret_val),
#         (X_te,  y_sig_te,  y_ret_te,  close_te),
#         samples_per_day,
#         day_id_tr, day_id_val, day_id_te
#     )


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
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],          
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],          
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  
    list,                                                    
    torch.Tensor, torch.Tensor, torch.Tensor                  
]:
    """
    Split flattened windows into train/val/test sets by calendar day.

    Functionality:
      1) Convert end_times to normalized calendar days and count windows per day.
      2) Determine how many days belong to train/val/test based on proportions
         (with train rounded up to full batches of train_batch).
      3) Compute cumulative window counts and slice X, y_sig, y_ret,
         and raw_close for the test set.
      4) Build per-window day_id tensors for each split, so DataLoader
         can move them to GPU alongside samples.

    Returns:
      (X_tr, y_sig_tr, y_ret_tr),
      (X_val, y_sig_val, y_ret_val),
      (X_te,  y_sig_te,  y_ret_te, raw_close_te),
      samples_per_day,
      day_id_tr, day_id_val, day_id_te
    """
    # 1) Count windows per normalized date
    dt_idx        = pd.to_datetime(end_times)
    normed        = dt_idx.normalize()
    days, counts  = np.unique(normed.values, return_counts=True)
    samples_per_day = counts.tolist()

    # Sanity check total windows matches tensor length
    total = sum(samples_per_day)
    if total != X.size(0):
        raise ValueError(f"Window count mismatch {total} vs {X.size(0)}")

    # 2) Determine day‐level cut points for train/val/test
    D            = len(samples_per_day)
    orig_tr_days = int(D * train_prop)
    full_batches = (orig_tr_days + train_batch - 1) // train_batch
    tr_days      = min(D, full_batches * train_batch)
    cut_train    = tr_days - 1
    cut_val      = int(D * (train_prop + val_prop))

    # 3) Slice by window counts
    cumsum      = np.concatenate([[0], np.cumsum(counts)])
    i_tr        = int(cumsum[tr_days])
    i_val       = int(cumsum[cut_val + 1])

    X_tr    = X[:i_tr];    y_sig_tr = y_sig[:i_tr];  y_ret_tr = y_ret[:i_tr]
    X_val   = X[i_tr:i_val]; y_sig_val = y_sig[i_tr:i_val]; y_ret_val = y_ret[i_tr:i_val]
    X_te    = X[i_val:];   y_sig_te = y_sig[i_val:];  y_ret_te = y_ret[i_val:]
    close_te = raw_close[i_val:]

    # 4) Build day_id tensors for each split
    def make_day_ids(start_day: int, end_day: int) -> torch.Tensor:
        cnts    = samples_per_day[start_day : end_day + 1]
        days_idx= torch.arange(start_day, end_day + 1, dtype=torch.long)
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


# class DayWindowDataset(Dataset):
#     """
#     Group sliding windows by calendar day and return per-day batches of:
#       • x_day     : input features, shape (1, W, look_back, F)
#       • y_day     : regression target (your precomputed signal), shape (1, W)
#       • y_sig_cls : binary label = 1 if signal > signal_thresh else 0
#       • ret_day   : true bar-to-bar returns, shape (1, W)
#       • y_ret_ter : ternary label ∈ {0,1,2} for (down, flat, up) based on ret_day and return_thresh
#       • rc        : optional raw close tensor, each shape (W,)
#       • weekday   : integer day-of-week
#       • end_ts    : timestamp of the last bar in the window
#     """
#     def __init__(
#         self,
#         X:              torch.Tensor,   # (N_windows, look_back, F)
#         y_signal:       torch.Tensor,   # (N_windows,)
#         y_return:       torch.Tensor,   # (N_windows,)
#         raw_close:      torch.Tensor,   # or None
#         end_times:      np.ndarray,     # (N_windows,), dtype datetime64[ns]
#         sess_start_time: time,          # cutoff for trading session
#         signal_thresh:  float,          # buy_threshold for y_sig_cls
#         return_thresh:  float           # dead-zone for up/down/flat
#     ):
#         self.signal_thresh = signal_thresh
#         self.return_thresh = return_thresh
#         self.has_raw       = raw_close is not None

#         # Filter windows by trading‐session start
#         valid = [
#             i for i, ts in enumerate(end_times)
#             if pd.Timestamp(ts).time() >= sess_start_time
#         ]
#         self.X          = X[valid]
#         self.y_signal   = y_signal[valid]
#         self.y_return   = y_return[valid]
#         self.end_times  = [pd.Timestamp(end_times[i]) for i in valid]

#         if self.has_raw:
#             self.raw_close = raw_close[valid]

#         # Build day‐boundaries for grouping windows by calendar date
#         dates        = pd.to_datetime(self.end_times).normalize()
#         days, counts = np.unique(dates.values, return_counts=True)
#         boundaries   = np.concatenate(([0], np.cumsum(counts)))
#         self.start   = torch.tensor(boundaries[:-1], dtype=torch.long)
#         self.end     = torch.tensor(boundaries[1:],  dtype=torch.long)
#         self.weekday = torch.tensor(
#             [d.dayofweek for d in pd.to_datetime(days)],
#             dtype=torch.long
#         )

#     def __len__(self):
#         return len(self.start)

#     def __getitem__(self, idx: int):
#         s, e    = self.start[idx].item(), self.end[idx].item()

#         # 1) inputs and regression target
#         x_day   = self.X[s:e].unsqueeze(0)            # (1, W, look_back, F)
#         y_day   = self.y_signal[s:e].unsqueeze(0)     # (1, W)

#         # 2) binary signal‐threshold label
#         y_sig_cls = (y_day > self.signal_thresh).float()

#         # 3) true returns + ternary return label
#         ret_day    = self.y_return[s:e].unsqueeze(0)  # (1, W)
#         # start all as “flat”=1
#         y_ret_ter  = torch.ones_like(ret_day, dtype=torch.long)
#         y_ret_ter[ret_day >  self.return_thresh] = 2  # “up”
#         y_ret_ter[ret_day < -self.return_thresh] = 0  # “down”

#         wd     = int(self.weekday[idx].item())
#         end_ts = self.end_times[e - 1]

#         if self.has_raw:
#             rc = self.raw_close[s:e]
#             return (
#                 x_day, y_day, y_sig_cls, ret_day, y_ret_ter, rc, wd, end_ts
#             )

#         return x_day, y_day, y_sig_cls, ret_day, y_ret_ter, wd, end_ts

class DayWindowDataset(Dataset):
    """
    Wrap pre-built sliding windows into per-day groups for DataLoader.

    Functionality:
      1) Accepts X, y_signal, y_return, optional raw_close and end_times—
         all already cut at session start.
      2) Groups windows by calendar date (using numpy datetime64).
      3) Computes start/end indices and weekday for each day.
      4) On __getitem__, returns:
         - x_day     : (1, W, look_back, F)
         - y_day     : (1, W)
         - y_sig_cls : binary labels from y_day > signal_thresh
         - ret_day   : (1, W) true returns
         - y_ret_ter : ternary labels from ret_day vs return_thresh
         - rc        : raw_close slice, if provided
         - weekday   : int day-of-week
         - end_ts    : numpy.datetime64 timestamp of last window
    """
    def __init__(
        self,
        X:              torch.Tensor,   # (N_windows, look_back, F)
        y_signal:       torch.Tensor,   # (N_windows,)
        y_return:       torch.Tensor,   # (N_windows,)
        raw_close:      torch.Tensor,   # or None
        end_times:      np.ndarray,     # (N_windows,), datetime64[ns]
        sess_start_time: time,          # unused: already filtered upstream
        signal_thresh:  float,
        return_thresh:  float
    ):
        self.signal_thresh = signal_thresh
        self.return_thresh = return_thresh
        self.has_raw       = raw_close is not None

        # 1) Store pre-filtered data buffers
        self.X         = X
        self.y_signal  = y_signal
        self.y_return  = y_return
        if self.has_raw:
            self.raw_close = raw_close
        self.end_times = end_times   # numpy datetime64[ns]

        # 2) Group by calendar day via numpy datetime64[D]
        days64 = end_times.astype("datetime64[D]")   # e.g. 2025-09-25
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
        # slice this day's windows
        s = self.start[idx].item()
        e = self.end[idx].item()

        x_day = self.X[s:e].unsqueeze(0)          # (1, W, look_back, F)
        y_day = self.y_signal[s:e].unsqueeze(0)   # (1, W)

        # binary signal label
        y_sig_cls = (y_day > self.signal_thresh).float()

        # true returns + ternary label
        ret_day   = self.y_return[s:e].unsqueeze(0)
        y_ret_ter = torch.ones_like(ret_day, dtype=torch.long)
        y_ret_ter[ret_day >  self.return_thresh] = 2  # up
        y_ret_ter[ret_day < -self.return_thresh] = 0  # down

        wd     = int(self.weekday[idx].item())
        end_ts = self.end_times[e - 1]  # numpy.datetime64[ns]

        if self.has_raw:
            rc = self.raw_close[s:e]
            return x_day, y_day, y_sig_cls, ret_day, y_ret_ter, rc, wd, end_ts

        return x_day, y_day, y_sig_cls, ret_day, y_ret_ter, wd, end_ts
        
#########################################################################################################


# def pad_collate(batch):
#     """
#     Pad a batch of per-day windows into fixed tensors and collect lengths.

#     Batch items are either train/val:
#       (x_day, y_day, y_sig_cls, ret_day, y_ret_ter, weekday, end_ts)
#     or test (has raw prices):
#       (x_day, y_day, y_sig_cls, ret_day, y_ret_ter,
#        rc, weekday, end_ts)

#     Returns:
#       x_pad      Tensor (B, max_W, look_back, F)
#       y_pad      Tensor (B, max_W)
#       y_sig_pad  Tensor (B, max_W)
#       ret_pad    Tensor (B, max_W)
#       y_ter_pad  LongTensor (B, max_W)
#       rc_pad  # only if has_raw
#       wd_tensor  LongTensor (B,)
#       ts_list    list of end_ts for each day
#       lengths    list[int] true window counts per day
#     """
#     has_raw = len(batch[0]) == 8

#     if has_raw:
#         (x_list, y_list, ysig_list, ret_list, yter_list, rc_list, wd_list, ts_list) = zip(*batch)
#     else:
#         x_list, y_list, ysig_list, ret_list, yter_list, wd_list, ts_list = zip(*batch)

#     # strip leading batch dim and collect sequences
#     xs    = [x.squeeze(0) for x in x_list]     # (W_i, look_back, F)
#     ys    = [y.squeeze(0) for y in y_list]     # (W_i,)
#     ysig  = [yc.squeeze(0) for yc in ysig_list]
#     rets  = [r.squeeze(0) for r in ret_list]
#     yter  = [t.squeeze(0) for t in yter_list]

#     lengths = [x.size(0) for x in xs]          # true W_i per day

#     # pad along time-axis
#     x_pad    = pad_sequence(xs,   batch_first=True)
#     y_pad    = pad_sequence(ys,   batch_first=True)
#     ysig_pad = pad_sequence(ysig, batch_first=True)
#     ret_pad  = pad_sequence(rets, batch_first=True)
#     yter_pad = pad_sequence(yter, batch_first=True)

#     wd_tensor = torch.tensor(wd_list, dtype=torch.long)

#     if has_raw:
#         rc_pad = pad_sequence(rc_list, batch_first=True)
#         return (
#             x_pad, y_pad, ysig_pad, ret_pad, yter_pad, rc_pad, wd_tensor, list(ts_list), lengths
#         )

#     return x_pad, y_pad, ysig_pad, ret_pad, yter_pad, wd_tensor, list(ts_list), lengths




def pad_collate(batch):
    """
    Pad variable-length per-day sequences into fixed tensors.

    Batch items (train/val):
      (x_day, y_day, y_sig_cls, ret_day, y_ret_ter, weekday, end_ts)
    or (test):
      (x_day, y_day, y_sig_cls, ret_day, y_ret_ter, rc, weekday, end_ts)

    Returns:
      x_pad   Tensor (B, max_W, look_back, F)
      y_pad   Tensor (B, max_W)
      ysig    Tensor (B, max_W)
      ret_pad Tensor (B, max_W)
      yter    LongTensor (B, max_W)
      rc_pad  Tensor (B, max_W) if has_raw
      wd      LongTensor (B,)
      ts_list list of end_ts per batch element
      lengths list of true window counts per day
    """
    has_raw = len(batch[0]) == 8

    if has_raw:
        x_list, y_list, ysig_list, ret_list, yter_list, rc_list, wd_list, ts_list = zip(*batch)
    else:
        x_list, y_list, ysig_list, ret_list, yter_list, wd_list, ts_list = zip(*batch)

    # remove leading batch dim, collect sequences
    xs   = [x.squeeze(0) for x in x_list]
    ys   = [y.squeeze(0) for y in y_list]
    ysig = [yc.squeeze(0) for yc in ysig_list]
    rets = [r.squeeze(0) for r in ret_list]
    yter = [t.squeeze(0) for t in yter_list]
    lengths = [seq.size(0) for seq in xs]

    # pad along time axis
    x_pad    = pad_sequence(xs,   batch_first=True)
    y_pad    = pad_sequence(ys,   batch_first=True)
    ysig_pad = pad_sequence(ysig, batch_first=True)
    ret_pad  = pad_sequence(rets, batch_first=True)
    yter_pad = pad_sequence(yter, batch_first=True)
    wd_tensor= torch.tensor(wd_list, dtype=torch.long)

    if has_raw:
        rc_pad = pad_sequence(rc_list, batch_first=True)
        return x_pad, y_pad, ysig_pad, ret_pad, yter_pad, rc_pad, wd_tensor, list(ts_list), lengths

    return x_pad, y_pad, ysig_pad, ret_pad, yter_pad, wd_tensor, list(ts_list), lengths
    
###################


# def split_to_day_datasets(
#     # train split tensors + times
#     X_tr,   y_sig_tr,  y_ret_tr,  end_times_tr,
#     # val split
#     X_val,  y_sig_val, y_ret_val, end_times_val,
#     # test split + raw-price arrays
#     X_te,   y_sig_te,  y_ret_te,  end_times_te,  raw_close_te,
#     *,
#     sess_start_time:       time,    # session cutoff
#     signal_thresh:         float,    # threshold for binary signal head
#     return_thresh:         float,    # dead-zone threshold for ternary return head
#     train_batch:           int = 32,
#     train_workers:         int = 0,
#     train_prefetch_factor: int = 1
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Build three DataLoaders that yield *per‐day* batches of the LSTM windows:
    
#       1) Instantiate DayWindowDataset for train, val, test.  
#          - train/val: raw_close = None  
#          - test:    raw_close = real price arrays  
#          Each dataset filters windows before sess_start_time and computes:
#            • y_sig_cls (binary from y_signal > signal_thresh)  
#            • y_ret_ter (ternary from y_return vs ±return_thresh)  

#       2) Wrap each dataset in a DataLoader:
#          - train_loader: batch_size=train_batch, shuffle=False, pad_collate,
#            num_workers=train_workers, pin_memory=True,
#            persistent_workers=(train_workers>0), prefetch_factor=...
#          - val_loader:   batch_size=1, shuffle=False, pad_collate
#          - test_loader:  batch_size=1, shuffle=False, pad_collate

#     Returns:
#       (train_loader, val_loader, test_loader)
#     """
#     # 1) Build the three DayWindowDatasets with a brief progress bar
#     splits = [
#         ("train", X_tr, y_sig_tr, y_ret_tr, end_times_tr, None),
#         ("val",   X_val, y_sig_val, y_ret_val, end_times_val,   None),
#         ("test",  X_te,  y_sig_te,  y_ret_te,  end_times_te,  raw_close_te)
#     ]

#     datasets = {}
#     for name, Xd, ys, yr, et, rc in tqdm(
#         splits, desc="Creating DayWindowDatasets", unit="split"
#     ):
#         datasets[name] = DayWindowDataset(
#             X=Xd,
#             y_signal=ys,
#             y_return=yr,
#             raw_close=rc,
#             end_times=et,
#             sess_start_time=sess_start_time,
#             signal_thresh=signal_thresh,
#             return_thresh=return_thresh
#         )

#     # 2) Wrap in DataLoaders
#     train_loader = DataLoader(
#         datasets["train"],
#         batch_size=train_batch,
#         shuffle=False,
#         drop_last=False,
#         collate_fn=pad_collate,
#         num_workers=train_workers,
#         pin_memory=True,
#         persistent_workers=(train_workers > 0),
#         prefetch_factor=(train_prefetch_factor if train_workers > 0 else None)
#     )

#     val_loader = DataLoader(
#         datasets["val"],
#         batch_size=1,
#         shuffle=False,
#         collate_fn=pad_collate,
#         num_workers=0,
#         pin_memory=True
#     )

#     test_loader = DataLoader(
#         datasets["test"],
#         batch_size=1,
#         shuffle=False,
#         collate_fn=pad_collate,
#         num_workers=0,
#         pin_memory=True
#     )

#     return train_loader, val_loader, test_loader




def split_to_day_datasets(
    X_tr, y_sig_tr, y_ret_tr, end_times_tr,
    X_val, y_sig_val, y_ret_val, end_times_val,
    X_te, y_sig_te, y_ret_te, end_times_te, raw_close_te,
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
      1) Build three DayWindowDataset objects—each already holds
         windows filtered by session start, grouped by calendar day,
         and applies binary/ternary thresholding on‐the‐fly.
      2) Wrap in DataLoader:
         - train: batch_size=train_batch, pad_collate, num_workers, pin_memory, prefetch
         - val/test: batch_size=1, pad_collate, single worker, pin_memory
    """
    splits = [
        ("train", X_tr, y_sig_tr, y_ret_tr, end_times_tr, None),
        ("val",   X_val, y_sig_val, y_ret_val, end_times_val,   None),
        ("test",  X_te,  y_sig_te,  y_ret_te,  end_times_te,  raw_close_te)
    ]

    datasets = {}
    for name, Xd, ys, yr, et, rc in tqdm(splits, desc="Creating DayWindowDatasets", unit="split"):
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


