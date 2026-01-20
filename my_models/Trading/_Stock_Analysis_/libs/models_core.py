from libs import plots, params

from typing import Sequence, List, Tuple, Optional, Union, Dict, Any
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
import matplotlib.pyplot as plt


#########################################################################################################


# def build_tensors(
#     df: pd.DataFrame,
#     look_back   = None,
#     sess_start = None,
#     *,
#     device     = torch.device("cpu"),
#     tmp_dir    = "/tmp/X_buf.dat",
#     thresh_gb  = params.thresh_gb,
# ) -> tuple[
#     torch.Tensor,  # X         shape=(N, look_back, F)
#     torch.Tensor,  # y_sig     shape=(N,)
#     torch.Tensor,  # y_ret     shape=(N,)
#     torch.Tensor,  # raw_close shape=(N,)
#     np.ndarray     # end_times shape=(N,) dtype=datetime64[ns]
# ]:
#     """
#     Build sliding-window tensors for an LSTM trading model (RAM-only, simple).

#     Behavior (complete)
#     - Select feature columns = df.columns minus params.label_col and "close_raw".
#     - First pass (per calendar day): extract per-day arrays (feats_np, sig_np, close_np),
#       compute per-bar log-returns safely, compute window-end timestamps aligned to
#       window ends (ends_np = index[look_back - 1:]) and a boolean mask for sess_start.
#       Record payloads only for days with mask.sum() > 0 and accumulate N_total.
#     - Allocate in-RAM numpy buffers sized by N_total and initialize to NaN/NaT so
#       unwritten slots are detectable.
#     - Second pass (parallel): produce sliding windows with
#       sliding_window_view(feats_np, window_shape=look_back, axis=0) which yields
#       (T - look_back + 1, look_back, F). Write wins[mask] into X_buf; writes are
#       guarded to accept dst layout (m, look_back, F) or (m, F, look_back) by transposing
#       when necessary. Progress updates happen on the main thread.
#     - Return CPU torch tensors (do not move to GPU here). Keep naming/variables unchanged.
#     """
#     # 1) Prepare DataFrame & feature list
#     df = df.copy()
#     feature_cols = [c for c in df.columns if c not in (params.label_col, "close_raw")]
#     F = len(feature_cols)

#     # session start -> seconds
#     cutoff_sec = sess_start.hour * 3600 + sess_start.minute * 60

#     # 2) First pass: per-day payloads and N_total
#     day_groups = df.groupby(df.index.normalize(), sort=False)
#     payloads = []
#     N_total = 0

#     for _, day_df in tqdm(day_groups, desc="Preparing days"):
#         day_df = day_df.sort_index()
#         T = len(day_df)
#         if T <= look_back:
#             continue

#         feats_np = day_df[feature_cols].to_numpy(np.float32)      # (T, F)
#         sig_np   = day_df[params.label_col].to_numpy(np.float32)  # (T,)
#         close_np = day_df["close_raw"].to_numpy(np.float32)       # (T,)

#         # safe log-returns; replace non-finite with 0
#         ret_full = np.empty_like(close_np, np.float32)
#         ret_full[0] = 0.0
#         with np.errstate(divide="ignore", invalid="ignore"):
#             ret_full[1:] = np.log(close_np[1:] / close_np[:-1])
#         bad = ~np.isfinite(ret_full)
#         if bad.any():
#             ret_full[bad] = 0.0

#         # window-end alignment: window [i : i+look_back] ends at i+look_back-1
#         ends_np = day_df.index.to_numpy()[look_back - 1:]        # (T - look_back + 1,)
#         secs = (ends_np - ends_np.astype("datetime64[D]")) / np.timedelta64(1, "s")
#         mask = secs >= cutoff_sec                                # (T - look_back + 1,)

#         m = int(mask.sum())
#         if m == 0:
#             continue

#         sig_end   = sig_np[look_back - 1:]
#         ret_end   = ret_full[look_back - 1:]
#         close_end = close_np[look_back - 1:]
#         payloads.append((feats_np, sig_end, ret_end, close_end, ends_np, mask, N_total))
#         N_total += m

#     # sanity and prints
#     assert N_total == sum(int(p[5].sum()) for p in payloads), "N_total mismatch after payload accumulation"
#     print("N_total:", N_total, "look_back:", look_back, "F:", F)
#     est_bytes = int(N_total) * int(look_back) * int(F) * 4
#     est_gb = est_bytes / (1024**3)
#     print(f"Estimated X_buf size: {est_bytes/1e9:.2f} GB — {'using memmap' if est_gb > thresh_gb else 'using RAM (in-memory)'} (thresh {thresh_gb} GiB)")

#     # 3) Allocate RAM buffers or memmap and initialize to NaN/NaT
#     if est_gb > thresh_gb:
#         print('initializing mmap...')
#         X_buf = np.memmap(tmp_dir, dtype=np.float32, mode="w+", shape=(N_total, look_back, F)); X_buf[:] = np.nan; X_buf.flush()
#     else:
#         X_buf = np.full((N_total, look_back, F), np.nan, dtype=np.float32)

#     y_buf = np.full((N_total,), np.nan, dtype=np.float32)
#     r_buf = np.full((N_total,), np.nan, dtype=np.float32)
#     c_buf = np.full((N_total,), np.nan, dtype=np.float32)
#     t_buf = np.full((N_total,), np.datetime64("NaT"), dtype="datetime64[ns]")

#     # 4) Second pass: build windows and write
#     pbar = tqdm(total=len(payloads), desc="Writing days")

#     def _write_np(payload):
#         feats_np, sig_end, ret_end, close_end, ends_np, mask, offset = payload
    
#         wins = np.lib.stride_tricks.sliding_window_view(feats_np, window_shape=look_back, axis=0)
#         assert wins.shape[0] == len(ends_np), "wins vs ends_np length mismatch"
#         assert mask.shape[0] == wins.shape[0], "mask length mismatch"
    
#         m = int(mask.sum())
#         if m == 0:
#             return None
    
#         wins_sel = wins[mask]               # usually (m, look_back, F)
#         dst = X_buf[offset : offset + m]    # expected (m, look_back, F)
    
#         # Short, explicit, no try/except: accept exact match or swapped feature/time axes
#         if dst.shape == wins_sel.shape:
#             dst[:] = wins_sel
#         elif dst.shape == (wins_sel.shape[0], wins_sel.shape[2], wins_sel.shape[1]):
#             dst[:] = wins_sel.transpose(0, 2, 1)
#         else:
#             raise ValueError(f"Axis mismatch writing wins -> X_buf: dst={dst.shape}, wins={wins_sel.shape}")
    
#         y_buf[offset : offset + m] = sig_end[mask]
#         r_buf[offset : offset + m] = ret_end[mask]
#         c_buf[offset : offset + m] = close_end[mask]
#         t_buf[offset : offset + m] = ends_np[mask]
    
#         return None

#     with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as exe:
#         for _ in exe.map(_write_np, payloads):
#             pbar.update(1)
#     pbar.close()

#     # 5) Wrap buffers as CPU torch tensors (do not move to GPU here)
#     X         = torch.from_numpy(X_buf)
#     y_sig     = torch.from_numpy(y_buf)
#     y_ret     = torch.from_numpy(r_buf)
#     raw_close = torch.from_numpy(c_buf)
#     end_times = t_buf.copy()

#     # 6) Light cleanup
#     gc.collect()
#     if device.type == "cuda":
#         torch.cuda.empty_cache()
    
#     # Defensive: fail if any target entries are non-finite
#     if not np.isfinite(y_buf).all():
#         bad_idx = np.where(~np.isfinite(y_buf))[0]
#         raise RuntimeError(f"build_tensors: non-finite y_sig at positions {bad_idx[:16].tolist()} (total {len(bad_idx)})")

#     return X, y_sig, y_ret, raw_close, end_times


def build_tensors(
    df,
    look_back,
    features_cols,
    *,
    device     = torch.device("cpu"),
    tmp_dir    = "/tmp/X_buf.dat",
    thresh_gb  = params.thresh_gb,
) -> tuple[
    torch.Tensor,  # X         shape=(N, look_back, F)
    torch.Tensor,  # y_sig     shape=(N,)
    torch.Tensor,  # raw_close shape=(N,)
    np.ndarray     # end_times shape=(N,) dtype=datetime64[ns]
]:
    """
    Build sliding-window tensors for an LSTM trading model.

    - Two-pass per-day builder:
        1) scan each day to collect per-day arrays and compute total windows (N_total);
        2) allocate buffers (RAM or memmap) sized to N_total and write sliding windows in parallel.
    - Produces X (N, look_back, F), y_sig, raw_close and end_times.
    - Windows are aligned to their end positions (index[look_back-1:]) and all window-ends are kept.
    """
    # number of features
    F = len(features_cols)

    # 1) First pass: collect per-day payloads and compute N_total
    day_groups = df.groupby(df.index.normalize(), sort=False)
    payloads = []
    N_total = 0

    for _, day_df in tqdm(day_groups, desc="Preparing days"):
        day_df = day_df.sort_index()
        T = len(day_df)
        if T <= look_back:
            continue  # not enough bars to form a single window

        # per-day arrays
        feats_np = np.ascontiguousarray(day_df[features_cols].to_numpy(np.float32))  # (T, F) C-contiguous
        sig_np   = day_df["signal_raw"].to_numpy(np.float32)   # (T,)
        close_np = day_df["close_raw"].to_numpy(np.float32)        # (T,)

        # window-end alignment: windows end at indices look_back-1 .. T-1
        ends_np = day_df.index.to_numpy()[look_back - 1:]           # (T - look_back + 1,)
        m = len(ends_np)                                            # number of windows for this day
        if m == 0:
            continue

        # end-aligned slices
        sig_end   = sig_np[look_back - 1:]
        close_end = close_np[look_back - 1:]

        # payload: per-day arrays and write offset (no mask)
        payloads.append((feats_np, sig_end, close_end, ends_np, N_total))
        N_total += m

    # summary and size estimate
    print("N_total:", N_total, "look_back:", look_back, "F:", F)
    est_bytes = int(N_total) * int(look_back) * int(F) * 4
    est_gb = est_bytes / (1024**3)
    print(f"Estimated X_buf size: {est_bytes/1e9:.2f} GB — {'using memmap' if est_gb > thresh_gb else 'using RAM (in-memory)'} (thresh {thresh_gb} GiB)")

    # 3) Allocate buffers (memmap if large) and initialize to NaN/NaT
    if est_gb > thresh_gb:
        print('initializing mmap...')
        X_buf = np.memmap(tmp_dir, dtype=np.float32, mode="w+", shape=(N_total, look_back, F))
        X_buf[:] = np.nan
        X_buf.flush()
    else:
        X_buf = np.full((N_total, look_back, F), np.nan, dtype=np.float32)

    y_buf = np.full((N_total,), np.nan, dtype=np.float32)
    r_buf = np.full((N_total,), np.nan, dtype=np.float32)
    c_buf = np.full((N_total,), np.nan, dtype=np.float32)
    t_buf = np.full((N_total,), np.datetime64("NaT"), dtype="datetime64[ns]")

    # 4) Second pass: build sliding windows and write contiguous blocks
    pbar = tqdm(total=len(payloads), desc="Writing days")

    def _write_np(payload):
        feats_np, sig_end, close_end, ends_np, offset = payload

        # build explicit windows with canonical layout (m, look_back, F)
        m = feats_np.shape[0] - look_back + 1
        if m <= 0:
            return None
        wins = np.stack([feats_np[i : i + look_back] for i in range(m)], axis=0)  # (m, look_back, F)
        dst = X_buf[offset : offset + m]
        dst[:] = wins

        # write targets and end-times (contiguous slices)
        y_buf[offset : offset + m] = sig_end[:m]
        c_buf[offset : offset + m] = close_end[:m]
        t_buf[offset : offset + m] = ends_np[:m]

        return None

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as exe:
        for _ in exe.map(_write_np, payloads):
            pbar.update(1)
    pbar.close()

    # 5) Wrap buffers as CPU torch tensors (do not move to GPU here)
    X         = torch.from_numpy(X_buf)
    y_sig     = torch.from_numpy(y_buf)
    raw_close = torch.from_numpy(c_buf)
    end_times = t_buf.copy()

    # 6) Light cleanup and non-finite reporting
    gc.collect()
    if not np.isfinite(y_buf).all():
        bad_idx = np.where(~np.isfinite(y_buf))[0]
        print(f"build_tensors: non-finite y_sig at positions {bad_idx[:16].tolist()} (total {len(bad_idx)})")

    return X, y_sig, raw_close, end_times



#########################################################################################################


# def chronological_split(
#     X:           torch.Tensor,
#     y_sig:       torch.Tensor,
#     y_ret:       torch.Tensor,
#     raw_close:   torch.Tensor,
#     end_times:   np.ndarray,      # (N,), dtype datetime64[ns]
#     *,
#     train_batch: int,
#     device       = torch.device("cpu")
# ) -> Tuple[
#     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],          
#     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],          
#     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  
#     list,                                                    
#     torch.Tensor, torch.Tensor, torch.Tensor                  
# ]:
#     """
#     Split flattened windows into train/val/test sets by calendar day,
#     returning raw_close for every split so downstream __getitem__
#     always has a tensor.

#     Functionality:
#       1) Normalize end_times to per-day bins and count windows per calendar day.
#       2) Compute how many days go to train/val/test (train rounded to full batches).
#       3) Build cumulative window‐count array, then derive slice indices i_tr, i_val.
#       4) Slice X, y_sig, y_ret, raw_close into:
#          - train  quadruple (X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr)
#          - val    quadruple (X_val, y_sig_val, y_ret_val, raw_close_val)
#          - test   quadruple (X_te,  y_sig_te,  y_ret_te,  raw_close_te)
#       5) Build per-window day‐id tensors for each split for GPU collation.
#       6) Return the three splits, samples_per_day list, and day_id_tr/val/te.
#     """
#     # 1) Count windows per normalized calendar day
#     dates_norm = pd.to_datetime(end_times).normalize().values
#     days, counts = np.unique(dates_norm, return_counts=True)
#     samples_per_day = counts.tolist()

#     # Sanity check total windows matches tensor length
#     total = sum(samples_per_day)
#     if total != X.size(0):
#         raise ValueError(f"Window count mismatch {total} vs {X.size(0)}")

#     # 2) Determine day‐level cut points for train/val/test
#     D             = len(samples_per_day)
#     orig_tr_days  = int(D * params.train_prop)
#     full_batches  = (orig_tr_days + train_batch - 1) // train_batch
#     tr_days       = min(D, full_batches * train_batch)
#     cut_train = max(0, tr_days - 1)
#     cut_val = min(D - 1, int(D * (params.train_prop + params.val_prop)))
    
#     # build cumsum then compute i_tr/i_val safely
#     cumsum = np.concatenate([[0], np.cumsum(counts)])
#     i_tr = int(cumsum[min(tr_days, D)])
#     i_val = int(cumsum[min(cut_val + 1, D)])
#     # sanity
#     assert 0 <= i_tr <= i_val <= X.size(0)

#     # 4) Slice into train/val/test (each gets raw_close slice)
#     X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr  = (
#         X[:i_tr],    y_sig[:i_tr],   y_ret[:i_tr],   raw_close[:i_tr]
#     )
#     X_val, y_sig_val, y_ret_val, raw_close_val = (
#         X[i_tr:i_val], y_sig[i_tr:i_val], y_ret[i_tr:i_val], raw_close[i_tr:i_val]
#     )
#     X_te,  y_sig_te,  y_ret_te,  raw_close_te  = (
#         X[i_val:],    y_sig[i_val:],   y_ret[i_val:],   raw_close[i_val:]
#     )

#     # 5) Build per-window day_id tensors for grouping
#     def make_day_ids(start_day: int, end_day: int) -> torch.Tensor:
#         # repeat each day index by its window count
#         day_counts = samples_per_day[start_day : end_day + 1]
#         day_idxs   = torch.arange(start_day, end_day + 1, dtype=torch.long)
#         return day_idxs.repeat_interleave(torch.tensor(day_counts, dtype=torch.long))

#     day_id_tr  = make_day_ids(0,          cut_train)
#     day_id_val = make_day_ids(cut_train+1, cut_val)
#     day_id_te  = make_day_ids(cut_val+1,  D - 1)

#     # 6) Return splits as 4-tuples + metadata
#     return (
#         (X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr),
#         (X_val, y_sig_val, y_ret_val, raw_close_val),
#         (X_te,  y_sig_te,  y_ret_te,  raw_close_te),
#         samples_per_day,
#         day_id_tr, day_id_val, day_id_te
#     )


def chronological_split(
    X:           torch.Tensor,
    y_sig:       torch.Tensor,
    raw_close:   torch.Tensor,
    end_times:   np.ndarray,      # (N,), dtype datetime64[ns]
    *,
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
      4) Slice X, y_sig, raw_close
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
    orig_tr_days  = int(D * params.train_prop)
    full_batches  = (orig_tr_days + train_batch - 1) // train_batch
    tr_days       = min(D, full_batches * train_batch)
    cut_train     = max(0, tr_days - 1)
    cut_val       = min(D - 1, int(D * (params.train_prop + params.val_prop)))
    
    # build cumsum then compute i_tr/i_val safely
    cumsum = np.concatenate([[0], np.cumsum(counts)])
    i_tr = int(cumsum[min(tr_days, D)])
    i_val = int(cumsum[min(cut_val + 1, D)])
    # sanity
    assert 0 <= i_tr <= i_val <= X.size(0)

    # 4) Slice into train/val/test (each gets raw_close slice)
    X_tr,  y_sig_tr, raw_close_tr  = (
        X[:i_tr],    y_sig[:i_tr],   raw_close[:i_tr]
    )
    X_val, y_sig_val, raw_close_val = (
        X[i_tr:i_val], y_sig[i_tr:i_val], raw_close[i_tr:i_val]
    )
    X_te,  y_sig_te,  raw_close_te  = (
        X[i_val:],    y_sig[i_val:],   raw_close[i_val:]
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

    # 6) Return splits as tuples + metadata
    return (
        (X_tr,  y_sig_tr,  raw_close_tr),
        (X_val, y_sig_val, raw_close_val),
        (X_te,  y_sig_te,  raw_close_te),
        samples_per_day,
        day_id_tr, day_id_val, day_id_te
    ) 

    
#########################################################################################################


# class DayWindowDataset(Dataset):
#     """
#     Return per-day tensors for the requested calendar day index.
    
#     Functionality
#     - Slices the precomputed sliding-window buffers for the day defined by idx and
#       returns the day's windows and labels without an extra leading day dimension.
#     - Produces per-day shapes that pad_collate expects:
#       - x        -> Tensor[W, T, F]       windows for that calendar day
#       - y_sig    -> Tensor[W]             per-window scalar targets
#       - y_ret    -> Tensor[W]             per-window returns
#       # - y_ret_ter-> LongTensor[W]         ternary labels computed from return_thresh
#       - rc       -> Tensor[W]             raw_close slice aligned to windows
#       - wd       -> int                   weekday index for the day
#       - end_ts   -> numpy.datetime64      timestamp of last window for the day
#     """
#     def __init__(
#         self,
#         X:               torch.Tensor,   # (N_windows, look_back, F)
#         y_signal:        torch.Tensor,   # (N_windows,)
#         # y_return:        torch.Tensor,   # (N_windows,)
#         raw_close:       torch.Tensor,   # (N_windows,)
#         end_times:       np.ndarray,     # (N_windows,), datetime64[ns]
#         # return_thresh:   float
#     ):
#         # self.return_thresh = return_thresh

#         # 1) Store all buffers (raw_close always provided now)
#         self.X         = X
#         self.y_signal  = y_signal
#         # self.y_return  = y_return
#         self.raw_close = raw_close
#         self.end_times = end_times   # numpy.datetime64[ns]

#         # 2) Group windows by calendar day
#         days64 = end_times.astype("datetime64[D]")       # e.g. 2025-09-25
#         days, counts = np.unique(days64, return_counts=True)
#         boundaries = np.concatenate(([0], np.cumsum(counts)))

#         # 3) Build start/end indices and weekday tensor
#         self.start   = torch.tensor(boundaries[:-1], dtype=torch.long)
#         self.end     = torch.tensor(boundaries[1:],  dtype=torch.long)
#         weekdays     = pd.to_datetime(days).dayofweek
#         self.weekday = torch.tensor(weekdays, dtype=torch.long)

#     def __len__(self):
#         return len(self.start)

#     def __getitem__(self, idx: int):
#         # Determine slice indices for this day
#         s = self.start[idx].item()
#         e = self.end[idx].item()
    
#         # 4) Slice out windows (no leading day dim)
#         x     = self.X[s:e]           # (W, look_back, F)
#         y_sig = self.y_signal[s:e]    # (W,)
    
#         # True returns + ternary label
#         # y_ret     = self.y_return[s:e]                     # (W,)
#         # y_ret_ter = torch.ones_like(y_ret, dtype=torch.long)
#         # y_ret_ter[y_ret >  self.return_thresh] = 2
#         # y_ret_ter[y_ret < -self.return_thresh] = 0
    
#         # Extract raw_close slice (length W, no leading dim)
#         rc = self.raw_close[s:e]   # (W,)
    
#         # Weekday index and last-window timestamp
#         wd     = int(self.weekday[idx].item())
#         end_ts = self.end_times[e - 1]  # numpy.datetime64[ns]
    
#         # Return the tuple with simplified shapes
#         return x, y_sig, y_ret, y_ret_ter, rc, wd, end_ts


class DayWindowDataset(Dataset):
    """
    Return per-day tensors for the requested calendar day index.
    
    Functionality
    - Slices the precomputed sliding-window buffers for the day defined by idx and
      returns the day's windows and labels without an extra leading day dimension.
    - Produces per-day shapes that pad_collate expects:
      - x        -> Tensor[W, T, F]       windows for that calendar day
      - y_sig    -> Tensor[W]             per-window scalar targets
      - rc       -> Tensor[W]             raw_close slice aligned to windows
      - wd       -> int                   weekday index for the day
      - end_ts   -> numpy.datetime64      timestamp of last window for the day
    """
    def __init__(
        self,
        X:               torch.Tensor,   # (N_windows, look_back, F)
        y_signal:        torch.Tensor,   # (N_windows,)
        raw_close:       torch.Tensor,   # (N_windows,)
        end_times:       np.ndarray,     # (N_windows,), datetime64[ns]
    ):
        # self.return_thresh = return_thresh

        # 1) Store all buffers (raw_close always provided now)
        self.X         = X
        self.y_signal  = y_signal
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
    
        # 4) Slice out windows (no leading day dim)
        x     = self.X[s:e]           # (W, look_back, F)
        y_sig = self.y_signal[s:e]    # (W,)
    
        # Extract raw_close slice (length W, no leading dim)
        rc = self.raw_close[s:e]   # (W,)
    
        # Weekday index and last-window timestamp
        wd     = int(self.weekday[idx].item())
        end_ts = self.end_times[e - 1]  # numpy.datetime64[ns]
    
        # Return the tuple with simplified shapes
        return x, y_sig, rc, wd, end_ts

        
######################


# def pad_collate(batch):
#     """
#     Pad and flatten a batch of per-day examples into canonical per-window tensors.

#     Args
#     - batch: iterable of DayWindowDataset items, 
#       where x is expected to be per-day windows shaped (W, T, F).

#     Returns
#     - x_flat     Tensor[N, T, F]    flattened windows (N = B * W_max)
#     - ysig_flat  Tensor[N]          flattened per-window scalar targets
#     - yret_flat  Tensor[N]          flattened per-window returns
#     - yter_flat  LongTensor[N]      flattened per-window ternary labels
#     - rc_flat    Tensor[N]          flattened per-window raw_close values
#     - wd_per_day list[int]          per-day weekday ints (length B)
#     - ts_list    list               per-day end timestamps 
#     - lengths    list[int]          true window counts per day 
#     """
#     # Unpack tuple structure
#     x_list, ysig_list, yret_list, yter_list, rc_list, wd_list, ts_list = zip(*batch)

#     xs      = list(x_list)        # expect (W, T, F)
#     ysig    = list(ysig_list)     # expect (W,)
#     yrets   = list(yret_list)
#     yter    = list(yter_list)
#     rc_seq  = list(rc_list)       # expect (W,)

#     lengths = [seq.size(0) for seq in xs]  # per-day window counts W_i

#     # Pad per-day along the window axis -> shapes (B, W_max, ...)
#     x_pad    = pad_sequence(xs,   batch_first=True)  # (B, W_max, T, F)
#     ysig_pad = pad_sequence(ysig,   batch_first=True) # (B, W_max)
#     yret_pad = pad_sequence(yrets, batch_first=True)  # (B, W_max)
#     yter_pad = pad_sequence(yter, batch_first=True)   # (B, W_max)
#     rc_pad   = pad_sequence(rc_seq, batch_first=True) # (B, W_max)

#     # Weekday per-day (B,)
#     wd_tensor = torch.tensor(wd_list, dtype=torch.long)

#     # Flatten first two dims to canonical per-window shapes (N = B * W_max)
#     B, W_max = ysig_pad.shape[0], ysig_pad.shape[1]
#     T = x_pad.shape[2]
#     F = x_pad.shape[3]

#     x_flat    = x_pad.contiguous().view(B * W_max, T, F)   # (N, T, F)
#     ysig_flat = ysig_pad.contiguous().view(B * W_max)      # (N,)
#     yret_flat = yret_pad.contiguous().view(B * W_max)      # (N,)
#     yter_flat = yter_pad.contiguous().view(B * W_max)      # (N,)
#     rc_flat   = rc_pad.contiguous().view(B * W_max)        # (N,)

#     # Keep per-day weekday vector (B,) for state resets; callers use lengths to align windows
#     wd_per_day = wd_tensor.tolist()  # list[int] length B

#     return x_flat, ysig_flat, yret_flat, yter_flat, rc_flat, wd_per_day, list(ts_list), lengths


def pad_collate(batch):
    """
    Pad and flatten a batch of per-day examples into canonical per-window tensors.

    Args
    - batch: iterable of DayWindowDataset items, 
      where x is expected to be per-day windows shaped (W, T, F).

    Returns
    - x_flat     Tensor[N, T, F]    flattened windows (N = B * W_max)
    - ysig_flat  Tensor[N]          flattened per-window scalar targets
    - rc_flat    Tensor[N]          flattened per-window raw_close values
    - wd_per_day list[int]          per-day weekday ints (length B)
    - ts_list    list               per-day end timestamps 
    - lengths    list[int]          true window counts per day 
    """
    # Unpack tuple structure
    x_list, ysig_list, rc_list, wd_list, ts_list = zip(*batch)

    xs      = list(x_list)        # expect (W, T, F)
    ysig    = list(ysig_list)     # expect (W,)
    rc_seq  = list(rc_list)       # expect (W,)

    lengths = [seq.size(0) for seq in xs]  # per-day window counts W_i

    # Pad per-day along the window axis -> shapes (B, W_max, ...)
    x_pad    = pad_sequence(xs,   batch_first=True)  # (B, W_max, T, F)
    ysig_pad = pad_sequence(ysig,   batch_first=True) # (B, W_max)
    rc_pad   = pad_sequence(rc_seq, batch_first=True) # (B, W_max)

    # Weekday per-day (B,)
    wd_tensor = torch.tensor(wd_list, dtype=torch.long)

    # Flatten first two dims to canonical per-window shapes (N = B * W_max)
    B, W_max = ysig_pad.shape[0], ysig_pad.shape[1]
    T = x_pad.shape[2]
    F = x_pad.shape[3]

    x_flat    = x_pad.contiguous().view(B * W_max, T, F)   # (N, T, F)
    ysig_flat = ysig_pad.contiguous().view(B * W_max)      # (N,)
    rc_flat   = rc_pad.contiguous().view(B * W_max)        # (N,)

    # Keep per-day weekday vector (B,) for state resets; callers use lengths to align windows
    wd_per_day = wd_tensor.tolist()  # list[int] length B

    return x_flat, ysig_flat, rc_flat, wd_per_day, list(ts_list), lengths

    
###############


# def split_to_day_datasets(
#     X_tr, y_sig_tr, y_ret_tr, raw_close_tr, end_times_tr,
#     X_val, y_sig_val, y_ret_val, raw_close_val, end_times_val,
#     X_te,  y_sig_te,  y_ret_te,  raw_close_te,  end_times_te,
#     *,
#     # return_thresh:         float,
#     train_batch:           int = 32,
#     train_workers:         int = 0,
#     train_prefetch_factor: int = 1
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Instantiate DayWindowDataset for train/val/test and wrap into DataLoaders.

#     Functionality:
#       1) Build three DayWindowDataset objects, each receiving raw_close tensor:
#          - train set gets raw_close_tr
#          - val   set gets raw_close_val
#          - test  set gets raw_close_te
#       2) Wrap each dataset in a DataLoader using pad_collate:
#          - train: batch_size=train_batch, num_workers=train_workers, prefetch_factor.
#          - val & test: batch_size=1, num_workers=0.
#       This ensures __getitem__ always sees a real raw_close tensor, never None.
#     """
#     splits = [
#         ("train", X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr,  end_times_tr),
#         ("val",   X_val, y_sig_val, y_ret_val, raw_close_val, end_times_val),
#         ("test",  X_te,  y_sig_te,  y_ret_te,  raw_close_te,  end_times_te),
#     ]

#     datasets = {}
#     for name, Xd, ys, yr, rc, et in tqdm(splits, desc="Creating DayWindowDatasets", unit="split"):
#         datasets[name] = DayWindowDataset(
#             X             = Xd,
#             y_signal      = ys,
#             raw_close     = rc,  
#             end_times     = et,
#         )

#     # Train loader: padded multi-day batches
#     train_loader = DataLoader(
#         datasets["train"],
#         batch_size           = train_batch,
#         shuffle              = False,
#         drop_last            = False,
#         collate_fn           = pad_collate,
#         num_workers          = train_workers,
#         pin_memory           = True,
#         persistent_workers   = (train_workers > 0),
#         prefetch_factor      = (train_prefetch_factor if train_workers > 0 else None),
#     )

#     # Validation loader: single-day batches
#     val_loader = DataLoader(
#         datasets["val"],
#         batch_size = 1,
#         shuffle    = False,
#         collate_fn = pad_collate,
#         num_workers= 0,
#         pin_memory = True,
#     )

#     # Test loader: single-day batches
#     test_loader = DataLoader(
#         datasets["test"],
#         batch_size = 1,
#         shuffle    = False,
#         collate_fn = pad_collate,
#         num_workers= 0,
#         pin_memory = True,
#     )

#     return train_loader, val_loader, test_loader


def split_to_day_datasets(
    X_tr, y_sig_tr, raw_close_tr, end_times_tr,
    X_val, y_sig_val, raw_close_val, end_times_val,
    X_te,  y_sig_te,  raw_close_te,  end_times_te,
    *,
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
        ("train", X_tr,  y_sig_tr,  raw_close_tr,  end_times_tr),
        ("val",   X_val, y_sig_val, raw_close_val, end_times_val),
        ("test",  X_te,  y_sig_te,  raw_close_te,  end_times_te),
    ]

    datasets = {}
    for name, Xd, ys, rc, et in tqdm(splits, desc="Creating DayWindowDatasets", unit="split"):
        datasets[name] = DayWindowDataset(
            X             = Xd,
            y_signal      = ys,
            raw_close     = rc,  
            end_times     = et,
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


# def model_core_pipeline(
#     df,                          # feature‐enriched DataFrame
#     look_back: int,              # how many ticks per window
#     train_batch: int,            # batch size for training
#     train_workers: int,          # DataLoader worker count
#     prefetch_factor: int,        # DataLoader prefetch_factor
#     features_cols: list,         # Kept features columns
# ) -> tuple:
#     """
#     Build DataLoaders end‐to‐end from raw df.

#     Steps & parameters:
#       1) build_tensors(df, look_back)
#       2) chronological_split(..., train_prop, val_prop, train_batch)
#       3) carve end_times into train/val/test
#       4) split_to_day_datasets(...)
#       cleans up all intermediate arrays before returning.
#     """
#     # 1) slide‐window tensorization
#     X, y_sig, y_ret, raw_close, end_times = build_tensors(
#         df            = df,
#         look_back     = look_back,
#         features_cols = features_cols
#     )

#     # 2) split into train/val/test by calendar day
#     (train_split, val_split, test_split,
#      samples_per_day,
#      day_id_tr, day_id_val, day_id_te) = chronological_split(
#          X, y_sig, y_ret, raw_close,
#          end_times   = end_times,
#          train_batch = train_batch
#     )
#     X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr  = train_split
#     X_val, y_sig_val, y_ret_val, raw_close_val = val_split
#     X_te,  y_sig_te,  y_ret_te,  raw_close_te  = test_split

#     # 3) carve end_times in the same proportions
#     n_tr  = day_id_tr .shape[0]
#     n_val = day_id_val.shape[0]
#     i_tr, i_val = n_tr, n_tr + n_val

#     end_times_tr  = end_times[:i_tr]
#     end_times_val = end_times[i_tr:i_val]
#     end_times_te  = end_times[i_val:]

#     # 4) DataLoader construction
#     train_loader, val_loader, test_loader = split_to_day_datasets(
#         # train split
#         X_tr,  y_sig_tr,  y_ret_tr,  raw_close_tr,  end_times_tr,
#         # val split
#         X_val, y_sig_val, y_ret_val, raw_close_val, end_times_val,
#         # test split
#         X_te,  y_sig_te,  y_ret_te,  raw_close_te,  end_times_te,

#         # return_thresh         = return_thresh,
#         train_batch           = train_batch,
#         train_workers         = train_workers,
#         train_prefetch_factor = prefetch_factor
#     )

#     # clean up large intermediates to free memory
#     del X, y_sig, y_ret, raw_close, end_times
#     del X_tr, y_sig_tr, y_ret_tr, raw_close_tr
#     del X_val, y_sig_val, y_ret_val, raw_close_val
#     del X_te, y_sig_te, y_ret_te, raw_close_te

#     return train_loader, val_loader, test_loader, end_times_tr, end_times_val, end_times_te


def model_core_pipeline(
    df,                          # feature‐enriched DataFrame
    look_back: int,              # how many ticks per window
    train_batch: int,            # batch size for training
    train_workers: int,          # DataLoader worker count
    prefetch_factor: int,        # DataLoader prefetch_factor
    features_cols: list,         # Kept features columns
) -> tuple:
    """
    Build DataLoaders end‐to‐end from raw df.

    Steps & parameters:
      1) build_tensors(df, look_back)
      2) chronological_split(..., train_prop, val_prop, train_batch)
      3) carve end_times into train/val/test
      4) split_to_day_datasets(...)
      cleans up all intermediate arrays before returning.
    """
    # 1) slide‐window tensorization
    X, y_sig, raw_close, end_times = build_tensors(
        df            = df,
        look_back     = look_back,
        features_cols = features_cols
    )

    # 2) split into train/val/test by calendar day
    (train_split, val_split, test_split,
     samples_per_day,
     day_id_tr, day_id_val, day_id_te) = chronological_split(
         X, y_sig, raw_close,
         end_times   = end_times,
         train_batch = train_batch
    )
    X_tr,  y_sig_tr,  raw_close_tr  = train_split
    X_val, y_sig_val, raw_close_val = val_split
    X_te,  y_sig_te,  raw_close_te  = test_split

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
        X_tr,  y_sig_tr,  raw_close_tr,  end_times_tr,
        # val split
        X_val, y_sig_val, raw_close_val, end_times_val,
        # test split
        X_te,  y_sig_te,  raw_close_te,  end_times_te,

        train_batch           = train_batch,
        train_workers         = train_workers,
        train_prefetch_factor = prefetch_factor
    )

    # clean up large intermediates to free memory
    del X, y_sig, raw_close, end_times
    del X_tr, y_sig_tr, raw_close_tr
    del X_val, y_sig_val, raw_close_val
    del X_te, y_sig_te, raw_close_te

    return train_loader, val_loader, test_loader, end_times_tr, end_times_val, end_times_te


#################################################


def mean_baseline_rmse(dl):
    '''
    The function returns two things:
    - The global mean of all targets (the baseline prediction).
    - The RMSE of using that mean as predictor (error of the mean baseline).
        '''
    # dl yields (x,y) or y
    s = n = 0
    for b in dl:
        y = b[1] if isinstance(b, (list,tuple)) and len(b)>1 else b
        y = torch.as_tensor(y).float()
        s += y.sum().item(); n += y.numel()
    m = s / n
    mse = sum(((torch.as_tensor(b[1] if isinstance(b, (list,tuple)) and len(b)>1 else b).float()-m)**2).sum().item() for b in dl)
    return m, math.sqrt(mse / n)


################################################


def summarize_split(name, loader, times):
    """
    Summary of loaders content, times, baseline
    """
    ds = loader.dataset
    X = ds.X
    L, F = X.shape[1], X.shape[2]
    Nw = X.shape[0]
    days, counts = np.unique(times.astype("datetime64[D]"), return_counts=True)

    print(f"--- {name.upper()} ---")
    print(f" calendar days : {len(days):3d}  ({days.min()} → {days.max()})")
    print(f" windows       : {Nw:4d}  (per-day min={counts.min():3d}, max={counts.max():3d}, mean={counts.mean():.1f})")
    print(f" window shape  : look_back={L}, n_features={F}")
    print(f" dataloader    : batches={len(loader):3d}, batch_size={loader.batch_size}, workers={loader.num_workers}, pin_memory={loader.pin_memory}")

    mean_val, rmse = mean_baseline_rmse(loader)
    print(f" baselines     : baseline prediction={mean_val:.6g}, baseline RMSE = {rmse:.6g}\n")


    
#########################################################################################################


def maybe_save_chkpt(
    models_dir: Path,
    model: torch.nn.Module,
    val_rmse: float,
    cur_best: float,
    model_feats, 
    model_hparams,
    tr: dict,
    val: dict,
    live_plot
) -> tuple[float, bool, dict, dict, dict | None]:
    """
    Compare `val_rmse` (current validation RMSE) against the best RMSE so far
    (cur_best) and on-disk checkpoints. If it’s an improvement, capture
    the model’s weights, metrics, and plot for both folder-best and
    in-run checkpointing.

    Returns:
      updated_best_rmse : new best RMSE (float)
      improved          : True if val_rmse < cur_best
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
    if val_rmse < cur_best:
        improved = True
        updated_best = val_rmse

        # Capture weights + metric snapshots
        best_state = model.state_dict()
        best_tr    = tr.copy()
        best_val    = val.copy()

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
                "hparams":          model_hparams,
                "features":         model_feats,
                "train_metrics":    best_tr,
                "val_metrics":      best_val,
                "train_plot_png":   plot_bytes,
            }
            ckpt_path = models_dir / fname
            torch.save(ckpt, ckpt_path)

            # Minimal audit line written at the moment of the save so logs
            # deterministically record which file was written for which val RMSE.
            try:
                _append_log(f"CHKPT SAVED val={updated_best:.3f} path={ckpt_path}", params.log_file)
            except Exception:
                # best-effort: do not fail the save on logging errors
                pass

            # Keep original user-visible print for immediate feedback
            print(f"🔖 Saved folder-best checkpoint (_chp): {fname}")

        return updated_best, improved, best_tr, best_val, best_state

    # No improvement
    return cur_best, False, {}, {}, None

    
################ 


def save_final_chkpt(
    models_dir: Path,
    best_state: dict,
    best_val_rmse: float,
    model_feats, 
    model_hparams,
    best_tr: dict,
    best_val: dict,
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
        "hparams":          model_hparams,
        "features":         model_feats,
        "train_metrics":    best_tr,
        "val_metrics":      best_val,
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


RUN_DIAGNOSTIC_EXPLANATION = (
    "\nSINGLE RUN DIAGNOSTIC FORMAT (explanatory)\n"
    "    BATCH_SHAPE B=7216 groups=1 seq_len_full=20 feat=20 : canonical input geometry used to build forwarded windows (snapshot.B, snapshot.groups, snapshot.seq_len_full, snapshot.feat_dim)\n"
    "    MICRODETAIL ms: <k=v ...> : single-line deterministic dump of snapshot keys printed as key=value (sorted); human-readable units used where applicable\n"
    "    -> B: number of batch samples used to build windows (snapshot.B)\n"
    "    -> groups: number of logical groups used when flattening windows (snapshot.groups)\n"
    "    -> seq_len_full: nominal full sequence length used when padding/truncating windows (snapshot.seq_len_full)\n"
    "    -> feat / feat_dim: feature dimension used to build windows (snapshot.feat_dim)\n"
    "    -> activation_mb: estimated activation footprint in MB computed from GPU allocation delta (snapshot.activation_mb)\n"
    "    -> backward_ms: backward pass ms placeholder (snapshot.backward_ms)\n"
    "    -> collector_ms: total wall-clock ms spent by the collector including sampling, CPU work and device synchronizations (snapshot.collector_ms)\n"
    "    -> cpu_copy_bytes: bytes copied to host for predictions shown human-readable in the log (snapshot.cpu_copy_bytes)\n"
    "    -> dataloader_ms: ms spent fetching the sampled batch from the dataloader (snapshot.dataloader_ms)\n"
    "    -> device_syncs_count: number of explicit device synchronizations performed during snapshot (snapshot.device_syncs_count)\n"
    "    -> env: small dict with python/torch/cuda/device_name strings (snapshot.env)\n"
    "    -> expected_segments: nominal B * groups used to estimate workload (snapshot.expected_segments)\n"
    "    -> full_forward_ms: wall-clock ms for the sampled forward over prepared windows (snapshot.full_forward_ms)\n"
    "    -> pred_extra_ms: estimated CPU-side post-forward ms = collector_ms - full_forward_ms - summed per-seg time (snapshot.pred_extra_ms)\n"
    "    -> gpu_allocated_bytes: raw GPU bytes allocated after forward (snapshot.gpu_allocated_bytes)\n"
    "    -> gpu_peak_mb: peak GPU memory in MB (snapshot.gpu_peak_mb)\n"
    "    -> gpu_reserved_bytes / gpu_reserved_mb: reserved GPU bytes and MB (snapshot.gpu_reserved_bytes, snapshot.gpu_reserved_mb)\n"
    "    -> grads: dict {'backbone': bool, 'head': bool} indicating gradient presence by name-bucket (snapshot.grads)\n"
    "    -> group_nonzero_counts: per-optimizer-group counts of parameters with non-None .grad (snapshot.group_nonzero_counts)\n"
    "    -> mean_seg_len: average per-segment time-series length in timesteps computed as sum_seg_lens / num_segments (snapshot.mean_seg_len)\n"
    "    -> num_segments: actual number of flattened segments forwarded (snapshot.num_segments)\n"
    "    -> sum_seg_lens: sum of non-empty segment lengths used to compute mean_seg_len (snapshot.sum_seg_lens)\n"
    "    -> out_bytes / out_dtype / out_numel / out_shape: model output bytes, dtype string, element count, and tuple shape (snapshot.out_bytes, snapshot.out_dtype, snapshot.out_numel, snapshot.out_shape)\n"
    "    -> param_bytes: total parameter memory in bytes (human-readable in log; raw int in snapshot.param_bytes)\n"
    "    -> total_params / trainable_params: canonical parameter counts stamped by init_log (snapshot.total_params, snapshot.trainable_params)\n"
    "    -> per_segment_p50_ms / per_segment_p90_ms: empirical per-segment forward-ms percentiles when sampling enabled (snapshot.per_segment_p50_ms, snapshot.per_segment_p90_ms)\n"
    "    -> raw_reg_shape: the raw detached regression output shape (snapshot.raw_reg_shape)\n"
    "    -> segments_per_sec: inferred throughput = num_segments / (full_forward_ms/1000.0) (snapshot.segments_per_sec)\n"
    "    -> windows_bytes: total bytes for the windows tensor (human-readable in log; raw int in snapshot.windows_bytes)\n"
    "\n"
    "PER-EPOCH LOG FORMAT (explanatory):\n"
    "  E{ep:02d}                : epoch number formatted with two digits\n"
    "  OPTS[{groups}:{lr_main}|cnts=[c1,c2,...]] : optimizer groups count; lr_main is the representative LR (first group); cnts lists per-group parameter counts\n"
    "  GN[name:val,...,TOT=val] : per-bucket gradient L2 norms printed as short_name:curr_norm; TOT is sqrt(sum squares over reported buckets)\n"
    "  GD[med,p90,max]         : gradient-norm distribution statistics printed as median, index-based 90th-percentile, and maximum\n"
    "  UR[med,max]             : update-ratio statistics (median,max) where update_ratio = lr * grad_norm / max(weight_norm,1e-8)\n"
    "  LR_MAIN={lr:.1e} | lr={lr:.1e} : representative main LR and explicit first-group lr printed in scientific notation\n"
    "  TR[...metrics...]       : training metrics reported for the epoch (rmse, mae, r2, etc.; see TR composition in code)\n"
    "  VAL[...metrics...]      : validation metrics reported for the epoch (rmse, mae, r2, etc.; see VAL composition in code)\n"
    "  SR={slope_rmse:.3f}     : slope RMSE computed on model.last_val_tot_preds/model.last_val_targs (trend calibration)\n"
    "  SL={slip:.2f},HR={hub_max:.3f} : slip fraction and hub max indicators derived from model.last_hub (defaults 0.00/0.000 when missing)\n"
    "  FMB={val:.4f}           : first-mini-batch diagnostic metric from the first-batch snapshot if set (snapshot.FMB)\n"
    "  T={elapsed:.1f}s,TP={throughput:.1f} : epoch elapsed seconds and throughput (segments/sec inferred from sampled forward)\n"
    "  chk={val}               : checkpoint marker token printed as chk in the line (implementation-specific; integerized when boolean)\n"
    "  GPU={GiB:.2f}GiB        : optional GPU memory high-water printed in GiB when CUDA available (torch.cuda.max_memory_allocated / (1024**3))\n"
    "  *CHKPT                  : optional marker when model._last_epoch_checkpoint is truthy (separate visual marker outside numeric chk token)\n"
    "  LAYER_GN[...]           : optional small set of monitored layer norms printed as name_short:curr_norm/ratio where ratio = curr / baseline_observed_on_first_epoch\n"
    "  G/U=name:grad/update_ratio,... : parameter buckets printed sorted by descending gradient norm; every short-name bucket is emitted in compact form\n"
    "      - format for each token is short_name:{g:.3e}/{u:.1e} where g is the gradient norm and u is the lr-scaled update ratio\n"
    "      - short_name uses the last 1–3 name segments (see short_name logic); entries are deduped by short_name keeping the highest-g representative\n"
    "      - the G/U list is printed every epoch (values change per epoch) and contains both large and very small gradients in scientific notation\n"
)


#############################################################


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
    seq_lengths = elems[-1]

    # Move inputs to device
    x_batch = x_batch.to(device)

    # Derive dims — assume layout (N, T, F)
    if x_batch.dim() != 3:
        raise RuntimeError(f"Expected x_batch.dim() == 3 (N, T, F), got {x_batch.dim()}")
    B, seq_len_full, feat_dim = x_batch.size()
    G = 1

    # when building segments
    segments = []
    seg_lens = []
    start = 0
    for i, L in enumerate(seq_lengths):
        Li = int(L.item()) if torch.is_tensor(L) else int(L)
        if Li <= 0:
            segments.append(x_batch.new_zeros((0, seq_len_full, feat_dim)))
            seg_lens.append(0)
            continue
        end = start + Li
        seg = x_batch[start:end]   # (Li, T, F)
        # pad in time dim to seq_len_full if needed
        cur_len = seg.size(1) if seg.dim() >= 2 else 0
        if cur_len < seq_len_full:
            pad_rows = seq_len_full - cur_len
            pad_tensor = seg.new_zeros((seg.size(0), pad_rows, feat_dim))
            seg = torch.cat([seg, pad_tensor], dim=1)
        segments.append(seg)
        seg_lens.append(min(Li, seq_len_full))
        start = end

    # Flatten to 3D: drop empty segments before concatenation
    flat_segments = [s for s in segments if s.numel() != 0]
    if not flat_segments:
        # preserve previous contract: return similar shape to earlier behaviour
        return None, None, prev_day

    windows_tensor = torch.cat(flat_segments, dim=0).to(device)

    # Compute consistent sum/num/mean for non-empty segments (num_segments equals windows_tensor.size(0))
    nonzero_seg_lens = [L for L in seg_lens if L > 0]
    sum_seg_lens = int(sum(nonzero_seg_lens)) if nonzero_seg_lens else 0
    num_segments = int(windows_tensor.size(0))
    
    mean_seg_len = float(sum_seg_lens) / num_segments if num_segments else float(seq_len_full)


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

    # ensure idxs is always defined
    idxs = []
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
    seg_times_sum = sum(seg_times) if seg_times else 0.0
    pred_extra_ms = max(0.0, float(collector_ms) - float(full_forward_ms) - float(seg_times_sum))

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
        "raw_reg_shape": out_shape,                
        "group_nonzero_counts": group_nonzero_counts,
        "grads": {"backbone": bool(backbone_has), "head": bool(head_has)},
        "full_forward_ms": float(full_forward_ms),
        "pred_extra_ms": pred_extra_ms,
        "num_segments": int(num_segments),
        "segments_per_sec": float(num_segments / (full_forward_ms / 1000.0)) if full_forward_ms and num_segments else None,
        "expected_segments": int(B * G),
        "sum_seg_lens": int(sum_seg_lens),
        "mean_seg_len": float(mean_seg_len),
        "gpu_peak_mb": int(gpu_peak_mb),
        "gpu_reserved_mb": int(gpu_reserved_mb) ,
        "gpu_allocated_bytes": int(gpu_allocated_bytes),
        "gpu_reserved_bytes": int(gpu_reserved_bytes) ,
        "cpu_copy_bytes": int(cpu_copy_bytes),
        "device_syncs_count": int(intentional_syncs),
        "per_segment_p50_ms": float(per_segment_p50_ms),
        "per_segment_p90_ms": float(per_segment_p90_ms),
        "activation_mb": int(activation_mb),
        "out_shape": out_shape,
        "out_dtype": out_dtype,
        "out_numel": int(out_numel),
        "out_bytes": int(out_bytes),
        "windows_bytes": int(windows_bytes),
        "collector_ms": float(collector_ms),
        "dataloader_ms": float(dataloader_ms),
        "env": env,
        "backward_ms": getattr(model, "_last_backward_ms", None)}

    model._micro_snapshot = snapshot
    model._microdetail_emitted = True

    if hasattr(model, "_micro_snapshot_in_progress"):
        delattr(model, "_micro_snapshot_in_progress")
    if model_was_training:
        model.train()

    return snapshot if return_snapshot else None


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

    Responsibilities
    - Emit run header, baselines and hyperparameters once.
    - Run a one-shot micro-snapshot collector (collect_or_run_forward_micro_snapshot).
    - Compute canonical parameter stats and produce a single authoritative
      BATCH_SHAPE + MICRODETAIL line derived from the numeric snapshot.

    Design notes
    - The collector performs sampling, forward, timing and raw numeric measurement
      and attaches a dict at model._micro_snapshot. init_log computes canonical
      parameter counts/bytes and performs all human-facing formatting.
    - Helpers are nested for single-file locality as requested.
    """
    # small helpers (single source of truth for param counts/bytes and formatting)
    def model_param_stats(model: torch.nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        def _elem_size_for(p):
            es = getattr(p, "element_size", None)
            if callable(es):
                return int(es())
            return 4

        param_bytes = int(sum(p.numel() * _elem_size_for(p) for p in model.parameters()))
        return int(total_params), int(trainable_params), int(param_bytes)

    def human_bytes(n):
        if n is None:
            return "None"
        n = int(n)
        if n >= 1024**2:
            return f"{n/1024**2:.2f}MB"
        if n >= 1024:
            return f"{n/1024:.1f}KB"
        return f"{n}B"

    global _RUN_STARTED, _RUN_DEBUG_DONE, _RUN_LOCK

    with _RUN_LOCK:
        if not _RUN_STARTED:
            sep = "-" * 150
            _append_log("\n" + sep, log_file)
            _append_log(f"RUN START: {datetime.utcnow().isoformat()}Z", log_file)

            _append_log(RUN_DIAGNOSTIC_EXPLANATION, log_file)

            if isinstance(baselines, dict) and baselines:
                _append_log("\nBASELINES:", log_file)
                _append_log(f"  TRAIN mean RMSE        = {baselines['bl_tr_mean']:.5f}", log_file)
                _append_log(f"  TRAIN persistence RMSE = {baselines['bl_tr_pers']:.5f}", log_file)
                _append_log(f"  VAL   mean RMSE        = {baselines['bl_val_mean']:.5f}", log_file)
                _append_log(f"  VAL   persistence RMSE = {baselines['bl_val_pers']:.5f}", log_file)

            if isinstance(hparams, dict) and hparams:
                _append_log("\nHYPERPARAMS:", log_file)
                for k, v in hparams.items():
                    _append_log(f"  {k} = {v}", log_file)
                _append_log("", log_file)

            # Compact runtime summary pieces
            debug_opt_line = None
            if optimizer is not None:
                opt_groups = len(optimizer.param_groups)
                opt_lrs = [g.get("lr", 0.0) for g in optimizer.param_groups]
                opt_counts = [sum(1 for _ in g["params"]) for g in optimizer.param_groups]
                debug_opt_line = f"DEBUG_OPT GROUPS={opt_groups} LRS={[f'{x:.1e}' for x in opt_lrs]} COUNTS={opt_counts}"

            # Canonical model counts computed once here so variables exist later
            if model is not None:
                total_params, trainable_params, computed_param_bytes = model_param_stats(model)
                frozen_params = total_params - trainable_params
                model_static_line = f"MODEL_STATIC: total_params={total_params:,} trainable={trainable_params:,} frozen={frozen_params:,}"
                param_bytes_line = f"param_bytes={human_bytes(computed_param_bytes)}"
            else:
                total_params = trainable_params = computed_param_bytes = None
                model_static_line = None
                param_bytes_line = None

            compact_parts = []
            if debug_opt_line:
                compact_parts.append(debug_opt_line)
            if model_static_line:
                compact_parts.append(model_static_line)
            if param_bytes_line:
                compact_parts.append(param_bytes_line)

            if compact_parts:
                _append_log(" | ".join(compact_parts), log_file)

            _RUN_STARTED = True

        # --- One-shot snapshot emission at most once ---
        if not _RUN_DEBUG_DONE:
            tloader = locals().get("train_loader", None) or globals().get("train_loader", None)
            prms = locals().get("params", None) or globals().get("params", None)
            opt_local = locals().get("optimizer", None) or globals().get("optimizer", None)

            # run collector (collector stays minimal: sampling/timing/forward)
            if callable(globals().get("collect_or_run_forward_micro_snapshot", None)):
                collect_or_run_forward_micro_snapshot(
                    model=model,
                    train_loader=tloader,
                    params=prms,
                    optimizer=opt_local,
                    log_file=log_file,
                )

            micro_ms = getattr(model, "_micro_snapshot", None)

            # stamp canonical param stats into snapshot and emit MICRODETAIL
            if isinstance(micro_ms, dict):
                micro_ms["total_params"] = int(total_params) if total_params is not None else None
                micro_ms["trainable_params"] = int(trainable_params) if trainable_params is not None else None
                micro_ms["param_bytes"] = int(computed_param_bytes) if computed_param_bytes is not None else None

                _append_log(
                    f"BATCH_SHAPE B={micro_ms.get('B')} groups={micro_ms.get('groups')} seq_len_full={micro_ms.get('seq_len_full')} feat={micro_ms.get('feat_dim')}",
                    log_file,
                )

                # format helpers used only for MICRODETAIL rendering
                def _hb(v):
                    if v is None: return "None"
                    v = int(v)
                    if v >= 1024**2: return f"{v/1024**2:.2f}MB"
                    if v >= 1024: return f"{v/1024:.1f}KB"
                    return f"{v}B"

                def _fmt(k, v):
                    if v is None:
                        return f"{k}=None"
                    # treat only param_bytes (and keys explicitly with "bytes") as byte quantities
                    if "bytes" in k or k == "param_bytes" or "cpu_copy" in k or "windows" in k:
                        return f"{k}={_hb(v)}"
                    if "shape" in k and not isinstance(v, str):
                        return f"{k}={tuple(v)}"
                    if isinstance(v, float):
                        return f"{k}={v:.2f}"
                    return f"{k}={v}"

                parts = [_fmt(k, micro_ms.get(k)) for k in sorted(micro_ms.keys())]
                _append_log("MICRODETAIL ms: " + " ".join(parts), log_file)

                _RUN_DEBUG_DONE = True


#################################################################################################################################


def log_epoch_feature_importance(model,
                                 feature_names: Optional[List[str]] = None,
                                 df = None,
                                 params = None,
                                 layer_token: Optional[str] = None,
                                 alpha: float = 0.9,
                                 mode: str = "combo",   # "combo" | "weights" | "grads"
                                 beta: Optional[float] = None,  # if set, update EWMA stored on model
                                 eps: float = 1e-12) -> Dict[str, Any]:
    """
    Compute per-feature scores from model.feature_proj.weight and its grad.

    - Prefer passing feature_names (ordered list). If None, will try df.columns, then no-op.
    - mode controls which normalized components to use:
      - "weights": normalized column weight norms only
      - "grads": normalized column grad norms only
      - "combo": alpha*w_norm + (1-alpha)*g_norm
    - Normalization: use log1p followed by min-max scaling for better visual spread.
    - Returns dict: top_token, items (list of tuples), score (np.array), w_norm, g_norm, layer_token.
    """
    # --- 1) resolve feature_names (simple checks only) ---
    if isinstance(feature_names, (list, tuple)):
        feature_names = list(feature_names)
    elif feature_names is None and df is not None and hasattr(df, "columns"):
        label_col = getattr(params, "label_col", "y") if params is not None else "y"
        feature_names = [c for c in df.columns if c not in (label_col, "close_raw")]
    else:
        feature_names = feature_names or []

    # --- 2) get projection parameter once --- #####################################################
    named = dict(model.named_parameters())
    # prefer per-raw-feature projection if present (columns align to original features)
    p = named.get("input_proj.weight") if "input_proj.weight" in named else named.get("feature_proj.weight")
    
    if p is None or len(feature_names) == 0:
        empty = np.zeros(0)
        return {"top_token": "", "items": [], "score": empty, "w_norm": empty.copy(), "g_norm": empty.copy(), "layer_token": layer_token or ""}

    # --- 3) read weights and grads on CPU and align names ---
    W = p.detach().cpu().numpy()                 # (out_dim, in_dim)
    in_dim = int(W.shape[1])

    # safer alignment: truncate or extend to exactly in_dim
    if len(feature_names) < in_dim:
        feature_names = list(feature_names) + [f"feat_{i}" for i in range(len(feature_names), in_dim)]
    else:
        feature_names = list(feature_names[:in_dim])

    w_norm = np.linalg.norm(W, axis=0)           # shape (in_dim,)
    if p.grad is None:
        g_norm = np.zeros_like(w_norm)
    else:
        G = p.grad.detach().cpu().numpy()
        g_norm = np.linalg.norm(G, axis=0)

    # --- 4) normalization helper (log1p + min-max) ---
    def _norm_vis(arr):
        if arr.size == 0:
            return arr
        a = np.log1p(arr)            # compress heavy tails
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + eps)

    w_s = _norm_vis(w_norm)
    g_s = _norm_vis(g_norm) if g_norm.size else g_norm

    # --- 5) build score according to mode ---
    if mode == "weights":
        score = w_s
    elif mode == "grads":
        score = g_s
    else:
        score = alpha * w_s + (1.0 - alpha) * g_s

    # --- 6) prepare items, top token and persist simple history ---
    items = list(zip(feature_names, score, w_norm, g_norm))
    items.sort(key=lambda x: x[1], reverse=True)
    top_token = ",".join(f"{n}:{s:.2e}" for n, s, _, _ in items)

    if not hasattr(model, "_feat_imp_history"):
        model._feat_imp_history = []
    model._feat_imp_history.append(score.copy())

    # preserve layer_token behavior if provided
    if top_token and layer_token is not None:
        layer_token = (layer_token if layer_token is not None else "") + f",FEAT_TOP[{top_token}]"

    return {"top_token": top_token, "items": items, "score": score, "w_norm": w_norm, "g_norm": g_norm, "layer_token": layer_token}


###################################################################################### 


_feat_history = []    
_param_history = []    
_epochs = []
_live_bars = plots.LiveFeatGuBars(top_feats=999, top_params=999)

def log_epoch_summary(
    epoch:            int,
    model:            torch.nn.Module,
    optimizer:        torch.optim.Optimizer,
    tr_tot_metrics:   dict,
    tr_base_metrics:  dict,
    tr_delta_metrics: dict,
    val_tot_metrics:  dict,
    val_base_metrics: float,
    val_tot_preds:    float,
    val_base_preds:   float,
    val_targs:        float,
    avg_base_loss:    float,
    avg_delta_loss:   float,
    log_file:         Path,
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
            "bl_tr_mean":  model.bl_tr_mean,
            "bl_tr_pers":  model.bl_tr_pers,
            "bl_val_mean": model.bl_val_mean,
            "bl_val_pers": model.bl_val_pers,
        },
        optimizer=optimizer,
        model=model,
    )

    # 2) detect top-level parameter groups ("heads") and their grad-norm totals — minimal
    recs = []; prefix_sq = {}; all_sq = 0.0
    lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
    
    for name, p in model.named_parameters():
        if getattr(p, "numel", lambda: 1)() == 0:
            continue
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
        pv, tv = getattr(model, "last_val_tot_preds", None), getattr(model, "last_val_targs", None)
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
    if val_tot_preds is not None and val_base_preds is not None and val_targs is not None:
        # ensure numpy arrays (1D) and aligned length
        vp = np.asarray(val_tot_preds, dtype=float).ravel()
        vb = np.asarray(val_base_preds, dtype=float).ravel()
        vt = np.asarray(val_targs, dtype=float).ravel()
        if vp.shape == vb.shape == vt.shape and vp.size > 0:
            fraction_model_better = float((np.abs(vp - vt) < np.abs(vb - vt)).mean())

    # 7) Pull train/val RMSE, R², MAE
    tr_rmse, tr_mae, tr_r2 = tr_tot_metrics["rmse"], tr_tot_metrics["mae"], tr_tot_metrics["r2"]
    val_rmse, val_mae, val_r2 = val_tot_metrics["rmse"], val_tot_metrics["mae"], val_tot_metrics["r2"]

    train_base_rmse = tr_base_metrics["rmse"] 
    train_delta_rmse = tr_delta_metrics["rmse"]
    val_base_rmse = val_base_metrics["rmse"]

    loss_tokens = ""
    loss_tokens = f"BASE_LOSS={avg_base_loss:.4e}"
    delta_ratio = avg_delta_loss / (avg_base_loss + 1e-12)
    loss_tokens += f",DELTA_LOSS={avg_delta_loss:.4e},DELTA_RATIO={delta_ratio:.3e}"
    loss_tokens += f",BASE_RMSE={train_base_rmse:.5f},DELTA_RMSE={train_delta_rmse:.5f}"

    # 8) compact Top: dedupe by short name, show all buckets sorted by g (compact format)
    def short_name(n):
        if "parametrizations" in n:
            return n.split('.')[-1]
        return ".".join(n.split('.')[-3:])
    
    seen = {}
    for name, g, u in recs:
        if "parametrizations" in name:
            continue
        s = short_name(name)
        # keep the entry with the largest g for each short name
        if s not in seen or g > seen[s][0]:
            seen[s] = (g, u)

    # sort all buckets by descending gradient and produce compact s:g/u tokens
    items = sorted(seen.items(), key=lambda kv: kv[1][0], reverse=True)
    G_U = ",".join(f"{s}:{g:.1e}/{u:.1e}" for s, (g, u) in items)

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
    if sched_obj is not None:
        total = getattr(sched_obj, "_total_steps", None) or getattr(sched_obj, "total_steps", None)
        # prefer explicit step counters
        step_idx = getattr(sched_obj, "last_epoch", None)
        if step_idx is None:
            step_idx = getattr(sched_obj, "_step_count", None)
        if total is None:
            # if total is missing but scheduler has been given expected_total elsewhere, try that attribute
            total = getattr(sched_obj, "expected_total", None)
        if total is not None and step_idx is not None and int(total) > 0:
            pct = min(100.0, max(0.0, 100.0 * float(step_idx) / float(total)))
            sched_pct_token = f"SCHED_PCT={pct:.1f}%"

    # 11) Optional timing / throughput (if the training loop stored them on the model)
    elapsed = getattr(model, "_last_epoch_elapsed", 0)
    samples = getattr(model, "_last_epoch_samples", 0)
    tp = (samples / elapsed) if (elapsed and elapsed > 0) else 0.0

    # 12) Optional checkpoint marker (if training loop marked it on the model)
    chk = getattr(model, "_last_epoch_checkpoint", False)

    # 13) Optional GPU memory high-water (low-noise, printed only if CUDA available)
    max_mem = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

    # 14) append FEAT_TOP after layer_token finalized
    feat_res = log_epoch_feature_importance(model, feature_names=model.feature_names,
                                           params=params, alpha=0.9, mode="combo") # some sensitivity to recent learning: score = 0.9norm_w + 0.1norm_g.

    # Final line assembly (compact, per-epoch changing values only)
    line = (
        f"\nE{epoch:02d} | "
        f"{opt_token} | "
        f"GN[{gn_items},TOT={GN_tot:.3f}] | "
        f"GD[{GD_med:.1e},{GD_p90:.1e},{GD_max:.1e}] | "
        f"UR[{UR_med:.1e},{UR_max:.1e}] | "
        f"lr={lr:.1e} | "
        f"TR[{tr_rmse:.4f},{tr_mae:.4f},{tr_r2:.4f},BASE_RMSE={train_base_rmse:.4f},DELTA_RMSE={train_delta_rmse:.4f}] | "
        f"VAL[{val_rmse:.4f},{val_mae:.4f},{val_r2:.4f},BASE_RMSE={val_base_rmse:.4f}] | "
        f"{loss_tokens} | "
        f"{sched_pct_token} | "
        f"SR={SR:.3f} | "
        f"SL={SL:.2f},HR={HR:.3f} | "
        f"FMB={fraction_model_better:.4f} | "
        f"T={elapsed:.1f}s,TP={tp:.1f}seg/s | "
        f"chk={int(bool(chk))} | "
        f"GPU={max_mem:.2f}GiB | "
        f"\nG/U={G_U} | "
        f"\nFEAT_TOP={feat_res['top_token']}"
    )
    _append_log(line, log_file)

    # plotting
    # build numeric dicts (no string parsing)
    gdict = {s: float(g) for s, (g, u) in items}
    
    featdict = {k: float(v) for k, v in (p.split(':',1) for p in feat_res['top_token'].split(','))}

    _live_bars.update(featdict, gdict, epoch)

