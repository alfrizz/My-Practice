from libs import params, plots, strategies 

import math
import pandas as pd
from pandas import Timestamp
import numpy as np
import glob
import os

from pathlib import Path 
from datetime import datetime, time, timedelta
import datetime as dt

import pytz
from typing import Optional, Dict, Tuple, List, Sequence, Union, Any
import matplotlib.pyplot as plt

from tqdm import tqdm


#########################################################################################################



def build_signal_per_day(
    df: pd.DataFrame,
    *,
    min_prof_thr: float = 0.5,             # minimum % gain to accept a swing
    max_down_prop: float = 0.25,           # base retracement threshold (fraction of move)
    gain_tightfact: float = 0.02,  # tighter retracement for larger gains
    tau_time: float = 60.0,                # minutes half-life for temporal decay
    tau_dur: float = 60.0,                 # minutes half-life for duration boost
    thresh_mode: Union[str, float] = "median_nonzero",
    thresh_window: Optional[int] = None,   # rolling window (bars) for rolling modes
) -> pd.DataFrame:
    """
    Build a continuous long-only signal per day (no reindexing).

    For each day:
      - Detect swings from local minima to subsequent maxima with a dynamic retracement break.
      - Accept swings with profit >= min_prof_thr (%).
      - For all bars up to each swing's sell, compute a decayed score:
            gap * exp(-mins_to_exit / tau_time) * (1 - exp(-dur_min / tau_dur))
        and take the max over all swings per bar -> signal_raw.
      - Add helper columns for sizing/logic:
            swing_dir (1 if last swing up), swing_gain_pct, last_buy, last_sell.
      - Compute a per-bar threshold per `thresh_mode`, and gap_to_thresh.

    Threshold modes:
      - "median_nonzero": median of positive signal_raw (per-day scalar).
      - "p90": 90th percentile of positive signal_raw (per-day scalar).
      - "mean_nonzero": mean of positive signal_raw (per-day scalar).
      - "roll_mean": rolling mean of signal_raw with window=thresh_window.
      - "roll_p90": rolling 90th percentile of signal_raw with window=thresh_window.
      - float/constant: use the given value.

    Returns:
      Single DataFrame with added columns:
        signal_raw, signal_thresh, gap_to_thresh,
        swing_dir, swing_gain_pct, last_buy, last_sell
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    parts = []

    for day, day_df in df.groupby(df.index.normalize()):
        if day_df.empty:
            parts.append(day_df)
            continue

        closes = day_df["close"].to_numpy()
        times  = day_df.index.to_numpy()
        n      = len(day_df)

        signal_raw = np.zeros(n, dtype=float)
        swing_dir = np.zeros(n, dtype=int)
        swing_gain_pct = np.zeros(n, dtype=float)
        last_buy = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        last_sell = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

        i = 1
        while i < n - 1:
            # local minimum as potential buy
            if closes[i] < closes[i-1] and closes[i] < closes[i+1]:
                buy_idx = i
                buy_price = closes[i]
                max_so_far = buy_price
                cand_sell_idx = None
                j = i + 1

                # walk forward to find peak and retracement break
                while j < n:
                    price = closes[j]
                    if price > max_so_far:
                        max_so_far = price
                        cand_sell_idx = j
                    elif max_so_far > buy_price:
                        gain_pc = 100 * (max_so_far - buy_price) / buy_price
                        dyn_prop = max_down_prop / (1 + gain_tightfact * gain_pc)
                        retracement = (max_so_far - price) / (max_so_far - buy_price)
                        if retracement > dyn_prop:
                            break
                    j += 1

                # if a valid swing is found and meets profit threshold
                if cand_sell_idx is not None:
                    sell_idx = cand_sell_idx
                    profit_pc = 100 * (closes[sell_idx] - buy_price) / buy_price
                    if profit_pc >= min_prof_thr:
                        exit_ts = times[sell_idx]
                        mask = times <= exit_ts
                        if mask.any():
                            gap = (closes[sell_idx] - closes[mask]) / closes[sell_idx]
                            mins_to_exit = (exit_ts - times[mask]) / np.timedelta64(1, "m")
                            decay_time = np.exp(-mins_to_exit / tau_time)
                            dur_min = (times[sell_idx] - times[buy_idx]) / np.timedelta64(1, "m")
                            boost_dur = 1 - np.exp(-dur_min / tau_dur)
                            score = gap * decay_time * boost_dur
                            signal_raw[mask] = np.maximum(signal_raw[mask], score)
                            swing_dir[mask] = 1
                            swing_gain_pct[mask] = profit_pc
                            last_buy[mask] = times[buy_idx]
                            last_sell[mask] = times[sell_idx]
                        i = sell_idx + 1
                        continue
                i = buy_idx + 1
            else:
                i += 1

        # attach computed columns
        day_df = day_df.copy()
        day_df["signal_raw"] = signal_raw
        day_df["swing_dir"] = swing_dir
        day_df["swing_gain_pct"] = swing_gain_pct
        day_df["last_buy"] = last_buy
        day_df["last_sell"] = last_sell

        # threshold computation
        if isinstance(thresh_mode, (int, float)):
            thresh_series = pd.Series(float(thresh_mode), index=day_df.index)
        elif thresh_mode == "median_nonzero":
            nz = signal_raw[signal_raw > 0]
            val = float(np.median(nz)) if len(nz) else 0.0
            thresh_series = pd.Series(val, index=day_df.index)
        elif thresh_mode == "p90":
            nz = signal_raw[signal_raw > 0]
            val = float(np.percentile(nz, 90)) if len(nz) else 0.0
            thresh_series = pd.Series(val, index=day_df.index)
        elif thresh_mode == "mean_nonzero":
            nz = signal_raw[signal_raw > 0]
            val = float(np.mean(nz)) if len(nz) else 0.0
            thresh_series = pd.Series(val, index=day_df.index)
        elif thresh_mode == "roll_mean":
            w = thresh_window or 20
            thresh_series = day_df["signal_raw"].rolling(window=w, min_periods=1).mean()
        elif thresh_mode == "roll_p90":
            w = thresh_window or 20
            thresh_series = day_df["signal_raw"].rolling(window=w, min_periods=1) \
                .apply(lambda x: np.percentile(x, 90), raw=False)
        else:
            raise ValueError(f"Unknown thresh_mode: {thresh_mode}")

        day_df["signal_thresh"] = thresh_series
        day_df["gap_to_thresh"] = day_df["signal_raw"] - day_df["signal_thresh"]

        parts.append(day_df)

    return pd.concat(parts).sort_index()
    

#########################################################################################################


def smooth_scale_saturate(
    series:   pd.Series,
    window:   int,
    beta_sat: float
) -> pd.Series:
    """
    1) Smoothing: centered rolling mean of width `window`.
    2) Proportional scale into [0,1]: divide by the global max of the smoothed series.
    3) Soft‐saturating exponential warp:
         h(u) = (1 - exp(-β·u)) / (1 - exp(-β))
       • h(0)=0, h(1)=1
       • concave: lifts the bulk (h(u)>u for u∈(0,1)), gently compresses the top end.
    Returns a new Series in [0,1], same index as `series`.
    """
    # 1) smooth
    sm = series.rolling(window=window, center=True, min_periods=1).mean()

    # 2) proportional scale → [0,1]
    u = sm / sm.max()

    # 3) soft‐saturate
    expb = np.exp(-beta_sat)
    # denominator = 1 - e^{-β}
    denom = 1.0 - expb
    warped = (1.0 - np.exp(-beta_sat * u)) / denom

    return pd.Series(warped, index=series.index)

    

