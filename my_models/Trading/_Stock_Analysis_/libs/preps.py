from libs import params, plots, strats 

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


def detect_and_adjust_splits(df, forward_threshold=0.5, reverse_threshold=2, tol=0.05, vol_fact=1, dst_transition_dates=None):
    """
    Detects forward and reverse splits and adjusts price columns and volume.
    Minimal addition: accepts dst_transition_dates (iterable of dates) and skips
    candidate split events that fall on those dates to avoid DST artifacts.
    """
    df_adj = df.copy()
    split_events = []
    price_columns = ['open', 'high', 'low', 'close', 'ask', 'bid']
    dst_transition_dates = set(dst_transition_dates or [])
    print(f"Executing [detect_and_adjust_splits]...")

    for i in range(1, len(df_adj)):
        prev_close = df_adj.iloc[i-1]['close']
        curr_close = df_adj.iloc[i]['close']
        ratio = curr_close / prev_close

        # Detect forward split (price drop)
        if ratio < forward_threshold:
            candidate_factor = round(1.0 / ratio)
            if candidate_factor >= 2:
                expected_ratio = 1.0 / candidate_factor
                if abs(ratio - expected_ratio) < tol:
                    event_time = df_adj.index[i]
                    # skip if event falls on a DST transition date
                    if event_time.date() in dst_transition_dates:
                        print(f"Skipping forward-split candidate at {event_time} due to DST transition")
                        continue
                    print(f"Detected forward split on {event_time} with factor {candidate_factor} (ratio: {ratio:.4f})")
                    split_events.append((event_time, candidate_factor, 'forward'))
                    df_adj.loc[:df_adj.index[i-1], price_columns] /= candidate_factor
                    df_adj.loc[:df_adj.index[i-1], 'volume'] *= candidate_factor * vol_fact

        # Detect reverse split (price jump)
        elif ratio > reverse_threshold:
            candidate_factor = round(ratio)
            if candidate_factor >= 2:
                expected_ratio = candidate_factor
                if abs(ratio - expected_ratio) < tol:
                    event_time = df_adj.index[i]
                    # skip if event falls on a DST transition date
                    if event_time.date() in dst_transition_dates:
                        print(f"Skipping reverse-split candidate at {event_time} due to DST transition")
                        continue
                    print(f"Detected reverse split on {event_time} with factor {candidate_factor} (ratio: {ratio:.4f})")
                    split_events.append((event_time, candidate_factor, 'reverse'))
                    df_adj.loc[:df_adj.index[i-1], price_columns] *= candidate_factor
                    df_adj.loc[:df_adj.index[i-1], 'volume'] /= candidate_factor

    return df_adj, split_events


#########################################################################################################


def process_splits(ticker):
    """Load intraday CSV, localize timestamps to US/Eastern, 
    detect & adjust corporate splits, and return the adjusted DataFrame."""
    print(f"[process_splits] Loading original alpaca csv")
    df = pd.read_csv(params.alpaca_csv, index_col=0, parse_dates=True)

    df.index = pd.to_datetime(df.index)
    
    # Create ask/bid from close using configured spread
    df['ask'] = round(df['close'] * (1 + params.bidask_spread_pct/100), 4)
    df['bid'] = round(df['close'] * (1 - params.bidask_spread_pct/100), 4)

    print("Plotting original data...")
    plots.plot_close_volume(df, title="Before Adjusting Splits: Close Price and Volume")

    # Minimal: if you don't want to compute DST transition dates here, pass an empty set
    transitions = set()
    df_adjusted, split_events = detect_and_adjust_splits(df=df, dst_transition_dates=transitions)

    if split_events:
        print("Splits detected. Plotting adjusted data...")
        plots.plot_close_volume(df_adjusted, title="After Adjusting Splits: Close Price and Volume")
    else:
        print("No splits detected.")

    return df_adjusted



#########################################################################################################


def prepare_interpolate_data(df: pd.DataFrame, tz_str: str = "US/Eastern") -> pd.DataFrame:
    """
    Normalize intraday timestamps to a consistent timezone and fill missing 1-minute bars.

    - Interprets df.index as timezone-aware and converts it to tz_str.
    - Drops exact duplicate rows once (prints count and timestamps).
    - Builds a per-day tz-aware 1-minute index and linearly interpolates missing bars.
    - Returns a DataFrame indexed by tz-aware 1-minute timestamps.
    """
    print(['executing prepare_interpolate_data'])
    df = df.copy()
    df[df.columns] = df[df.columns].astype(np.float64)

    # normalize: convert index to datetimes (keep naive index for storage)
    print('normalize data...')
    idx = pd.to_datetime(df.index)
    # subtract 1h for standard-time (winter) timestamps, leave DST unchanged
    _e = idx.tz_localize(tz_str, ambiguous="infer", nonexistent="shift_forward") if idx.tz is None else idx.tz_convert(tz_str)
    df.index = idx - pd.to_timedelta((_e.map(lambda t: t.dst()) == pd.Timedelta(0)).astype(int), unit="h")

    # single, early duplicate drop with clear audit print
    dup_mask = df.index.duplicated(keep=False)
    if dup_mask.any():
        dup_times = df.index[dup_mask].unique()
        print(
            "prepare_interpolate_data: dropping",
            int(dup_mask.sum()),
            "exact duplicate rows at:",
            ", ".join(t.strftime("%Y-%m-%d %H:%M:%S") for t in dup_times),
        )
        df = df[~df.index.duplicated(keep="first")]

    df.sort_index(inplace=True)

    # per-day 1-min grid and linear interpolation where needed (naive index)
    print('interpolate data...')
    filled = []
    for day, grp in df.groupby(df.index.normalize()):
        idx_start = grp.index.min()
        idx_end = grp.index.max()
        full_idx = pd.date_range(start=idx_start, end=idx_end, freq="1min")
        day_filled = grp.reindex(full_idx).interpolate(method="linear", limit_direction="both")
        filled.append(day_filled)

    df_out = pd.concat(filled).sort_index()

    # drop flat days if desired (keeps original behavior)
    df_out = df_out.groupby(df_out.index.normalize()).filter(lambda g: g['close'].nunique() > 1)
    return df_out


#########################################################################################################


def build_signal_per_day(
    df: pd.DataFrame,
    *,
    min_prof_thr: float = 0.5,
    max_down_prop: float = 0.25,
    gain_tightfact: float = 0.02,
    tau_time: float = 60.0,
    tau_dur: float = 60.0,
    col_close: str = "close"
) -> pd.DataFrame:
    """
    Detect up-swings per trading day and produce per-bar columns:
      - targ_signal, swing_dir, swing_gain_pct, last_buy, last_sell

    This is the original detection/score logic only. It does NOT compute thresholds.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    parts = []
    for day, day_df in df.groupby(df.index.normalize()):
        if day_df.empty:
            parts.append(day_df)
            continue

        closes = day_df[col_close].to_numpy()
        times = day_df.index.to_numpy()
        n = len(day_df)

        targ_signal = np.zeros(n, dtype=float)
        swing_dir = np.zeros(n, dtype=int)
        swing_gain_pct = np.zeros(n, dtype=float)
        last_buy = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        last_sell = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

        i = 1
        while i < n - 1:
            if closes[i] < closes[i - 1] and closes[i] < closes[i + 1]:
                buy_idx = i
                buy_price = closes[i]
                max_so_far = buy_price
                cand_sell_idx = None
                j = i + 1

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
                            score = gap * decay_time * boost_dur * 1000.0
                            targ_signal[mask] = np.maximum(targ_signal[mask], score)
                            swing_dir[mask] = 1
                            swing_gain_pct[mask] = profit_pc
                            last_buy[mask] = times[buy_idx]
                            last_sell[mask] = times[sell_idx]
                        i = sell_idx + 1
                        continue
                i = buy_idx + 1
            else:
                i += 1

        out = day_df.copy()
        out["targ_signal"] = targ_signal
        out["swing_dir"] = swing_dir
        out["swing_gain_pct"] = swing_gain_pct
        out["last_buy"] = last_buy
        out["last_sell"] = last_sell

        parts.append(out)

    return pd.concat(parts).sort_index()


#########################################################################################################


def apply_thresholds_per_day(
    df: pd.DataFrame,
    *,
    col_signal: str = "targ_signal",
    thresh_mode: Union[str, float] = "median_nonzero",
    thresh_window: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute per-day (or per-day rolling) thresholds for `col_signal` and add:
      - <col_signal>_thresh  (named 'signal_thresh' for compatibility)
      - gap_to_thresh         = <col_signal> - signal_thresh

    Supported scalar modes:
      - numeric (int/float)
      - "median_nonzero", "mean_nonzero", "p90", "p95", "p99"
      - "median", "mean"

    Supported rolling modes (per-day):
      - "roll_mean", "roll_median", "roll_p90", "roll_p95"

    Returns a new DataFrame with 'signal_thresh' and 'gap_to_thresh'.
    """
    if col_signal not in df.columns:
        raise ValueError(f"DataFrame must contain '{col_signal}' column")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    def day_thresh(arr: np.ndarray) -> float:
        nz = arr[arr > 0]
        if isinstance(thresh_mode, (int, float)):
            return float(thresh_mode)
        if thresh_mode == "median_nonzero":
            return float(np.median(nz)) if len(nz) else 0.0
        if thresh_mode == "mean_nonzero":
            return float(np.mean(nz)) if len(nz) else 0.0
        if thresh_mode == "p90":
            return float(np.percentile(nz, 90)) if len(nz) else 0.0
        if thresh_mode == "p95":
            return float(np.percentile(nz, 95)) if len(nz) else 0.0
        if thresh_mode == "p99":
            return float(np.percentile(nz, 99)) if len(nz) else 0.0
        if thresh_mode == "median":
            return float(np.median(arr))
        if thresh_mode == "mean":
            return float(np.mean(arr))
        raise ValueError(f"Unknown scalar thresh_mode: {thresh_mode}")

    parts = []
    for day, day_df in tqdm(df.groupby(df.index.normalize()), desc="Thresh per day", leave=True):
        out = day_df.copy()
        series = out[col_signal].to_numpy()

        if isinstance(thresh_mode, (int, float)) or thresh_mode in {
            "median_nonzero", "mean_nonzero", "p90", "p95", "p99", "median", "mean"
        }:
            val = day_thresh(series)
            out["signal_thresh"] = val
        else:
            w = thresh_window or 20
            if thresh_mode == "roll_mean":
                out["signal_thresh"] = out[col_signal].rolling(window=w, min_periods=1).mean()
            elif thresh_mode == "roll_median":
                out["signal_thresh"] = out[col_signal].rolling(window=w, min_periods=1) \
                    .apply(lambda x: np.median(x), raw=False)
            elif thresh_mode in ("roll_p90", "roll_p95"):
                q = 90 if thresh_mode == "roll_p90" else 95
                out["signal_thresh"] = out[col_signal].rolling(window=w, min_periods=1) \
                    .apply(lambda x: np.percentile(x, q), raw=False)
            else:
                raise ValueError(f"Unknown rolling thresh_mode: {thresh_mode}")

        out["gap_to_thresh"] = out[col_signal] - out["signal_thresh"]
        parts.append(out)

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