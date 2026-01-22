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


# def build_signal_per_day(
#     df: pd.DataFrame,
#     *,
#     min_prof_thr: float = 0.5,             # minimum % gain to accept a swing
#     max_down_prop: float = 0.25,           # base retracement threshold (fraction of move)
#     gain_tightfact: float = 0.02,  # tighter retracement for larger gains
#     tau_time: float = 60.0,                # minutes half-life for temporal decay
#     tau_dur: float = 60.0,                 # minutes half-life for duration boost
#     thresh_mode: Union[str, float] = "median_nonzero",
#     thresh_window: Optional[int] = None,   # rolling window (bars) for rolling modes
#     col_close: str = "close"
# ) -> pd.DataFrame:
#     """
#     Build a continuous long-only signal per day (no reindexing).

#     For each day:
#       - Detect swings from local minima to subsequent maxima with a dynamic retracement break.
#       - Accept swings with profit >= min_prof_thr (%).
#       - For all bars up to each swing's sell, compute a decayed score:
#             gap * exp(-mins_to_exit / tau_time) * (1 - exp(-dur_min / tau_dur))
#         and take the max over all swings per bar -> signal_raw.
#       - Add helper columns for sizing/logic:
#             swing_dir (1 if last swing up), swing_gain_pct, last_buy, last_sell.
#       - Compute a per-bar threshold per `thresh_mode`, and gap_to_thresh.

#     Threshold modes:
#       - "median_nonzero": median of positive signal_raw (per-day scalar).
#       - "p90": 90th percentile of positive signal_raw (per-day scalar).
#       - "mean_nonzero": mean of positive signal_raw (per-day scalar).
#       - "roll_mean": rolling mean of signal_raw with window=thresh_window.
#       - "roll_p90": rolling 90th percentile of signal_raw with window=thresh_window.
#       - float/constant: use the given value.

#     Returns:
#       Single DataFrame with added columns:
#         signal_raw, signal_thresh, gap_to_thresh,
#         swing_dir, swing_gain_pct, last_buy, last_sell
#     """
#     if not isinstance(df.index, pd.DatetimeIndex):
#         df.index = pd.to_datetime(df.index)

#     parts = []

#     for day, day_df in df.groupby(df.index.normalize()):
#         if day_df.empty:
#             parts.append(day_df)
#             continue

#         closes = day_df[col_close].to_numpy()
#         times  = day_df.index.to_numpy()
#         n      = len(day_df)

#         signal_raw = np.zeros(n, dtype=float)
#         swing_dir = np.zeros(n, dtype=int)
#         swing_gain_pct = np.zeros(n, dtype=float)
#         last_buy = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
#         last_sell = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

#         i = 1
#         while i < n - 1:
#             # local minimum as potential buy
#             if closes[i] < closes[i-1] and closes[i] < closes[i+1]:
#                 buy_idx = i
#                 buy_price = closes[i]
#                 max_so_far = buy_price
#                 cand_sell_idx = None
#                 j = i + 1

#                 # walk forward to find peak and retracement break
#                 while j < n:
#                     price = closes[j]
#                     if price > max_so_far:
#                         max_so_far = price
#                         cand_sell_idx = j
#                     elif max_so_far > buy_price:
#                         gain_pc = 100 * (max_so_far - buy_price) / buy_price
#                         dyn_prop = max_down_prop / (1 + gain_tightfact * gain_pc)
#                         retracement = (max_so_far - price) / (max_so_far - buy_price)
#                         if retracement > dyn_prop:
#                             break
#                     j += 1

#                 # if a valid swing is found and meets profit threshold
#                 if cand_sell_idx is not None:
#                     sell_idx = cand_sell_idx
#                     profit_pc = 100 * (closes[sell_idx] - buy_price) / buy_price
#                     if profit_pc >= min_prof_thr:
#                         exit_ts = times[sell_idx]
#                         mask = times <= exit_ts
#                         if mask.any():
#                             gap = (closes[sell_idx] - closes[mask]) / closes[sell_idx]
#                             mins_to_exit = (exit_ts - times[mask]) / np.timedelta64(1, "m")
#                             decay_time = np.exp(-mins_to_exit / tau_time)
#                             dur_min = (times[sell_idx] - times[buy_idx]) / np.timedelta64(1, "m")
#                             boost_dur = 1 - np.exp(-dur_min / tau_dur)
#                             score = gap * decay_time * boost_dur * 1000 # scaling by a factor of 1000 for better signal visibility
#                             signal_raw[mask] = np.maximum(signal_raw[mask], score)
#                             swing_dir[mask] = 1
#                             swing_gain_pct[mask] = profit_pc
#                             last_buy[mask] = times[buy_idx]
#                             last_sell[mask] = times[sell_idx]
#                         i = sell_idx + 1
#                         continue
#                 i = buy_idx + 1
#             else:
#                 i += 1

#         # attach computed columns
#         day_df = day_df.copy()
#         day_df["signal_raw"] = signal_raw
#         day_df["swing_dir"] = swing_dir
#         day_df["swing_gain_pct"] = swing_gain_pct
#         day_df["last_buy"] = last_buy
#         day_df["last_sell"] = last_sell

#         # threshold computation
#         if isinstance(thresh_mode, (int, float)):
#             thresh_series = pd.Series(float(thresh_mode), index=day_df.index)
#         elif thresh_mode == "median_nonzero":
#             nz = signal_raw[signal_raw > 0]
#             val = float(np.median(nz)) if len(nz) else 0.0
#             thresh_series = pd.Series(val, index=day_df.index)
#         elif thresh_mode == "p90":
#             nz = signal_raw[signal_raw > 0]
#             val = float(np.percentile(nz, 90)) if len(nz) else 0.0
#             thresh_series = pd.Series(val, index=day_df.index)
#         elif thresh_mode == "mean_nonzero":
#             nz = signal_raw[signal_raw > 0]
#             val = float(np.mean(nz)) if len(nz) else 0.0
#             thresh_series = pd.Series(val, index=day_df.index)
#         elif thresh_mode == "roll_mean":
#             w = thresh_window or 20
#             thresh_series = day_df["signal_raw"].rolling(window=w, min_periods=1).mean()
#         elif thresh_mode == "roll_p90":
#             w = thresh_window or 20
#             thresh_series = day_df["signal_raw"].rolling(window=w, min_periods=1) \
#                 .apply(lambda x: np.percentile(x, 90), raw=False)
#         else:
#             raise ValueError(f"Unknown thresh_mode: {thresh_mode}")

#         day_df["signal_thresh"] = thresh_series
#         day_df["gap_to_thresh"] = day_df["signal_raw"] - day_df["signal_thresh"]

#         parts.append(day_df)

#     return pd.concat(parts).sort_index()
    

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
    col_close: str = "close"
) -> pd.DataFrame:
    """
    Generate a per-bar continuous long-only signal from intraday price series.

    This function scans each trading day independently, finds up-swings
    (local minima → subsequent maxima) using a dynamic retracement rule,
    and computes a decayed, duration-weighted score for every bar that is
    covered by an accepted swing. It returns the original data augmented
    with per-bar signal columns and simple swing metadata.

    Key behavior:
      - Detect candidate swings from local minima to later peaks.
      - Apply a dynamic retracement break and a minimum profit threshold
        to accept swings.
      - For each accepted swing, compute a per-bar score that decays
        with time-to-exit and increases with swing duration; keep the
        maximum score per bar across overlapping swings.
      - Produce helper columns useful for sizing and downstream logic:
        swing_dir, swing_gain_pct, last_buy, last_sell.
      - Compute a per-bar threshold according to `thresh_mode` and
        expose gap_to_thresh = signal_raw - signal_thresh.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    parts = []

    for day, day_df in df.groupby(df.index.normalize()):
        if day_df.empty:
            parts.append(day_df)
            continue

        closes = day_df[col_close].to_numpy()
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
                            score = gap * decay_time * boost_dur * 1000 # scaling by a factor of 1000 for better signal visibility
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