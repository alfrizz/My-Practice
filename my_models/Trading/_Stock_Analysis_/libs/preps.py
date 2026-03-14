from libs import params, plots, strats 

# Load Alpaca tools
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

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

from tqdm.auto import tqdm


#########################################################################################################


def fetch_bars(symbol: str, client, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetches historical data in yearly chunks to provide a progress bar (tqdm).
    Saves the final result using params.to_parquet_with_progress.
    """
    all_chunks = []
    
    # Create yearly intervals for the progress bar
    years = range(start.year, end.year + 1)
    pbar = tqdm(years, desc=f"🚀 Downloading {symbol}")

    for year in pbar:
        # Define the start and end for this specific chunk
        chunk_start = max(start, datetime(year, 1, 1, tzinfo=pytz.UTC))
        chunk_end   = min(end, datetime(year, 12, 31, 23, 59, tzinfo=pytz.UTC))
        
        # If the chunk start is after our overall end date, skip
        if chunk_start > end:
            continue

        pbar.set_description(f"📡 Fetching {symbol} for {year}")
        
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=chunk_start,
            end=chunk_end
        )
        
        res = client.get_stock_bars(req)
        
        # Only append if data exists for that year (Safe based on our diagnostic test)
        if not res.df.empty:
            all_chunks.append(res.df)

    print("\n✅ Download Complete. Consolidating data...")
    
    # --- SAFETY NET ADDED HERE ---
    if not all_chunks:
        print(f"⚠️ No data found for {symbol} in the given date range.")
        return pd.DataFrame()
    
    # 1. Combine all chunks
    df = pd.concat(all_chunks).sort_index()
    
    # 2. Cleanup: Extract symbol and convert timezone
    df = df.xs(symbol, level=0)
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    
    return df


# def detect_and_adjust_splits(df, forward_threshold=0.5, reverse_threshold=2, tol=0.05, vol_fact=1, dst_transition_dates=None):
#     """
#     Detects forward and reverse splits and adjusts price columns and volume.
#     Minimal addition: accepts dst_transition_dates (iterable of dates) and skips
#     candidate split events that fall on those dates to avoid DST artifacts.
#     """
#     df_adj = df.copy()
#     split_events = []
#     price_columns = ['open', 'high', 'low', 'close', 'ask', 'bid']
#     dst_transition_dates = set(dst_transition_dates or [])
#     print(f"Executing [detect_and_adjust_splits]...")

#     for i in range(1, len(df_adj)):
#         prev_close = df_adj.iloc[i-1]['close']
#         curr_close = df_adj.iloc[i]['close']
#         ratio = curr_close / prev_close

#         # Detect forward split (price drop)
#         if ratio < forward_threshold:
#             candidate_factor = round(1.0 / ratio)
#             if candidate_factor >= 2:
#                 expected_ratio = 1.0 / candidate_factor
#                 if abs(ratio - expected_ratio) < tol:
#                     event_time = df_adj.index[i]
#                     # skip if event falls on a DST transition date
#                     if event_time.date() in dst_transition_dates:
#                         print(f"Skipping forward-split candidate at {event_time} due to DST transition")
#                         continue
#                     print(f"Detected forward split on {event_time} with factor {candidate_factor} (ratio: {ratio:.4f})")
#                     split_events.append((event_time, candidate_factor, 'forward'))
#                     df_adj.loc[:df_adj.index[i-1], price_columns] /= candidate_factor
#                     df_adj.loc[:df_adj.index[i-1], 'volume'] *= candidate_factor * vol_fact

#         # Detect reverse split (price jump)
#         elif ratio > reverse_threshold:
#             candidate_factor = round(ratio)
#             if candidate_factor >= 2:
#                 expected_ratio = candidate_factor
#                 if abs(ratio - expected_ratio) < tol:
#                     event_time = df_adj.index[i]
#                     # skip if event falls on a DST transition date
#                     if event_time.date() in dst_transition_dates:
#                         print(f"Skipping reverse-split candidate at {event_time} due to DST transition")
#                         continue
#                     print(f"Detected reverse split on {event_time} with factor {candidate_factor} (ratio: {ratio:.4f})")
#                     split_events.append((event_time, candidate_factor, 'reverse'))
#                     df_adj.loc[:df_adj.index[i-1], price_columns] *= candidate_factor
#                     df_adj.loc[:df_adj.index[i-1], 'volume'] /= candidate_factor

#     return df_adj, split_events


def detect_and_adjust_splits(df: pd.DataFrame, forward_threshold=0.55, reverse_threshold=1.8, tol=0.05, vol_fact=1, dst_transition_dates=None):
    """
    Detects forward and reverse splits and adjusts price columns and volume.
    Vectorized for high-performance execution on large datasets.
    """
    df_adj = df.copy()
    split_events = []
    
    # Safely select columns that actually exist in the dataframe
    possible_price_cols = ['open', 'high', 'low', 'close', 'ask', 'bid']
    price_columns = [c for c in possible_price_cols if c in df_adj.columns]
    
    dst_transition_dates = set(dst_transition_dates or [])
    print(f"Executing [detect_and_adjust_splits]...")

    # 1. Vectorized ratio calculation (Instantaneous across millions of rows)
    # Using shift(1) divides today's close by yesterday's close
    ratios = df_adj['close'] / df_adj['close'].shift(1)

    # 2. Filter down to only the timestamps that breach our thresholds
    # We use <= and >= to catch exact integer splits (like 0.5 for 2-for-1)
    candidates = ratios[(ratios <= forward_threshold) | (ratios >= reverse_threshold)].dropna()

    # 3. Process only the handful of anomaly candidates
    for event_time, ratio in candidates.items():
        if event_time.date() in dst_transition_dates:
            print(f"Skipping split candidate at {event_time} due to DST transition")
            continue

        # --- Detect Forward Split (Price drop) ---
        if ratio <= forward_threshold:
            candidate_factor = round(1.0 / ratio)
            if candidate_factor >= 2:
                expected_ratio = 1.0 / candidate_factor
                if abs(ratio - expected_ratio) < tol:
                    print(f"Detected forward split on {event_time} with factor {candidate_factor} (ratio: {ratio:.4f})")
                    split_events.append((event_time, candidate_factor, 'forward'))
                    
                    mask = df_adj.index < event_time
                    df_adj.loc[mask, price_columns] /= candidate_factor
                    if 'volume' in df_adj.columns:
                        df_adj.loc[mask, 'volume'] *= (candidate_factor * vol_fact)

        # --- Detect Reverse Split (Price jump) ---
        elif ratio >= reverse_threshold:
            candidate_factor = round(ratio)
            if candidate_factor >= 2:
                expected_ratio = float(candidate_factor)
                if abs(ratio - expected_ratio) < tol:
                    print(f"Detected reverse split on {event_time} with factor {candidate_factor} (ratio: {ratio:.4f})")
                    split_events.append((event_time, candidate_factor, 'reverse'))
                    
                    mask = df_adj.index < event_time
                    df_adj.loc[mask, price_columns] *= candidate_factor
                    if 'volume' in df_adj.columns:
                        df_adj.loc[mask, 'volume'] /= candidate_factor

    return df_adj, split_events

    
#########################################################################################################


# def process_splits(df, ticker: str):
#     """
#     Load intraday Parquet data, detect & adjust corporate splits, 
#     and return the adjusted DataFrame.
#     """
    
#     # Ensure index is datetime
#     if not pd.api.types.is_datetime64_any_dtype(df.index):
#         df.index = pd.to_datetime(df.index)
    
#     # Create ask/bid using vectorization (faster than round() in a loop)
#     # We use conservative spread to simulate slippage
#     df['ask'] = (df['close'] * (1 + params.bidask_spread_pct/100)).round(4)
#     df['bid'] = (df['close'] * (1 - params.bidask_spread_pct/100)).round(4)

#     print("Plotting original data...")
#     plots.plot_close_volume(df, title="Before Adjusting Splits: Close Price and Volume")

#     # Pass empty set for transitions as requested
#     transitions = set()
    
#     # Assuming detect_and_adjust_splits is defined in the same file or imported
#     df_adjusted, split_events = detect_and_adjust_splits(df=df, dst_transition_dates=transitions)

#     if split_events:
#         print(f"Splits detected ({len(split_events)}). Plotting adjusted data...")
#         plots.plot_close_volume(df_adjusted, title="After Adjusting Splits: Close Price and Volume")
#     else:
#         print("No splits detected.")

#     return df_adjusted


def process_splits(df, ticker: str):
    """
    Adjusts data for splits FIRST, then applies session-aware spreads.
    This ensures slippage is calculated on the price relevant to the trade.
    """
    # 1. Ensure Index Integrity
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    
    # 2. DETECT AND ADJUST SPLITS FIRST
    # We must have clean, split-adjusted 'close' prices before calculating spreads.
    transitions = set()
    df, split_events = detect_and_adjust_splits(
        df=df, 
        dst_transition_dates=transitions
    )

    # 3. Identify Sessions
    times = df.index.time
    is_reg_session = (times >= params.sess_start_reg) & (times <= params.sess_end)
    
    # 4. Multi-Tiered Factors
    ext_factor = params.bidask_spread_pct_ext / 100.0  
    reg_factor = params.bidask_spread_pct_reg / 100.0  
    factors = np.where(is_reg_session, reg_factor, ext_factor)
    
    # 5. Calculate Ask/Bid on ADJUSTED Close
    df['ask'] = (df['close'] * (1.0 + factors)).round(4)
    df['bid'] = (df['close'] * (1.0 - factors)).round(4)

    # 6. Tick Guard
    df['ask'] = np.where(df['ask'] <= df['bid'], df['bid'] + 0.0001, df['ask'])

    if split_events:
        print(f"SUCCESS: Adjusted {len(split_events)} split events for {ticker}.")
    
    return df


#########################################################################################################


def prepare_interpolate_data(df: pd.DataFrame, tz_str: str = "US/Eastern") -> pd.DataFrame:
    """
    Normalize timestamps and linearly interpolate missing 1-minute bars.
    Optimized to avoid slow lambda maps.
    """
    print('[executing prepare_interpolate_data]')
    df = df.copy()
    
    # Ensure numerical precision for calculations
    df = df.astype(np.float64)

    # 1. Timezone Normalization (Fixed & Vectorized)
    print('Normalizing timezones...')
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Convert to target timezone
    if df.index.tz is None:
        idx_tz = df.index.tz_localize("UTC").tz_convert(tz_str)
    else:
        idx_tz = df.index.tz_convert(tz_str)

    # In Pandas, the fastest way to get offsets from a DatetimeIndex is 
    # calculating the difference between the localized index and its UTC version.
    # Standard Time (Winter) has a larger negative offset than DST (Summer).
    offsets = idx_tz.view('int64') - idx_tz.tz_convert(None).view('int64')
    is_standard_time = (offsets == offsets.min())
    
    # Standard Time Correction: subtract 1h for winter bars to maintain continuity
    # Using .values here ensures we are doing a fast numpy-level subtraction
    df.index = df.index - pd.to_timedelta(is_standard_time.astype(int), unit="h")

    # 2. Duplicate Check
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        print(f"prepare_interpolate_data: dropping {dup_count} duplicate rows.")
        df = df[~df.index.duplicated(keep="first")]

    df.sort_index(inplace=True)

    # 3. Fast Interpolation
    print('Interpolating missing bars...')
    # Using groupby.apply with a lambda is actually quite fast for daily grids
    def fill_day(grp):
        full_idx = pd.date_range(start=grp.index.min(), end=grp.index.max(), freq="1min")
        return grp.reindex(full_idx).interpolate(method="linear", limit_direction="both")

    df_out = df.groupby(df.index.normalize(), group_keys=False).apply(fill_day)

    # 4. Filter out 'Flat' days (no price movement/holidays)
    print('Filtering flat sessions...')
    df_out = df_out.groupby(df_out.index.normalize()).filter(lambda g: g['close'].nunique() > 1)
    
    return df_out

    
#########################################################################################################


def build_signal_per_day(
    df: pd.DataFrame,
    *,
    min_prof_thr: float,
    max_down_prop: float,
    gain_tightfact: float,
    tau_time: float,
    tau_dur: float,
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
    
    # --- ADDED TQDM HERE ---
    grouped = df.groupby(df.index.normalize())
    for day, day_df in tqdm(grouped, desc="Building Oracle Signal", leave=False):
        if day_df.empty:
            parts.append(day_df)
            continue

        closes = day_df[col_close].to_numpy()
        closes = np.where((~np.isfinite(closes)) | (closes == 0), 1e-12, closes) # replace with eps if NaN or zero

        times = day_df.index.to_numpy()
        n = len(day_df)

        target_signal = np.zeros(n, dtype=float)
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
                        # This ensures the score only applies from the bottom to the top
                        mask = (times >= times[buy_idx]) & (times <= exit_ts)
                        if mask.any():
                            gap = (closes[sell_idx] - closes[mask]) / closes[sell_idx]
                            mins_to_exit = (exit_ts - times[mask]) / np.timedelta64(1, "m")
                            decay_time = np.exp(-mins_to_exit / tau_time)
                            dur_min = (times[sell_idx] - times[buy_idx]) / np.timedelta64(1, "m")
                            boost_dur = 1 - np.exp(-dur_min / tau_dur)
                            score = gap * decay_time * boost_dur * 1000.0
                            target_signal[mask] = np.maximum(target_signal[mask], score)
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
        out["targ_signal"] = target_signal
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
    col_signal: str,
    thresh_mode: Union[str, float], # Accepts string OR float directly
    thresh_window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute per-day or per-day rolling thresholds for col_signal.
    thresh_mode may be a string token ("median_nonzero") or a numeric float value.
    thresh_window is used only for rolling modes.
    """
    def day_thresh(arr: np.ndarray) -> float:
        nz = arr[arr > 0]
        # It natively handles the float right here
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
        raise ValueError(f"Unknown thresh_mode: {thresh_mode}")

    parts: List[pd.DataFrame] = []
    for day, day_df in tqdm(df.groupby(df.index.normalize()), desc="Thresh per day", leave=False):
        out = day_df.copy()
        series = out[col_signal].to_numpy()

        if isinstance(thresh_mode, (int, float)) or thresh_mode in {
            "median_nonzero", "mean_nonzero", "p90", "p95", "p99", "median", "mean"
        }:
            out["signal_thresh"] = day_thresh(series)
        else:
            if thresh_mode == "roll_mean":
                out["signal_thresh"] = out[col_signal].rolling(window=thresh_window, min_periods=1).mean()
            elif thresh_mode == "roll_median":
                out["signal_thresh"] = out[col_signal].rolling(window=thresh_window, min_periods=1).median()
            elif thresh_mode in ("roll_p90", "roll_p95"):
                q = 0.90 if thresh_mode == "roll_p90" else 0.95
                out["signal_thresh"] = out[col_signal].rolling(window=thresh_window, min_periods=1).quantile(q)
            else:
                raise ValueError(f"Unknown rolling thresh_mode: {thresh_mode}")

        out["gap_to_thresh"] = out[col_signal] - out["signal_thresh"]
        parts.append(out)

    return pd.concat(parts).sort_index()

#########################################################################################################


# def smooth_scale_saturate(
#     series:   pd.Series,
#     window:   int,
#     beta_sat: float
# ) -> pd.Series:
#     """
#     1) Smoothing: centered rolling mean of width `window`.
#     2) Proportional scale into [0,1]: divide by the global max of the smoothed series.
#     3) Soft‐saturating exponential warp:
#          h(u) = (1 - exp(-β·u)) / (1 - exp(-β))
#        • h(0)=0, h(1)=1
#        • concave: lifts the bulk (h(u)>u for u∈(0,1)), gently compresses the top end.
#     Returns a new Series in [0,1], same index as `series`.
#     """
#     # 1) smooth
#     sm = series.rolling(window=window, center=True, min_periods=1).mean()

#     # 2) proportional scale → [0,1]
#     u = sm / sm.max()

#     # 3) soft‐saturate
#     expb = np.exp(-beta_sat)
#     # denominator = 1 - e^{-β}
#     denom = 1.0 - expb
#     warped = (1.0 - np.exp(-beta_sat * u)) / denom

#     return pd.Series(warped, index=series.index)