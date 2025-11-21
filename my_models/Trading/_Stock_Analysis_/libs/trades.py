from libs import params, plots

import math
import pandas as pd
from pandas import Timestamp
import numpy as np
import glob
import os

from pathlib import Path 
from datetime import datetime
import datetime as dt

import pytz
from typing import Optional, Dict, Tuple, List, Sequence, Union, Any
import matplotlib.pyplot as plt

from tqdm import tqdm

#########################################################################################################


def detect_and_adjust_splits(df, forward_threshold=0.5, reverse_threshold=2, tol=0.05, vol_fact=1):
    """
    Detects forward and reverse splits in the DataFrame and adjusts the price
    columns (plus volume) accordingly.
    
    Parameters:
      df (pd.DataFrame): DataFrame with at least the columns:
          ['open', 'high', 'low', 'close', 'ask', 'bid', 'volume']
      forward_threshold (float): Triggers a forward split check if current_close / prev_close < threshold.
      reverse_threshold (float): Triggers a reverse split check if current_close / prev_close > threshold.
      tol (float): Tolerance to check the "roundness" of the candidate split factor.
      
    Returns:
      df_adj (pd.DataFrame): The adjusted DataFrame.
      split_events (list): List of detected split events as (event timestamp, candidate_factor, split_type).
    """
    df_adj = df.copy()
    split_events = []
    price_columns = ['open', 'high', 'low', 'close', 'ask', 'bid']
    
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
                    print(f"Detected forward split on {event_time} with factor {candidate_factor} (ratio: {ratio:.4f})")
                    split_events.append((event_time, candidate_factor, 'forward'))
                    df_adj.loc[:df_adj.index[i-1], price_columns] /= candidate_factor
                    df_adj.loc[:df_adj.index[i-1], 'volume'] *= candidate_factor * vol_fact # additonal manual volume factor necessary in some cases (?)
        
        # Detect reverse split (price jump)
        elif ratio > reverse_threshold:
            candidate_factor = round(ratio)
            if candidate_factor >= 2:
                expected_ratio = candidate_factor
                if abs(ratio - expected_ratio) < tol:
                    event_time = df_adj.index[i]
                    print(f"Detected reverse split on {event_time} with factor {candidate_factor} (ratio: {ratio:.4f})")
                    split_events.append((event_time, candidate_factor, 'reverse'))
                    df_adj.loc[:df_adj.index[i-1], price_columns] *= candidate_factor
                    df_adj.loc[:df_adj.index[i-1], 'volume'] /= candidate_factor
                    
    return df_adj, split_events


##################


def process_splits(folder, ticker, bidask_spread_pct):
    """
    Processes the intraday CSV file for the given ticker from the specified folder.
    
    It looks for the unique CSV file in 'folder' whose name starts with '{ticker}_'.
    Then, it checks if the processed file already exists in 'dfs training/{ticker}_base.csv'.
      - If it exists, the function reads it, plots its data, and returns the DataFrame.
      - Otherwise, it reads the intraday CSV, keeps only necessary columns,
        creates the 'ask' and 'bid' columns using the provided bidask_spread_pct,
        plots the original data, detects and adjusts splits (also adjusting volume),
        plots the adjusted data if splits are detected, saves the resulting DataFrame, and returns it.
    
    Parameters:
      folder (str): The folder where the intraday CSV file is stored (e.g. "Intraday stocks").
      ticker (str): The ticker symbol used to locate the file and name the output.
      +    bidasktoclose_pct (percent): The one‐way per‐leg spread in percent.
      
    Returns:
      pd.DataFrame: The processed DataFrame.
    """

    # Find the unique intraday CSV file using glob.
    pattern = os.path.join(folder, f"{ticker}_*.csv")
    matching_files = glob.glob(pattern)
    if not matching_files:
        raise FileNotFoundError(f"No file found matching pattern: {pattern}")
    if len(matching_files) > 1:
        print(f"Warning: More than one file found for pattern {pattern}. Using the first one.")
    intraday_csv = matching_files[0]
    print(f"Reading data from {intraday_csv}")

    # Read the intraday CSV, using the 'datetime' column for dates.
    df = pd.read_csv(intraday_csv, index_col=0, parse_dates=["datetime"])
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # Create 'ask' and 'bid' columns using the given spread 
    df['ask'] = round(df['close'] * (1 + params.bidask_spread_pct/100), 4)
    df['bid'] = round(df['close'] * (1 - params.bidask_spread_pct/100), 4)
    
    # Plot the original data.
    print("Plotting original data...")
    plots.plot_close_volume(df, title="Before Adjusting Splits: Close Price and Volume")
    
    # Detect and adjust splits.
    df_adjusted, split_events = detect_and_adjust_splits(df=df)
    
    if split_events:
        print("Splits detected. Plotting adjusted data...")
        plots.plot_close_volume(df_adjusted, title="After Adjusting Splits: Close Price and Volume")
    else:
        print("No splits detected.")
  
    return df_adjusted

    
#########################################################################################################


def is_dst_for_day(day, tz_str="US/Eastern"):
    """
    Given a day (as a Timestamp or string), determine if that day is in DST for the specified timezone.
    We use noon on that day (to avoid ambiguity) and return True if DST is in effect.
    """
    tz = pytz.timezone(tz_str)
    dt = pd.Timestamp(day).replace(hour=12, minute=0, second=0, microsecond=0)
    # Localize if naive.
    if dt.tzinfo is None:
        dt = tz.localize(dt)
    else:
        dt = dt.astimezone(tz)
    return bool(dt.dst())



def prepare_interpolate_data(
    df,
    sess_premark,           # str or datetime.time: grid lower‐bound each day
    sess_start,             # str or datetime.time: official session open
    sess_end,               # str or datetime.time: official session close
    red_pretr_win=1,        # legacy argument (unused)
    tz_str="US/Eastern"     # timezone name for DST logic
) -> pd.DataFrame:
    """
    Exactly your old per‐day interpolation, but applied in one pass over
    the full multi‐day DataFrame. Steps:

    0) Clone input so we never mutate the caller’s df.
    1) Cast all columns to float64 (your “working” series).
    2) Shift each day’s timestamps forward 1h if that calendar day is in DST.
    3) For each calendar day:
       a) Build a 1-minute index from min(raw, sess_premark)
          through max(raw, sess_end).
       b) Reindex the day’s bars to that grid and linearly interpolate
          forward & backward.
    4) Concatenate all per-day frames, sort by timestamp, and drop
       duplicate timestamps (printing how many were removed).
    5) Finally, keep only the bars whose time lies between
       sess_premark and sess_end .
    """

    # 0) Clone
    df = df.copy()

    # 1) Cast all columns to float64 for numeric ops
    all_cols = df.columns.tolist()
    df[all_cols] = df[all_cols].astype(np.float64)

    # 2) DST adjustment: shift each day +1h if in DST
    ts = df.index.to_series()
    for day, grp in df.groupby(df.index.normalize()):
        add = pd.Timedelta(hours=1) if is_dst_for_day(day, tz_str) else pd.Timedelta(0)
        ts.loc[grp.index] = grp.index + add
    df.index = ts.sort_values()
    df.sort_index(inplace=True)

    # 3) Loop per calendar day to build & fill a minute grid
    filled_days = []
    for day, grp in df.groupby(df.index.normalize()):
        day_str = day.strftime("%Y-%m-%d")

        # a) grid bounds: earliest bar vs shifted start → latest bar vs close
        grid_start  = pd.Timestamp(f"{day_str} {sess_premark}")
        session_end = pd.Timestamp(f"{day_str} {sess_end}")

        idx_start = min(grp.index.min(), grid_start)
        idx_end   = max(grp.index.max(), session_end)

        # b) reindex on a full 1-min range and interpolate
        full_idx = pd.date_range(start=idx_start, end=idx_end, freq="1min")
        day_filled = grp.reindex(full_idx).interpolate(
            method="linear", limit_direction="both"
        )
        filled_days.append(day_filled)

    # 4) Concatenate all days, sort, and drop duplicates
    df_out = pd.concat(filled_days).sort_index()
    before = len(df_out)
    df_out = df_out[~df_out.index.duplicated(keep="first")]
    removed = before - len(df_out)
    print(f"prepare_interpolate_data: removed {removed} duplicate timestamps.")

    # 5) Slice to only [sess_premark, sess_end] each day
    df_out = df_out.between_time(sess_premark, sess_end)

    # 6) drop any calendar day whose close is perfectly flat (to avoid to get extra days as saturdays)
    df_out = (
        df_out.groupby(df_out.index.normalize())
          .filter(lambda grp: grp['close'].nunique() > 1)
    )

    return df_out


##########################################################################################################


def identify_trades_by_day(
    df: pd.DataFrame,
    min_prof_thr: float,
    max_down_prop: float,
    gain_tightening_factor: float,
    merging_retracement_thr: float,
    merging_time_gap_thr: float,
    sess_premark: str,
    sess_end: str
) -> Dict[dt.date, Tuple[pd.DataFrame, List]]:
    """
    Scan price data day by day to detect, filter and merge trades.

    1) Ensure the DataFrame has a DateTimeIndex.
    2) For each calendar day:
       a) Slice the session interval [sess_premark, sess_end] at 1min frequency.
       b) Identification phase:
          • Find local minima as buy points.
          • Forward-scan to local maxima, apply dynamic retracement rule:
            retracement > max_down_prop / (1 + gain_tightening_factor * gain_pct).
          • If profit% >= min_prof_thr, record (buy_date, sell_date, profit%).
       c) Merging phase:
          • Iteratively merge consecutive trades when:
            - second peak > first peak,
            - intermediate retracement ≤ merging_retracement_thr,
            - time gap ratio ≤ merging_time_gap_thr.
       d) Store (day_df, merged_trades) under the day’s date.
    Returns a dict mapping each trading date to its minute-sliced DataFrame
    and the list of merged-trade triples.
    """
    # 1) Enforce datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    results: Dict[dt.date, Tuple[pd.DataFrame, List]] = {}

    # 2) Process each day separately
    for day, group in df.groupby(df.index.normalize()):
        day_str  = day.strftime("%Y-%m-%d")
        start_ts = pd.Timestamp(f"{day_str} {sess_premark}")
        end_ts   = pd.Timestamp(f"{day_str} {sess_end}")
        idx      = pd.date_range(start=start_ts, end=end_ts, freq="1min")

        # a) Build the minute-sampled day DataFrame
        day_df = group.reindex(idx)
        if day_df.empty:
            results[day.date()] = (day_df, [])
            continue

        closes = day_df["close"].values
        dates  = day_df.index
        n      = len(day_df)
        i      = 1
        raw_trades = []

        # b) Identification Phase: local-min → local-max → retracement test
        while i < n - 1:
            if closes[i] < closes[i - 1] and closes[i] < closes[i + 1]:
                buy_idx   = i
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
                        gain_pc      = 100 * (max_so_far - buy_price) / buy_price
                        dyn_prop     = max_down_prop / (1 + gain_tightening_factor * gain_pc)
                        retracement = (max_so_far - price) / (max_so_far - buy_price)
                        if retracement > dyn_prop:
                            break
                    j += 1

                if cand_sell_idx is not None:
                    sell_idx   = cand_sell_idx
                    profit_pc  = 100 * (closes[sell_idx] - buy_price) / buy_price
                    if profit_pc >= min_prof_thr:
                        raw_trades.append((
                            (dates[buy_idx], dates[sell_idx]),
                            (buy_price, closes[sell_idx]),
                            profit_pc
                        ))
                    i = sell_idx + 1
                else:
                    i = buy_idx + 1
            else:
                i += 1

        # c) Merging Phase: combine neighbor trades by retracement & time-gap
        merged = raw_trades[:]
        changed = True
        while changed:
            changed = False
            new_list = []
            k = 0
            while k < len(merged):
                if k < len(merged) - 1:
                    t1, t2 = merged[k], merged[k+1]
                    (b1, s1), (_, _), _ = t1
                    (b2, s2), (_, _), _ = t2
                    price1 = t1[1][1]
                    price2 = t2[1][1]
                    if price2 > price1:
                        # retracement ratio
                        p1_low, p1_high = t1[1]
                        retr = ((price1 - t2[1][0]) / (price1 - p1_low)) if (price1 - p1_low) else 1.0
                        # time-gap ratio
                        dur1 = (t1[0][1] - t1[0][0]).total_seconds()
                        dur2 = (t2[0][1] - t2[0][0]).total_seconds()
                        gap  = (t2[0][0] - t1[0][1]).total_seconds()
                        time_gap = gap / (dur1 + dur2) if (dur1 + dur2) else 1.0
                        if retr <= merging_retracement_thr and time_gap <= merging_time_gap_thr:
                            # merge into one trade
                            new_buy, new_sell = t1[0][0], t2[0][1]
                            new_price_buy, new_price_sell = p1_low, price2
                            new_profit = 100 * (new_price_sell - new_price_buy) / new_price_buy
                            new_list.append(((new_buy, new_sell),
                                             (new_price_buy, new_price_sell),
                                             new_profit))
                            k += 2
                            changed = True
                            continue
                new_list.append(merged[k])
                k += 1
            merged = new_list

        # d) Store results for the day
        results[day.date()] = (day_df, merged)

    return results


##########################################################################################################


def compute_continuous_signal(
    day_df: pd.DataFrame,
    trades: List[Tuple[Tuple[dt.datetime, dt.datetime],
                       Tuple[float, float],
                       float]],
    tau_time: float,        # minutes “half‐life” for temporal decay of past scores
    tau_dur: float,         # minutes “half‐life” for duration‐based trade boost
    # smoothing_window: Optional[int] = None  # size of centered rolling window to smooth final signal
) -> pd.DataFrame:
    """
    Build raw and smoothed per‐minute signal from past trades.

    1) For each trade:
       a) gap[t]          = (sell_price – close[t]) / sell_price
       b) decay_time[t]   = exp(– mins_to_exit[t] / tau_time)     # in (0,1]
       c) duration_boost  = 1 – exp(– trade_duration_min / tau_dur) # in [0,1)
       d) raw_score[t]    = gap[t] * decay_time[t] * duration_boost
       e) signal_raw[t]   = max(prev_signal_raw[t], raw_score[t])
    """
    df = day_df.copy()
    n  = len(df)
    # initialize raw‐score array
    signal_raw = np.zeros(n, dtype=float)
    times      = df.index.to_numpy()

    for (buy_dt, sell_dt), (_, sell_price), _ in trades:
        exit_ts      = np.datetime64(sell_dt)
        mask         = times <= exit_ts
        if not mask.any():
            continue

        closes       = df["close"].to_numpy()[mask]
        # a) gap vector
        gap          = (sell_price - closes) / sell_price

        # b) temporal decay vector
        mins_to_exit = (exit_ts - times[mask]) / np.timedelta64(1, "m")
        decay_time   = np.exp(- mins_to_exit / tau_time)

        # c) duration‐boost scalar
        dur_min      = (sell_dt - buy_dt).total_seconds() / 60.0
        boost_dur    = 1 - np.exp(- dur_min / tau_dur)

        # d) raw score for this trade
        score        = gap * decay_time * boost_dur

        # e) keep the maximum raw score per minute
        signal_raw[mask] = np.maximum(signal_raw[mask], score)

    # assign raw signal series
    df["signal_raw"] = signal_raw

    return df


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

    
#########################################################################################################


# def generate_trade_actions(
#     df: pd.DataFrame,
#     col_signal: str,
#     col_action: str,
#     buy_threshold: float,
#     trailing_stop_pct: float,
#     sess_start: dt.time,
#     col_close: str
# ) -> pd.DataFrame:
#     """
#     From a continuous signal series, produce discrete trade actions.

#     1) Initialize action column to 0 (no position).
#     2) Monitor signal and price per minute:
#        • If not in trade and signal ≥ buy_threshold after sess_start → action=1 (enter).
#        • If in trade:
#          – Track peak price seen.
#          – Compute trailing stop level.
#          – If price drops below stop level and signal < buy_threshold → action=-1 (exit).
#          – Else action=0 (hold).
#     3) If still in trade at end of session, force action=-1 on last row.
#     Returns a DataFrame with the new action column.
#     """
#     df = df.copy()
#     n = len(df)
#     df[col_action] = 0

#     sig = df[col_signal].to_numpy()
#     closes = df[col_close].to_numpy()
#     times = df.index.time
#     stop_thresh = trailing_stop_pct / 100.0

#     in_trade = False
#     peak_price = 0.0

#     for i in range(n):
#         price = closes[i]
#         t = times[i]

#         if not in_trade:
#             if sig[i] >= buy_threshold and t >= sess_start:
#                 df.iat[i, df.columns.get_loc(col_action)] = 1
#                 in_trade = True
#                 peak_price = price
#         else:
#             peak_price = max(peak_price, price)
#             stop_level = peak_price * (1 - stop_thresh)
#             if price < stop_level and sig[i] < buy_threshold:
#                 df.iat[i, df.columns.get_loc(col_action)] = -1
#                 in_trade = False
#             else:
#                 df.iat[i, df.columns.get_loc(col_action)] = 0

#     if in_trade:
#         df.iat[-1, df.columns.get_loc(col_action)] = -1

#     return df

def generate_trade_actions(
    df: pd.DataFrame,
    col_signal: str,
    col_action: str,
    sell_minidx: int,
    buy_threshold: float,
    trail_stop_pct: float,
    sess_start: dt.time
) -> pd.DataFrame:
    """
    From a continuous signal series, produce discrete trade actions.

    - Enter when signal >= buy_threshold (after sess_start): action = 1.
    - While in trade track peak marketable sell price and compute trailing stop.
    - Exit (action = -1) when the marketable sell price falls below the trailing stop AND signal < buy_threshold.
    - Forced sell index via sell_minidx:
      * 0 -> no forced sell (do not auto-exit at end).
      * >0 -> absolute 0-based row index (clipped to last row).
      * <0 -> relative to end (-1 = last row, -2 = penultimate).
    - After the forced-sell index is reached, no further entries are allowed.
    """
    df = df.copy()
    n = len(df)
    df[col_action] = 0
    df["trailstop_price"] = float("nan")

    sig = df[col_signal].to_numpy()
    asks = df["ask"].to_numpy()
    bids = df["bid"].to_numpy()
    times = df.index.time
    stop_thresh = trail_stop_pct / 100.0

    # compute forced-sell target (None means no forced sell)
    if sell_minidx == 0:
        target = None
    elif sell_minidx > 0:
        target = min(sell_minidx, n - 1)
    else:
        target = max(0, n + sell_minidx)

    in_trade = False
    peak_price = 0.0
    entries_allowed = True

    for i in range(n):
        entry_price = asks[i]
        exit_price = bids[i]

        if target is not None and i == target:
            if in_trade:
                df.iat[i, df.columns.get_loc(col_action)] = -1
                in_trade = False
            entries_allowed = False
            continue

        stop_level = None  # will be set if/when in_trade becomes True for this row

        if not in_trade:
            if entries_allowed and sig[i] >= buy_threshold and times[i] >= sess_start:
                df.iat[i, df.columns.get_loc(col_action)] = 1
                in_trade = True
                peak_price = entry_price
                stop_level = peak_price * (1 - stop_thresh)
        else:
            peak_price = max(peak_price, exit_price)
            stop_level = peak_price * (1 - stop_thresh)
            if exit_price < stop_level and sig[i] < buy_threshold:
                df.iat[i, df.columns.get_loc(col_action)] = -1
                in_trade = False
            else:
                df.iat[i, df.columns.get_loc(col_action)] = 0

        if in_trade and stop_level is not None:
            df.iat[i, df.columns.get_loc("trailstop_price")] = stop_level
            
    return df

#########################################################################################################


# def fees_for_one_share(price: float, side: str,
#                        alpaca_comm_per_share: float = 0.0040,
#                        finra_taf_per_share: float = 0.000166,
#                        cat_per_share: float = 0.000009,
#                        sec_fee_per_dollar: float = 0.00013810,
#                        per_trade_minimum: float = 0.01) -> dict:
#     """
#     Compute regulatory + commission fees for a single share.
#     Return per‑share commission + regulatory fees for one executed share.

#     - price: share price used to convert SEC per‑dollar rate to per‑share.
#     - side: "buy" or "sell".
#     - sec_fee_per_dollar is $ per $1 (e.g. 0.00013810 => $138.10 / $1,000,000).
#     - FINRA/CAT small components are bumped to per_trade_minimum if >0 and <minimum.
#     """
#     assert side in ("buy", "sell")

#     # Alpaca commission (per share, both sides)
#     alpaca_comm = float(alpaca_comm_per_share)

#     # SEC fee: per-dollar of sale, applies only on sells. Convert to per-share by multiplying price.
#     sec_raw = float(sec_fee_per_dollar) * float(price) if side == "sell" else 0.0

#     # FINRA TAF: per-share on sells only
#     finra_raw = float(finra_taf_per_share) if side == "sell" else 0.0

#     # CAT: per executed equivalent share (both sides)
#     cat_raw = float(cat_per_share)

#     # Apply per-trade minimums for very small components (broker typically rounds small regulator fees)
#     finra_billed = finra_raw
#     cat_billed = cat_raw
#     if 0 < finra_raw < per_trade_minimum:
#         finra_billed = float(per_trade_minimum)
#     if 0 < cat_raw < per_trade_minimum:
#         cat_billed = float(per_trade_minimum)

#     regulatory_billed = sec_raw + finra_billed + cat_billed

#     total_per_share_billed = alpaca_comm + regulatory_billed

#     return {
#         "alpaca_comm": round(alpaca_comm, 8),
#         "sec_raw": round(sec_raw, 8),
#         "finra_billed": round(finra_billed, 8),
#         "cat_billed": round(cat_billed, 8),
#         "regulatory_billed": round(regulatory_billed, 8),
#         "total_per_share_billed": round(total_per_share_billed, 8),
#     }


def fees_for_one_share(price: float, side: str,
                       alpaca_comm_per_share: float = 0.0040,
                       finra_taf_per_share: float = 0.000166,
                       cat_per_share: float = 0.000009,
                       sec_fee_per_dollar: float = 0.00013810) -> dict:
    """
    Compute per‑share commission + regulatory fees for one executed share.

    - price: share price used to convert SEC per‑dollar rate to per‑share.
    - side: "buy" or "sell".
    - Returns exact per‑share components.
    """
    assert side in ("buy", "sell")

    alpaca_comm = float(alpaca_comm_per_share)
    sec_raw = float(sec_fee_per_dollar) * float(price) if side == "sell" else 0.0
    finra_raw = float(finra_taf_per_share) if side == "sell" else 0.0
    cat_raw = float(cat_per_share)

    # Keep exact per-share values.
    finra_billed = finra_raw
    cat_billed = cat_raw

    regulatory_billed = sec_raw + finra_billed + cat_billed
    total_per_share_billed = alpaca_comm + regulatory_billed

    return {
        "alpaca_comm": round(alpaca_comm, 8),
        "sec_raw": round(sec_raw, 8),
        "finra_billed": round(finra_billed, 8),
        "cat_billed": round(cat_billed, 8),
        "regulatory_billed": round(regulatory_billed, 8),
        "total_per_share_billed": round(total_per_share_billed, 8),
    }


##################################


# def simulate_trading(
#     results_by_day_sign: Dict[dt.date, Tuple[pd.DataFrame, List]],
#     col_action: str,
#     sess_start: dt.time,
#     sess_end: dt.time,
#     shares_per_trade: int = 1
# ) -> Dict[dt.date, Tuple[pd.DataFrame, List, Dict[str, object]]]:
#     """
#     Simulate minute-level P&L over multiple days for a single ticker.

#     - Trades `shares_per_trade` per Buy/Sell signal (default 1).
#     - Applies per‑share fees via fees_for_one_share (scaled by qty) when updating cash.
#     - Returns same structure: {date: (df_sim, trades, perf_stats)}.
#     """
#     updated_results: Dict[dt.date, Tuple[pd.DataFrame, List, Dict[str, object]]] = {}

#     items = list(results_by_day_sign.items())
#     if len(items) > 1:
#         items = tqdm(items, desc="Simulating trading days", unit="day")

#     for day, (session_df, _) in items:
#         # a) Prepare and sort day's DataFrame
#         df_day = session_df.sort_index().copy()
#         position, cash = 0, 0.0
#         session_open_price = None

#         # Buffers for per-bar metrics
#         positions, cash_balances = [], []
#         net_values, actions = [], []
#         traded_amounts = []
#         bh_running, st_running = [], []

#         # b) Iterate through each minute bar
#         for ts, row in df_day.iterrows():
#             price_bid, price_ask = row["bid"], row["ask"]
#             sig = int(row[col_action])
#             now = ts.time()

#             if sess_start <= now < sess_end:
#                 if session_open_price is None:
#                     session_open_price = price_ask

#                 if sig == 1:  # Buy shares_per_trade
#                     qty = int(shares_per_trade)
#                     position += qty
#                     # compute fees per share and scale to qty
#                     fee_detail = fees_for_one_share(price=price_ask, side="buy")
#                     total_fees = fee_detail["total_per_share_billed"] * qty
#                     # charge cash: price * qty + fees
#                     cash -= (price_ask * qty) + total_fees
#                     action, amt = "Buy", qty

#                 elif sig == -1 and position > 0:  # Sell up to shares_per_trade
#                     qty = min(int(shares_per_trade), position)
#                     position -= qty
#                     fee_detail = fees_for_one_share(price=price_bid, side="sell")
#                     total_fees = fee_detail["total_per_share_billed"] * qty
#                     # add cash: proceeds minus fees
#                     cash += (price_bid * qty) - total_fees
#                     action, amt = "Sell", -qty

#                 else:
#                     action, amt = "Hold", 0
#             else:
#                 action, amt = "No trade", 0

#             # Record per-bar state
#             positions.append(position)
#             cash_balances.append(round(cash, 6))
#             net_val = round(cash + position * price_bid, 6)
#             net_values.append(net_val)
#             actions.append(action)
#             traded_amounts.append(amt)

#             # Running buy-and-hold vs. strategy P&L
#             if session_open_price is not None:
#                 bh = round(price_bid - session_open_price, 6)
#                 st = net_val
#             else:
#                 bh = st = 0.0

#             bh_running.append(bh)
#             st_running.append(round(st, 6))

#         # c) Build simulation DataFrame with all metrics
#         df_sim = df_day.copy()
#         df_sim["Position"]        = positions
#         df_sim["Cash"]            = cash_balances
#         df_sim["NetValue"]        = net_values
#         df_sim["Action"]          = actions
#         df_sim["TradedAmount"]    = traded_amounts
#         df_sim["BuyHoldEarning"]  = bh_running
#         df_sim["StrategyEarning"] = st_running
#         df_sim["EarningDiff"]     = df_sim["StrategyEarning"] - df_sim["BuyHoldEarning"]

#         # d) Identify round-trip trades and compute % returns
#         trades: List[Tuple[Tuple[pd.Timestamp, pd.Timestamp],
#                           Tuple[float, float],
#                           float]] = []
#         entry_price = None
#         entry_ts    = None

#         for ts, row in df_sim.iterrows():
#             if row["Action"] == "Buy" and entry_price is None:
#                 entry_price, entry_ts = row["ask"], ts
#             elif row["Action"] == "Sell" and entry_price is not None:
#                 exit_price, exit_ts = row["bid"], ts
#                 gain_pct = 100 * (exit_price - entry_price) / entry_price
#                 trades.append((
#                     (entry_ts, exit_ts),
#                     (entry_price, exit_price),
#                     round(gain_pct, 3)
#                 ))
#                 entry_price = entry_ts = None

#         # Force liquidate any open position at end-of-day
#         if entry_price is not None:
#             exit_price, exit_ts = df_sim["bid"].iat[-1], df_sim.index[-1]
#             gain_pct = 100 * (exit_price - entry_price) / entry_price
#             trades.append((
#                 (entry_ts, exit_ts),
#                 (entry_price, exit_price),
#                 round(gain_pct, 3)
#             ))

#         # e) Compute day-level performance statistics
#         session = df_sim.between_time(sess_start, sess_end)
#         if not session.empty:
#             open_ask, close_bid = session["ask"].iloc[0], session["bid"].iloc[-1]
#             bh_gain = round(close_bid - open_ask, 3)
#             st_gain = round(session["NetValue"].iloc[-1], 3)
#         else:
#             bh_gain = st_gain = 0.0

#         perf_stats = {
#             "Buy & Hold Return ($)": bh_gain,
#             "Strategy Return ($)"  : st_gain,
#             "Trades Returns ($)"   : [
#                 round((pct / 100) * prices[0], 3)
#                 for (_, _), prices, pct in trades
#             ]
#         }

#         updated_results[day] = (df_sim, trades, perf_stats)

#     return updated_results


def simulate_trading(
    results_by_day_sign: Dict[dt.date, Tuple[pd.DataFrame, List]],
    col_action: str,
    sess_start: dt.time,
    sess_end: dt.time,
    shares_per_trade: int = 1
) -> Dict[dt.date, Tuple[pd.DataFrame, List, Dict[str, object]]]:
    """
    Simulate minute-level P&L per day applying per-share fees, accumulate raw fees
    during the day and round up the day total to cents at EOD (Alpaca behaviour).
    Returns per-day (df_sim, trades, perf_stats).
    """
    import math

    updated_results: Dict[dt.date, Tuple[pd.DataFrame, List, Dict[str, object]]] = {}

    items = list(results_by_day_sign.items())
    if len(items) > 1:
        items = tqdm(items, desc="Simulating trading days", unit="day")

    for day, (session_df, _) in items:
        df_day = session_df.sort_index().copy()
        position, cash = 0, 0.0
        session_open_price = None

        positions, cash_balances = [], []
        net_values, actions = [], []
        traded_amounts = []
        bh_running, st_running = [], []

        day_fee_acc = 0.0  # accumulate raw per-trade fees across the day

        for ts, row in df_day.iterrows():
            price_bid, price_ask = row["bid"], row["ask"]
            sig = int(row[col_action])
            now = ts.time()

            if sess_start <= now < sess_end:
                if session_open_price is None:
                    session_open_price = price_ask

                if sig == 1:
                    qty = int(shares_per_trade)
                    position += qty
                    fee_detail = fees_for_one_share(price=price_ask, side="buy")
                    total_fees = fee_detail["total_per_share_billed"] * qty
                    day_fee_acc += total_fees
                    cash -= (price_ask * qty) + total_fees
                    action, amt = "Buy", qty

                elif sig == -1 and position > 0:
                    qty = min(int(shares_per_trade), position)
                    position -= qty
                    fee_detail = fees_for_one_share(price=price_bid, side="sell")
                    total_fees = fee_detail["total_per_share_billed"] * qty
                    day_fee_acc += total_fees
                    cash += (price_bid * qty) - total_fees
                    action, amt = "Sell", -qty

                else:
                    action, amt = "Hold", 0
            else:
                action, amt = "No trade", 0

            positions.append(position)
            cash_balances.append(round(cash, 6))
            net_val = round(cash + position * price_bid, 6)
            net_values.append(net_val)
            actions.append(action)
            traded_amounts.append(amt)

            if session_open_price is not None:
                bh = round(price_bid - session_open_price, 6)
                st = net_val
            else:
                bh = st = 0.0

            bh_running.append(bh)
            st_running.append(round(st, 6))

        # At EOD: round up the day's accumulated fees to cents and adjust cash so net fees == rounded amount
        day_fee_rounded = math.ceil(day_fee_acc * 100.0) / 100.0
        cash += (day_fee_acc - day_fee_rounded)

        # Build simulation DataFrame (rest unchanged)
        df_sim = df_day.copy()
        df_sim["Position"]        = positions
        df_sim["Cash"]            = cash_balances
        df_sim["NetValue"]        = net_values
        df_sim["Action"]          = actions
        df_sim["TradedAmount"]    = traded_amounts
        df_sim["BuyHoldEarning"]  = bh_running
        df_sim["StrategyEarning"] = st_running
        df_sim["EarningDiff"]     = df_sim["StrategyEarning"] - df_sim["BuyHoldEarning"]

        trades: List[Tuple[Tuple[pd.Timestamp, pd.Timestamp],
                          Tuple[float, float],
                          float]] = []
        entry_price = None
        entry_ts    = None

        for ts, row in df_sim.iterrows():
            if row["Action"] == "Buy" and entry_price is None:
                entry_price, entry_ts = row["ask"], ts
            elif row["Action"] == "Sell" and entry_price is not None:
                exit_price, exit_ts = row["bid"], ts
                gain_pct = 100 * (exit_price - entry_price) / entry_price
                trades.append((
                    (entry_ts, exit_ts),
                    (entry_price, exit_price),
                    round(gain_pct, 3)
                ))
                entry_price = entry_ts = None

        if entry_price is not None:
            exit_price, exit_ts = df_sim["bid"].iat[-1], df_sim.index[-1]
            gain_pct = 100 * (exit_price - entry_price) / entry_price
            trades.append((
                (entry_ts, exit_ts),
                (entry_price, exit_price),
                round(gain_pct, 3)
            ))

        session = df_sim.between_time(sess_start, sess_end)
        if not session.empty:
            open_ask, close_bid = session["ask"].iloc[0], session["bid"].iloc[-1]
            bh_gain = round(close_bid - open_ask, 3)
            st_gain = round(session["NetValue"].iloc[-1], 3)
        else:
            bh_gain = st_gain = 0.0

        perf_stats = {
            "Buy & Hold Return ($)": bh_gain,
            "Strategy Return ($)"  : st_gain,
            "Trades Returns ($)"   : [
                round((pct / 100) * prices[0], 3)
                for (_, _), prices, pct in trades
            ]
        }

        updated_results[day] = (df_sim, trades, perf_stats)

    return updated_results

#########################################################################################################


def run_trading_pipeline(
    df: pd.DataFrame,
    col_signal: str,
    col_action: str,
    min_prof_thr: float,
    max_down_prop: float,
    gain_tightening_factor: float,
    merging_retracement_thr: float,
    merging_time_gap_thr: float,
    tau_time: float,
    tau_dur: float,
    trailing_stop_pct: float,
    buy_threshold: float,
    smoothing_window: int,
    beta_sat: float,
    col_close: str = "close"
) -> Optional[Dict[dt.date, Tuple[pd.DataFrame, List, Dict[str, Any]]]]:
    """
    End-to-end trading pipeline:

    1) Detect & merge candidate trades per day.
    2) Compute per-day raw continuous signals (unsmoothed).
    3) Concatenate all days’ raw signals, then apply:
       a) smoothing (centered rolling mean, window=smoothing_window)
       b) proportional scale into [0,1] by dividing by the global max
       c) soft‐saturating exponential warp with curvature beta_sat
       → yields one global “warped” pd.Series over the full index.
    4) Split that warped series back into each day’s df_sig[col_signal].
    5) Generate discrete actions & simulate P&L.

    Returns mapping day → (sim_df, trades, perf_stats).
    """
    # 1) Detect & merge
    print("Detecting & merging trades by day …")
    trades_by_day = identify_trades_by_day(
        df,
        min_prof_thr,
        max_down_prop,
        gain_tightening_factor,
        merging_retracement_thr,
        merging_time_gap_thr,
        params.sess_premark,
        params.sess_end,
    )

    # 2) Compute raw continuous signals
    print("Computing raw continuous signals …")
    raw_signals: Dict[dt.date, Tuple[pd.DataFrame, List]] = {}
    for day, (day_df, trades) in trades_by_day.items():
        df_sig = compute_continuous_signal(
            day_df,
            trades,
            tau_time=tau_time,
            tau_dur=tau_dur
        )
        raw_signals[day] = (df_sig, trades)

    # 3) Build one global Series of raw signals across all days
    print(f"Smoothing, scaling & soft-saturating …")
    # concatenate in chronological order
    all_raw = pd.concat(
        [df_sig["signal_raw"] for df_sig, _ in raw_signals.values()]
    ).sort_index()

    # apply the combined smoothing / scaling / saturating
    warped = smooth_scale_saturate(
        series=all_raw,
        window=smoothing_window,
        beta_sat=beta_sat
    )

    # 4) assign back per day and generate actions
    signaled: Dict[dt.date, Tuple[pd.DataFrame, List]] = {}
    for day, (df_sig, trades) in raw_signals.items():
        # pick the warped values matching this day's timestamps
        df_sig[col_signal] = warped.loc[df_sig.index]

        # generate trade actions on the warped [0,1] signal
        df_act = generate_trade_actions(
            df_sig,
            col_signal,
            col_action,
            buy_threshold,
            trailing_stop_pct,
            params.sess_start,
            col_close,
        )
        signaled[day] = (df_act, trades)

    # 5) simulate P&L
    sim_results = simulate_trading(
        results_by_day_sign=signaled,
        col_action=col_action,
        sess_start=params.sess_start,
        sess_end=params.sess_end,
    )

    return sim_results

