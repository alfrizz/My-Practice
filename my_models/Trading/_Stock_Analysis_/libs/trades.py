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
#     trail_stop_pct: float,
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
#     Returns a DataFrame with the new action column.
#     """
#     df = df.copy()
#     n = len(df)
#     df[col_action] = 0

#     sig = df[col_signal].to_numpy()
#     closes = df[col_close].to_numpy()
#     times = df.index.time
#     stop_thresh = trail_stop_pct / 100.0

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

#     return df



def generate_trade_actions(
    df: pd.DataFrame,
    col_signal: str,
    col_action: str,
    buy_threshold: float,
    trail_stop_pct: float,
    sess_start: dt.time,
    col_close: str
) -> pd.DataFrame:
    """
    From a continuous signal series, produce discrete per-bar trade actions and
    a per-bar trailing-stop level.

    Responsibilities (minimal, single-pass):
      - Create a discrete action column (0 default, 1 = entry, -1 = exit).
      - Maintain in_trade and peak_price state to compute a simple trailing stop.
      - Return the input DataFrame with two new columns:
          * col_action (int):  1 = Buy at this bar, -1 = Sell at this bar, 0 = no action
          * "trailstop_price" (float): trailing stop price while in trade; 0 before/after trades
    """

    df = df.copy()
    n = len(df)
    df[col_action] = 0  # default no-action

    sig = df[col_signal].to_numpy()
    closes = df[col_close].to_numpy()
    times = df.index.time
    stop_thresh = trail_stop_pct / 100.0

    in_trade = False
    peak_price = 0.0

    # minimal trail buffer (0.0 before/after trades)
    trail = np.zeros(n, dtype=float)

    for i in range(n):
        price = closes[i]
        t = times[i]

        if not in_trade:
            # entry condition: signal >= threshold and after session start
            if sig[i] >= buy_threshold and t >= sess_start:
                df.iat[i, df.columns.get_loc(col_action)] = 1
                in_trade = True
                peak_price = price
                # initialize trailing stop at entry bar
                trail[i] = peak_price * (1 - stop_thresh)
        else:
            # update peak and stop while in trade
            peak_price = max(peak_price, price)
            stop_level = peak_price * (1 - stop_thresh)
            trail[i] = stop_level
            # exit condition: price below stop and signal dropped below threshold
            if price < stop_level and sig[i] < buy_threshold:
                df.iat[i, df.columns.get_loc(col_action)] = -1
                in_trade = False
            else:
                df.iat[i, df.columns.get_loc(col_action)] = 0

    # attach canonical column expected by plotting code
    df["trailstop_price"] = trail
    return df


#########################################################################################################



def fees_for_one_share(price: float, side: str,
                       alpaca_comm_per_share: float = 0.0040,
                       finra_taf_per_share: float = 0.000166,
                       cat_per_share: float = 0.000009,
                       sec_fee_per_dollar: float = 0.00013810) -> dict:
    """
    Compute per-share commission + regulatory fees for one executed share.

    Returns precise per-share components and a rounded total for display.
    """
    assert side in ("buy", "sell")

    alpaca_comm = float(alpaca_comm_per_share)
    sec_raw = float(sec_fee_per_dollar) * float(price) if side == "sell" else 0.0
    finra_raw = float(finra_taf_per_share) if side == "sell" else 0.0
    cat_raw = float(cat_per_share)

    finra_billed = finra_raw
    cat_billed = cat_raw

    regulatory_billed = sec_raw + finra_billed + cat_billed
    total_per_share_billed = alpaca_comm + regulatory_billed

    return {
        "alpaca_comm": alpaca_comm,
        "sec_raw": sec_raw,
        "finra_billed": finra_billed,
        "cat_billed": cat_billed,
        "regulatory_billed": regulatory_billed,
        "total_per_share_billed": total_per_share_billed,
        "total_per_share_billed_rounded": round(total_per_share_billed, 8),
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



# def simulate_trading(
#     results_by_day_sign: Dict[dt.date, Tuple[pd.DataFrame, List]],
#     col_action: str,
#     sess_start: dt.time,
#     sess_end: dt.time,
#     shares_per_trade: int = 1
# ) -> Dict[dt.date, Tuple[pd.DataFrame, List, Dict[str, str]]]:
#     """
#     Minimal simulator: returns (df_sim, trades, perf) where perf contains only
#     formatted strings: BUY & HOLD, TRADES (list of strings), STRATEGY.
#     """
#     updated = {}
#     items = list(results_by_day_sign.items())
#     if len(items) > 1:
#         items = tqdm(items, desc="Simulating trading days", unit="day")

#     def _r3(x): return round(float(x), 3)

#     for day, (session_df, _) in items:
#         df = session_df.sort_index().copy()
#         position, cash = 0, 0.0
#         session_open_price = None

#         pos_buf, cash_buf, net_buf, act_buf, amt_buf = [], [], [], [], []

#         for ts, row in df.iterrows():
#             bid, ask = row["bid"], row["ask"]
#             sig = int(row[col_action]); now = ts.time()

#             # snapshot pre-trade state
#             pos_buf.append(position)
#             cash_buf.append(cash)
#             net_buf.append(cash + position * bid)
#             act_buf.append("")    # placeholder
#             amt_buf.append(0)     # placeholder

#             # apply trade
#             if sess_start <= now < sess_end:
#                 if session_open_price is None:
#                     session_open_price = ask
#                 if sig == 1:
#                     q = int(shares_per_trade); position += q
#                     fee = fees_for_one_share(price=ask, side="buy")["total_per_share_billed"] * q
#                     cash -= (ask * q) + fee
#                     action, amt = "Buy", q
#                 elif sig == -1 and position > 0:
#                     q = min(int(shares_per_trade), position); position -= q
#                     fee = fees_for_one_share(price=bid, side="sell")["total_per_share_billed"] * q
#                     cash += (bid * q) - fee
#                     action, amt = "Sell", -q
#                 else:
#                     action, amt = "Hold", 0
#             else:
#                 action, amt = "No trade", 0

#             act_buf[-1] = action
#             amt_buf[-1] = amt

#         # build df_sim
#         df_sim = df.copy()
#         df_sim["Position"] = pos_buf
#         df_sim["Cash"] = cash_buf
#         df_sim["NetValue"] = net_buf
#         df_sim["Action"] = act_buf
#         df_sim["TradedAmount"] = amt_buf

#         # detect round-trip trades (ask entry, bid exit)
#         trades = []
#         entry_p = entry_ts = None
#         for ts, row in df_sim.iterrows():
#             if row["Action"] == "Buy" and entry_p is None:
#                 entry_p, entry_ts = row["ask"], ts
#             elif row["Action"] == "Sell" and entry_p is not None:
#                 exit_p, exit_ts = row["bid"], ts
#                 pct = round(100 * (exit_p - entry_p) / entry_p, 3)
#                 trades.append(((entry_ts, exit_ts), (entry_p, exit_p), pct))
#                 entry_p = entry_ts = None
#         if entry_p is not None:
#             exit_p, exit_ts = df_sim["bid"].iat[-1], df_sim.index[-1]
#             pct = round(100 * (exit_p - entry_p) / entry_p, 3)
#             trades.append(((entry_ts, exit_ts), (entry_p, exit_p), pct))

#         # session slice and BH numbers (for formatted line)
#         session = df_sim.between_time(sess_start, sess_end)
#         if not session.empty:
#             open_ask = float(session["ask"].iloc[0]); close_bid = float(session["bid"].iloc[-1])
#             buy_fee = fees_for_one_share(price=open_ask, side="buy")["total_per_share_billed"]
#             sell_fee = fees_for_one_share(price=close_bid, side="sell")["total_per_share_billed"]
#             bh_gain_exact = (close_bid - open_ask) - (buy_fee + sell_fee)
#         else:
#             open_ask = close_bid = 0.0
#             buy_fee = sell_fee = 0.0
#             bh_gain_exact = 0.0

#         # Build TRADES formatted strings and compute per-trade PnL only for formatting
#         trades_lines = []
#         realized_vals = []
#         for i, ((_b, _s), (ep, xp), pct) in enumerate(trades, start=1):
#             buy_fee = fees_for_one_share(price=ep, side="buy")["total_per_share_billed"]
#             sell_fee = fees_for_one_share(price=xp, side="sell")["total_per_share_billed"]
#             total_fee = buy_fee + sell_fee
#             pnl = (xp - ep) - total_fee
#             r = _r3(pnl)
#             realized_vals.append(r)
#             trades_lines.append(f"Bid({i}) {_r3(xp)} - Ask({i}) {_r3(ep)} - Fee({i}) {_r3(total_fee)} = {_r3(pnl)}")

#         # Strategy bracket line (formatted only)
#         if realized_vals:
#             parts = " + ".join([f"Trade({i}) {_r3(v)}" for i, v in enumerate(realized_vals, start=1)])
#             strategy_line = f"[{parts} = {_r3(sum(realized_vals))}]"
#         else:
#             strategy_line = "[ = 0.000]"

#         bh_line = f"Bid(0) {_r3(close_bid)} - Ask(0) {_r3(open_ask)} - Fee(0) {_r3(buy_fee + sell_fee)} = {_r3(bh_gain_exact)}"

#         # PERF: only formatted strings (no numeric lists/fields)
#         perf = {
#             "BUY&HOLD": bh_line,
#             "TRADES": trades_lines,
#             "STRATEGY": strategy_line
#         }

#         updated[day] = (df_sim, trades, perf)

#     return updated


def simulate_trading(
    results_by_day_sign: Dict[dt.date, Tuple[pd.DataFrame, List]],
    col_action: str,
    sess_start: dt.time,
    sess_end: dt.time,
    shares_per_trade: int = 1
) -> Dict[dt.date, Tuple[pd.DataFrame, List, Dict[str, str]]]:
    """
    Minimal intraday simulator (short, canonical).

    - Snapshots pre-trade state per bar (Position, Cash, NetValue) so DF is
      useful for debugging and plotting.
    - Executes simple market buys (ask) and sells (bid) inside session window.
    - Computes per-round-trip trades (entry = ask on Buy, exit = bid on Sell).
    - Returns updated[day] = (df_sim, trades, perf) where perf contains ONLY
      readable strings:
        "BUY&HOLD" : "Bid(0) ... = <x>"
        "TRADES"   : [ "Bid(i) ... = <xi>", ... ]  (each on its own line)
        "STRATEGY" : "[Trade(1) x1 + Trade(2) x2 = <sum>]"
    - All numeric formatting rounded to 3 decimals.
    """
    updated = {}
    items = list(results_by_day_sign.items())
    if len(items) > 1:
        items = tqdm(items, desc="Simulating trading days", unit="day")

    def _r3(x): return round(float(x), 3)

    for day, (session_df, _) in items:
        df = session_df.sort_index().copy()
        position, cash = 0, 0.0
        session_open_price = None

        pos_buf, cash_buf, net_buf, act_buf, amt_buf = [], [], [], [], []

        for ts, row in df.iterrows():
            bid, ask = row["bid"], row["ask"]
            sig = int(row[col_action]); now = ts.time()

            # 1) snapshot pre-trade state (for df_sim)
            pos_buf.append(position)
            cash_buf.append(cash)
            net_buf.append(cash + position * bid)
            act_buf.append("")    # placeholder
            amt_buf.append(0)     # placeholder

            # 2) apply trade if in session
            if sess_start <= now < sess_end:
                if session_open_price is None:
                    session_open_price = ask

                if sig == 1:
                    q = int(shares_per_trade); position += q
                    fee = fees_for_one_share(price=ask, side="buy")["total_per_share_billed"] * q
                    cash -= (ask * q) + fee
                    action, amt = "Buy", q
                elif sig == -1 and position > 0:
                    q = min(int(shares_per_trade), position); position -= q
                    fee = fees_for_one_share(price=bid, side="sell")["total_per_share_billed"] * q
                    cash += (bid * q) - fee
                    action, amt = "Sell", -q
                else:
                    action, amt = "Hold", 0
            else:
                action, amt = "No trade", 0

            # 3) record action/amt into placeholders
            act_buf[-1] = action
            amt_buf[-1] = amt

        # build df_sim for plotting/debug
        df_sim = df.copy()
        df_sim["Position"] = pos_buf
        df_sim["Cash"] = cash_buf
        df_sim["NetValue"] = net_buf
        df_sim["Action"] = act_buf
        df_sim["TradedAmount"] = amt_buf

        # detect round-trip trades (Buy ask -> Sell bid)
        trades = []
        entry_p = entry_ts = None
        for ts, row in df_sim.iterrows():
            if row["Action"] == "Buy" and entry_p is None:
                entry_p, entry_ts = row["ask"], ts
            elif row["Action"] == "Sell" and entry_p is not None:
                exit_p, exit_ts = row["bid"], ts
                pct = round(100 * (exit_p - entry_p) / entry_p, 3)
                trades.append(((entry_ts, exit_ts), (entry_p, exit_p), pct))
                entry_p = entry_ts = None
        if entry_p is not None:
            exit_p, exit_ts = df_sim["bid"].iat[-1], df_sim.index[-1]
            pct = round(100 * (exit_p - entry_p) / entry_p, 3)
            trades.append(((entry_ts, exit_ts), (entry_p, exit_p), pct))

        # session buy&hold for formatted line
        session = df_sim.between_time(sess_start, sess_end)
        if not session.empty:
            open_ask = float(session["ask"].iloc[0]); close_bid = float(session["bid"].iloc[-1])
            buy_fee = fees_for_one_share(price=open_ask, side="buy")["total_per_share_billed"]
            sell_fee = fees_for_one_share(price=close_bid, side="sell")["total_per_share_billed"]
            bh_gain_exact = (close_bid - open_ask) - (buy_fee + sell_fee)
        else:
            open_ask = close_bid = 0.0
            buy_fee = sell_fee = 0.0
            bh_gain_exact = 0.0

        # build TRADES formatted strings (compute per-trade pnl only for formatting)
        trades_lines = []
        realized_vals = []
        for i, ((_b, _s), (ep, xp), pct) in enumerate(trades, start=1):
            buy_fee = fees_for_one_share(price=ep, side="buy")["total_per_share_billed"]
            sell_fee = fees_for_one_share(price=xp, side="sell")["total_per_share_billed"]
            total_fee = buy_fee + sell_fee
            pnl = (xp - ep) - total_fee
            r = _r3(pnl)
            realized_vals.append(r)
            trades_lines.append(f"Bid({i}) {_r3(xp)} - Ask({i}) {_r3(ep)} - Fee({i}) {_r3(total_fee)} = {_r3(pnl)}")

        # formatted Strategy  (strings only)
        if realized_vals:
            parts = " + ".join([f"Trade({i}) {_r3(v)}" for i, v in enumerate(realized_vals, start=1)])
            strategy_line = f"{parts} = {_r3(sum(realized_vals))}"
        else:
            strategy_line = "0.000"

        bh_line = f"Bid(0) {_r3(close_bid)} - Ask(0) {_r3(open_ask)} - Fee(0) {_r3(buy_fee + sell_fee)} = {_r3(bh_gain_exact)}"

        # PERF: only formatted strings (no numeric lists/fields)
        # use key without space as requested
        perf = {
            "BUY&HOLD": bh_line,
            "TRADES": trades_lines,
            "STRATEGY": strategy_line
        }

        updated[day] = (df_sim, trades, perf)

    return updated


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
    trail_stop_pct: float,
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
            trail_stop_pct,
            params.sess_start
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

