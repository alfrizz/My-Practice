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



def process_splits(folder, ticker, bidasktoclose_pct):
    """
    Processes the intraday CSV file for the given ticker from the specified folder.
    
    It looks for the unique CSV file in 'folder' whose name starts with '{ticker}_'.
    Then, it checks if the processed file already exists in 'dfs training/{ticker}_base.csv'.
      - If it exists, the function reads it, plots its data, and returns the DataFrame.
      - Otherwise, it reads the intraday CSV, keeps only necessary columns,
        creates the 'ask' and 'bid' columns using the provided bidasktoclose_spread,
        plots the original data, detects and adjusts splits (also adjusting volume),
        plots the adjusted data if splits are detected, saves the resulting DataFrame, and returns it.
    
    Parameters:
      folder (str): The folder where the intraday CSV file is stored (e.g. "Intraday stocks").
      ticker (str): The ticker symbol used to locate the file and name the output.
      +    bidasktoclose_pct (percent): The one‐way per‐leg spread in percent (realistic 0.5%).
      
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

    bidasktoclose_spread = bidasktoclose_pct / 100
    
    # Create 'ask' and 'bid' columns using the given spread 
    df['ask'] = round(df['close'] * (1 + bidasktoclose_spread), 4)
    df['bid'] = round(df['close'] * (1 - bidasktoclose_spread), 4)
    
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


#########################################################################################################

# B1.1
def identify_trades(
    df,                     # DataFrame with a datetime index and a 'close' column
    min_prof_thr,           # minimum profit percent to accept a trade
    max_down_prop,          # base maximum retracement fraction
    gain_tightening_factor, # factor tightening retracement as gains grow
    merging_retracement_thr,# max retracement ratio allowed to merge two trades
    merging_time_gap_thr    # max time‐gap ratio allowed to merge two trades
):
    """
    Detect and merge intraday trades using a dynamic retracement rule.

    Identification Phase
      • Buy at each local minimum in the 'close' series.
      • Track the highest subsequent price; if price falls
        by more than max_down_prop/(1+gain_tightening_factor·gain%)
        from that peak, close the trade at the peak.
      • Record trades whose profit ≥ min_prof_thr.

     Merging Phase (repeat until no changes)
          • Only consider merging when the second trade’s peak > first trade’s peak.
          • Compute intermediate retracement:
              (first_peak – second_buy) / (first_peak – first_buy)
            and require ≤ merging_retracement_thr.
          • Compute time-gap ratio:
              (second_buy_time – first_sell_time) /
              (duration1 + duration2)
            and require ≤ merging_time_gap_thr.
          • If both conditions hold, merge into one trade from the first buy to the second sell,
            then recalculate profit.
    """
    trades = []
    # Extract the close prices and timestamps.
    closes = df['close'].values
    dates = df.index
    n = len(df)
    i = 1  # Start from the second element to allow a look-back.

    # --- Identification Phase ---
    # Scan through the DataFrame to find local minima as candidate buy points and then determine the trade exit.
    while i < n - 1:
        # Look for a local minimum, where the price is lower than its neighbors.
        if closes[i] < closes[i - 1] and closes[i] < closes[i + 1]:
            buy_index = i
            buy_date = dates[buy_index]
            buy_price = closes[buy_index]
            
            # Initialize candidate sell using the buy price.
            max_so_far = buy_price
            candidate_sell_index = None
            
            # Scan forward to find the candidate sell (the local maximum) while monitoring for retracement.
            j = i + 1
            while j < n:
                current_price = closes[j]
                if current_price > max_so_far:
                    max_so_far = current_price
                    candidate_sell_index = j
                else:
                    # Only check retracement if there's been an increase.
                    if max_so_far > buy_price:
                        gain_pc = 100 * (max_so_far - buy_price) / buy_price
                        # Dynamically tighten the allowed retracement as gain increases.
                        dyn_prop = max_down_prop / (1 + gain_tightening_factor * gain_pc)
                        
                        retracement = (max_so_far - current_price) / (max_so_far - buy_price)
                        if retracement > dyn_prop:
                            break
                j += 1

            # If a candidate sell is found, record the trade.
            if candidate_sell_index is not None:
                sell_index = candidate_sell_index
                sell_date = dates[sell_index]
                sell_price = closes[sell_index]
                profit_pc = ((sell_price - buy_price) / buy_price) * 100
                if profit_pc >= min_prof_thr:
                    trades.append(((buy_date, sell_date), (buy_price, sell_price), profit_pc))
                # Jump past the sell to avoid overlapping trades.
                i = sell_index + 1
            else:
                i = buy_index + 1
        else:
            i += 1

    # --- Merging Phase ---
    # Iteratively merge consecutive trades based on intermediate retracement and time gap criteria.
    merged_trades = trades[:]  # Copy the identified trades.
    changed = True
    while changed:
        changed = False
        new_trades = []
        i = 0
        while i < len(merged_trades):
            # Consider the possibility of merging with the next trade if available.
            if i < len(merged_trades) - 1:
                trade1 = merged_trades[i]
                trade2 = merged_trades[i + 1]
                
                # Unpack trade details.
                (buy_date1, sell_date1), (buy_price1, sell_price1), profit1 = trade1
                (buy_date2, sell_date2), (buy_price2, sell_price2), profit2 = trade2
                
                # Only consider merging if the second trade's peak (sell price) is higher than the first’s.
                if sell_price2 > sell_price1:
                    # --- Check Intermediate Retracement ---
                    # Compute the retracement between the first trade's peak and the second trade's entry.
                    profit_range = sell_price1 - buy_price1
                    if profit_range > 0:
                        retracement_ratio = (sell_price1 - buy_price2) / profit_range
                    else:
                        retracement_ratio = 1.0  # Fail-safe
                    
                    # --- Check Time Gap Using Active Durations ---
                    # Calculate the active duration (from buy to sell) for each trade.
                    active_duration1 = (sell_date1 - buy_date1).total_seconds()
                    active_duration2 = (sell_date2 - buy_date2).total_seconds()
                    total_active_duration = active_duration1 + active_duration2
                    # Calculate the gap between the first trade's sell and the second trade's buy.
                    gap_seconds = (buy_date2 - sell_date1).total_seconds()
                    if total_active_duration > 0:
                        time_gap_ratio = gap_seconds / total_active_duration
                    else:
                        time_gap_ratio = 1.0  # Fail-safe
                    
                    # Merge the two trades if both conditions are met.
                    if retracement_ratio <= merging_retracement_thr and time_gap_ratio <= merging_time_gap_thr:
                        # The merged trade uses the buy data from trade 1 and the sell data from trade 2.
                        new_buy_date = buy_date1
                        new_sell_date = sell_date2
                        new_buy_price = buy_price1
                        new_sell_price = sell_price2
                        new_profit = ((new_sell_price - new_buy_price) / new_buy_price) * 100
                        merged_trade = ((new_buy_date, new_sell_date), (new_buy_price, new_sell_price), new_profit)
                        new_trades.append(merged_trade)
                        i += 2  # Skip the next trade since it has been merged.
                        changed = True
                        continue
            # If no merge occurred at this position, simply add the current trade.
            new_trades.append(merged_trades[i])
            i += 1
        merged_trades = new_trades

    return merged_trades


#########################################################################################################
# B1
def identify_trades_daily(
    df: pd.DataFrame,  
    min_prof_thr: float,
    max_down_prop: float,
    gain_tightening_factor: float,
    merging_retracement_thr: float,
    merging_time_gap_thr: float,
    sess_premark: str,
    sess_end: str,
    day_to_check: Optional[str] = None
) -> Dict[dt.date, Tuple[pd.DataFrame, List]]:
    """
    1) Slice df into daily sessions (sess_premark→sess_end).
    2) Identify trades via local-extrema + retracement logic.
    3) If day_to_check is set, only keep that date in the returned dict.
    Returns a dict mapping date -> (day_df, trades), even if trades == [].
    """
    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    results: Dict[dt.date, Tuple[pd.DataFrame, List]] = {}
    for day, group in df.groupby(df.index.normalize()):
        # early filter: skip all but the target date
        if day_to_check and day.strftime("%Y-%m-%d") != day_to_check:
            continue

        day_str  = day.strftime("%Y-%m-%d")
        start_ts = pd.Timestamp(f"{day_str} {sess_premark}")
        end_ts   = pd.Timestamp(f"{day_str} {sess_end}")
        idx      = pd.date_range(start=start_ts, end=end_ts, freq="1min")

        day_df = group.reindex(idx)
        if day_df.empty:
            continue

        trades = identify_trades(
            df                      = day_df,
            min_prof_thr            = min_prof_thr,
            max_down_prop           = max_down_prop,
            gain_tightening_factor  = gain_tightening_factor,
            merging_retracement_thr = merging_retracement_thr,
            merging_time_gap_thr    = merging_time_gap_thr
        )

        # minimum fix: record every day, even when trades == []
        results[day.date()] = (day_df, trades)

    return results


#########################################################################################################

#B2.1
def compute_continuous_signal(
    day_df: pd.DataFrame,
    trades: list,
    pre_entry_decay: float,
    short_penal_decay: float,
    smoothing_window: int
) -> pd.DataFrame:
    """
    Build an un‐normalized per‐bar confidence curve for one trading day.

    1) As before, compute raw “signal” = profit_gap * time_decay * dur_penalty.
    2) Optionally apply a centered moving‐average smoother.
    3) Return df["signal"] as raw (>=0) values—no scaling to [0,1] here.
    """
    df = day_df.copy()
    n = len(df)
    signal = np.zeros(n, dtype=float)
    times = df.index.to_numpy()

    for (buy_dt, sell_dt), (_, sell_price), _ in trades:
        exit_ts = np.datetime64(sell_dt)
        mask = times <= exit_ts
        if not mask.any():
            continue

        closes = df["close"].to_numpy()[mask]
        gap = np.maximum((sell_price - closes) / sell_price, 0.0)

        mins_to_exit = (exit_ts - times[mask]) / np.timedelta64(1, "m")
        decay_time = np.exp(-pre_entry_decay * mins_to_exit)

        dur_min = max((sell_dt - buy_dt).total_seconds() / 60.0, 1.0)
        decay_dur = np.exp(-short_penal_decay / dur_min)

        trade_score = gap * decay_time * decay_dur
        signal[mask] = np.maximum(signal[mask], trade_score)

    df["signal"] = signal

    # --- smoothing step ---
    if smoothing_window:
        df["signal"] = (
            df["signal"]
            .rolling(window=smoothing_window, center=True, min_periods=1)
            .mean()
        )

    return df


#########################################################################################################

# B2.2
def generate_trade_actions(
    df,                     # DataFrame w/ DatetimeIndex and a 'close' column
    col_signal,             # name of the input signal column to use
    col_action,             # name for the output action column to add
    buy_threshold,          # signal cutoff to enter a trade
    trailing_stop_pct,      # trailing stop distance in percent (e.g. 0.5 for 0.5%)
    sess_start,             # earliest time to allow entries (datetime.time or "HH:MM" string)
    col_close="close"       # ← name of the column to use for price
):
    """
    Generate per-bar trade actions (+1=buy, 0=hold, -1=sell) for one trading day
    using a fixed entry threshold and a simple trailing-stop exit rule.

    Functionality:
      1) Work on a copy, initialize action column to 0.
      2) Convert trailing_stop_pct to a decimal stop threshold.
      3) Normalize sess_start to a time object if passed as string.
      4) Loop through each bar:
         - If not in a trade:
             • If signal ≥ buy_threshold AND time ≥ sess_start → mark buy (+1),
               set in_trade=True, record entry price as max_price.
         - If in a trade:
             • Update max_price to the highest close since entry.
             • Compute stop_level = max_price * (1 - trailing_stop_thresh).
             • If close < stop_level AND signal < buy_threshold → mark sell (-1),
               set in_trade=False.
      5) At end of day, if still in_trade, force a sell on the last bar.
    """
    
    # Work on a copy to avoid side-effects
    df = df.copy()
    n = len(df)

    # Initialize trade_action and state
    df[col_action] = 0
    
    # Convert trailing stop percentages into decimal thresholds
    trailing_stop_thresh = trailing_stop_pct / 100.0
    
    in_trade = False
    max_price = None

    # Convert sess_start to time if needed
    if isinstance(sess_start, str):
        sess_start = pd.to_datetime(sess_start).time()

    # Extract series for speed
    sig = df[col_signal].values
    times = df.index.time

    closes = df[col_close].values
    for i in range(n):
        t = times[i]
        price = closes[i]

        if not in_trade:
            # Check buy condition: signal crosses threshold, and time ≥ sess_start
            if sig[i] >= buy_threshold and t >= sess_start:
                df.iat[i, df.columns.get_loc(col_action)] = 1
                in_trade = True
                max_price = price
            # else: remain flat (0)
        else:
            # We’re in a trade: update the highest price seen so far
            if price > max_price:
                max_price = price

            # Compute simple trailing-stop level
            stop_level = max_price * (1 - trailing_stop_thresh)

            # Sell if price falls below the stop
            if price < stop_level and sig[i] < buy_threshold:
                df.iat[i, df.columns.get_loc(col_action)] = -1
                in_trade = False
            else:
                df.iat[i, df.columns.get_loc(col_action)] = 0

    # If still in trade at end-of-day, force a sell on last row
    if in_trade:
        df.iat[-1, df.columns.get_loc(col_action)] = -1

    return df


#########################################################################################################

# B2
# def add_trade_signal_to_results(
#     results_by_day_trad,    # dict(day -> (day_df, trades)) from identify_trades
#     col_signal,             # name for the signal column to use
#     col_action,             # name for the trade-action column to add
#     min_prof_thr,           # minimum profit % (unused here, kept for API consistency)
#     sess_start,             # session start time for generate_trade_actions
#     pre_entry_decay,        # decay rate applied before trade entry
#     short_penal_decay,      # decay rate penalizing missed early entries
#     trailing_stop_pct,      # trailing stop distance in percent
#     buy_threshold,          # signal threshold to enter trades in generate_trade_actions
#     top_percentile=1,       # percentile of bars to cap at signal=1.0
#     smoothing_window=False  # optional window size to smooth raw signal
# ):
#     """
#     Compute continuous trading signals and generate discrete actions for each day.

#     Functionality:
#       1) First pass: for each day in results_by_day_trad
#          - Call compute_continuous_signal(day_df, trades, pre_entry_decay,
#            short_penal_decay, smoothing_window) to produce df_sig with 'signal'.
#          - Accumulate all df_sig["signal"] values across days.
#       2) Determine a global cutoff = (100 - top_percentile)th percentile of all signals.
#          Compute scale = 1.0 / cutoff (zero if cutoff=0).
#       3) Second pass: for each day
#          - Multiply df_sig['signal'] by scale, cap at 1.0, store in col_signal.
#          - Call generate_trade_actions(df_sig, col_signal, col_action,
#            buy_threshold, trailing_stop_pct, sess_start) to get df_actions.
#       4) Return a new dict mapping each day to (df_actions, trades).
#     """
#     # First pass: compute & gather raw signals
#     raw_results: Dict[dt.date, Tuple[pd.DataFrame, any]] = {}
#     all_vals: list = []

#     for day, (day_df, trades) in results_by_day_trad.items():
#         df_sig = compute_continuous_signal(
#             day_df            = day_df,
#             trades            = trades,
#             pre_entry_decay   = pre_entry_decay,
#             short_penal_decay = short_penal_decay,
#             smoothing_window  = smoothing_window
#         )
#         raw_results[day] = (df_sig, trades)
#         all_vals.append(df_sig["signal"].to_numpy())

#     if not all_vals:
#         return {}  # no data at all

#     # Flatten and compute global cutoff
#     flat_vals = np.concatenate(all_vals)
#     pct       = 100.0 - top_percentile
#     threshold = np.percentile(flat_vals, pct)

#     # Compute one linear scale factor
#     scale = 1.0 / threshold if threshold > 0 else 0.0

#     # Second pass: scale & cap, then generate actions
#     updated_results: Dict[dt.date, Tuple[pd.DataFrame, any]] = {}
#     for day, (df_sig, trades) in raw_results.items():
#         # apply the same multiplier to every bar
#         df_sig[col_signal] = df_sig["signal"] * scale
#         # cap at 1.0
#         df_sig[col_signal] = np.minimum(df_sig[col_signal], 1.0)

#         df_actions = generate_trade_actions(
#             df_sig,
#             col_signal           = col_signal,
#             col_action           = col_action,
#             buy_threshold        = buy_threshold,
#             trailing_stop_pct    = trailing_stop_pct,
#             sess_start           = sess_start
#         )
#         updated_results[day] = (df_actions, trades)

#     return updated_results


def add_trade_signal_to_results(
    results_by_day_trad: Dict[dt.date, Tuple[pd.DataFrame, Any]],
    col_signal:           str,         # name for the signal column to use
    col_action:           str,         # name for the trade-action column to add
    sess_start,                        # session start time for entries
    pre_entry_decay:      float,
    short_penal_decay:    float,
    trailing_stop_pct:    float,
    buy_threshold:        float,
    top_percentile:       float = 1.0, # percentile of raw signal to cap at 1.0
    smoothing_window=False             # optional smoothing window size
) -> Dict[dt.date, Tuple[pd.DataFrame, Any]]:
    """
    For each trading day:
      1) Compute a continuous “signal” series (via compute_continuous_signal).
      2) Collect all raw signal values to determine a global scale factor 
         based on the (100−top_percentile)th percentile.
      3) Apply the same linear scaling and cap to the series in-place 
         (writing back into col_signal).
      4) Generate discrete trade actions (+1/0/−1) using the scaled signal, 
         a fixed buy_threshold, and a trailing-stop rule.
    
    Returns a new dict mapping each day to (df_with_actions, trades).
    """
    # --- Pass 1: compute df_sig and gather raw signals ---
    raw_results     = {}
    all_raw_signals = []

    for day, (day_df, trades) in results_by_day_trad.items():
        df_sig = compute_continuous_signal(
            day_df            = day_df,
            trades            = trades,
            pre_entry_decay   = pre_entry_decay,
            short_penal_decay = short_penal_decay,
            smoothing_window  = smoothing_window
        )
        raw_results[day] = (df_sig, trades)
        # copy out the raw values before overwriting
        all_raw_signals.append(df_sig[col_signal].to_numpy())

    if not all_raw_signals:
        return {}

    # --- Compute global cutoff & scale factor ---
    flat_vals = np.concatenate(all_raw_signals)
    pct       = 100.0 - top_percentile
    threshold = np.percentile(flat_vals, pct)
    scale     = (1.0 / threshold) if threshold > 0 else 0.0

    # --- Pass 2: scale in-place and generate actions ---
    updated_results = {}
    for day, (df_sig, trades) in raw_results.items():
        # overwrite the same column with scaled & capped values
        df_sig[col_signal] = df_sig[col_signal] * scale
        df_sig[col_signal] = np.minimum(df_sig[col_signal], 1.0)

        # generate discrete buy/hold/sell actions
        df_actions = generate_trade_actions(
            df_sig,
            col_signal        = col_signal,
            col_action        = col_action,
            buy_threshold     = buy_threshold,
            trailing_stop_pct = trailing_stop_pct,
            sess_start        = sess_start
        )
        updated_results[day] = (df_actions, trades)

    return updated_results

#########################################################################################################

# B3

def simulate_trading(
    results_by_day_sign: Union[Dict[pd.Timestamp, Tuple[pd.DataFrame, List]], pd.DataFrame],
    col_action: str,
    sess_start: dt.time,
    sess_end: dt.time,
    ticker: str
) -> Dict[pd.Timestamp, Tuple[pd.DataFrame, List, Dict[str, object]]]:
    """
    Simulate minute‐level P&L driven by discrete buy/sell signals.

    Functionality:
      1) If passed a single DataFrame, split it into per‐day slices.
      2) For each calendar day (optionally with a progress bar if >1 day):
         a) Sort the minute bars and initialize position, cash, and session open price.
         b) Iterate each bar:
            - Within session hours, apply +1/-1/0 signals to adjust position and cash.
            - Outside hours, record “No trade.”
            - Track per‐bar Position, Cash, NetValue, Action, TradedAmount.
            - Track running buy‐and‐hold vs. strategy P&L.
         c) Assemble a simulation DataFrame with all those time‐series metrics.
         d) Scan the simulated actions to identify round‐trip trades and compute % returns.
         e) Compute day‐level performance stats: strategy return, buy‐&‐hold return, and trade‐by‐trade dollar gains.
      3) Return a dict mapping each date to (df_sim, trades_list, performance_stats).
    """
    # 1) Accept either dict[date→(df, trades)] or a single combined DataFrame
    if isinstance(results_by_day_sign, pd.DataFrame):
        df_all = results_by_day_sign.sort_index()
        per_day = {}
        for date, df_day in df_all.groupby(df_all.index.normalize(), sort=False):
            per_day[date] = (df_day.copy(), [])
        results_by_day_sign = per_day

    updated_results = {}

    # 2) Process each day; show tqdm bar only if multiple days (i.e. training mode)
    items = results_by_day_sign.items()
    if len(results_by_day_sign) > 1:
        items = tqdm(items, desc="Simulating trading days", unit="day")

    for day, val in items:
        # Unpack input tuple: (day_df, trades_list[, perf_stats])
        if len(val) == 2:
            session_df, prior_trades = val
        elif len(val) == 3:
            session_df, prior_trades, _ = val
        else:
            raise ValueError(f"Expected tuple of length 2 or 3 for {day}; got {len(val)}")

        # 2a) Prepare the day's data
        session_df = session_df.sort_index().copy()
        position           = 0       # current long position
        cash               = 0.0     # cumulative cash P&L
        session_open_price = None    # first ask price to seed buy‐and‐hold

        # Buffers for per‐bar metrics
        positions      = []
        cash_balances  = []
        net_values     = []
        actions        = []
        traded_amounts = []
        bh_running     = []  # buy‐and‐hold P&L per bar
        st_running     = []  # strategy P&L per bar

        # 2b) Iterate through each minute bar
        for ts, row in session_df.iterrows():
            bid, ask = row['bid'], row['ask']
            sig       = int(row[col_action])
            now       = ts.time()

            if sess_start <= now < sess_end:
                if session_open_price is None:
                    session_open_price = ask

                if sig == 1:
                    position += 1
                    cash     -= ask
                    action, amt = "Buy",  1
                elif sig == -1 and position > 0:
                    position -= 1
                    cash     += bid
                    action, amt = "Sell", -1
                else:
                    action, amt = "Hold",  0
            else:
                action, amt = "No trade", 0

            # Record state
            positions.append(position)
            cash_balances.append(np.round(cash, 3))
            net_val = np.round(cash + position * bid, 3)
            net_values.append(net_val)
            actions.append(action)
            traded_amounts.append(amt)

            # Running buy‐and‐hold vs. strategy P&L
            if session_open_price is not None:
                bh = bid - session_open_price
                st = net_val
            else:
                bh = st = 0.0

            bh_running.append(np.round(bh, 3))
            st_running.append(np.round(st, 3))

        # 2c) Build simulation DataFrame
        df_sim = session_df.copy()
        df_sim['Position']        = positions
        df_sim['Cash']            = cash_balances
        df_sim['NetValue']        = net_values
        df_sim['Action']          = actions
        df_sim['TradedAmount']    = traded_amounts
        df_sim['BuyHoldEarning']  = bh_running
        df_sim['StrategyEarning'] = st_running
        df_sim['EarningDiff']     = df_sim['StrategyEarning'] - df_sim['BuyHoldEarning']

        # 2d) Identify round‐trip trades and compute % returns
        trades = []
        entry_price = None
        entry_ts    = None
        for ts, row in df_sim.iterrows():
            if row['Action'] == "Buy" and entry_price is None:
                entry_price, entry_ts = row['ask'], ts
            elif row['Action'] == "Sell" and entry_price is not None:
                exit_price, exit_ts = row['bid'], ts
                gain_pct = 100 * (exit_price - entry_price) / entry_price
                trades.append(((entry_ts, exit_ts),
                               (entry_price, exit_price),
                               np.round(gain_pct, 3)))
                entry_price = None
                entry_ts    = None

        # Liquidate any open position at day‐end
        if entry_price is not None:
            exit_price, exit_ts = df_sim['bid'].iat[-1], df_sim.index[-1]
            gain_pct = 100 * (exit_price - entry_price) / entry_price
            trades.append(((entry_ts, exit_ts),
                           (entry_price, exit_price),
                           np.round(gain_pct, 3)))

        # 2e) Compute daily performance vs. buy‐and‐hold
        session = df_sim.between_time(sess_start, sess_end)
        if not session.empty:
            open_ask, close_bid = session['ask'].iloc[0], session['bid'].iloc[-1]
            buy_hold_gain = close_bid - open_ask
            strat_gain    = session['NetValue'].iloc[-1]
        else:
            buy_hold_gain = strat_gain = 0.0

        performance_stats = {
            'Buy & Hold Return ($)' : np.round(buy_hold_gain, 3),
            'Strategy Return ($)'   : np.round(strat_gain,     3),
            'Trades Returns ($)'    : [
                round((p/100)*prices[0], 3) for (_, _), prices, p in trades
            ]
        }

        updated_results[day] = (df_sim, trades, performance_stats)

    return updated_results

#########################################################################################################


def run_trading_pipeline(
    df,
    col_signal,
    col_action,
    min_prof_thr,
    max_down_prop,
    gain_tightening_factor,
    merging_retracement_thr,
    merging_time_gap_thr,
    pre_entry_decay,
    short_penal_decay,
    trailing_stop_pct, 
    buy_threshold,
    top_percentile,
    smoothing_window: Optional[str] = False,
    day_to_check: Optional[str] = None
):

    print("Identify_trades_daily …")
    trades_by_day = identify_trades_daily(
        df                     = df,
        min_prof_thr           = min_prof_thr,
        max_down_prop          = max_down_prop,
        gain_tightening_factor = gain_tightening_factor,
        merging_retracement_thr= merging_retracement_thr,
        merging_time_gap_thr   = merging_time_gap_thr,
        sess_premark           = params.sess_premark,
        sess_end               = params.sess_end,
        day_to_check           = day_to_check
    )

    print("Add_trade_signal_to_results …")
    signaled = add_trade_signal_to_results(
        results_by_day_trad  = trades_by_day,
        col_signal           = col_signal,
        col_action           = col_action,
        sess_start           = params.sess_start, 
        pre_entry_decay      = pre_entry_decay,
        short_penal_decay    = short_penal_decay,
        trailing_stop_pct    = trailing_stop_pct,
        buy_threshold        = buy_threshold,
        top_percentile       = top_percentile,
        smoothing_window     = smoothing_window
    )
    
    print("Simulate_trading …")
    sim_results = simulate_trading(
        results_by_day_sign = signaled,
        col_action          = col_action,
        sess_start          = params.sess_start, 
        sess_end            = params.sess_end,
        ticker              = params.ticker
    )

    if day_to_check:
        # single-day: return the only triple
        triple = next(iter(sim_results.values()), None)
        return triple
    # else  
    return sim_results

