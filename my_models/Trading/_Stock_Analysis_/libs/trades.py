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
from typing import Optional, Dict, Tuple, List, Sequence, Union
import matplotlib.pyplot as plt

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
    regular_start_shifted,  # str or datetime.time: grid lower‐bound each day
    regular_start,          # str or datetime.time: official session open
    regular_end,            # str or datetime.time: official session close
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
       a) Build a 1-minute index from min(raw, regular_start_shifted)
          through max(raw, regular_end).
       b) Reindex the day’s bars to that grid and linearly interpolate
          forward & backward.
    4) Concatenate all per-day frames, sort by timestamp, and drop
       duplicate timestamps (printing how many were removed).
    5) Finally, keep only the bars whose time lies between
       regular_start_shifted and regular_end (session‐plus‐lookback).
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
        grid_start  = pd.Timestamp(f"{day_str} {regular_start_shifted}")
        session_end = pd.Timestamp(f"{day_str} {regular_end}")

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

    # 5) Slice to only [regular_start_shifted, regular_end] each day
    df_out = df_out.between_time(regular_start_shifted, regular_end)

    # 6) drop any calendar day whose close is perfectly flat (to avoid to get extra days as saturdays)
    df_out = (
        df_out.groupby(df_out.index.normalize())
          .filter(lambda grp: grp['close'].nunique() > 1)
    )

    return df_out


#########################################################################################################

# B1.1
def identify_trades(df,                     # DataFrame with a datetime index and a 'close' column.
                    min_prof_thr,           # Minimum profit required
                    max_down_prop,          # Maximum allowed retracement (as a fraction of the gain)
                    gain_tightening_factor, # As the gain increases, the allowed retracement is tightened by this factor
                    merging_retracement_thr,# Maximum allowed intermediate retracement ratio (relative to the profit range of the first trade) to allow merging.
                    merging_time_gap_thr):  # Maximum allowed time gap ratio for merging, where the gap is measured relative to the sum of the
                                            # active durations (buy-to-sell times) of both trades.
    """
    Identifies trades from a one-minute bars DataFrame using a retracement rule and then iteratively merges
    consecutive trades based on two conditions:
    
    Identification Phase:
      - A buy candidate is a local minimum (the price is lower than its immediate neighbors).
      - For a candidate buy, the algorithm scans forward, updating the highest price encountered (candidate sell).
      - If the price later retraces by more than max_down_prop (adjusted via gain_tightening_factor according to
        the overall gain from buy to the high), the trade is closed immediately, with the sell point taken from
        the highest price seen.
      - The trade is recorded only if the profit percentage exceeds min_prof_thr.
    
    Merging Phase (Iterative):
      - Two consecutive trades are eligible for merging only if the second trade's peak (sell price) exceeds that
        of the first.
      - Minimal Intermediate Retracement:
            (first_trade_sell_price - second_trade_buy_price) / (first_trade_sell_price - first_trade_buy_price)
        must be <= merging_retracement_thr.  
        (In other words, the drop from the peak of the first trade to the entry of the second trade must be limited.)
      - Short Time Gap Between Trades:
            (second_trade_buy_date - first_trade_sell_date) /
            (active_duration_trade1 + active_duration_trade2)
        must be <= merging_time_gap_thr, where each active duration is the time from trade buy to trade sell.
      
      - If both conditions are met, the trades are merged by using the buy data from the first trade and 
        the sell data from the second trade. After merging, the trade’s profit is recalculated.
      - The merging is performed iteratively until no further merges can be done.
    
    Returns:
      merged_trades : list
         A list of tuples, each trade represented as:
           ((buy_date, sell_date), (buy_price, sell_price), profit_pc)
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
    regular_start_shifted: str,
    regular_end: str,
    day_to_check: Optional[str] = None
) -> Dict[dt.date, Tuple[pd.DataFrame, List]]:
    """
    1) Slice df into daily sessions (regular_start_shifted→regular_end).
    2) Identify trades via your local-extrema + retracement logic.
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
        start_ts = pd.Timestamp(f"{day_str} {regular_start_shifted}")
        end_ts   = pd.Timestamp(f"{day_str} {regular_end}")
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
    df,
    col_signal,
    col_action,
    buy_threshold,
    trailing_stop_pct,
    regular_start
):
    """
    Generates per-bar trade actions for one day:

    - Buy when signal ≥ buy_threshold (only after regular_start time).
    - Update highest price since entry.
    - Compute stop_level = max_price * (1 - trailing_stop_thresh).
    - Sell when price < stop_level or at end of day.

    Parameters:
      df: pd.DataFrame with DatetimeIndex and 'close' column
      col_signal: str, name of signal column
      col_action: str, name for output action column
      buy_threshold: float, entry cutoff (signal domain)
      trailing_stop_pct: percent, decimal stop distance (eg 0.05% conservative minimum threshold)
      regular_start: time or "HH:MM" string to begin new trades

    Returns:
      pd.DataFrame with new col_action: +1=buy, 0=hold, -1=sell
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

    # Convert regular_start to time if needed
    if isinstance(regular_start, str):
        regular_start = pd.to_datetime(regular_start).time()

    # Extract series for speed
    sig = df[col_signal].values
    closes = df["close"].values
    times = df.index.time

    for i in range(n):
        t = times[i]
        price = closes[i]

        if not in_trade:
            # Check buy condition: signal crosses threshold, and time ≥ regular_start
            if sig[i] >= buy_threshold and t >= regular_start:
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
def add_trade_signal_to_results(
    results_by_day_trad: Dict[dt.date, Tuple[pd.DataFrame, any]],
    col_signal: str,
    col_action: str,
    min_prof_thr: float,
    regular_start: dt.time,
    pre_entry_decay: float,
    short_penal_decay: float,
    trailing_stop_pct: float,
    buy_threshold: float,
    top_percentile: float = 1,
    smoothing_window: Optional[int] = False
) -> Dict[dt.date, Tuple[pd.DataFrame, any]]:
    """
    1) First pass: compute raw signals (with smoothing if set), collect all signal values.
    2) Compute threshold = (100 - top_percentile)th percentile of all raw signals.
    3) Second pass: scale each day’s signal by a single factor and cap at 1.0,
       so that exactly top_percentile% of minute‐bars become 1.0.
    """
    # First pass: compute & gather raw signals
    raw_results: Dict[dt.date, Tuple[pd.DataFrame, any]] = {}
    all_vals: list = []

    for day, (day_df, trades) in results_by_day_trad.items():
        df_sig = compute_continuous_signal(
            day_df            = day_df,
            trades            = trades,
            pre_entry_decay   = pre_entry_decay,
            short_penal_decay = short_penal_decay,
            smoothing_window  = smoothing_window
        )
        raw_results[day] = (df_sig, trades)
        all_vals.append(df_sig["signal"].to_numpy())

    if not all_vals:
        return {}  # no data at all

    # Flatten and compute global cutoff
    flat_vals = np.concatenate(all_vals)
    pct       = 100.0 - top_percentile
    threshold = np.percentile(flat_vals, pct)

    # Compute one linear scale factor
    scale = 1.0 / threshold if threshold > 0 else 0.0

    # Second pass: scale & cap, then generate actions
    updated_results: Dict[dt.date, Tuple[pd.DataFrame, any]] = {}
    for day, (df_sig, trades) in raw_results.items():
        # apply the same multiplier to every bar
        df_sig[col_signal] = df_sig["signal"] * scale
        # cap at 1.0
        df_sig[col_signal] = np.minimum(df_sig[col_signal], 1.0)

        df_actions = generate_trade_actions(
            df_sig,
            col_signal           = col_signal,
            col_action           = col_action,
            buy_threshold        = buy_threshold,
            trailing_stop_pct    = trailing_stop_pct,
            regular_start        = regular_start
        )
        updated_results[day] = (df_actions, trades)

    return updated_results







#########################################################################################################

# B3
def simulate_trading(
    results_by_day_sign: Union[Dict[pd.Timestamp, Tuple[pd.DataFrame, List]], pd.DataFrame],
    col_action: str,
    regular_start: dt.time,
    regular_end: dt.time,
    ticker: str
) -> Dict[pd.Timestamp, Tuple[pd.DataFrame, List, Dict[str, object]]]:
    """
    Simulate minute-level trading P&L driven by a discrete signal.

    Parameters
    ----------
    results_by_day_sign : dict or DataFrame
        Either a dict mapping each date → (day_df, trades_list[, perf_stats]),
        or a single DataFrame with a DatetimeIndex and signal column.
    col_action : str
        Column name holding discrete trading signals: +1=buy, ‑1=sell, 0=hold.
    regular_start : datetime.time
        Inclusive start of regular session.
    regular_end   : datetime.time
        Exclusive end of regular session.
    ticker : str
        Asset symbol (only for logging/extensibility).

    Returns
    -------
    updated_results : dict
        Maps date → (
          df_sim: minute-level DataFrame with Position, Cash, NetValue, etc.,
          trades: list of ((buy_ts, sell_ts), (buy_price, sell_price), pct_return),
          performance_stats: {
            'Strategy Return ($)'   : final P&L,
            'Buy & Hold Return ($)' : baseline P&L,
            'Trades Returns ($)'    : list of dollar gains per trade
          }
        )
    """

    # 1) If user passed one big DataFrame, split it by calendar date
    if isinstance(results_by_day_sign, pd.DataFrame):
        df_all = results_by_day_sign.sort_index()
        per_day = {}
        for date, df_day in df_all.groupby(df_all.index.normalize(), sort=False):
            per_day[date] = (df_day.copy(), [])
        results_by_day_sign = per_day

    updated_results = {}

    # 2) Process each day in isolation
    for day, val in results_by_day_sign.items():
        # Unpack: (day_df, trades_list[, perf_stats])
        if len(val) == 2:
            session_df, prior_trades = val
        elif len(val) == 3:
            session_df, prior_trades, _ = val
        else:
            raise ValueError(f"Expected tuple of length 2 or 3 for {day}; got {len(val)}")

        # Ensure minute bars are sorted
        session_df = session_df.sort_index().copy()

        # Initialize state
        position           = 0       # current long position
        cash               = 0.0     # cash P&L
        session_open_price = None    # first ask at or after open

        # Buffers for minute-by-minute tracking
        positions      = []
        cash_balances  = []
        net_values     = []
        actions        = []
        traded_amounts = []
        bh_running     = []  # buy-and-hold P&L per bar
        st_running     = []  # strategy P&L per bar

        # 3) Iterate each minute bar
        for ts, row in session_df.iterrows():
            bid, ask = row['bid'], row['ask']
            sig       = int(row[col_action])
            now       = ts.time()

            # Only trade within regular hours
            if regular_start <= now < regular_end:
                # lock in first ask once
                if session_open_price is None:
                    session_open_price = ask

                if sig == 1:
                    position += 1
                    cash     -= ask
                    action   = "Buy"
                    amt      = +1
                elif sig == -1 and position > 0:
                    position -= 1
                    cash     += bid
                    action   = "Sell"
                    amt      = -1
                else:
                    action   = "Hold"
                    amt      = 0
            else:
                action = "No trade"
                amt    = 0

            # record state
            positions.append(position)
            cash_balances.append(np.round(cash, 3))
            net_val = np.round(cash + position * bid, 3)
            net_values.append(net_val)
            actions.append(action)
            traded_amounts.append(amt)

            # running buy-and-hold vs strategy
            if session_open_price is not None:
                bh = bid - session_open_price
                st = net_val
            else:
                bh = st = 0.0

            bh_running.append(np.round(bh, 3))
            st_running.append(np.round(st, 3))

        # assemble minute-level DataFrame
        df_sim = session_df.copy()
        df_sim['Position']        = positions
        df_sim['Cash']            = cash_balances
        df_sim['NetValue']        = net_values
        df_sim['Action']          = actions
        df_sim['TradedAmount']    = traded_amounts
        df_sim['BuyHoldEarning']  = bh_running
        df_sim['StrategyEarning'] = st_running
        df_sim['EarningDiff']     = df_sim['StrategyEarning'] - df_sim['BuyHoldEarning']

        # 4) Compute per-trade round-trip results with timestamps & % return
        trades = []
        entry_price = None
        entry_ts    = None

        for ts, row in df_sim.iterrows():
            if row['Action'] == "Buy" and entry_price is None:
                entry_price = row['ask']
                entry_ts    = ts

            elif row['Action'] == "Sell" and entry_price is not None:
                exit_price = row['bid']
                exit_ts    = ts
                gain_pct   = 100 * (exit_price - entry_price) / entry_price

                trades.append((
                    (entry_ts, exit_ts),
                    (entry_price, exit_price),
                    np.round(gain_pct, 3)
                ))
                entry_price = None
                entry_ts    = None

        # if still long at end, liquidate at last bid
        if entry_price is not None:
            exit_price = df_sim['bid'].iat[-1]
            exit_ts    = df_sim.index[-1]
            gain_pct   = 100 * (exit_price - entry_price) / entry_price

            trades.append((
                (entry_ts, exit_ts),
                (entry_price, exit_price),
                np.round(gain_pct, 3)
            ))

        # 5) Compute day-level performance
        session = df_sim.between_time(regular_start, regular_end)

        if not session.empty:
            open_ask  = session['ask'].iloc[0]
            close_bid = session['bid'].iloc[-1]
            buy_hold_gain = close_bid - open_ask
            strat_gain    = session['NetValue'].iloc[-1]
        else:
            buy_hold_gain = strat_gain = 0.0

        performance_stats = {
            'Buy & Hold Return ($)' : np.round(buy_hold_gain, 3),
            'Strategy Return ($)'   : np.round(strat_gain,    3),
            'Trades Returns ($)'    : [round((p/100)*(price[0]), 3)
                                       for (_, _), price, p in trades]
        }

        # store final results
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
    regular_start_shifted,
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
        regular_start_shifted  = regular_start_shifted,
        regular_end            = params.regular_end,
        day_to_check           = day_to_check
    )

    print("Add_trade_signal_to_results …")
    signaled = add_trade_signal_to_results(
        results_by_day_trad=trades_by_day,
        col_signal           = col_signal,
        col_action           = col_action,
        min_prof_thr         = min_prof_thr,
        regular_start        = params.regular_start, 
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
        regular_start       = params.regular_start, 
        regular_end         = params.regular_end,
        ticker              = params.ticker
    )

    if day_to_check:
        # single-day: return the only triple
        triple = next(iter(sim_results.values()), None)
        return triple
    # else  
    return sim_results

