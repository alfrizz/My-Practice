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



def process_splits(folder, ticker, bidasktoclose_spread):
    """
    Processes the intraday CSV file for the given ticker from the specified folder.
    
    It looks for the unique CSV file in 'folder' whose name starts with '{ticker}_'.
    Then, it checks if the processed file already exists in 'dfs training/{ticker}_base.csv'.
      - If it exists, the function reads it, plots its data, and returns the DataFrame.
      - Otherwise, it reads the intraday CSV, keeps only necessary columns,
        creates the 'ask' and 'bid' columns using the provided bidasktoclose_spread (e.g. 0.03 for 3%),
        plots the original data, detects and adjusts splits (also adjusting volume),
        plots the adjusted data if splits are detected, saves the resulting DataFrame, and returns it.
    
    Parameters:
      folder (str): The folder where the intraday CSV file is stored (e.g. "Intraday stocks").
      ticker (str): The ticker symbol used to locate the file and name the output.
      bidasktoclose_spread (float): The spread fraction; e.g., 0.03 means 3% (do not divide by 100).
      
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
    
    # Create 'ask' and 'bid' columns using the given spread (e.g. 0.03 means 3%).
    df['ask'] = round(df['close'] * (1 + bidasktoclose_spread/100), 4)
    df['bid'] = round(df['close'] * (1 - bidasktoclose_spread/100), 4)
    
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
def identify_trades(df, # DataFrame with a datetime index and a 'close' column.
                    min_prof_thr, # Minimum profit required
                    max_down_prop, # Maximum allowed retracement (as a fraction of the gain)
                    gain_tightening_factor, # As the gain increases, the allowed retracement is tightened by this factor
                    merging_retracement_thr, # Maximum allowed intermediate retracement ratio (relative to the profit range of the first trade) to allow merging.
                    merging_time_gap_thr): # Maximum allowed time gap ratio for merging, where the gap is measured relative to the sum of the
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
    Returns a dict mapping date -> (day_df, trades).
    """
    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    results: Dict[dt.date, Tuple[pd.DataFrame, List]] = {}
    for day, group in df.groupby(df.index.normalize()):
        # early filter: skip all but the target date
        if day_to_check and day.strftime("%Y-%m-%d") != day_to_check:
            continue

        day_str = day.strftime("%Y-%m-%d")
        start_ts = pd.Timestamp(f"{day_str} {regular_start_shifted}")
        end_ts   = pd.Timestamp(f"{day_str} {regular_end}")
        idx      = pd.date_range(start=start_ts, end=end_ts, freq="1min")

        day_df = group.reindex(idx)
        if day_df.empty:
            continue

        trades = identify_trades(
            df                     = day_df,
            min_prof_thr           = min_prof_thr,
            max_down_prop          = max_down_prop,
            gain_tightening_factor = gain_tightening_factor,
            merging_retracement_thr= merging_retracement_thr,
            merging_time_gap_thr   = merging_time_gap_thr
        )
        if trades:
            results[day.date()] = (day_df, trades)

    return results

#########################################################################################################

#B2.1
def compute_continuous_signal(
    day_df: pd.DataFrame,
    trades,
    smooth_win_sig: int,
    pre_entry_decay: float,
    short_penalty: float,       # penalty factor applied at 1-minute trades
    is_centered: bool,
    ref_profit: float           # benchmark profit in dollars (e.g. avg(sell_price−buy_price))
) -> pd.DataFrame:
    """
    Build a 0→1 intraday signal with per-trade scaling, decay and duration penalty:

    1) HARD-CODED DURATIONS
       • short_duration = 1 minute
       • long_duration  = 1 hour

    2) REFERENCE SCALE
       inv_ref = 1/ref_profit ensures a dollar-gain == ref_profit → combined_raw ≈ 1

    3) PER-TRADE “COMBINED” CURVE
       For each trade ((buy_dt, sell_dt), (buy_price, sell_price), profit_pct):
       
       a) Mask bars up to sell_dt:
            mask  = df.index ≤ sell_dt
            times = df.index[mask]

       b) Raw dollar gaps & exponential decay:
            closes      = df.loc[mask, "close"]
            profit_gaps = max(sell_price − closes, 0)
            dt_min      = minutes from each bar to sell_dt
            decayed     = profit_gaps × exp(−(pre_entry_decay/100)·dt_min)

       c) Duration‐penalty (no if/else):
            dur       = sell_dt − buy_dt
            time_frac = min(dur / long_duration, 1.0)
            time_pen  = 1.0 − sign(short_penalty)*(1−short_penalty)*(1−time_frac)

       d) Apply reference scale & penalty:
            combined_raw = decayed × inv_ref × time_pen

       e) Squash to (0,1):
            combined = combined_raw / (1 + combined_raw)

       f) Merge via point-wise max:
            signal_raw[t] = max over trades of combined[t]

    4) BROADCAST FOR COMPATIBILITY
       signal_scaled = signal_raw

    5) SMOOTH
       signal_smooth = rolling_mean(signal_scaled,
                                    window=smooth_win_sig,
                                    center=is_centered,
                                    min_periods=1)

    Returns df with columns:
      • signal_raw
      • signal_scaled
      • signal_smooth
    """

    # 2) Copy DataFrame and init raw signal
    df = day_df.copy()
    df["signal_raw"] = 0.0

    # 3) Build and merge per-trade “combined” curves
    for ((buy_dt, sell_dt), (buy_price, sell_price), profit_pct) in trades:
        # 3a) mask bars up to exit
        mask  = df.index <= sell_dt
        times = df.index[mask]
        if times.empty:
            continue

        # 3b) raw gaps & exponential decay
        closes      = df.loc[mask, "close"].to_numpy()
        profit_gaps = np.maximum(sell_price - closes, 0.0)
        dt_min      = (sell_dt - times).total_seconds() / 60.0
        λ           = pre_entry_decay / 100.0
        decayed     = profit_gaps * np.exp(-λ * dt_min)

        # 3c) duration‐based penalty
        dur            = sell_dt - buy_dt
        long_dur  = dt.timedelta(hours=1)
        time_pen = min(dur / long_dur, 1.0)
        short_pen = time_pen * (1 - short_penalty)

        # 3d) apply reference scaling & penalty
        curr_profit = sell_price - buy_price
        combined_raw =  decayed * (curr_profit / ref_profit) * short_pen

        # 3e) squash into (0,1)
        combined = combined_raw / (1.0 + combined_raw)

        # 3f) point-wise max into signal_raw
        existing = df.loc[mask, "signal_raw"].to_numpy()
        wins     = combined > existing
        ts_upd   = times[wins]
        df.loc[ts_upd, "signal_raw"] = combined[wins]

    # 5) smooth the final signal
    df["signal_smooth"] = (
        df["signal_raw"]
          .rolling(window=smooth_win_sig, center=is_centered, min_periods=1)
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
    trailing_stop_thresh,
    regular_start
):
    """
    Given a DataFrame for one trading day that already contains:
        • col_signal           — signal to use for trading
        • col_action           — name to assign to the action column
        • "close"              — the price series
    
    This function strips out any normalization or scaling steps and simply:
      1) Buys when the smoothed signal ≥ buy_threshold (only at/after regular_start)
      2) Tracks a simple trailing stop of trailing_stop_thresh% off the highest price since entry
      3) Sells when price falls below that trailing stop
      4) Forces a sell at EOD if still in trade

    Parameters:
      df : pd.DataFrame
          Must have a DatetimeIndex, and column "close".
      buy_threshold : float
          Cut-off on signal at which to enter a trade.
      trailing_stop_thresh : float
          Percent trailing stop (e.g. 1.5 means 1.5%).
      regular_start : datetime.time or str
          Don’t enter a new trade before this time (e.g. "14:30").

    Returns:
      pd.DataFrame : original df plus col_action column:
                     +1 = buy, 0 = hold, -1 = sell
    """
    
    # Work on a copy to avoid side-effects
    df = df.copy()
    n = len(df)

    # Initialize trade_action and state
    df[col_action] = 0
    in_trade = False
    entry_price = None
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
                entry_price = price
                max_price = price
            # else: remain flat (0)
        else:
            # We’re in a trade: update the highest price seen so far
            if price > max_price:
                max_price = price

            # Compute simple trailing-stop level
            stop_level = max_price * (1 - trailing_stop_thresh / 100.0)

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
    results_by_day_trad: dict,
    col_signal: str,
    col_action: str,
    ref_profit: float,
    min_prof_thr: float,
    regular_start: dt.time,
    smooth_win_sig: int,
    pre_entry_decay: float,
    short_penalty: float,
    trailing_stop_thresh: float,
    buy_threshold: float,
    is_centered: bool,
    clip_quantiles=(0.01, 0.99),
    scale_bounds=(0.1, 2.0)
) -> dict:
    """
    (A) Derive ref_return from the average profit% of all identified trades.
    (B) For each day, build the continuous signal (using ref_return in Step 1A),
        then generate discrete trade actions via trailing‐stop logic.

    Returns updated_results: { day → (df_with_signals&actions, trades) }
    """

    updated_results = {}
    for day, (day_df, trades) in results_by_day_trad.items():

        # Step 1–3: continuous signal w/ per‐trade scaling
        df_signal = compute_continuous_signal(
            day_df          = day_df,
            trades          = trades,
            smooth_win_sig  = smooth_win_sig,
            pre_entry_decay = pre_entry_decay,
            short_penalty   = short_penalty,
            is_centered     = is_centered,
            ref_profit      = ref_profit,
        )

        # Step B: discrete buy/sell via trailing stop
        df_actions = generate_trade_actions(
            df_signal,
            col_signal          = col_signal,
            col_action          = col_action,
            buy_threshold       = buy_threshold,
            trailing_stop_thresh= trailing_stop_thresh,
            regular_start       = regular_start
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
            'Strategy Return ($)'   : np.round(strat_gain,    3),
            'Buy & Hold Return ($)' : np.round(buy_hold_gain, 3),
            'Trades Returns ($)'    : [round((p/100)*(price[0]), 3)
                                       for (_, _), price, p in trades]
        }

        # store final results
        updated_results[day] = (df_sim, trades, performance_stats)

    return updated_results


#########################################################################################################

def compute_global_ref_profit(
    df: pd.DataFrame,  
    min_prof_thr           = params.min_prof_thr_tick,
    max_down_prop          = params.max_down_prop_tick,
    gain_tightening_factor = params.gain_tightening_factor_tick,
    merging_retracement_thr= params.merging_retracement_thr_tick,
    merging_time_gap_thr   = params.merging_time_gap_thr_tick,
    regular_start_pred     = params.regular_start_pred, # we only calculate the reference over the trading time for which we are computing the signal
    regular_end            = params.regular_end
) -> float:
    """
    1) Calls identify_trades_daily over ALL days (day_to_check=None)
       to get every day’s trades.
    2) Flattens every trade’s (sell_price - buy_price) into a list.
    3) Returns the median of that full list.
    """
    # collect trades for every day
    all_trades = identify_trades_daily(
        df                     = df,
        min_prof_thr           = min_prof_thr,
        max_down_prop          = max_down_prop,
        gain_tightening_factor = gain_tightening_factor,
        merging_retracement_thr= merging_retracement_thr,
        merging_time_gap_thr   = merging_time_gap_thr,
        regular_start_shifted  = regular_start_pred, # using regular_start_pred
        regular_end            = regular_end,
        day_to_check           = None      # no filtering here
    )

    # flatten all trade gains
    flat_profits = [
        sell_price - buy_price
        for (_df, trades) in all_trades.values()
        for ((_, _), (buy_price, sell_price), _) in trades
    ]

    # single median across the entire dataset 
    return float(np.median(flat_profits))


#########################################################################################################


def run_trading_pipeline(
    df,
    col_signal,
    col_action,
    ref_profit,           
    min_prof_thr           = params.min_prof_thr_tick,
    max_down_prop          = params.max_down_prop_tick,
    gain_tightening_factor = params.gain_tightening_factor_tick,
    merging_retracement_thr= params.merging_retracement_thr_tick,
    merging_time_gap_thr   = params.merging_time_gap_thr_tick,
    smooth_win_sig         = params.smooth_win_sig_tick,
    pre_entry_decay        = params.pre_entry_decay_tick,
    short_penalty          = params.short_penalty_tick,
    trailing_stop_thresh   = params.trailing_stop_thresh_tick,
    buy_threshold          = params.buy_threshold_tick,
    regular_start_shifted  = params.regular_start_shifted, # we want it here, because in the optuna signal optimization it can vary
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
        col_signal        = col_signal,
        col_action        = col_action,
        ref_profit        = ref_profit,
        min_prof_thr      = min_prof_thr,
        regular_start     = params.regular_start, 
        smooth_win_sig    = smooth_win_sig,
        pre_entry_decay   = pre_entry_decay,
        short_penalty     = short_penalty,
        trailing_stop_thresh= trailing_stop_thresh,
        buy_threshold     = buy_threshold,
        is_centered       = params.is_centered
    )
    
    print("Simulate_trading …")
    sim_results = simulate_trading(
        results_by_day_sign=signaled,
        col_action        = col_action,
        regular_start     = params.regular_start, 
        regular_end       = params.regular_end,
        ticker            = params.ticker
    )

    if day_to_check:
        # single-day: return the only triple
        triple = next(iter(sim_results.values()), None)
        return triple
    # else  
    return sim_results







