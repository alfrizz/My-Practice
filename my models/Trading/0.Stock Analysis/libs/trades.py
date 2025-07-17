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

from libs import params


#########################################################################################################

def plot_close_volume(df, title="Close Price and Volume"):
    """
    Quickly plots the 'close' and 'volume' columns from the DataFrame
    using a secondary y-axis for volume.
    """
    ax = df[['close', 'volume']].plot(secondary_y=['volume'], figsize=(10, 5), title=title, alpha=0.7)
    ax.set_xlabel("Date")
    plt.show()


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
                    df_adj.loc[:df_adj.index[i-1], 'volume'] *= candidate_factor * vol_fact # additonal manual volume factor necessary in some cases
        
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


def process_splits(folder, ticker, bidasktoclose_spread, vol_fact):
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
    output_file = f"dfs training/{ticker}_base.csv"
    
    # Check if the processed file already exists.
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Reading and plotting the processed data...")
        df_existing = pd.read_csv(output_file, index_col=0, parse_dates=True)
        plot_close_volume(df_existing, title="Processed Data: Close Price and Volume")
        return df_existing

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
    plot_close_volume(df, title="Before Adjusting Splits: Close Price and Volume")
    
    # Detect and adjust splits.
    df_adjusted, split_events = detect_and_adjust_splits(df=df, vol_fact=vol_fact)
    
    if split_events:
        print("Splits detected. Plotting adjusted data...")
        plot_close_volume(df_adjusted, title="After Adjusting Splits: Close Price and Volume")
    else:
        print("No splits detected.")
    
    # Save the resulting DataFrame.
    os.makedirs("dfs training", exist_ok=True)
    df_adjusted.to_csv(output_file, index=True)
    print(f"Processed data saved to {output_file}")
    
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
    regular_start_shifted,  # str or time: lower‐bound of minute grid (e.g. look-back start)
    regular_start,          # str or time: official session open (for reg_mask)
    regular_end,            # str or time: official session close
    red_pretr_win=1,        # int: factor for smoothing pre/post-market by vol ratio
    tz_str="US/Eastern"     # str: timezone name for DST adjustment
):
    """
    Prepares and fills minute‐bar data for each calendar day, with two distinct “starts”:
      • regular_start_shifted: defines how early to begin your minute grid
      • regular_start: classifies which bars count as “in‐session” vs pre/post‐market

    Steps:
      1. DST‐shift timestamps forward 1h on DST days.
      2. Mark bars inside [regular_start, regular_end] as regular, others non‐regular.
      3. Compute a small rolling‐mean window for non‐regular bars by volume ratio.
      4. Apply that rolling‐mean to non‐regular bars only.
      5. For each calendar day:
           • Build a 1-minute index from the earliest raw tick or regular_start_shifted
           • through the latest raw tick or regular_end.
           • Reindex the DataFrame to that grid and linearly interpolate both ways.
      6. Concatenate all days, sort, and drop any duplicate timestamps.

    Returns
    -------
    pd.DataFrame
        A DataFrame reindexed to every minute in each day’s span,
        with non‐regular bars smoothed and gaps filled.
    """

    # 0) Clone to avoid mutating the caller’s df, and keep a copy of raw values
    df = df.copy()
    df_orig = df.add_suffix("_orig")
    df = df.join(df_orig)

    # 1) Cast working columns (non-_orig) to float64
    working = [c for c in df.columns if not c.endswith("_orig")]
    df[working] = df[working].astype(np.float64)

    # 2) DST adjustment: shift each day’s timestamps +1h if in DST
    ts = df.index.to_series()
    for day, grp in df.groupby(df.index.normalize()):
        shift = pd.Timedelta(hours=1) if is_dst_for_day(day, tz_str) else pd.Timedelta(0)
        ts.loc[grp.index] = grp.index + shift
    df.index = ts.sort_values()
    df.sort_index(inplace=True)

    # 3) Identify regular vs non‐regular bars by official session times
    #    *regular_start_shifted* is NOT used here; only for grid bounds later.
    start_t = (pd.to_datetime(regular_start).time()
               if isinstance(regular_start, str) else regular_start)
    end_t   = (pd.to_datetime(regular_end).time()
               if isinstance(regular_end,   str) else regular_end)

    times       = df.index.time
    reg_mask    = (times >= start_t) & (times <= end_t)
    nonreg_mask = ~reg_mask


    # 4) Build minute‐grid per day, clamped by regular_start_shifted & regular_end
    filled = []
    for day, grp in df.groupby(df.index.normalize()):
        day_str = day.strftime("%Y-%m-%d")

        # compute grid bounds
        grid_start_idx   = pd.Timestamp(f"{day_str} {regular_start_shifted}")
        session_end_idx  = pd.Timestamp(f"{day_str} {regular_end}")

        # allow raw-data to extend beyond either bound
        idx_start = min(grp.index.min(), grid_start_idx)
        idx_end   = max(grp.index.max(), session_end_idx)

        full_idx = pd.date_range(start=idx_start, end=idx_end, freq="1min")

        # reindex → insert NaNs → interpolate both forward/back
        grp_f = grp.reindex(full_idx).interpolate(
            method="linear", limit_direction="both"
        )

        filled.append(grp_f)

    # 5) Concatenate all days, sort, and drop any duplicate timestamps
    out = pd.concat(filled).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    
    return out
    

#########################################################################################################

# B1.1
def identify_trades(df, min_prof_thr, max_down_prop, gain_tightening_factor, merging_retracement_thr, merging_time_gap_thr):
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
    
    Parameters:
      df : pd.DataFrame
         DataFrame with a datetime index and a 'close' column.
      min_prof_thr : float
         Minimum profit percentage required (e.g., 1.5 for 1.5%).
      max_down_prop : float
         Maximum allowed retracement (as a fraction of the gain). For example, 0.5 means that if the price
         falls more than 50% of (max_so_far - buy_price) after the buy, the trade is closed.
      gain_tightening_factor : float
         As the gain increases, the allowed retracement is tightened by this factor.
      merging_retracement_thr : float
         Maximum allowed intermediate retracement ratio (relative to the profit range of the first trade)
         to allow merging.
      merging_time_gap_thr : float
         Maximum allowed time gap ratio for merging, where the gap is measured relative to the sum of the
         active durations (buy-to-sell times) of both trades.
    
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
    df,                          # pd.DataFrame: minute-indexed, DST-adjusted, interpolated
    min_prof_thr,                # float: minimum profit percentage (%) to retain a trade
    max_down_prop,               # float: maximum allowed retracement proportion from peak
    gain_tightening_factor,      # float: factor tightening retracement allowance as gains grow
    regular_start_shifted,       # str: session open time (e.g. "09:30"), includes pre-market if shifted
    regular_end,                 # str: session close time (e.g. "16:00")
    merging_retracement_thr,     # float: retracement threshold for merging adjacent trades
    merging_time_gap_thr,        # float: time-gap ratio threshold for merging adjacent trades
    day_to_check: Optional[str] = None  # "YYYY-MM-DD" to process only that day, else all days
) -> Dict[datetime.date, Tuple[pd.DataFrame, List]]:
    """
    For each calendar day in `df`, slice out the full minute-bar session
    from `regular_start_shifted` to `regular_end` (gapless), then identify
    buy/sell trades via local extrema + retracement logic.
    
    Returns a dict mapping each date to (day_df, trades).
    """

    # 1) Ensure we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    results_by_day_trad: Dict[datetime.date, Tuple[pd.DataFrame, List]] = {}

    # 2) Iterate per calendar day
    for day, group in df.groupby(df.index.normalize()):
        # If a specific day is requested, skip all others
        if day_to_check and day.strftime('%Y-%m-%d') != day_to_check:
            continue

        # 2a) Build the exact session minute grid
        day_str       = day.strftime('%Y-%m-%d')
        start_ts      = pd.Timestamp(f"{day_str} {regular_start_shifted}")
        end_ts        = pd.Timestamp(f"{day_str} {regular_end}")
        trading_index = pd.date_range(start=start_ts, end=end_ts, freq='1min')

        # 2b) Slice out those minutes (no NaNs because df was pre-filled)
        # making sure  DataFrame has exactly one row per minute in that window
        day_df = group.reindex(trading_index)

        # 2c) Skip days with no data
        if day_df.empty:
            continue

        # 3) Identify trades on this day’s minute bars
        trades = identify_trades(
            df=day_df,
            min_prof_thr=min_prof_thr,
            max_down_prop=max_down_prop,
            gain_tightening_factor=gain_tightening_factor,
            merging_retracement_thr=merging_retracement_thr,
            merging_time_gap_thr=merging_time_gap_thr
        )

        # 4) Keep only days that produced at least one trade
        if trades:
            results_by_day_trad[day.date()] = (day_df, trades)

    return results_by_day_trad


#########################################################################################################

#B2.1
# def compute_continuous_signal(
#     day_df: pd.DataFrame,
#     trades,
#     smooth_win_sig: int,
#     pre_entry_decay: float,
#     short_penalty: float,       # penalty at 1-minute trades
#     is_centered: bool,
#     ref_return: float,          # your trade-gain benchmark in same units as profit_pct
#     clip_quantiles=(0.01, 0.99),
#     scale_bounds=(0.1, 1.0)
# ) -> pd.DataFrame:
#     """
#     1) RAW SIGNAL (per-trade gains scaled & penalized + decay)
#       a) scale each trade:  profit_pct/ref_return  (clipped to scale_bounds)
#       b) decay gap = (sell_price – close[t])·exp(−λ·dt_min)
#       c) penalty by duration: 1m→short_penalty, 1h→1.0, interp in between
#       d) combined[t] = decay[t] · raw_scale · penalty
#       e) signal_raw[t] = max over trades of combined[t]

#     2) (OPTIONAL) NORMALIZATION — commented out so you keep absolute scale  
#        # lo,hi = signal_raw.quantile(clip_quantiles);  
#        # signal_norm = (signal_raw−lo)/(hi−lo)  

#     3) BROADCAST for backward compatibility  
#        signal_scaled = signal_raw  

#     4) SMOOTH  
#        signal_smooth = rolling_mean(signal_scaled, window=smooth_win_sig)
#     """

#     # hardcode durations
#     short_duration = dt.timedelta(minutes=1)
#     long_duration  = dt.timedelta(hours=1)

#     df = day_df.copy()
#     df["signal_raw"] = 0.0
#     λ = pre_entry_decay / 100.0

#     # 1) build per-trade curves
#     for ((buy_dt, sell_dt), (_, sell_price), profit_pct) in trades:
#         # 1a) reference scaling
#         raw_scale = profit_pct / ref_return if ref_return else 1.0
#         raw_scale = np.clip(raw_scale, *scale_bounds)

#         # 1b) mask bars up to sell
#         mask  = df.index <= sell_dt
#         times = df.index[mask]
#         if times.empty:
#             continue

#         closes  = df.loc[mask, "close"].to_numpy()
#         dt_min  = (sell_dt - times).total_seconds() / 60.0

#         # 1c) exponential decay on profit gap
#         gaps    = np.maximum(sell_price - closes, 0.0)
#         decayed = gaps * np.exp(-λ * dt_min)

#         # 1d) duration penalty
#         dur  = sell_dt - buy_dt
#         frac = (dur - short_duration) / (long_duration - short_duration)
#         frac = min(max(frac, 0.0), 1.0)
#         pen  = short_penalty + frac * (1.0 - short_penalty)

#         # 1e) combined per-trade
#         combined = decayed * raw_scale * pen

#         # 1f) merge into raw signal by max()
#         existing = df.loc[mask, "signal_raw"].to_numpy()
#         wins     = combined > existing
#         ts_upd   = times[wins]
#         df.loc[ts_upd, "signal_raw"] = combined[wins]

#     # 2) NORMALIZATION — comment out if you want absolute scaling
#     # lo, hi = df["signal_raw"].quantile(clip_quantiles).to_list()
#     # if hi > lo:
#     #     clipped            = df["signal_raw"].clip(lo, hi)
#     #     df["signal_norm"]  = (clipped - lo) / (hi - lo)
#     # else:
#     #     df["signal_norm"]  = 0.0

#     # 3) BROADCAST raw → scaled for backward compatibility
#     df["signal_scaled"] = df["signal_raw"]

#     # 4) SMOOTH the (un-normalized) scaled signal
#     df["signal_smooth"] = (
#         df["signal_scaled"]
#           .rolling(window=smooth_win_sig, center=is_centered, min_periods=1)
#           .mean()
#     )

#     return df


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
          Don’t enter a new trade before this time (e.g. "13:30").

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
# def add_trade_signal_to_results(results_by_day_trad, col_signal, col_action, ref_return, min_prof_thr, regular_start,
#                                 smooth_win_sig, pre_entry_decay, short_penalty, buy_threshold, trailing_stop_thresh, is_centered):
#     """
#     Updates the input dictionary (results_by_day_trad) by applying two steps:
    
#        (A) Compute the continuous trading signal for each day (using compute_continuous_signal).
#        (B) Generate discrete trade actions based on the continuous signal and a trailing stop loss 
#            mechanism (using generate_trade_actions).
    
#     The continuous signal uses the formula:
#          raw_value = (P_max - close(t)) - min_prof_thr,
#     with exponential decay applied prior to the optimal entry time. The signal is later smoothed.
    
#     The discrete trade actions use the normalized continuous signal to identify candidate peaks.
#     A candidate peak (shifted by half the smoothing window) is taken as the entry (buy signal) if the
#     raw close is a local minimum. Then, while in a trade, a trailing stop loss is applied: when
#          (max_price - current_close) / (max_price - buy_price) * 100 >= trailing_stop_thresh,
#     a sell signal is issued.
    
#     Parameters:
#       results_by_day_trad : dict
#           Dictionary mapping each trading day (Timestamp) to a tuple (day_df, trades).
#       col_signal : float
#           Signal column
#       col_action : int
#           Trade action column
#       min_prof_thr : float
#           Minimum profit threshold used in continuous signal calculation.
#       smooth_win_sig : int, default 5
#           Rolling window size for smoothing signals.
#       pre_entry_decay : float, default 0.01
#           Decay rate applied before the trade entry for signal calculation.
#       buy_threshold : float, default 0.6
#           Minimum level in the smoothed normalized signal to consider a candidate peak for a trade.
#       trailing_stop_thresh : float, default 0.5 percent of max stock price
#           Trailing stop loss threshold. A sell signal is triggered when the retracement in raw close
#           prices meets/exceeds this value.
    
#     Returns:
#       updated_results : dict
#           The results_by_day_sign dictionary updated 
#     """
#     updated_results = {}
    
#     for day, (day_df, trades) in results_by_day_trad.items():
        
#         # Step (A): Compute the continuous trading signal.
#         df = compute_continuous_signal(day_df, trades, smooth_win_sig, pre_entry_decay, short_penalty, is_centered, ref_return) 
        
#         # Step (B): Generate discrete trade actions (using trailing stop loss logic).
#         df = generate_trade_actions(df, col_signal, col_action, buy_threshold, trailing_stop_thresh, regular_start)
        
#         updated_results[day] = (df, trades)
    
#     return updated_results


import numpy as np

def add_trade_signal_to_results(
    results_by_day_trad: dict,
    col_signal: str,
    col_action: str,
    min_prof_thr: float,
    regular_start: dt.time,
    smooth_win_sig: int,
    pre_entry_decay: float,
    short_penalty: float,
    buy_threshold: float,
    trailing_stop_thresh: float,
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

    # (A) collect every trade's dollar‐gain
    profits = []
    for (_day_df, trades) in results_by_day_trad.values():
        for ((buy_dt, sell_dt), (buy_price, sell_price), _profit_pct) in trades:
            profits.append(sell_price - buy_price)

    # ref_profit = np.mean(profits) # mean, median or percentiles
    ref_profit = np.percentile(profits, 75)

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
    results_by_day_sign: Union[dict, pd.DataFrame],
    col_action: str,
    regular_start: dt.time,
    regular_end:   dt.time,
    ticker:        str
) -> dict:
    """
    Simulate per‐minute trading P&L driven by a discrete signal column.

    Accepts either:
      1) A dict mapping each date → (day_df, trades[, perf_stats]), where
         day_df has columns ['bid','ask', col_action].
      2) A single DataFrame with a DatetimeIndex and col_action column,
         which is split into per‐day slices internally.

    For each day we produce:
      • df_sim: minute‐level DataFrame with columns
          Position, Cash, NetValue, Action, TradedAmount,
          BuyHoldEarning, StrategyEarning, EarningDiff
      • trades_list: list of (gain $, gain %) for each completed round‐trip
      • performance_stats: dict with keys

    Returns
    -------
    updated_results : dict
        Maps date → (df_sim, trades_list, performance_stats)
    """

    # 1) If a single DataFrame was passed, split into a dict by calendar day
    if isinstance(results_by_day_sign, pd.DataFrame):
        df_all = results_by_day_sign.sort_index()
        rebuilt = {}
        for date, df_day in df_all.groupby(df_all.index.normalize(), sort=False):
            rebuilt[date] = (df_day.copy(), [])
        results_by_day_sign = rebuilt

    updated_results = {}

    # 2) Process each day separately
    for day, val in results_by_day_sign.items():
        # Unpack tuple (day_df, trades_list[, perf_stats])
        if len(val) == 2:
            day_df, trades_list = val
        elif len(val) == 3:
            day_df, trades_list, _ = val
        else:
            raise ValueError(
                f"Expected tuple of length 2 or 3 for {day}; got {len(val)}"
            )

        session_df = day_df.sort_index().copy()

        # --- Initialize trading state --------------------------------------
        position          = 0       # current long position (number of shares)
        cash              = 0.0     # cash P&L
        session_open_price = None   # first ask price when market opens

        # Lists to record minute-by-minute state
        positions         = []
        cash_balances     = []
        net_values        = []
        actions           = []
        traded_amounts    = []
        buy_hold_earnings = []
        strat_earnings    = []

        # 3) Loop over each minute-bar
        for ts, row in session_df.iterrows():
            bid = row['bid']
            ask = row['ask']
            sig = int(row[col_action])  # discrete signal: -1,0,+1
            now = ts.time()

            # 3a) Determine trade action only during regular hours
            if regular_start <= now < regular_end:
                if sig == 1:
                    position += 1
                    cash     -= ask
                    action    = "Buy"
                    amt       = +1
                elif sig == -1 and position > 0:
                    position -= 1
                    cash     += bid
                    action    = "Sell"
                    amt       = -1
                else:
                    action = "Hold"
                    amt    = 0
            else:
                action = "No trade"
                amt    = 0

            # 3b) Record position & cash
            positions.append(position)
            cash_balances.append(np.round(cash, 3))
            # mark-to-market P&L: cash + position × current bid
            net_val = np.round(cash + position * bid, 3)
            net_values.append(net_val)
            actions.append(action)
            traded_amounts.append(amt)

            # 3c) Compute per-minute earnings relative to buy‐and‐hold
            if now >= regular_start:
                if session_open_price is None:
                    session_open_price = ask   # lock in open price
                # buy‐and‐hold profit = current bid − initial ask
                bh = bid - session_open_price
                # strategy profit = net P&L
                st = net_val
            else:
                bh = 0.0
                st = 0.0

            buy_hold_earnings.append(np.round(bh, 3))
            strat_earnings.append(np.round(st, 3))

        # --- Assemble the per-minute simulation DataFrame ---------------
        df_sim = session_df.copy()
        df_sim['Position']        = positions
        df_sim['Cash']            = cash_balances
        df_sim['NetValue']        = net_values
        df_sim['Action']          = actions
        df_sim['TradedAmount']    = traded_amounts
        df_sim['BuyHoldEarning']  = buy_hold_earnings
        df_sim['StrategyEarning'] = strat_earnings
        df_sim['EarningDiff']     = df_sim['StrategyEarning'] - df_sim['BuyHoldEarning']

        # --- Compute per‐trade round‐trip gains -------------------------
        trade_gains     = []
        trade_gains_pct = []
        entry_price     = None

        for ts, row in df_sim.iterrows():
            act = row['Action']
            if act == "Buy" and entry_price is None:
                entry_price = row['ask']
            elif act == "Sell" and entry_price is not None:
                exit_price = row['bid']
                gain   = exit_price - entry_price
                pct    = 100 * gain / entry_price
                trade_gains.append(np.round(gain, 3))
                trade_gains_pct.append(np.round(pct, 3))
                entry_price = None

        # if still long at end of day, liquidate at last bid
        if entry_price is not None:
            final_bid = df_sim['bid'].iat[-1]
            gain      = final_bid - entry_price
            pct       = 100 * gain / entry_price
            trade_gains.append(np.round(gain, 3))
            trade_gains_pct.append(np.round(pct, 3))

        # --- Compute overall performance metrics ------------------------
        # mask of minutes during regular trading session
        mask = (
            (df_sim.index.time >= regular_start) &
            (df_sim.index.time <  regular_end)
        )
        if mask.any() and session_open_price is not None:
            idxs        = np.where(mask)[0]
            start_i     = idxs[0]
            end_i       = idxs[-1]
            baseline_ask    = session_open_price
            final_net_value = net_values[end_i]
            final_bid_price = session_df['bid'].iat[end_i]
        else:
            # if no trading minutes, safe‐default to end‐of‐day
            baseline_ask    = session_open_price or np.nan
            final_net_value = net_values[-1]
            final_bid_price = session_df['bid'].iat[-1]

        # --- Compute overall performance ------------------------------------
        # baseline ask = session_open_price (first ask when market opened)
        # final bid = bid at last regular‐hour bar
        buy_hold_gain   = final_bid_price - session_open_price
        profit_diff     = final_net_value - buy_hold_gain
        final_ret_pct   = 100 * final_net_value      / session_open_price
        buy_hold_ret_pct= 100 * buy_hold_gain        / session_open_price
        improve_pct     = final_ret_pct - buy_hold_ret_pct

        # build performance_stats with your exact keys & order
        performance_stats = {
            'Strategy Return ($)'            : np.round(final_net_value, 3),
            'Buy & Hold Return ($)'          : np.round(buy_hold_gain, 3),
            'Trades Returns ($)'             : trade_gains,
        }
            
        updated_results[day] = (df_sim, trades_list, performance_stats)
        
    return updated_results

#########################################################################################################


def run_trading_pipeline(
    df_prep,
    col_signal,
    col_action,
    min_prof_thr=params.min_prof_thr_man, 
    max_down_prop=params.max_down_prop_man, 
    gain_tightening_factor=params.gain_tightening_factor_man, 
    smooth_win_sig=params.smooth_win_sig_man, 
    pre_entry_decay=params.pre_entry_decay_man,
    short_penalty=params.short_penalty_man,
    buy_threshold=params.buy_threshold_man, 
    trailing_stop_thresh=params.trailing_stop_thresh_man, 
    merging_retracement_thr=params.merging_retracement_thr_man, 
    merging_time_gap_thr=params.merging_time_gap_thr_man,
    day_to_check=None
):

    print("Step B1: identify_trades_daily …")
    trades_by_day = identify_trades_daily(
        df=df_prep,
        min_prof_thr=min_prof_thr,
        max_down_prop=max_down_prop,
        gain_tightening_factor=gain_tightening_factor,
        regular_start_shifted=params.regular_start_shifted,
        regular_end=params.regular_end,
        merging_retracement_thr=merging_retracement_thr,
        merging_time_gap_thr=merging_time_gap_thr,
        day_to_check=day_to_check
    )

    print("Step B2: add_trade_signal_to_results …")
    signaled = add_trade_signal_to_results(
        results_by_day_trad=trades_by_day,
        col_signal=col_signal,
        col_action=col_action,
        min_prof_thr=min_prof_thr,
        regular_start=params.regular_start,
        smooth_win_sig=smooth_win_sig,
        pre_entry_decay=pre_entry_decay,
        short_penalty=short_penalty,
        buy_threshold=buy_threshold,
        trailing_stop_thresh=trailing_stop_thresh,
        is_centered=params.is_centered
    )
    
    print("Step B3: simulate_trading …")
    sim_results = simulate_trading(
        results_by_day_sign=signaled,
        col_action=col_action,
        regular_start=params.regular_start,
        regular_end=params.regular_end,
        ticker=params.ticker
    )

    if day_to_check:
        target = pd.to_datetime(day_to_check).date()
        for ts, triple in sim_results.items():
            # normalize ts to a date for comparison
            ts_date = ts.date() if hasattr(ts, "date") else ts
            if ts_date == target:
                return triple
        return None
        
    return sim_results






