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

def compute_reference_gain(df):
    """
    Computes the *relative* reference gain (%) as the average daily % range:
       daily_pct_range = (high.max - low.min) / low.min
    Returns a float like 0.025 if on average days swing 2.5%.
    (used afterwards to adjust the smoothed continuous signal , while normalizing it)
    """
    # Ensure your index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # For each calendar day compute (high.max - low.min) / low.min
    daily_pct = (
        df.groupby(df.index.date)
          .apply(lambda d: (d["high"].max() - d["low"].min()) / d["low"].min())
    )
    # If you want it in percent points, multiply by 100 here.
    return daily_pct.median() # or mean()


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
    regular_start,   # str or time: session open (e.g. "08:00" or your pre-market shifted start)
    regular_end,     # str or time: session close (e.g. "16:00")
    red_pretr_win=1, # int: factor for smoothing pre/post-market bars by volume ratio
    tz_str="US/Eastern"  # str: timezone for DST adjustment
):
    """
    1) Shift timestamps on DST days
    2) Smooth non-regular (pre/post-market) bars by volume ratio
    3) For each calendar day, build a minute grid that covers both:
         • the raw-data span (first→last bar)
         • the official session span (regular_start→regular_end)
       then reindex & linearly interpolate both directions to fill all gaps.
    """

    # Make a copy and preserve originals
    df = df.copy()
    df_orig = df.add_suffix("_orig")
    df = df.join(df_orig)

    # Cast working columns to float64
    working = [c for c in df.columns if not c.endswith("_orig")]
    df[working] = df[working].astype(np.float64)

    # 1) DST adjustment
    ts = df.index.to_series()
    for day, grp in df.groupby(df.index.normalize()):
        shift = pd.Timedelta(hours=1) if is_dst_for_day(day, tz_str) else pd.Timedelta(0)
        ts.loc[grp.index] = grp.index + shift
    df.index = ts.sort_values()
    df = df.sort_index()

    # 2) Identify regular vs non-regular bars
    start_t = (pd.to_datetime(regular_start).time()
               if isinstance(regular_start, str) else regular_start)
    end_t   = (pd.to_datetime(regular_end).time()
               if isinstance(regular_end, str)   else regular_end)
    times   = df.index.time
    reg_mask    = (times >= start_t) & (times <= end_t)
    nonreg_mask = ~reg_mask

    # 3) Compute smoothing window (pre/post-market) by volume ratio
    v_reg = df.loc[reg_mask,    "volume_orig"].mean()
    v_non = df.loc[nonreg_mask, "volume_orig"].mean()
    if pd.isna(v_non) or v_non == 0:
        v_non = red_pretr_win
    ratio = v_reg / v_non if v_non else np.nan
    w_non  = 1 if pd.isna(ratio) else max(int(ratio / red_pretr_win), 1)

    # 4) Smooth non-regular bars
    for col in working:
        vals = df.loc[nonreg_mask, col].reset_index(drop=True)
        sm   = vals.rolling(window=w_non, min_periods=1).mean()
        if col != "volume":
            df.loc[nonreg_mask, col] = np.round(sm, 4).values
        else:
            df.loc[nonreg_mask, col] = np.rint(sm).astype(np.int64).values

    # 5) Reindex & interpolate per day over combined span
    filled = []
    for day, grp in df.groupby(df.index.normalize()):
        day_str = day.strftime("%Y-%m-%d")

        # official session window
        start_idx = pd.Timestamp(f"{day_str} {regular_start}")
        end_idx   = pd.Timestamp(f"{day_str} {regular_end}")

        # also cover any raw-data bars outside that window
        idx_start = min(grp.index.min(), start_idx)
        idx_end   = max(grp.index.max(), end_idx)

        full_idx = pd.date_range(start=idx_start, end=idx_end, freq="1min")

        # two-way linear interpolation to fill all NaNs
        grp_f = grp.reindex(full_idx).interpolate(
            method="linear", limit_direction="both"
        )

        filled.append(grp_f)

    return pd.concat(filled)



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

# B2.1
def compute_continuous_signal(
    day_df,
    trades,
    smooth_win_sig,
    decay,
    is_centered,
    reference_gain,
    regular_start=params.regular_start,
    regular_end=params.regular_end,
    clip_quantiles=(0.01, 0.99),
    scale_bounds=(0.5, 2.0)
):
    """
    1) Build raw decayed signal → signal_raw
    2) Robust min–max normalize signal_raw → signal_norm
    3) Compute today’s intraday range robustly → raw_range
    4) Compute and clip scale relative to reference_gain → scale
    5) Apply scale → signal_scaled
    6) Smooth the scaled curve → signal_smooth
    """
    # Make a working copy
    df = day_df.copy()

    # 1) Build raw decayed signal: for each sell event, backfill a decaying value
    df["signal_raw"] = 0.0
    for (_, sell_dt), (_, sell_price), _ in trades:
        mask = df.index <= sell_dt
        # time difference in minutes
        delta_mins = (sell_dt - df.index[mask]).total_seconds() / 60.0
        # positive profit diffs only
        diffs = (sell_price - df.loc[mask, "close"]).clip(lower=0)
        # exponential decay
        vals = diffs * np.exp(-decay * delta_mins)
        # take the maximum decay-value per timestamp across all sells
        df.loc[mask, "signal_raw"] = np.maximum(df.loc[mask, "signal_raw"], vals)

    # 2) Robust normalize to [0,1] using clipped percentiles
    lo_q, hi_q = clip_quantiles
    lo = df["signal_raw"].quantile(lo_q)
    hi = df["signal_raw"].quantile(hi_q)
    if hi > lo:
        clipped = df["signal_raw"].clip(lower=lo, upper=hi)
        df["signal_norm"] = (clipped - lo) / (hi - lo)
    else:
        df["signal_norm"] = 0.0

    # 3) Compute today's intraday high/low robustly (quantile-based)
    reg = df.between_time(regular_start, regular_end)
    highs = reg["high"] if not reg.empty else df["high"]
    lows  = reg["low"]  if not reg.empty else df["low"]
    # use 99th/1st percentiles to ignore bar-level spikes
    day_hi = highs.quantile(0.99)
    day_lo = lows.quantile(0.01)

    # relative range for the day
    raw_range = (day_hi - day_lo) / day_lo if day_lo else 0.0

    # 4) Compute scale as ratio to reference_gain, then clip to bounds
    if reference_gain:
        raw_scale = raw_range / reference_gain
    else:
        raw_scale = 1.0
    min_s, max_s = scale_bounds
    scale = np.clip(raw_scale, min_s, max_s)

    # 5) Apply scale to normalized signal
    df["signal_scaled"] = df["signal_norm"] * scale

    # 6) Smooth the scaled signal
    df["signal_smooth"] = (
        df["signal_scaled"]
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
def add_trade_signal_to_results(results_by_day_trad, col_signal, col_action, min_prof_thr, regular_start, reference_gain,
                                smooth_win_sig, pre_entry_decay, buy_threshold, trailing_stop_thresh, is_centered):
    """
    Updates the input dictionary (results_by_day_trad) by applying two steps:
    
       (A) Compute the continuous trading signal for each day (using compute_continuous_signal).
       (B) Generate discrete trade actions based on the continuous signal and a trailing stop loss 
           mechanism (using generate_trade_actions).
    
    The continuous signal uses the formula:
         raw_value = (P_max - close(t)) - min_prof_thr,
    with exponential decay applied prior to the optimal entry time. The signal is later smoothed.
    
    The discrete trade actions use the normalized continuous signal to identify candidate peaks.
    A candidate peak (shifted by half the smoothing window) is taken as the entry (buy signal) if the
    raw close is a local minimum. Then, while in a trade, a trailing stop loss is applied: when
         (max_price - current_close) / (max_price - buy_price) * 100 >= trailing_stop_thresh,
    a sell signal is issued.
    
    Parameters:
      results_by_day_trad : dict
          Dictionary mapping each trading day (Timestamp) to a tuple (day_df, trades).
      col_signal : float
          Signal column
      col_action : int
          Trade action column
      min_prof_thr : float
          Minimum profit threshold used in continuous signal calculation.
      smooth_win_sig : int, default 5
          Rolling window size for smoothing signals.
      pre_entry_decay : float, default 0.01
          Decay rate applied before the trade entry for signal calculation.
      buy_threshold : float, default 0.6
          Minimum level in the smoothed normalized signal to consider a candidate peak for a trade.
      trailing_stop_thresh : float, default 0.5 percent of max stock price
          Trailing stop loss threshold. A sell signal is triggered when the retracement in raw close
          prices meets/exceeds this value.
    
    Returns:
      updated_results : dict
          The results_by_day_sign dictionary updated 
    """
    updated_results = {}
    
    for day, (day_df, trades) in results_by_day_trad.items():
        
        # Step (A): Compute the continuous trading signal.
        df = compute_continuous_signal(day_df, trades, smooth_win_sig, pre_entry_decay, is_centered, reference_gain)
        
        # Step (B): Generate discrete trade actions (using trailing stop loss logic).
        df = generate_trade_actions(df, col_signal, col_action, buy_threshold, trailing_stop_thresh, regular_start)
        
        updated_results[day] = (df, trades)
    
    return updated_results

#########################################################################################################

# B3
def simulate_trading(
    results_by_day_sign: Union[dict, pd.DataFrame],
    col_action: int,
    regular_start: dt.time,
    regular_end: dt.time,
    ticker: str
) -> dict:
    """
    Simulate per‐minute trading P&L driven by a discrete signal column.

    Accepts either:
      1) A dict mapping each date → (day_df, trades[, perf_stats]),
         where day_df has columns 'bid', 'ask', and your discrete col_action (+1/0/-1).
      2) A single DataFrame with a DatetimeIndex and col_action column,
         which is split into per‐day slices internally.

    For each day we produce a minute‐level DataFrame df_sim with columns:
      Position, Cash, NetValue, Action, TradedAmount,
      BuyHoldEarning, StrategyEarning, EarningDiff
    Plus per‐trade lists:
      Trade Gains ($), Trade Gains (%)
    And overall performance_stats:
      Final Net Value ($), Buy & Hold Gain ($), Strategy Profit Difference ($),
      Final Net Return (%), Buy & Hold Return (%), Strategy Improvement (%),
      Trade Gains ($), Trade Gains (%)

    Returns
    -------
    updated_results : dict
        Maps date → (df_sim, trades_list, performance_stats)
    """
    # If passed a full DataFrame, split into dict of (day_df, empty trades)
    if isinstance(results_by_day_sign, pd.DataFrame):
        df_all = results_by_day_sign.sort_index()
        rebuilt = {}
        for date, df_day in df_all.groupby(df_all.index.normalize(), sort=False):
            rebuilt[date] = (df_day.copy(), [])
        results_by_day_sign = rebuilt

    updated_results = {}

    # Process each day
    for day, val in results_by_day_sign.items():
        # Unpack
        if len(val) == 2:
            day_df, trades_list = val
        elif len(val) == 3:
            day_df, trades_list, _ = val
        else:
            raise ValueError(f"Expected day tuple of length 2 or 3; got {len(val)} for {day}")

        # We'll simulate on a copy
        session_df = day_df.copy()

        # --- Initialize state ------------------------------------------------
        position = 0          # shares held
        cash     = 0.0        # cash balance
        session_open_price = None  # first ask ≥ regular_start

        # History lists
        positions         = []
        cash_balances     = []
        net_values        = []
        actions           = []
        traded_amounts    = []
        buy_hold_earnings = []
        strat_earnings    = []

        # Loop minute by minute
        for ts, row in session_df.iterrows():
            bid   = row['bid']
            ask   = row['ask']
            sig   = int(row[col_action])  # must be -1, 0, or +1
            now   = ts.time()

            # Determine trade action
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
                    action    = "Hold"
                    amt       = 0
            else:
                action    = "No trade"
                amt       = 0

            # Record state
            positions.append(position)
            cash_balances.append(np.round(cash, 3))
            net_val = np.round(cash + position * bid, 3)
            net_values.append(net_val)
            actions.append(action)
            traded_amounts.append(amt)

            # Compute earnings only after session opens
            if now >= regular_start:
                # lock in open price once
                if session_open_price is None:
                    session_open_price = ask
                bh = bid - session_open_price
                st = net_val
            else:
                bh = 0.0
                st = 0.0

            buy_hold_earnings.append(np.round(bh, 3))
            strat_earnings.append(np.round(st, 3))

        # --- Build simulation DataFrame --------------------------------------
        df_sim = session_df.copy()
        df_sim['Position']        = positions
        df_sim['Cash']            = cash_balances
        df_sim['NetValue']        = net_values
        df_sim['Action']          = actions
        df_sim['TradedAmount']    = traded_amounts
        df_sim['BuyHoldEarning']  = buy_hold_earnings
        df_sim['StrategyEarning'] = strat_earnings
        df_sim['EarningDiff']     = df_sim['StrategyEarning'] - df_sim['BuyHoldEarning']

        # --- Compute per-trade gains -----------------------------------------
        trade_gains     = []
        trade_gains_pct = []
        entry_price     = None
        entry_ts        = None

        for ts, row in df_sim.iterrows():
            act = row['Action']
            if act == "Buy" and entry_price is None:
                # open new trade
                entry_price = row['ask']
                entry_ts    = ts
            elif act == "Sell" and entry_price is not None:
                # close trade
                exit_price = row['bid']
                gain       = exit_price - entry_price
                pct_gain   = 100 * gain / entry_price

                trade_gains.append(np.round(gain, 3))
                trade_gains_pct.append(np.round(pct_gain, 3))
                # reset entry
                entry_price = None
                entry_ts    = None

        # if still holding at end, liquidate at last bid
        if entry_price is not None:
            final_bid = df_sim['bid'].iloc[-1]
            gain      = final_bid - entry_price
            pct_gain  = 100 * gain / entry_price
            trade_gains.append(np.round(gain, 3))
            trade_gains_pct.append(np.round(pct_gain, 3))

        # --- Compute overall performance ------------------------------------
        # Select all minutes within session window
        mask = (df_sim.index.time >= regular_start) & (df_sim.index.time < regular_end)
        if mask.any() and session_open_price is not None:
            # First and last valid indices
            valid_idxs = np.where(mask)[0]
            start_i    = valid_idxs[0]
            end_i      = valid_idxs[-1]

            baseline_ask    = session_open_price
            final_net_value = net_values[end_i]
            final_bid_price = session_df['bid'].iat[end_i]
        else:
            # No valid trading minutes → zeroed stats
            baseline_ask    = session_open_price or np.nan
            final_net_value = net_values[-1]
            final_bid_price = session_df['bid'].iat[-1]

        buy_hold_gain     = final_bid_price - baseline_ask
        profit_diff       = final_net_value - buy_hold_gain
        final_ret_pct     = 100 * final_net_value / baseline_ask
        buy_hold_ret_pct  = 100 * buy_hold_gain   / baseline_ask
        improve_pct       = final_ret_pct - buy_hold_ret_pct

        performance_stats = {
            'Final Net Value ($)'          : np.round(final_net_value, 3),
            'Buy & Hold Gain ($)'          : np.round(buy_hold_gain, 3),
            'Strategy Profit Difference ($)': np.round(profit_diff, 3),
            'Final Net Return (%)'         : np.round(final_ret_pct, 3),
            'Buy & Hold Return (%)'        : np.round(buy_hold_ret_pct, 3),
            'Strategy Improvement (%)'     : np.round(improve_pct, 3),
            'Trade Gains ($)'              : trade_gains,
            'Trade Gains (%)'              : trade_gains_pct
        }

        # Store results
        updated_results[day] = (df_sim, trades_list, performance_stats)

        # Optionally save per-day CSV
        df_sim.to_csv(f'csv simul/df_sim_{ticker}.csv', index=True)

    return updated_results



#########################################################################################################


def run_trading_pipeline(
    df_prep,
    col_signal,
    col_action,
    reference_gain,
    min_prof_thr=params.min_prof_thr_man, 
    max_down_prop=params.max_down_prop_man, 
    gain_tightening_factor=params.gain_tightening_factor_man, 
    smooth_win_sig=params.smooth_win_sig_man, 
    pre_entry_decay=params.pre_entry_decay_man,
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
        reference_gain=reference_gain,
        smooth_win_sig=smooth_win_sig,
        pre_entry_decay=pre_entry_decay,
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






