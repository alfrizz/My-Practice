import math
import pandas as pd
from pandas import Timestamp
import numpy as np
import glob
import os
from datetime import datetime
import datetime as dt
import pytz
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import display, HTML

#########################################################################################################
ticker = 'GOOGL'

label_col      = "signal_smooth_adjusted"
feature_cols   = ["open", "high", "low", "close", "volume"]

look_back = 60
red_pretr_win = 1 # factor to reduce the smoothing of the pretrade smoothing window
is_centered = True # smoothing and centering using past and future data (True) or only with past data without centering (False)

# Market Session	        US Market Time (ET)	             Corresponding Time in Datasheet (UTC)
# Premarket             	~4:00 AM – 9:30 AM	             9:00 – 14:30
# Regular Trading	        9:30 AM – 4:00 PM	             14:30 – 21:00
# After-Hours	           ~4:00 PM – 7:00 PM	             21:00 – 00:00

premarket_start  = datetime.strptime('09:00', '%H:%M').time()   

regular_start  = datetime.strptime('14:30', '%H:%M').time()   

regular_start_shifted = dt.time(*divmod(regular_start.hour * 60 + regular_start.minute - look_back, 60))

regular_end = datetime.strptime('21:00' , '%H:%M').time()   

afterhours_end = datetime.strptime('00:00' , '%H:%M').time()  

#########################################################################################################

def signal_parameters(ticker):
    '''
    # to define the trades
    min_prof_thr ==> # percent of minimum profit to define a potential trade
    max_down_prop ==> # float (percent/100) of maximum allowed drop of a potential trade
    gain_tightening_factor ==> # as gain grows, tighten the stop 'max_down_prop' by this factor.
    merging_retracement_thr ==> # intermediate retracement, relative to the first trade's full range
    merging_time_gap_thr ==> # time gap between trades, relative to the first and second trade durations
    
    # to define the smoothed signal
    smooth_win_sig ==> # smoothing window of the signal used for the identification of the final trades 
    pre_entry_decay ==> # pre-trade decay of the final trades' raw signal
    
    # to define the final buy and sell triggers
    buy_threshold ==> # float (percent/100) threshold of the smoothed signal to trigger the final trade
    trailing_stop_thresh ==> # percent of the trailing stop loss of the final trade
    '''
    if ticker == 'AAPL':
        # to define the initial trades:
        min_prof_thr=0.2 
        max_down_prop=0.4
        gain_tightening_factor=0.1
        merging_retracement_thr=0.9
        merging_time_gap_thr=0.7
        # to define the smoothed signal:
        smooth_win_sig=5
        pre_entry_decay=0.77
        # to define the final buy and sell triggers:
        buy_threshold=0.1
        trailing_stop_thresh=0.16
        
    if ticker == 'GOOGL':
        # to define the initial trades:
        min_prof_thr=0.2 
        max_down_prop=0.5
        gain_tightening_factor=0.1
        merging_retracement_thr=0.9
        merging_time_gap_thr=0.7
        # to define the smoothed signal:
        smooth_win_sig=15
        pre_entry_decay=0.005
        # to define the final buy and sell triggers:
        buy_threshold=0.25
        trailing_stop_thresh=0.25
        
    if ticker == 'TSLA':
        # to define the initial trades:
        min_prof_thr=0.45 
        max_down_prop=0.3
        gain_tightening_factor=0.02
        merging_retracement_thr=0.9
        merging_time_gap_thr=0.7
        # to define the smoothed signal:
        smooth_win_sig=3  
        pre_entry_decay=0.6
        # to define the final buy and sell triggers:
        buy_threshold=0.1 
        trailing_stop_thresh=0.1 

    return min_prof_thr, max_down_prop, gain_tightening_factor, smooth_win_sig, pre_entry_decay, \
        buy_threshold, trailing_stop_thresh, merging_retracement_thr, merging_time_gap_thr
    
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

def smooth_prepost_trading_data(df, regular_start, regular_end, red_pretr_win, tz_str="US/Eastern"):
    """
    Modifies the input DataFrame (using a copy) by first adjusting each day's timestamps based solely
    on DST detection. For every day, if DST is in effect (as determined by is_dst_for_day), we assume
    the day’s timestamps are recorded in solar time and add one hour so that they line up with legal time.
    
    Then, the function smooths all data outside the regular trading session (defined by regular_start and 
    regular_end) using a moving average. The smoothing window is computed from the ratio of average volumes 
    in the regular and non-regular sessions. The original columns are preserved with an "_orig" suffix.
    
    Within the smoothing loop:
      - Price columns (open, high, low, close, ask, bid) are rounded to 4 decimals.
      - The volume column is rounded to the nearest integer.
    
    Finally, the DataFrame's index is updated to the adjusted timestamps.
    
    Parameters:
      df : pd.DataFrame
          DataFrame with columns: open, high, low, close, volume, ask, bid and a DatetimeIndex.
      regular_start : str or datetime.time
          The target start time of the regular session (e.g. "13:30" or a time object).
      regular_end : str or datetime.time
          The target end time of the regular session (e.g. "20:00" or a time object).
      red_pretr_win : float
          Factor used to reduce the pre-trading smoothing window.
      tz_str : str, default "US/Eastern"
          The timezone used to detect DST.
    
    Returns:
      df : pd.DataFrame
          A modified copy of the DataFrame with updated (shifted) columns and index.
    """
    # Work on a copy.
    df = df.copy()
    df_orig = df.add_suffix("_orig")  # preserve actual original columns
    df = df.join(df_orig)
    
    # Identify the working columns (those not ending with "_orig") and convert them to float.
    working_cols = [col for col in df.columns if not col.endswith("_orig")]
    df[working_cols] = df[working_cols].astype(np.float64)
    
    # --- DST Adjustment: update timestamps for each day ---
    # Simply detect if the day is in DST. If yes, assume its timestamps are in solar time and *add* one hour.
    adj_times = df.index.to_series().copy()
    for day, group in df.groupby(df.index.normalize()):
        if is_dst_for_day(day, tz_str):
            # For days in DST, add 1 hour (shifting in the opposite direction than previously)
            adj_times.loc[group.index] = group.index + pd.Timedelta(hours=1)
        else:
            adj_times.loc[group.index] = group.index

    # Create masks for regular vs. non-regular rows using the adjusted times.
    target_start = pd.to_datetime(regular_start).time() if isinstance(regular_start, str) else regular_start
    target_end   = pd.to_datetime(regular_end).time() if isinstance(regular_end, str) else regular_end
    regular_mask = (adj_times.dt.time >= target_start) & (adj_times.dt.time <= target_end)
    nonregular_mask = ~regular_mask

    # Compute the smoothing window size based on volume ratios.
    avg_vol_regular = df.loc[regular_mask, "volume_orig"].mean()
    avg_vol_nonregular = df.loc[nonregular_mask, "volume_orig"].mean()
    if pd.isna(avg_vol_nonregular) or avg_vol_nonregular == 0:
        avg_vol_nonregular = red_pretr_win
    ratio = avg_vol_regular / avg_vol_nonregular
    window_nonregular = 1 if pd.isna(ratio) else max(int(ratio / red_pretr_win), 1)
    
    # Smooth each working column for non-regular rows.
    for col in working_cols:
        series_nr = df.loc[nonregular_mask, col].reset_index(drop=True)
        smoothed = series_nr.rolling(window=window_nonregular, min_periods=1).mean().values.astype(np.float64)
        if col != "volume":
            df.loc[nonregular_mask, col] = np.round(smoothed, 4).astype(np.float64)
        else:
            df.loc[nonregular_mask, col] = np.rint(smoothed).astype(np.int64)
    
    # Finally, update the DataFrame's index to use the adjusted times.
    df.index = adj_times

    return df



#########################################################################################################


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

def identify_trades_daily(df, min_prof_thr, max_down_prop, gain_tightening_factor, regular_start_shifted, regular_end, 
                          merging_retracement_thr, merging_time_gap_thr, day_to_check=None):
    """
    Identifies all trades for each trading day in a multi-day DataFrame and returns,
    for each day, a DataFrame reindexed to exactly cover the trading hours (at one-minute intervals)
    along with the list of identified trades for that day.

    Process:
      1. Ensure the DataFrame index is a DatetimeIndex.
      2. Group the DataFrame by day (using the normalized date).
      3. For each day:
           a. Build a fixed date_range from regular_start_shifted to regular_end.
           b. Filter the day’s data to these trading hours using between_time().
           c. Reindex the filtered data to the fixed minute index and forward-fill missing values.
           d. Identify trades using identify_trades().
      4. Only store days where at least one trade is found.

    Parameters:
      df : pd.DataFrame
          A DataFrame with a datetime index (or a column convertible to datetime) and at least a 'close' column.
      min_prof_thr : float
          The minimum profit percentage required to record a trade.
      max_down_prop : float
          The maximum allowed retracement in each trade.
      regular_start_shifted: datetime
          The start time for the trading session (e.g., '13:00' for shifted timestamps).
      regular_end : datetime
          The end time for the trading session (e.g., '20:00').
      day_to_check : str, optional
          A specific day (in 'YYYY-MM-DD' format) to process. Only that day will be processed
          if provided. Default is None (process all days).

    Returns:
      results_by_day_trad : dict
          A dictionary mapping each trading day (Timestamp) to a tuple:
              (day_df, trades)
          where day_df is the DataFrame strictly covering the trading hours (with one-minute frequency)
          and trades is a list of identified trade tuples.
    """

    # Ensure the index is a DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    results_by_day_trad = {}

    # Group the DataFrame by day
    for day, group in df.groupby(df.index.normalize()):
        if day_to_check is not None and day.strftime('%Y-%m-%d') != day_to_check:
            continue # If not the specific selected day, skip.

        # Build the fixed trading session index for the day.
        day_str = day.strftime('%Y-%m-%d')
        day_start = pd.Timestamp(f"{day_str} {regular_start_shifted}")
        day_end = pd.Timestamp(f"{day_str} {regular_end}")
        trading_index = pd.date_range(start=day_start, end=day_end, freq='min')

        # Filter the day's data to the defined trading hours.
        day_filtered = group.between_time(regular_start_shifted, regular_end)
        # Reindex using the fixed trading session index (forward fill missing minutes).
        day_df = day_filtered.reindex(trading_index).ffill()

        if day_df.empty:
            continue # If no valid data is present, skip this day.

        # Call the helper function using the day's filtered data.
        trades = identify_trades(day_df, min_prof_thr, max_down_prop, gain_tightening_factor, merging_retracement_thr, merging_time_gap_thr)

        # Only store days that contain at least one identified trade.
        if trades:
            results_by_day_trad[day] = (day_df, trades)
    
    return results_by_day_trad

#########################################################################################################

def compute_continuous_signal(
    day_df,
    trades,
    smooth_win_sig,
    decay,
    is_centered,
    reference_gain
):
    """
    Build and scale a minute‐by‐minute “convenience‐to‐buy” signal for one calendar day.

    Steps:
      1) Raw signal:
           For each merged trade ((buy_dt, sell_dt), (_, sell_price), _):
             • P_max = sell_price
             • For each timestamp t ≤ sell_dt:
                 raw_diff = P_max − close[t]
                 if raw_diff > 0:
                     Δ = minutes from t to sell_dt
                     signal[t] = max(signal[t], raw_diff * exp(−decay * Δ))

      2) Smoothing:
           signal_smooth = rolling‐mean(signal, window=smooth_win_sig,
                                         center=is_centered, min_periods=1)

      3) Normalization:
           signal_norm        = (signal − min(signal)) / (max(signal) − min(signal))
           signal_smooth_norm = (signal_smooth − min(signal_smooth)) / (max(signal_smooth) − min(signal_smooth))

      4) Price‐based scaling:
           daily_price_range = day_df["high"].max() − day_df["low"].min()
           scale             = daily_price_range / reference_gain
           signal_smooth_adjusted = signal_smooth_norm * scale

    Parameters:
      day_df               : pd.DataFrame for one trading day. Must include:
                               - DatetimeIndex matching minute bars
                               - columns "close", "high", "low"
      trades               : list of merged trades for that day,
                             each ((buy_dt, sell_dt), (buy_p, sell_p), profit_pc)
      smooth_win_sig       : int     – rolling window size in minutes
      decay                : float   – exponential‐decay rate per minute
      is_centered          : bool    – whether to center the rolling window
      reference_gain       : float   – long‐term avg daily price range
                               (e.g. mean over days of high.max−low.min)

    Returns:
      pd.DataFrame : copy of day_df augmented with columns:
                       • "signal"
                       • "signal_smooth"
                       • "signal_norm"
                       • "signal_smooth_norm"
                       • "signal_smooth_adjusted"
    """
    # 0) work on a copy; initialize raw signal column
    df = day_df.copy()
    df["signal"] = 0.0

    # 1) build the raw decayed signal up to the trade peaks
    for (buy_dt, sell_dt), (_, sell_price), _ in trades:
        P_max = sell_price
        # restrict to times up to the sell timestamp
        mask = df.index <= sell_dt
        for t in df.index[mask]:
            raw_diff = P_max - df.at[t, "close"]
            if raw_diff <= 0:
                continue
            # apply exponential decay based on distance to the peak
            delta_min = (sell_dt - t).total_seconds() / 60.0
            penalized = raw_diff * math.exp(-decay * delta_min)
            # keep the maximum across overlapping trades
            df.at[t, "signal"] = max(df.at[t, "signal"], penalized)

    # 2) smooth the raw signal with a rolling average
    df["signal_smooth"] = (
        df["signal"]
          .rolling(window=smooth_win_sig,
                   center=is_centered,
                   min_periods=1)
          .mean()
    )

    # 3) per‐day min–max normalization to [0,1]
    lo, hi = df["signal"].min(), df["signal"].max()
    if hi > lo:
        df["signal_norm"] = (df["signal"] - lo) / (hi - lo)
    else:
        df["signal_norm"] = 0.0

    slo, shi = df["signal_smooth"].min(), df["signal_smooth"].max()
    if shi > slo:
        df["signal_smooth_norm"] = (df["signal_smooth"] - slo) / (shi - slo)
    else:
        df["signal_smooth_norm"] = 0.0

    # 4) relative price‐range scaling
    day_hi = day_df["high"].max()
    day_lo = day_df["low"].min()
    daily_rel_range = (day_hi - day_lo) / day_lo

    scale = daily_rel_range / reference_gain
    df["signal_smooth_adjusted"] = df["signal_smooth_norm"] * scale
    
    return df



#########################################################################################################


def generate_trade_actions(
    df,
    buy_threshold,
    trailing_stop_thresh,
    regular_start
):
    """
    Given a DataFrame for one trading day that already contains:
        • "signal_smooth"  — your smoothed continuous signal
        • "close"          — the price series
    
    This function strips out any normalization or scaling steps and simply:
      1) Buys when the smoothed signal ≥ buy_threshold (only at/after regular_start)
      2) Tracks a simple trailing stop of trailing_stop_thresh% off the highest price since entry
      3) Sells when price falls below that trailing stop
      4) Forces a sell at EOD if still in trade

    Parameters:
      df : pd.DataFrame
          Must have a DatetimeIndex, and columns "signal_smooth" and "close".
      buy_threshold : float
          Cut-off on signal_smooth at which to enter a trade.
      trailing_stop_thresh : float
          Percent trailing stop (e.g. 1.5 means 1.5%).
      regular_start : datetime.time or str
          Don’t enter a new trade before this time (e.g. "13:30").

    Returns:
      pd.DataFrame : original df plus "trade_action" column:
                     +1 = buy, 0 = hold, -1 = sell
    """
    # Work on a copy to avoid side-effects
    df = df.copy()
    n = len(df)

    # Initialize trade_action and state
    df["trade_action"] = 0
    in_trade = False
    entry_price = None
    max_price = None

    # Convert regular_start to time if needed
    if isinstance(regular_start, str):
        regular_start = pd.to_datetime(regular_start).time()

    # Extract series for speed
    sig = df["signal_smooth_adjusted"].values
    closes = df["close"].values
    times = df.index.time

    for i in range(n):
        t = times[i]
        price = closes[i]

        if not in_trade:
            # Check buy condition: signal crosses threshold, and time ≥ regular_start
            if sig[i] >= buy_threshold and t >= regular_start:
                df.iat[i, df.columns.get_loc("trade_action")] = 1
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
                df.iat[i, df.columns.get_loc("trade_action")] = -1
                in_trade = False
            else:
                df.iat[i, df.columns.get_loc("trade_action")] = 0

    # If still in trade at end-of-day, force a sell on last row
    if in_trade:
        df.iat[-1, df.columns.get_loc("trade_action")] = -1

    return df


#########################################################################################################


def add_trade_signal_to_results(results_by_day_trad, min_prof_thr, regular_start, reference_gain,
                                smooth_win_sig, pre_entry_decay,
                                buy_threshold, trailing_stop_thresh, is_centered):
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
          The results_by_day_sign dictionary updated so that each day's DataFrame now includes:
              "signal", "signal_smooth", "signal_norm", "signal_smooth_norm", and "trade_action".
    """
    updated_results = {}
    
    for day, (day_df, trades) in results_by_day_trad.items():
        
        # Step (A): Compute the continuous trading signal.
        df = compute_continuous_signal(day_df, trades, smooth_win_sig, pre_entry_decay, is_centered, reference_gain)
        
        # Step (B): Generate discrete trade actions (using trailing stop loss logic).
        df = generate_trade_actions(df, buy_threshold, trailing_stop_thresh, regular_start)
        
        updated_results[day] = (df, trades)
    
    return updated_results

#########################################################################################################

def simulate_trading(results_by_day_sign, regular_start, regular_end, ticker):
    """
    Processes the results_by_day_sign dictionary by simulating 
    trading for each day's DataFrame. It uses the precomputed "trade_action" column to drive 
    the simulation. The updated DataFrame is augmented with the following new columns:
      - "Position": cumulative number of shares held.
      - "Cash": evolving cash balance.
      - "NetValue": mark-to-market net asset value (cash + position * bid price).
      - "Action": a string describing the action ("Buy", "Sell", "Hold", "No trade").
      - "TradedAmount": the number of shares traded (+1 for buy, -1 for sell, 0 otherwise).
      - "StrategyEarning": the current net value during the session (0 before regular start).
      - "BuyHoldEarning": profit (or loss) if one had bought at the session’s start ask and sold at the current bid (0 before regular start).
      - "EarningDiff": the difference between StrategyEarning and BuyHoldEarning.
      
    Additionally, it computes two lists:
      - "Trade Gains ($)": a list of gains for each completed trade.
      - "Trade Gains (%)": a list of percentage gains for each completed trade.
      
    All preexisting columns in the input DataFrame are preserved.
    
    Parameters:
      results_by_day_sign : dict
          A dictionary mapping each trading day (Timestamp) to a tuple. The tuple can be either:
              (day_df, trades)
          or
              (day_df, trades, performance_stats)
          where day_df is the DataFrame covering the trading hours (with one-minute frequency)
          and trades is a list of identified trade tuples.
      regular_start : datetime.time 
          Starting time of the trading session (e.g. '13:00').
      regular_end : datetime.time 
          Ending time of the trading session (e.g. '20:00').
    
    Returns:
      updated_results_by_day : dict
          A dictionary mapping each trading day (Timestamp) to a tuple:
              (df_sim, trades, performance_stats)
          where df_sim is the updated simulation DataFrame for that day.
    """
    
    updated_results = {}
    
    # Process each day. Unpack values depending on their length.
    for day, value in results_by_day_sign.items():
        if len(value) == 2:
            day_df, trades = value
        elif len(value) == 3:
            day_df, trades, _ = value   # Ignore existing performance_stats if present.
        else:
            raise ValueError(f"Expected tuple of length 2 or 3 for key {day}, got {len(value)}")
        
        session_df = day_df.copy()  # Work with a copy.
        
        # Initialize simulation variables.
        position = 0   # shares held
        cash = 0       # starting cash
        
        positions = []      # cumulative position per minute
        cash_balances = []  # cash balance per minute
        net_values = []     # net asset value: cash + position * bid
        actions = []        # descriptive action: "Buy", "Sell", "Hold", "No trade"
        traded_amounts = [] # numeric traded amount: +1, -1, or 0
        
        # Lists for the earnings:
        buy_hold_earnings = []     # Earning without the strategy (buy-and-hold)
        strategy_earnings = []     # Earning with the strategy (actual simulation net value)
        
        session_initial_trade_price = None  # Will be set at the first row with time >= regular_start
        
        # Loop over each minute (row) in the session.
        for timestamp, row in session_df.iterrows():
            bid_price = row['bid']
            ask_price = row['ask']
            action_num = row['trade_action']   # Precomputed signal: +1 (buy), -1 (sell), 0 (hold)
            current_time = timestamp.time()
            
            # Only execute trades if within session hours.
            if regular_start <= current_time < regular_end:
                if action_num == 1:
                    position += 1
                    cash -= ask_price  # buy at ask price
                    action = "Buy"
                    traded_amt = 1
                elif action_num == -1:
                    if position > 0:
                        position -= 1
                        cash += bid_price  # sell at bid price
                        action = "Sell"
                        traded_amt = -1
                    else:
                        action = "Hold"
                        traded_amt = 0
                else:
                    action = "Hold"
                    traded_amt = 0
            else:
                action = "No trade"
                traded_amt = 0
            
            positions.append(position)
            cash_balances.append(np.round(cash, 3))
            net_val = np.round(cash + position * bid_price, 3)
            net_values.append(net_val)
            actions.append(action)
            traded_amounts.append(traded_amt)
            
            # Compute earnings only if current_time >= regular_start.
            if current_time >= regular_start:
                if session_initial_trade_price is None:
                    # Set the reference (buy-and-hold) price at the first row of the session.
                    session_initial_trade_price = ask_price
                current_bh = bid_price - session_initial_trade_price
                current_strat = net_val  # Strategy net value.
            else:
                current_bh = 0
                current_strat = 0
            buy_hold_earnings.append(np.round(current_bh, 3))
            strategy_earnings.append(np.round(current_strat, 3))
        
        # Build the simulation DataFrame (preserving all preexisting columns).
        df_sim = session_df.copy()
        df_sim['Position'] = positions
        df_sim['Cash'] = cash_balances
        df_sim['NetValue'] = net_values
        df_sim['Action'] = actions
        df_sim['TradedAmount'] = traded_amounts
        
        # Append the new earnings columns.
        df_sim['StrategyEarning'] = strategy_earnings
        df_sim['BuyHoldEarning'] = buy_hold_earnings
        df_sim['EarningDiff'] = df_sim['StrategyEarning'] - df_sim['BuyHoldEarning']
        
        # --------------------------------------------------------------------
        # NEW: Compute separate gains (in $ and in %) for each trade using the "Action" column.
        # A trade is defined as a "Buy" (recorded from the Ask price) followed by the next "Sell" (using the Bid).
        trade_gains = []
        trade_gains_perc = []
        buy_price = None  # Reference price for the current trade.
        for timestamp, row in df_sim.iterrows():
            if row['Action'] == "Buy":
                buy_price = row['ask']
            elif row['Action'] == "Sell" and buy_price is not None:
                gain = row['bid'] - buy_price
                perc_gain = (gain / buy_price) * 100
                trade_gains.append(np.round(gain, 3))
                trade_gains_perc.append(np.round(perc_gain, 3))
                buy_price = None
        # NEW: If there's an open trade at the end, simulate a sale using the final bid price.
        if buy_price is not None:
            final_bid = df_sim.iloc[-1]['bid']
            gain = final_bid - buy_price
            perc_gain = (gain / buy_price) * 100
            trade_gains.append(np.round(gain, 3))
            trade_gains_perc.append(np.round(perc_gain, 3))
        # --------------------------------------------------------------------
        
        # --- Compute performance statistics using realistic prices ---
        idx_start = df_sim.index.get_loc(next(ts for ts in df_sim.index if ts.time() >= regular_start))
        baseline_price = session_df.iloc[idx_start]['ask']
        if len(session_df) > 1:
            final_liquidation_price = session_df.iloc[-2]['bid']  # penultimate row's bid.
            buy_hold_gain = final_liquidation_price - baseline_price
            final_net_value = net_values[-2]
        else:
            final_liquidation_price = session_df.iloc[-1]['bid']
            buy_hold_gain = final_liquidation_price - baseline_price
            final_net_value = net_values[-1]
        
        profit_diff = final_net_value - buy_hold_gain
        final_net_return_pct = (final_net_value / baseline_price) * 100
        buy_hold_return_pct = (buy_hold_gain / baseline_price) * 100
        
        strategy_improve_pct = final_net_return_pct - buy_hold_return_pct
        
        performance_stats = {
            'Final Net Value ($)': np.round(final_net_value, 3),
            'Buy & Hold Gain ($)': np.round(buy_hold_gain, 3),
            'Strategy Profit Difference ($)': np.round(profit_diff, 3),
            'Final Net Return (%)': np.round(final_net_return_pct, 3),
            'Buy & Hold Return (%)': np.round(buy_hold_return_pct, 3),
            'Strategy Improvement (%)': np.round(strategy_improve_pct, 3),
            'Trade Gains ($)': trade_gains,            # List of individual trade gains (in dollars).
            'Trade Gains (%)': trade_gains_perc          # List of individual trade gain percentages.
        }
        
        updated_results[day] = (df_sim, trades, performance_stats)

        # Save the simulation DataFrame.
        df_sim.to_csv(f'backtest/df_label_{ticker}.csv', index=True)
    
    return updated_results

