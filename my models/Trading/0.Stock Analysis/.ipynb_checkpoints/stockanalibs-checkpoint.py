import math
import pandas as pd
from pandas import Timestamp
import numpy as np
import glob
import os
from datetime import datetime
import datetime as dt

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import display, HTML

#########################################################################################################
ticker = 'GOOGL'

label_col      = "signal_smooth_adjusted"
feature_cols   = ["open", "high", "low", "close", "volume"]

look_back = 60
red_pretr_win = 3 # factor to reduce the pretrade smoothing window
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
        min_prof_thr=0.25 
        max_down_prop=0.5
        gain_tightening_factor=0.1
        merging_retracement_thr=0.9
        merging_time_gap_thr=0.7
        # to define the smoothed signal:
        smooth_win_sig=15
        pre_entry_decay=0.1
        # to define the final buy and sell triggers:
        buy_threshold=0.3
        trailing_stop_thresh=0.15
        
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

def smooth_prepost_trading_data(df, regular_start, regular_end, red_pretr_win):
    """
    Modifies the input DataFrame in place by shifting entire days that begin in the 8:00 hour
    by +1 hour so that the trading times become aligned (solar times), and then smoothing
    all data outside the regular trading session continuously across day boundaries.
    
    The smoothing window (in minutes) is computed from the ratio of average volumes in the 
    regular and non-regular sessions.
    
    All columns (open, high, low, close, volume, ask, bid) are then smoothed using a moving
    average over the non-regular rows (i.e. those outside the session defined by regular_start 
    and regular_end). The original columns are preserved with an "_orig" suffix.
    
    Within the smoothing loop:
      - Price columns (open, high, low, close, ask, bid) are rounded to 4 decimals.
      - The volume column is rounded to the nearest integer.
    
    Additionally, the DataFrame's index is updated to reflect the shifted (solar) times.
    
    Parameters:
      df : pd.DataFrame
          DataFrame with columns: open, high, low, close, volume, ask, bid and a DatetimeIndex.
      regular_start : The start time of the regular session.
      regular_end :  The end time of the regular session.
      red_pretr_win : factor to reduce the pretrade smoothing window
    
    Returns:
      df : pd.DataFrame
          The same DataFrame (modified in place) with updated columns and index.
    """
    
    df = df.copy()
    df_orig = df.add_suffix("_orig")            # every col -> <col>_orig
    df = df.join(df_orig)                       # now both copies coexist

    # Identify the working columns (those not ending with "_orig") and convert them to float.
    working_cols = [col for col in df.columns if not col.endswith("_orig")]
    df[working_cols] = df[working_cols].astype(np.float64)

    # Compute adjusted times (temporary Series) by shifting days with an 8:00 hour start.
    adj_times = df.index.to_series().copy()
    for day, group in df.groupby(df.index.normalize()):
        if group.index.min().hour == 8:  # if first timestamp is anywhere between 08:00 and 08:59
            adj_times.loc[group.index] = group.index + pd.Timedelta(hours=1)
        else:
            adj_times.loc[group.index] = group.index

    # Create masks for regular vs. non-regular rows using the adjusted times.
    regular_mask = (adj_times.dt.time >= regular_start) & (adj_times.dt.time <= regular_end)
    nonregular_mask = ~regular_mask

    # Compute the smoothing window size based on the ratio of average volumes.

    avg_vol_regular = df.loc[regular_mask, "volume_orig"].mean()
    avg_vol_nonregular = df.loc[nonregular_mask, "volume_orig"].mean()

    # ratio = avg_vol_regular / (avg_vol_nonregular if avg_vol_nonregular != 0 else red_pretr_win)
    # window_nonregular = int(ratio / red_pretr_win)

    # If avg_vol_nonregular is NaN or zero, replace it with red_pretr_win.
    if pd.isna(avg_vol_nonregular) or avg_vol_nonregular == 0:
        avg_vol_nonregular = red_pretr_win
    
    # Calculate the ratio.
    ratio = avg_vol_regular / avg_vol_nonregular
    
    # If ratio is NaN, default the window size to 1; otherwise, compute normally.
    if pd.isna(ratio):
        window_nonregular = 1
    else:
        window_nonregular = max(int(ratio / red_pretr_win), 1)
    
    # Loop only through the working columns
    for col in working_cols:
        # Extract non‑regular rows (resetting the index to get a continuous Series).
        series_nr = df.loc[nonregular_mask, col].reset_index(drop=True)
        smoothed = series_nr.rolling(window=window_nonregular, min_periods=1).mean().values.astype(np.float64)
        if col != "volume":
            df.loc[nonregular_mask, col] = np.round(smoothed, 4).astype(np.float64)
        else:
            df.loc[nonregular_mask, col] = np.rint(smoothed).astype(np.int64)

    # Update the DataFrame's index to the adjusted times.
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
      results_by_day : dict
          A dictionary mapping each trading day (Timestamp) to a tuple:
              (day_df, trades)
          where day_df is the DataFrame strictly covering the trading hours (with one-minute frequency)
          and trades is a list of identified trade tuples.
    """

    # Ensure the index is a DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    results_by_day = {}

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
            results_by_day[day] = (day_df, trades)

    return results_by_day

#########################################################################################################

def compute_continuous_signal(day_df, trades, min_prof_thr, smooth_win_sig, pre_entry_decay, is_centered):
    """
    Computes the continuous trading signal for a single trading day based on the provided trades.
    
    For each trade, assume:
      - t_min = buy_date (optimal entry)
      - P_max = sell_price (assumed maximum price reached during the trade)
      
    The raw signal at time t is defined as:
         raw_value = (P_max - close(t)) - min_prof_thr.
         
    For raw_value <= 0, the signal is 0.
    For t < t_min, an exponential penalty is applied:
         signal(t) = exp(-pre_entry_decay * delta_minutes) * raw_value,
    where delta_minutes is the minutes between t and t_min.
    For t >= t_min (up to sell_date), no penalty is applied.
    For overlapping trades the maximum signal is used.
    
    After calculating the continuous signal (stored in "signal"), a smoothed version is computed,
    stored in "signal_smooth" using a rolling window.
    
    Parameters:
      day_df : pd.DataFrame
          DataFrame for the day (must include a "close" column and a datetime index).
      trades : list
          List of trades for the day. Each trade is expected to be a tuple:
          ((buy_date, sell_date), (buy_price, sell_price), profit_pc)
      min_prof_thr : float
          Minimum profit threshold above which potential profit is counted.
      smooth_win_sig : int
          Rolling window size (in minutes) for smoothing the continuous signal.
      pre_entry_decay : float
          Decay rate for applying an exponential penalty to points before the trade entry.
          Increasing the pre_entry_decay value makes the signal drop off more steeply before trade entry
    
    Returns:
      df : pd.DataFrame
          The input DataFrame updated with:
             "signal"         -- raw continuous trading signal.
             "signal_smooth"  -- smoothed continuous trading signal.
    """
    df = day_df.copy()
    df["signal"] = 0.0  # initialize continuous signal column

    # Process each trade and update the signal accordingly.
    for trade in trades:
        (buy_date, sell_date), (buy_price, sell_price), profit_pc = trade
        P_max = sell_price  # assumed maximum reached in trade
        t_min = buy_date    # optimal entry time
        
        # Only calculate signal for time points up to the sell_date.
        mask = df.index <= sell_date
        for t in df.index[mask]:
            price = df.at[t, "close"]
            raw_value = (P_max - price) - min_prof_thr
            if raw_value <= 0:
                signal_value = 0.0
            else:
                # For times before the trade entry, apply exponential penalty.
                if t < t_min:
                    delta_minutes = (t_min - t).total_seconds() / 60.0
                    penalty = math.exp(-pre_entry_decay * delta_minutes)
                    signal_value = penalty * raw_value
                else:
                    signal_value = raw_value
            # If there are overlapping trades, store the maximum signal.
            df.at[t, "signal"] = max(df.at[t, "signal"], signal_value)
    
    # Smooth the continuous signal with a rolling window.
    # df['signal_smooth'] = df['signal'].ewm(span=smooth_win_sig, adjust=is_centered).mean()
    df['signal_smooth'] = df['signal'].rolling(window=smooth_win_sig, center=is_centered).mean()
    
    return df

#########################################################################################################

def generate_trade_actions(df, smooth_win_sig, buy_threshold, trailing_stop_thresh, regular_start, reference_gain):
    """
    Generates discrete trade actions for a single day, incorporating an adjustment that scales the intra-day
    normalized signal with respect to a reference gain level (e.g. an average daily gain).

    The method works as follows:
      1. Normalize the pre-computed "signal" column using min–max normalization to produce "signal_norm".
      2. Normalize the already-smoothed signal ("signal_smooth") via min–max normalization to obtain "signal_smooth_norm".
      3. Compute the day’s overall gain in the raw signal as: 
              daily_gain = max(signal) - min(signal)
         Then, calculate a scaling factor as:
              scale_factor = daily_gain / reference_gain
         and create an adjusted smoothed signal:
              signal_smooth_adjusted = signal_smooth_norm * scale_factor
      4. Using this adjusted signal, the function iterates over time points:
         - A buy is triggered the first time when the adjusted smoothed signal crosses above (or equals)
           the buy_threshold, but only if the current time is at/after the specified regular_start.
         - Once in a trade, the raw trade entry price is recorded and the maximum close since entry is tracked.
         - The trailing stop level is computed dynamically as: 
                 trailing_stop_level = trade_max_price * (1 - (trailing_stop_thresh * (1 + adjusted_signal))/100)
           A sell is triggered when the current raw close falls below this level (and the adjusted signal is below the buy threshold).

    Parameters:
      df : pd.DataFrame
         DataFrame that already contains columns "signal", "signal_smooth" and "close". It should represent one day.
      smooth_win_sig : int
          Rolling window size used for smoothing the signal (assumed already applied to create "signal_smooth").
      buy_threshold : float
          The threshold which, when crossed or met by the adjusted smoothed signal, triggers a buy.
      trailing_stop_thresh : float
          Trailing stop loss percent used in computing a dynamic trailing stop level.
      regular_start : datetime.time or str
          The earliest time of day at which a buy signal is allowed (e.g. "13:30" or datetime.time(13,30)).
      reference_gain : float
          A reference gain parameter computed externally (e.g. an average daily gain) that is used to adjust the normalized signal.
    
    Returns:
      df : pd.DataFrame
          The input DataFrame updated with additional columns:
             "signal_norm"            -- the normalized raw signal.
             "signal_smooth_norm"     -- the normalized smoothed signal.
             "signal_smooth_adjusted" -- the adjusted normalized smoothed signal (scaled by daily gain/reference_gain).
             "trade_action"           -- discrete trade actions:
                                          +1 for buy, 0 for hold, -1 for sell.
    """
    n = len(df)
    
    # --- Step 1: Normalize the raw "signal" ---
    sig_min = df["signal"].min()
    sig_max = df["signal"].max()
    if sig_max == sig_min:
        df["signal_norm"] = 0.0
    else:
        df["signal_norm"] = (df["signal"] - sig_min) / (sig_max - sig_min)
    
    # --- Step 2: Normalize the smoothed signal ---
    sig_sm_min = df["signal_smooth"].min()
    sig_sm_max = df["signal_smooth"].max()
    if sig_sm_max == sig_sm_min:
        df["signal_smooth_norm"] = 0.0
    else:
        df["signal_smooth_norm"] = (df["signal_smooth"] - sig_sm_min) / (sig_sm_max - sig_sm_min)
    
    # --- Step 3: Adjust the normalized smoothed signal by the day's gain ---
    # Compute the day's gain in the raw signal. This represents the overall profit potential for the day.
    daily_gain = sig_max - sig_min

    # Compute the scale factor based on the reference_gain.
    scale_factor = daily_gain / reference_gain

    # Adjust the normalized smoothed signal by multiplying by the scale factor.
    df["signal_smooth_adjusted"] = df["signal_smooth_norm"] * scale_factor
    
    # --- Initialize trade action column and trade state variables ---
    df["trade_action"] = 0  # default: hold/no action
    in_trade = False
    trade_buy_price = None   # raw close price at entry
    trade_max_price = None   # maximum raw close reached since entry
    entry_signal = 0.0       # capture the entry (adjusted) signal when buy condition is met

    pending_buy = False      # flag to indicate a buy condition has been met (even if before regular_start)
    pending_buy_signal = 0.0 # stores the adjusted signal value when condition was met
    
    # Retrieve the adjusted smoothed signal as a NumPy array for faster access.
    smooth_adj_signal = df["signal_smooth_adjusted"].values
    
    # --- Trade Generation Phase: iterate over each time point ---
    for i in range(n):
        # Get the current time from the index (assumes df.index is a DatetimeIndex).
        current_time = df.iloc[i].name.time()
        
        if not in_trade:
            # Check if the adjusted smoothed signal has crossed above or equals the buy threshold.
            # For the very first buy, we also check that the signal rises from below.
            if smooth_adj_signal[i] >= buy_threshold and ((df["trade_action"] == 1).any() == False or 
                                                          (i > 0 and smooth_adj_signal[i-1] < buy_threshold)):
                # Mark a pending buy condition.
                pending_buy = True
                pending_buy_signal = smooth_adj_signal[i]
            
            # If a buy condition was met and the current time is allowed (>= regular_start), trigger buy.
            if pending_buy and current_time >= regular_start and smooth_adj_signal[i] >= buy_threshold:
                df.iloc[i, df.columns.get_loc("trade_action")] = 1  # Trigger buy.
                trade_buy_price = df["close"].iloc[i]
                trade_max_price = trade_buy_price
                entry_signal = pending_buy_signal  # Store the signal level at buy entry.
                in_trade = True
                pending_buy = False
        else:
            # We are in a trade. Update the maximum reached price.
            current_price = df["close"].iloc[i]
            if current_price > trade_max_price:
                trade_max_price = current_price

            # Compute a dynamic trailing stop level.
            # The trailing stop is adjusted by the current adjusted signal.
            dynamic_trailing_thresh = trailing_stop_thresh * (1 + smooth_adj_signal[i])
            trailing_stop_level = trade_max_price * (1 - dynamic_trailing_thresh / 100)
            
            # If the current price falls below the trailing stop level—and the signal is not indicating a buy,
            # we trigger a sell.
            if current_price < trailing_stop_level and smooth_adj_signal[i] < buy_threshold:
                df.iloc[i, df.columns.get_loc("trade_action")] = -1  # Trigger sell.
                in_trade = False
            else:
                df.iloc[i, df.columns.get_loc("trade_action")] = 0  # Hold.
    
    # Optionally force a sell at the end of the day if still in trade.
    if in_trade:
        df.iloc[-1, df.columns.get_loc("trade_action")] = -1
    
    return df


#########################################################################################################

def add_trade_signal_to_results(results_by_day, min_prof_thr, regular_start, reference_gain,
                                smooth_win_sig, pre_entry_decay,
                                buy_threshold, trailing_stop_thresh, is_centered):
    """
    Updates the input dictionary (results_by_day) by applying two steps:
    
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
      results_by_day : dict
          Dictionary mapping each trading day (Timestamp) to a tuple (day_df, trades).
      min_prof_thr : float
          Minimum profit threshold used in continuous signal calculation.
      smooth_win_sig : int, default 5
          Rolling window size for smoothing signals.
      pre_entry_decay : float, default 0.05
          Decay rate applied before the trade entry for signal calculation.
      buy_threshold : float, default 0.6
          Minimum level in the smoothed normalized signal to consider a candidate peak for a trade.
      trailing_stop_thresh : float, default 0.5 percent of max stock price
          Trailing stop loss threshold. A sell signal is triggered when the retracement in raw close
          prices meets/exceeds this value.
    
    Returns:
      updated_results : dict
          The results_by_day dictionary updated so that each day's DataFrame now includes:
              "signal", "signal_smooth", "signal_norm", "signal_smooth_norm", and "trade_action".
    """
    updated_results = {}
    
    for day, (day_df, trades) in results_by_day.items():
        # Step (A): Compute the continuous trading signal.
        df = compute_continuous_signal(day_df, trades, min_prof_thr, smooth_win_sig, pre_entry_decay, is_centered)
        
        # Step (B): Generate discrete trade actions (using trailing stop loss logic).
        df = generate_trade_actions(df, smooth_win_sig, buy_threshold, trailing_stop_thresh, regular_start, reference_gain)
        
        updated_results[day] = (df, trades)
    
    return updated_results

#########################################################################################################

def simulate_trading(results_by_day, regular_start, regular_end, ticker):
    """
    Processes the results_by_day dictionary (produced by identify_trades_daily) by simulating 
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
      results_by_day : dict
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
    for day, value in results_by_day.items():
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

