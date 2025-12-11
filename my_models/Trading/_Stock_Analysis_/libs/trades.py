from libs import params, plots, strategies

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

    
# #########################################################################################################

# ##################
# _last_trail = 0.0
# ##################

# def generate_trade_actions(
#     df: pd.DataFrame,
#     col_signal: str,
#     buy_thresh: float,
#     trailstop_pct: float,
#     sellmin_idx: int,
#     sess_start: time,
#     col_price: str = "close",
#     col_smoothsign: float = None
# ) -> pd.DataFrame:
#     """
#     Generate discrete trade actions (Buy/Sell/Hold) and a trailing-stop price series from signals and price data.
#     """
#     global _last_trail

#     # work on a copy to avoid mutating caller's dataframe
#     df = df.copy()
#     df["action"] = 0 # HOLD

#     # extract arrays for faster indexed access inside the loop
#     sig = df[col_signal].to_numpy()
#     times = df.index.time
#     prices = df[col_price].to_numpy(dtype=float)
#     stop_frac = float(trailstop_pct) / 100.0  # convert percent to fraction

#     # prepare an array to hold computed trailing stop prices (NaN when no open trade)
#     trail_arr = np.full(len(df), np.nan, dtype=float)
#     # open_flag indicates whether we started the day with a carried open trade
#     open_flag = (_last_trail > 0)

#     # allow scalar or per-row threshold
#     buy_arr = None if np.ndim(buy_thresh) == 0 else np.asarray(buy_thresh)

#     # iterate over rows by index to compute actions and trailing stops
#     for i in range(len(df)):
#         cur_thresh = float(buy_arr[i]) if buy_arr is not None else float(buy_thresh)
        
#         # if already in-trade (previous trail exists), compute peak from previous trail
#         if i > 0 and np.isfinite(trail_arr[i - 1]) and df["action"].iat[i - 1] != -1:
#             # compute the running peak using previous trail value and current price
#             peak = max(prices[i], trail_arr[i - 1] / (1.0 - stop_frac))
#             trail_val = peak * (1.0 - stop_frac)
#             trail_arr[i] = trail_val
            
#             # check exit SELL condition:  
#             if ((sig[i] < cur_thresh) # signal dropped below threshold
#                 and (df["bid"].iat[i] < trail_val) #  bid price below running trail stop price
#                 and (times[i] >= sess_start)): # sell only during trading time
#                 df.at[df.index[i], "action"] = -1 # SELL
#                 open_flag = False
#             continue

#         # not in trade: possible entry BUY
#         if (((sig[i] >= cur_thresh) 
#             and (times[i] >= sess_start)) # buy only during trading time ...
#                  or (_last_trail > 0)): # ... unless open trade carried from previous day
#             df.at[df.index[i], "action"] = 1 # BUY
#             # seed peak with either current price or carried last trail (converted back to peak)
#             peak = max(prices[i], _last_trail / (1.0 - stop_frac))
#             _last_trail = 0.0  # consume carried trail so it's used only once
#             trail_arr[i] = peak * (1.0 - stop_frac)
#             open_flag = True

#     # attach computed trailing stop series to dataframe
#     df["trailstop_price"] = pd.Series(trail_arr, index=df.index)

#     # persist last trail: last value if day ended with open trade, else 0
#     if sellmin_idx is None:
#         _last_trail = float(df["trailstop_price"].iloc[-1]) if open_flag else 0.0

#     return df


# ####################################

# def generate_tradact_inds(
#     df: pd.DataFrame,
#     col_signal: str,
#     buy_thresh: float,
#     trailstop_pct: float,
#     sellmin_idx: int,
#     sess_start: time,
#     col_price: str = "close",
#     col_smoothsign: float = None
# ) -> pd.DataFrame:
#     """
#     Generate discrete trade actions (Buy/Sell/Hold) and a trailing-stop price series from signals and price data.
#     """
#     global _last_trail

#     # work on a copy to avoid mutating caller's dataframe
#     df = df.copy()
#     df["action"] = 0 # HOLD

#     # extract arrays for faster indexed access inside the loop
#     sig = df[col_signal].to_numpy()
#     times = df.index.time
#     prices = df[col_price].to_numpy(dtype=float)
#     stop_frac = float(trailstop_pct) / 100.0  # convert percent to fraction

#     # prepare an array to hold computed trailing stop prices (NaN when no open trade)
#     trail_arr = np.full(len(df), np.nan, dtype=float)
#     # open_flag indicates whether we started the day with a carried open trade
#     open_flag = (_last_trail > 0)

#     # allow scalar or per-row threshold
#     buy_arr = None if np.ndim(buy_thresh) == 0 else np.asarray(buy_thresh)

#     # iterate over rows by index to compute actions and trailing stops
#     for i in range(len(df)):
#         cur_thresh = float(buy_arr[i]) if buy_arr is not None else float(buy_thresh)
        
#         # if already in-trade (previous trail exists), compute peak from previous trail
#         if i > 0 and np.isfinite(trail_arr[i - 1]) and df["action"].iat[i - 1] != -1:
#             # compute the running peak using previous trail value and current price
#             peak = max(prices[i], trail_arr[i - 1] / (1.0 - stop_frac))
#             trail_val = peak * (1.0 - stop_frac)
#             trail_arr[i] = trail_val
            
#             # check exit SELL condition:  
#             if ((sig[i] < cur_thresh) # signal dropped below threshold
#                 and (df["bid"].iat[i] < trail_val) #  bid price below running trail stop price
#                 and (times[i] >= sess_start)): # sell only during trading time
#                 df.at[df.index[i], "action"] = -1 # SELL
#                 open_flag = False
#             continue

#         # not in trade: possible entry BUY
#         if (((sig[i] >= cur_thresh) 
#             and (times[i] >= sess_start)) # buy only during trading time ...
#                  or (_last_trail > 0)): # ... unless open trade carried from previous day
#             df.at[df.index[i], "action"] = 1 # BUY
#             # seed peak with either current price or carried last trail (converted back to peak)
#             peak = max(prices[i], _last_trail / (1.0 - stop_frac))
#             _last_trail = 0.0  # consume carried trail so it's used only once
#             trail_arr[i] = peak * (1.0 - stop_frac)
#             open_flag = True

#     # attach computed trailing stop series to dataframe
#     df["trailstop_price"] = pd.Series(trail_arr, index=df.index)

#     # persist last trail: last value if day ended with open trade, else 0
#     if sellmin_idx is None:
#         _last_trail = float(df["trailstop_price"].iloc[-1]) if open_flag else 0.0

#     return df

    
# #########################################################################################################


# def fees_for_one_share(price: float, side: str,
#                        alpaca_comm_per_share: float = 0.0040,
#                        finra_taf_per_share: float = 0.000166,
#                        cat_per_share: float = 0.000009,
#                        sec_fee_per_dollar: float = 0.00013810) -> dict:
#     """
#     Compute per-share commission + regulatory fees for one executed share.

#     Returns precise per-share components and a rounded total for display.
#     """
#     assert side in ("buy", "sell")

#     alpaca_comm = float(alpaca_comm_per_share)
#     sec_raw = float(sec_fee_per_dollar) * float(price) if side == "sell" else 0.0
#     finra_raw = float(finra_taf_per_share) if side == "sell" else 0.0
#     cat_raw = float(cat_per_share)

#     finra_billed = finra_raw
#     cat_billed = cat_raw

#     regulatory_billed = sec_raw + finra_billed + cat_billed
#     total_per_share_billed = alpaca_comm + regulatory_billed

#     return {
#         "alpaca_comm": alpaca_comm,
#         "sec_raw": sec_raw,
#         "finra_billed": finra_billed,
#         "cat_billed": cat_billed,
#         "regulatory_billed": regulatory_billed,
#         "total_per_share_billed": total_per_share_billed,
#         "total_per_share_billed_rounded": round(total_per_share_billed, 8),
#     }


# ##################################

# ###############
# _last_bid = 0.0
# ###############

# def simulate_trading(
#     day,
#     df,
#     sellmin_idx: int,
#     sess_start: time,
#     shares_per_trade: int = 1
# ) -> dict:
#     """
#     Minimal simulator that only executes generator actions.
#     - Trusts generator actions in "action" (1=Buy, -1=Sell, 0=Hold).
#     - Does not compute or enforce any trailing stop or make entry/exit decisions.
#     - Executes actions as-is, records trades and snapshots, and closes any open entry at the last bar.
#     """
#     global _last_bid, _last_trail
#     updated = {}
#     _r3 = lambda x: round(float(x), 3)
#     buy_fee_per = lambda p: fees_for_one_share(price=p, side="buy")["total_per_share_billed"]
#     sell_fee_per = lambda p: fees_for_one_share(price=p, side="sell")["total_per_share_billed"]

#     # work on a sorted copy so we don't mutate caller's dataframe and iteration is stable
#     df = df.sort_index().copy()

#     # running state for the simulator
#     position, cash, buy_fee = 0, 0.0, 0.0
#     ask_p, ask_ts, open_ask, close_bid = None, None, None, None
#     pos_buf, cash_buf, net_buf, act_buf, amt_buf, trades = [], [], [], [], [], []

#     # marker: True if generator indicated we started the day with a carried open trade
#     carried_open = (_last_trail > 0)   # True if we start the day with a carried open trade

#     # determine the timestamp that marks the day's final sell (based on sellmin_idx)
#     if sellmin_idx is None: 
#         last_bid_ts = df.index[-1] 
#     elif sellmin_idx >= 0: 
#         last_bid_ts = df[df.index.time >= sess_start].index[sellmin_idx] 
#     else: # negative
#         last_bid_ts = df.index[sellmin_idx] 
#     # force a sell action from last_bid_ts onward to close any open position at day end
#     df.loc[last_bid_ts:, "action"] = -1

#     # iterate bars; execute only the actions written in "action"
#     for ts, row in df.iterrows():
#         sell_fee = 0.0
#         amt = 0.0
#         bid, ask = float(row["bid"]), float(row["ask"])
#         action, amt = "Hold", 0 # HOLD
        
#         # BUY branch: only enter if no current position
#         if row["action"] == 1 and position == 0: # BUY
#             q = shares_per_trade
#             position += q
#             action, amt = "Buy", q
#             # if there is a carried bid from previous day, treat this as continuation
#             if _last_bid > 0: # it´s the continuation of a carried open trade
#                 ask_p = _last_bid
#                 buy_fee = 0.0
#                 _last_bid = 0.0
#             else:
#                 # normal fresh buy: charge buy fee based on current ask
#                 ask_p = ask
#                 buy_fee = buy_fee_per(ask) * q
#             ask_ts = ts
#             # debit cash for the buy price plus buy fee
#             cash -= (ask_p * q) + buy_fee

#         # SELL branch: only execute if we have a position to close
#         elif row["action"] == -1 and position > 0:  # SELL
#             q = min(shares_per_trade, position)
#             position -= q
#             action, amt = "Sell", -q
#             # waive fee only for a carried open trade that is closed at/after last_bid_ts (the day's final sell)
#             if carried_open and ts >= last_bid_ts:
#                 sell_fee = 0.0
#                 carried_open = False
#             else:
#                 # normal sell: compute sell fee based on current bid
#                 sell_fee = sell_fee_per(bid) * q
#             # credit cash for the sale net of sell fee
#             cash += (bid * q) - sell_fee

#         # record round-trip when Sell closes an entry
#         if action == "Sell" and ask_p is not None:
#             bid_p = bid
#             bid_ts = ts
#             pct = round(100 * (bid_p - ask_p) / ask_p, 3)
#             # append trade tuple: ((ask_ts, bid_ts), (ask_p, bid_p), pct, buy_fee, sell_fee)
#             trades.append(((ask_ts, bid_ts), (ask_p, bid_p), pct, buy_fee, sell_fee))
#             # clear entry tracking and reset fees for next trade
#             ask_p, ask_ts = None, None
#             buy_fee = sell_fee = 0.0

#         # record entry on Buy if not already tracked
#         if action == "Buy" and ask_p is None:
#             ask_p, ask_ts = ask, ts

#         # snapshot state for this bar (position, cash, net value, action, traded amount)
#         pos_buf.append(position)
#         cash_buf.append(cash)
#         net_buf.append(cash + position * bid)
#         act_buf.append(action)
#         amt_buf.append(amt)

#     # build df_sim from recorded snapshots
#     df_sim = df.assign(Position=pos_buf, Cash=cash_buf, NetValue=net_buf, Action=act_buf, TradedAmount=amt_buf)

#     # persist last bid for potential carry into next day (only when sellmin_idx is None)
#     if sellmin_idx is None:
#         _last_bid = float(df.iloc[-1]["bid"] if _last_trail > 0 else 0) # computed at the end of the day, to be reused the next day 
    
#     # compute buy-and-hold fees explicitly for the day's open_ask and final close_bid
#     mask = (df.index.time >= sess_start) & (df.index.time <= params.sess_end)
#     open_ask = float(df.loc[mask, "ask"].iloc[0])
#     close_bid = float(df.loc[mask, "bid"].iloc[-1])
#     bh_buy_fee = buy_fee_per(open_ask) * shares_per_trade
#     bh_sell_fee = sell_fee_per(close_bid) * shares_per_trade
#     bh_line = f"Bid(0) {_r3(close_bid)} - Ask(0) {_r3(open_ask)} - Fee(0) [{_r3(bh_buy_fee)}+{_r3(bh_sell_fee)}] = {_r3((close_bid - open_ask) - (bh_buy_fee + bh_sell_fee))}"

#     # performance summary
#     trades_lines = []; 
#     realized_vals = []
#     for i, trade in enumerate(trades, 1):
#         (_, _), (ep, xp), pct, buy_fee, sell_fee = trade
#         pnl = _r3((xp - ep) - (buy_fee + sell_fee))
#         realized_vals.append(pnl)
#         trades_lines.append(f"Bid({i}) {_r3(xp)} - Ask({i}) {_r3(ep)} - Fee({i}) [{_r3(buy_fee)}+{_r3(sell_fee)}] = {pnl}")
        
#     # aggregate strategy line (sum of realized trade PnLs)
#     strategy_line = (" + ".join([f"Trade({i}) {v}" for i,v in enumerate(realized_vals,1)]) + f" = {_r3(sum(realized_vals))}") if realized_vals else "0.000"
#     perf = {"BUY&HOLD": bh_line, "TRADES": trades_lines, "STRATEGY": strategy_line}
#     updated[day] = (df_sim, trades, perf)

#     return updated


#########################################################################################################


def run_trading_pipeline(
    df: pd.DataFrame,
    min_prof_thr: float,
    max_down_prop: float,
    gain_tightening_factor: float,
    merging_retracement_thr: float,
    merging_time_gap_thr: float,
    tau_time: float,
    tau_dur: float,
    trailstop_pct: float,
    sign_thresh: float,
    smoothing_window: int,
    beta_sat: float,
    sess_start: time,
    sellmin_idx: float,
    col_signal:str = "signal",
    col_price: str = "close"
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
        df                      = df,
        min_prof_thr            = min_prof_thr,
        max_down_prop           = max_down_prop,
        gain_tightening_factor  = gain_tightening_factor,
        merging_retracement_thr = merging_retracement_thr,
        merging_time_gap_thr    = merging_time_gap_thr,
        sess_premark            = params.sess_premark,
        sess_end                = params.sess_end,
        )

    # 2) Compute raw continuous signals
    print("Computing raw continuous signals …")
    raw_signals: Dict[dt.date, Tuple[pd.DataFrame, List]] = {}
    for day, (day_df, trades) in trades_by_day.items():
        df_sig = compute_continuous_signal(
            day_df              = day_df,
            trades              = trades,
            tau_time            = tau_time,
            tau_dur             = tau_dur
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
        series                  = all_raw,
        window                  = smoothing_window,
        beta_sat                = beta_sat
        )

    # 4) assign back per day and generate actions
    signaled: Dict[dt.date, Tuple[pd.DataFrame, List]] = {}
    sim_results: Dict[dt.date, Tuple[pd.DataFrame, List, Dict[str, Any]]] = {}  
    for day, (df_sig, trades) in tqdm(raw_signals.items(), desc="Generate trade action, simulate trading …"):
        # pick the warped values matching this day's timestamps
        df_sig[col_signal] = warped.loc[df_sig.index]

        # generate trade actions on the warped [0,1] signal
        df_act = strategies.generate_trade_actions(
            df                  = df_sig,
            col_signal          = col_signal,
            sellmin_idx         = sellmin_idx,
            sign_thresh         = sign_thresh,
            trailstop_pct       = trailstop_pct,
            sess_start          = sess_start
            )
        signaled[day] = (df_act, trades)

        # 5) simulate P&L
        sim_results.update(
            strategies.simulate_trading(
                df                  = df_act,
                day                 = day,
                sess_start          = sess_start,
                sellmin_idx         = sellmin_idx,
            )
        )

    return sim_results

