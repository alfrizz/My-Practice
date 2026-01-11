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


def detect_and_adjust_splits(df, forward_threshold=0.5, reverse_threshold=2, tol=0.05, vol_fact=1, dst_transition_dates=None):
    """
    Detects forward and reverse splits and adjusts price columns and volume.
    Minimal addition: accepts dst_transition_dates (iterable of dates) and skips
    candidate split events that fall on those dates to avoid DST artifacts.
    """
    df_adj = df.copy()
    split_events = []
    price_columns = ['open', 'high', 'low', 'close', 'ask', 'bid']
    dst_transition_dates = set(dst_transition_dates or [])
    print(f"Executing [detect_and_adjust_splits]...")

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
                    # skip if event falls on a DST transition date
                    if event_time.date() in dst_transition_dates:
                        print(f"Skipping forward-split candidate at {event_time} due to DST transition")
                        continue
                    print(f"Detected forward split on {event_time} with factor {candidate_factor} (ratio: {ratio:.4f})")
                    split_events.append((event_time, candidate_factor, 'forward'))
                    df_adj.loc[:df_adj.index[i-1], price_columns] /= candidate_factor
                    df_adj.loc[:df_adj.index[i-1], 'volume'] *= candidate_factor * vol_fact

        # Detect reverse split (price jump)
        elif ratio > reverse_threshold:
            candidate_factor = round(ratio)
            if candidate_factor >= 2:
                expected_ratio = candidate_factor
                if abs(ratio - expected_ratio) < tol:
                    event_time = df_adj.index[i]
                    # skip if event falls on a DST transition date
                    if event_time.date() in dst_transition_dates:
                        print(f"Skipping reverse-split candidate at {event_time} due to DST transition")
                        continue
                    print(f"Detected reverse split on {event_time} with factor {candidate_factor} (ratio: {ratio:.4f})")
                    split_events.append((event_time, candidate_factor, 'reverse'))
                    df_adj.loc[:df_adj.index[i-1], price_columns] *= candidate_factor
                    df_adj.loc[:df_adj.index[i-1], 'volume'] /= candidate_factor

    return df_adj, split_events


##################


def process_splits(ticker):
    """Load intraday CSV, localize timestamps to US/Eastern, 
    detect & adjust corporate splits, and return the adjusted DataFrame."""
    print(f"[process_splits] Loading original alpaca csv")
    df = pd.read_csv(params.alpaca_csv, index_col=0, parse_dates=True)

    df.index = pd.to_datetime(df.index)
    
    # Create ask/bid from close using configured spread
    df['ask'] = round(df['close'] * (1 + params.bidask_spread_pct/100), 4)
    df['bid'] = round(df['close'] * (1 - params.bidask_spread_pct/100), 4)

    print("Plotting original data...")
    plots.plot_close_volume(df, title="Before Adjusting Splits: Close Price and Volume")

    # Minimal: if you don't want to compute DST transition dates here, pass an empty set
    transitions = set()
    df_adjusted, split_events = detect_and_adjust_splits(df=df, dst_transition_dates=transitions)

    if split_events:
        print("Splits detected. Plotting adjusted data...")
        plots.plot_close_volume(df_adjusted, title="After Adjusting Splits: Close Price and Volume")
    else:
        print("No splits detected.")

    return df_adjusted


##########################################################################################################


def prepare_interpolate_data(df: pd.DataFrame, tz_str: str = "US/Eastern") -> pd.DataFrame:
    """
    Normalize intraday timestamps to a consistent timezone and fill missing 1-minute bars.

    - Interprets df.index as timezone-aware and converts it to tz_str.
    - Drops exact duplicate rows once (prints count and timestamps).
    - Builds a per-day tz-aware 1-minute index and linearly interpolates missing bars.
    - Returns a DataFrame indexed by tz-aware 1-minute timestamps.
    """
    print(['executing prepare_interpolate_data'])
    df = df.copy()
    df[df.columns] = df[df.columns].astype(np.float64)

    # normalize: convert index to datetimes (keep naive index for storage)
    print('normalize data...')
    idx = pd.to_datetime(df.index)
    # subtract 1h for standard-time (winter) timestamps, leave DST unchanged
    _e = idx.tz_localize(tz_str, ambiguous="infer", nonexistent="shift_forward") if idx.tz is None else idx.tz_convert(tz_str)
    df.index = idx - pd.to_timedelta((_e.map(lambda t: t.dst()) == pd.Timedelta(0)).astype(int), unit="h")

    # single, early duplicate drop with clear audit print
    dup_mask = df.index.duplicated(keep=False)
    if dup_mask.any():
        dup_times = df.index[dup_mask].unique()
        print(
            "prepare_interpolate_data: dropping",
            int(dup_mask.sum()),
            "exact duplicate rows at:",
            ", ".join(t.strftime("%Y-%m-%d %H:%M:%S") for t in dup_times),
        )
        df = df[~df.index.duplicated(keep="first")]

    df.sort_index(inplace=True)

    # per-day 1-min grid and linear interpolation where needed (naive index)
    print('interpolate data...')
    filled = []
    for day, grp in df.groupby(df.index.normalize()):
        idx_start = grp.index.min()
        idx_end = grp.index.max()
        full_idx = pd.date_range(start=idx_start, end=idx_end, freq="1min")
        day_filled = grp.reindex(full_idx).interpolate(method="linear", limit_direction="both")
        filled.append(day_filled)

    df_out = pd.concat(filled).sort_index()

    # drop flat days if desired (keeps original behavior)
    df_out = df_out.groupby(df_out.index.normalize()).filter(lambda g: g['close'].nunique() > 1)
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

