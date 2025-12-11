from libs import params, plots

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


##################
_last_trail = 0.0
_last_bid = 0.0
##################

def generate_trade_actions(
    df: pd.DataFrame,
    col_signal: str,
    trailstop_pct: float,
    sellmin_idx: int,
    sess_start: time,
    sign_thresh,
    col_price: str = "close"
) -> pd.DataFrame:
    """
    Generate discrete trade actions (Buy/Sell/Hold) and a trailing-stop price series from signals and price data.
    """
    global _last_trail

    # work on a copy to avoid mutating caller's dataframe
    df = df.copy()
    df["action"] = 0 # HOLD

    # extract arrays for faster indexed access inside the loop
    col_signal = df[col_signal].to_numpy()
    sign_thresh = df[sign_thresh].to_numpy() if isinstance(sign_thresh, str) else sign_thresh
    times = df.index.time
    prices = df[col_price].to_numpy(dtype=float)
    stop_frac = float(trailstop_pct) / 100.0  # convert percent to fraction

    # prepare an array to hold computed trailing stop prices (NaN when no open trade)
    trail_arr = np.full(len(df), np.nan, dtype=float)
    # open_flag indicates whether we started the day with a carried open trade
    open_flag = (_last_trail > 0)

    # allow scalar or per-row threshold
    cur_thr = None if np.ndim(sign_thresh) == 0 else np.asarray(sign_thresh)

    # iterate over rows by index to compute actions and trailing stops
    for i in range(len(df)):
        cur_thresh = float(cur_thr[i]) if cur_thr is not None else float(sign_thresh)
        
        # if already in-trade (previous trail exists), compute peak from previous trail
        if i > 0 and np.isfinite(trail_arr[i - 1]) and df["action"].iat[i - 1] != -1:
            # compute the running peak using previous trail value and current price
            peak = max(prices[i], trail_arr[i - 1] / (1.0 - stop_frac))
            trail_val = peak * (1.0 - stop_frac)
            trail_arr[i] = trail_val
            
            # check exit SELL condition:  
            if ((col_signal[i] < cur_thresh) # signal dropped below threshold
                and (df["bid"].iat[i] < trail_val) #  bid price below running trail stop price
                and (times[i] >= sess_start)): # sell only during trading time
                df.at[df.index[i], "action"] = -1 # SELL
                open_flag = False
            continue

        # not in trade: possible entry BUY
        if (((col_signal[i] >= cur_thresh) 
            and (times[i] >= sess_start)) # buy only during trading time ...
                 or (_last_trail > 0)): # ... unless open trade carried from previous day
            df.at[df.index[i], "action"] = 1 # BUY
            # seed peak with either current price or carried last trail (converted back to peak)
            peak = max(prices[i], _last_trail / (1.0 - stop_frac))
            _last_trail = 0.0  # consume carried trail so it's used only once
            trail_arr[i] = peak * (1.0 - stop_frac)
            open_flag = True

    # attach computed trailing stop series to dataframe
    df["trailstop_price"] = pd.Series(trail_arr, index=df.index)

    # persist last trail: last value if day ended with open trade, else 0
    if sellmin_idx is None:
        _last_trail = float(df["trailstop_price"].iloc[-1]) if open_flag else 0.0

    return df


####################################

def generate_tradact_inds(
    df: pd.DataFrame,
    col_signal: str,
    sign_thresh: float,
    trailstop_pct: float,
    sellmin_idx: int,
    sess_start: time,
    col_price: str = "close",
) -> pd.DataFrame:
    """
    Generate discrete trade actions (Buy/Sell/Hold) and a trailing-stop price series from signals and price data.
    """
    global _last_trail

    # work on a copy to avoid mutating caller's dataframe
    df = df.copy()
    df["action"] = 0 # HOLD

    # extract arrays for faster indexed access inside the loop
    col_signal = df[col_signal].to_numpy()
    sign_thresh = df[sign_thresh].to_numpy() if isinstance(sign_thresh, str) else sign_thresh
    times = df.index.time
    prices = df[col_price].to_numpy(dtype=float)
    stop_frac = float(trailstop_pct) / 100.0  # convert percent to fraction

    # prepare an array to hold computed trailing stop prices (NaN when no open trade)
    trail_arr = np.full(len(df), np.nan, dtype=float)
    # open_flag indicates whether we started the day with a carried open trade
    open_flag = (_last_trail > 0)

    # allow scalar or per-row threshold
    cur_thr = None if np.ndim(sign_thresh) == 0 else np.asarray(sign_thresh)

    # iterate over rows by index to compute actions and trailing stops
    for i in range(len(df)):
        cur_thresh = float(cur_thr[i]) if cur_thr is not None else float(sign_thresh)
        
        # if already in-trade (previous trail exists), compute peak from previous trail
        if i > 0 and np.isfinite(trail_arr[i - 1]) and df["action"].iat[i - 1] != -1:
            # compute the running peak using previous trail value and current price
            peak = max(prices[i], trail_arr[i - 1] / (1.0 - stop_frac))
            trail_val = peak * (1.0 - stop_frac)
            trail_arr[i] = trail_val
            
            # check exit SELL condition:  
            if ((col_signal[i] < cur_thresh) # signal dropped below threshold
                and (df["bid"].iat[i] < trail_val) #  bid price below running trail stop price
                and (times[i] >= sess_start)): # sell only during trading time
                df.at[df.index[i], "action"] = -1 # SELL
                open_flag = False
            continue

        # not in trade: possible entry BUY
        if (((col_signal[i] >= cur_thresh) 
            and (times[i] >= sess_start)) # buy only during trading time ...
                 or (_last_trail > 0)): # ... unless open trade carried from previous day
            df.at[df.index[i], "action"] = 1 # BUY
            # seed peak with either current price or carried last trail (converted back to peak)
            peak = max(prices[i], _last_trail / (1.0 - stop_frac))
            _last_trail = 0.0  # consume carried trail so it's used only once
            trail_arr[i] = peak * (1.0 - stop_frac)
            open_flag = True

    # attach computed trailing stop series to dataframe
    df["trailstop_price"] = pd.Series(trail_arr, index=df.index)

    # persist last trail: last value if day ended with open trade, else 0
    if sellmin_idx is None:
        _last_trail = float(df["trailstop_price"].iloc[-1]) if open_flag else 0.0

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


def simulate_trading(
    day,
    df,
    sellmin_idx: int,
    sess_start: time,
    shares_per_trade: int = 1
) -> dict:
    """
    Minimal simulator that only executes generator actions.
    - Trusts generator actions in "action" (1=Buy, -1=Sell, 0=Hold).
    - Does not compute or enforce any trailing stop or make entry/exit decisions.
    - Executes actions as-is, records trades and snapshots, and closes any open entry at the last bar.
    """
    global _last_bid, _last_trail
    updated = {}
    _r3 = lambda x: round(float(x), 3)
    buy_fee_per = lambda p: fees_for_one_share(price=p, side="buy")["total_per_share_billed"]
    sell_fee_per = lambda p: fees_for_one_share(price=p, side="sell")["total_per_share_billed"]

    # work on a sorted copy so we don't mutate caller's dataframe and iteration is stable
    df = df.sort_index().copy()

    # running state for the simulator
    position, cash, buy_fee = 0, 0.0, 0.0
    ask_p, ask_ts, open_ask, close_bid = None, None, None, None
    pos_buf, cash_buf, net_buf, act_buf, amt_buf, trades = [], [], [], [], [], []

    # marker: True if generator indicated we started the day with a carried open trade
    carried_open = (_last_trail > 0)   # True if we start the day with a carried open trade

    # determine the timestamp that marks the day's final sell (based on sellmin_idx)
    if sellmin_idx is None: 
        last_bid_ts = df.index[-1] 
    elif sellmin_idx >= 0: 
        last_bid_ts = df[df.index.time >= sess_start].index[sellmin_idx] 
    else: # negative
        last_bid_ts = df.index[sellmin_idx] 
    # force a sell action from last_bid_ts onward to close any open position at day end
    df.loc[last_bid_ts:, "action"] = -1

    # iterate bars; execute only the actions written in "action"
    for ts, row in df.iterrows():
        sell_fee = 0.0
        amt = 0.0
        bid, ask = float(row["bid"]), float(row["ask"])
        action, amt = "Hold", 0 # HOLD
        
        # BUY branch: only enter if no current position
        if row["action"] == 1 and position == 0: # BUY
            q = shares_per_trade
            position += q
            action, amt = "Buy", q
            # if there is a carried bid from previous day, treat this as continuation
            if _last_bid > 0: # it´s the continuation of a carried open trade
                ask_p = _last_bid
                buy_fee = 0.0
                _last_bid = 0.0
            else:
                # normal fresh buy: charge buy fee based on current ask
                ask_p = ask
                buy_fee = buy_fee_per(ask) * q
            ask_ts = ts
            # debit cash for the buy price plus buy fee
            cash -= (ask_p * q) + buy_fee

        # SELL branch: only execute if we have a position to close
        elif row["action"] == -1 and position > 0:  # SELL
            q = min(shares_per_trade, position)
            position -= q
            action, amt = "Sell", -q
            # waive fee only for a carried open trade that is closed at/after last_bid_ts (the day's final sell)
            if carried_open and ts >= last_bid_ts:
                sell_fee = 0.0
                carried_open = False
            else:
                # normal sell: compute sell fee based on current bid
                sell_fee = sell_fee_per(bid) * q
            # credit cash for the sale net of sell fee
            cash += (bid * q) - sell_fee

        # record round-trip when Sell closes an entry
        if action == "Sell" and ask_p is not None:
            bid_p = bid
            bid_ts = ts
            pct = round(100 * (bid_p - ask_p) / ask_p, 3)
            # append trade tuple: ((ask_ts, bid_ts), (ask_p, bid_p), pct, buy_fee, sell_fee)
            trades.append(((ask_ts, bid_ts), (ask_p, bid_p), pct, buy_fee, sell_fee))
            # clear entry tracking and reset fees for next trade
            ask_p, ask_ts = None, None
            buy_fee = sell_fee = 0.0

        # record entry on Buy if not already tracked
        if action == "Buy" and ask_p is None:
            ask_p, ask_ts = ask, ts

        # snapshot state for this bar (position, cash, net value, action, traded amount)
        pos_buf.append(position)
        cash_buf.append(cash)
        net_buf.append(cash + position * bid)
        act_buf.append(action)
        amt_buf.append(amt)

    # build df_sim from recorded snapshots
    df_sim = df.assign(Position=pos_buf, Cash=cash_buf, NetValue=net_buf, Action=act_buf, TradedAmount=amt_buf)

    # persist last bid for potential carry into next day (only when sellmin_idx is None)
    if sellmin_idx is None:
        _last_bid = float(df.iloc[-1]["bid"] if _last_trail > 0 else 0) # computed at the end of the day, to be reused the next day 
    
    # compute buy-and-hold fees explicitly for the day's open_ask and final close_bid
    mask = (df.index.time >= sess_start) & (df.index.time <= params.sess_end)
    open_ask = float(df.loc[mask, "ask"].iloc[0])
    close_bid = float(df.loc[mask, "bid"].iloc[-1])
    bh_buy_fee = buy_fee_per(open_ask) * shares_per_trade
    bh_sell_fee = sell_fee_per(close_bid) * shares_per_trade
    bh_line = f"Bid(0) {_r3(close_bid)} - Ask(0) {_r3(open_ask)} - Fee(0) [{_r3(bh_buy_fee)}+{_r3(bh_sell_fee)}] = {_r3((close_bid - open_ask) - (bh_buy_fee + bh_sell_fee))}"

    # performance summary
    trades_lines = []; 
    realized_vals = []
    for i, trade in enumerate(trades, 1):
        (_, _), (ep, xp), pct, buy_fee, sell_fee = trade
        pnl = _r3((xp - ep) - (buy_fee + sell_fee))
        realized_vals.append(pnl)
        trades_lines.append(f"Bid({i}) {_r3(xp)} - Ask({i}) {_r3(ep)} - Fee({i}) [{_r3(buy_fee)}+{_r3(sell_fee)}] = {pnl}")
        
    # aggregate strategy line (sum of realized trade PnLs)
    strategy_line = (" + ".join([f"Trade({i}) {v}" for i,v in enumerate(realized_vals,1)]) + f" = {_r3(sum(realized_vals))}") if realized_vals else "0.000"
    perf = {"BUY&HOLD": bh_line, "TRADES": trades_lines, "STRATEGY": strategy_line}
    updated[day] = (df_sim, trades, perf)

    return updated