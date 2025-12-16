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
_last_position = 0.0
##################


def generate_trade_actions(
    df: pd.DataFrame,
    col_signal: str,
    trailstop_pct: float,
    sellmin_idx: int,
    sess_start: time,
    sign_thresh,
    col_close: str = "close"
) -> pd.DataFrame:
    """
    Generate discrete trade actions (1=Buy, -1=Sell, 0=NotInTrade, 2=InTrade) and a trailing-stop series.
    - Works row-by-row: seeds entries, updates a running peak, computes a trailing stop
      as `trail = peak * (1 - stop_frac)`, and issues sells when signal < threshold
      and bid < trail (only during session).
    - Supports scalar or per-row `sign_thresh`. Preserves carry-in via global _last_trail.
    """
    global _last_trail

    df = df.copy()
    df["action"] = 0  # not in trade by default

    signal = df[col_signal].to_numpy(dtype=float)
    close = df[col_close].to_numpy(dtype=float)
    bid = df["bid"].to_numpy(dtype=float)
    
    sign_thresh = df[sign_thresh].to_numpy() if isinstance(sign_thresh, str) else sign_thresh # the threshold field can be string or numeric
    is_series = np.ndim(sign_thresh) != 0 # the threshold can be a constant or a series
    
    trail_arr = np.full(len(df), np.nan, dtype=float)
    stop_frac = float(trailstop_pct) / 100.0

    # iterate rows and compute actions + trailing stop
    for i in range(len(df)):
        sign_thr = float(sign_thresh[i]) if is_series else float(sign_thresh)
        prev_action = df["action"].iat[i - 1] if i > 0 else (2 if _last_trail > 0 else 0)

        # compute a single candidate peak and its trail value once per row
        peak = close[i]
        if i > 0 and prev_action in [1, 2]: 
            peak = max(peak, trail_arr[i - 1] / (1.0 - stop_frac))
        elif _last_trail > 0:
            peak = max(peak, _last_trail / (1.0 - stop_frac))
            df.at[df.index[i], "action"] = 2 # in trade
            _last_trail = 0.0  # consume carried trail

        trail_arr[i] = peak * (1.0 - stop_frac) # running trail

        # possible BUY
        if prev_action in [0,-1]:
            if (signal[i] >= sign_thr 
                 and df.index.time[i] >= sess_start):
                df.at[df.index[i], "action"] = 1 # buy action
            else: 
                df.at[df.index[i], "action"] = 0 # not in trade
                trail_arr[i] = np.nan

        # possible SELL
        if prev_action in [1,2]:
            if (signal[i] < sign_thr
                and bid[i] < trail_arr[i] 
                and df.index.time[i] >= sess_start):
                df.at[df.index[i], "action"] = -1 # sell action
            else: 
                df.at[df.index[i], "action"] = 2 # in trade

    df["trail_stop_price"] = pd.Series(trail_arr, index=df.index)
    
    if sellmin_idx is None: # persist previous day trail value if still in trade
        _last_trail = float(df["trail_stop_price"].iat[-1]) if int(df["action"].iat[-1]) in [1, 2] else 0.0

    return df


####################################


def generate_tradact_elab(
    df: pd.DataFrame,
    col_signal: str,
    sign_thresh,
    trailstop_pct: float,
    sellmin_idx: int,
    sess_start: time,
    col_close: str       = "close",
    col_atr: str         = "atr_14",
    col_rsi: str         = "rsi_6",
    col_vwap: str        = "vwap_14",
    rsi_thresh: int      = 50,
    atr_mult: float      = 1.0,
    vwap_atr_mult: float = 0.5,
) -> pd.DataFrame:
    """
    Generate discrete trade actions (1=Buy, -1=Sell, 0=NotInTrade, 2=InTrade), a trailing-stop series,
    and an ATR-based stop series.
    - Row-by-row generator: seeds entries, updates a running peak, computes a trailing
      stop as `trail = peak * (1 - stop_frac)`, and issues sells when signal < threshold
      and bid < trail (only during session).
    - Also computes `atr_stop_price = peak - atr_mult * atr` for optional ATR-based exits.
    - Supports scalar or per-row `sign_thresh`. Preserves carry-in via global _last_trail.
    """
    global _last_trail

    df = df.copy()
    df["action"] = 0  # not in trade by default

    signal = df[col_signal].to_numpy(dtype=float)
    close = df[col_close].to_numpy(dtype=float)
    atr = df[col_atr].to_numpy(dtype=float)
    rsi = df[col_rsi].to_numpy(dtype=float)
    vwap = df[col_vwap].to_numpy(dtype=float)
    bid = df["bid"].to_numpy(dtype=float)
    
    sign_thresh = df[sign_thresh].to_numpy() if isinstance(sign_thresh, str) else sign_thresh # the threshold field can be string or numeric
    is_series = np.ndim(sign_thresh) != 0 # the threshold can be a constant or a series

    trail_arr = np.full(len(df), np.nan, dtype=float)
    atr_arr = np.full(len(df), np.nan, dtype=float)
    vwap_arr = np.full(len(df), np.nan, dtype=float)
    stop_frac = float(trailstop_pct) / 100.0

    # iterate rows and compute actions + trailing stop + atr stop
    for i in range(len(df)):
        sign_thr = float(sign_thresh[i]) if is_series else float(sign_thresh)
        prev_action = df["action"].iat[i - 1] if i > 0 else (2 if _last_trail > 0 else 0)

        # compute a single candidate peak and its trail value once per row
        peak = close[i]
        if i > 0 and prev_action in [1, 2]: 
            peak = max(peak, trail_arr[i - 1] / (1.0 - stop_frac))
        elif _last_trail > 0:
            peak = max(peak, _last_trail / (1.0 - stop_frac))
            df.at[df.index[i], "action"] = 2 # in trade
            _last_trail = 0.0  # consume carried trail

        trail_arr[i] = peak * (1.0 - stop_frac) # running trail
        atr_arr[i]   = peak - atr_mult * atr[i] # running atr
        vwap_arr[i] = vwap[i] + vwap_atr_mult * atr[i] # running vwap
        
        # possible BUY                
        if prev_action in [0,-1]:
            if (signal[i] >= sign_thr 
                 and (rsi[i] > rsi_thresh 
                 or close[i] > vwap_arr[i])
                 and df.index.time[i] >= sess_start): 
                df.at[df.index[i], "action"] = 1 # buy action
            else: 
                df.at[df.index[i], "action"] = 0 # not in trade
                trail_arr[i] = atr_arr[i] = np.nan

        # possible SELL 
        if prev_action in [1,2]:
            if (signal[i] < sign_thr
                and (bid[i] < trail_arr[i]
                or bid[i] < atr_arr[i])
                and df.index.time[i] >= sess_start):
                df.at[df.index[i], "action"] = -1 # sell action
            else: 
                df.at[df.index[i], "action"] = 2 # in trade
                
    df["trail_stop_price"] = pd.Series(trail_arr, index=df.index)
    df["atr_stop_price"] = pd.Series(atr_arr, index=df.index)
    df["vwap_stop_price"] = pd.Series(vwap_arr, index=df.index)

    if sellmin_idx is None: # persist previous day trail value if still in trade
        _last_trail = float(df["trail_stop_price"].iat[-1]) if int(df["action"].iat[-1]) in [1, 2] else 0.0
      
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



# def simulate_trading(
#     day,
#     df,
#     sellmin_idx: int,
#     sess_start: time,
#     shares: int = 1 # shares per trade
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
#     position, cash, buy_fee = 0.0, 0.0, 0.0
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

#     # iterate bars; execute only the actions written in "action"
#     for ts, row in df.iterrows():
#         sell_fee = 0.0
#         amt = 0.0
#         bid, ask = float(row["bid"]), float(row["ask"])
#         action, amt = "Hold", 0 # HOLD
        
#         # BUY branch: only enter if no current position
#         if row["action"] == 1 and position == 0: # BUY
#             position += shares
#             action, amt = "Buy", shares
#             # if there is a carried bid from previous day, treat this as continuation
#             if _last_bid > 0: # it´s the continuation of a carried open trade
#                 ask_p = _last_bid
#                 buy_fee = 0.0
#                 _last_bid = 0.0
#                 ask_ts = df.index[0]
#             else:
#                 # normal fresh buy: charge buy fee based on current ask
#                 ask_p = ask
#                 buy_fee = buy_fee_per(ask) * shares
#                 ask_ts = ts
#             # debit cash for the buy price plus buy fee
#             cash -= (ask_p * shares) + buy_fee
        
#         # SELL branch: only execute if we have a position to close, or to force a 'fake' sell action for a open position at day end
#         if ((row["action"] == -1) or (carried_open and ts == last_bid_ts)) and position > 0:
#             # shares to close (never more than current position)
#             shares = min(shares, position)
#             # update position and record the executed action/amount
#             position -= shares
#             action, amt = "Sell", -shares
#             # waive fee only when closing a carried trade at/after the last_bid_ts
#             if carried_open and ts >= last_bid_ts:
#                 sell_fee = 0.0
#                 carried_open = False  # mark carried trade as closed
#             else:
#                 sell_fee = sell_fee_per(bid) * shares
#             # credit cash net of sell fee
#             cash += (bid * shares) - sell_fee

#         # record round-trip when Sell closes an entry
#         if action == "Sell" and ask_p is not None:
#             bid_p = bid
#             bid_ts = ts
#             pct = round(100 * (bid_p - ask_p) / ask_p, 3)
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
#     bh_buy_fee = buy_fee_per(open_ask) * shares
#     bh_sell_fee = sell_fee_per(close_bid) * shares
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

def simulate_trading(
    day,
    df,
    sellmin_idx: int,
    sess_start: time,
    shares: int = 1 # shares per trade
) -> dict:
    """
    Minimal simulator that only executes generator actions.
    - Trusts generator actions in "action" (1=Buy, -1=Sell, 0=Hold).
    - Does not compute or enforce any trailing stop or make entry/exit decisions.
    - Executes actions as-is, records trades and snapshots, and closes any open entry at the last bar.
    """
    global _last_bid, _last_trail, _last_position # , _last_bid_ts
    updated = {}
    _r3 = lambda x: round(float(x), 3)
    buy_fee_per = lambda p: fees_for_one_share(price=p, side="buy")["total_per_share_billed"]
    sell_fee_per = lambda p: fees_for_one_share(price=p, side="sell")["total_per_share_billed"]

    # work on a sorted copy so we don't mutate caller's dataframe and iteration is stable
    df = df.sort_index().copy()

    # running state for the simulator
    position, cash, buy_fee = float(_last_position), 0.0, 0.0
    ask_p, ask_ts, open_ask, close_bid = None, None, None, None
    pos_buf, cash_buf, net_buf, act_buf, amt_buf, trades = [], [], [], [], [], []

    # determine the timestamp that marks the day's final sell (based on sellmin_idx)
    if sellmin_idx is None: 
        last_bid_ts = df.index[-1] 
    elif sellmin_idx >= 0: 
        last_bid_ts = df[df.index.time >= sess_start].index[sellmin_idx] 
    else: # negative
        last_bid_ts = df.index[sellmin_idx] 

    # iterate bars; execute only the actions written in "action"
    for ts, row in df.iterrows():
        sell_fee = 0.0
        amt = 0.0
        bid, ask = float(row["bid"]), float(row["ask"])
        action= "Hold"

        # # BUY branch
        # if row["action"] == 1 and position == 0: # new trade
        #     position += shares
        #     action = "Buy"
        #     amt = shares
        #     ask_p = ask
        #     ask_ts = ts
        #     buy_fee = buy_fee_per(ask) * shares
        #     cash -= (ask_p * shares) + buy_fee
        # elif row["action"] == 2 and _last_bid > 0: # carried trade with open position ('fake' buy)
        #     ask_p = float(_last_bid) # last bid of previous day same trade
        #     ask_ts =  df.index[0] # it should be _last_bid_ts, but the plot must be adjusted
        #     buy_fee= 0.0 # no additional buy fees for an already open trade
        #     _last_bid = 0.0 # reset global variable

        # BUY branch
        if row["action"] == 1 and position == 0:  # new trade
            position += shares
            action, amt, ask_p, ask_ts = "Buy", shares, ask, ts
            buy_fee = buy_fee_per(ask) * shares
            cash -= (ask_p * shares) + buy_fee
        elif row["action"] == 2 and _last_bid > 0:  # carried trade with open position ('fake' buy)
            ask_p, ask_ts, buy_fee = float(_last_bid), df.index[0], 0.0  # no additional buy fee
            _last_bid = 0.0  # reset global variable
    
        # # SELL branch
        # if ((row["action"] == -1) # regular sell
        #     or (_last_trail > 0 and ts == last_bid_ts) # 'fake' sell action for a open position at day end
        # and position > 0):
        #     # shares to close (never more than current position)
        #     shares = min(shares, position)
        #     amt = -shares
        #     # waive fee only when closing a carried trade at the last_bid_ts
        #     if _last_trail > 0 and ts == last_bid_ts: # 'fake' sell
        #         sell_fee = 0.0 # the sell fee will be charged at real sell the day after
        #         action = "Hold"
        #     else: # regular sell
        #         sell_fee = sell_fee_per(bid) * shares
        #         position -= shares
        #         action = "Sell"
        #     cash += (bid * shares) - sell_fee
        #     bid_p, bid_ts = bid, ts
        #     pct = round(100 * (bid_p - ask_p) / ask_p, 3)
        #     trades.append(((ask_ts, bid_ts), (ask_p, bid_p), pct, buy_fee, sell_fee))
        #     # clear entry tracking and reset fees for next trade
        #     ask_p, ask_ts = None, None
        #     buy_fee = sell_fee = 0.0

        # SELL branch
        if position > 0 and (
            row["action"] == -1 or (_last_trail > 0 and ts == last_bid_ts)
        ):
            shares = min(shares, position)  # shares to close (max: current position)
            amt = -shares
            if _last_trail > 0 and ts == last_bid_ts:  # 'fake' sell
                sell_fee, action = 0.0, "Hold"  # fee waived, charged later on real sell
            else:  # regular sell
                sell_fee = sell_fee_per(bid) * shares
                position -= shares
                action = "Sell"
        
            cash += (bid * shares) - sell_fee
            bid_p, bid_ts = bid, ts
            pct = round(100 * (bid_p - ask_p) / ask_p, 3)
            trades.append(((ask_ts, bid_ts), (ask_p, bid_p), pct, buy_fee, sell_fee))
            ask_p, ask_ts, buy_fee, sell_fee = None, None, 0.0, 0.0  # reset for next trade

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
        # _last_bid_ts = df.index[-1]

    _last_position = float(position)

    # compute buy-and-hold fees explicitly for the day's open_ask and final close_bid
    mask = (df.index.time >= sess_start) & (df.index.time <= params.sess_end)
    open_ask = float(df.loc[mask, "ask"].iloc[0])
    close_bid = float(df.loc[mask, "bid"].iloc[-1])
    bh_buy_fee = buy_fee_per(open_ask) * shares
    bh_sell_fee = sell_fee_per(close_bid) * shares
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

