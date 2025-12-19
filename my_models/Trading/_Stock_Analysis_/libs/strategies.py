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


#########################################################################################################


def fees_for_one_share(price: float, 
                       side: str, # buy or sell
                       alpaca_comm_per_share: float = 0.0040,
                       finra_taf_per_share: float = 0.000166,
                       cat_fee_per_trade: float = 0.000009,
                       sec_fee_per_dollar: float = 0.0000229) -> dict:

    """
    Compute per-share commission + regulatory fees for one executed share.

    Returns precise per-share components and a rounded total for display.
    """
    alpaca_comm = float(alpaca_comm_per_share)
    
    # SEC fee applies only on sells, proportional to trade value
    sec_raw = sec_fee_per_dollar * price if side == "sell" else 0.0

    # FINRA TAF applies only on sells, per share, but billed per trade
    if side == "sell":
        # round UP to nearest cent, cap at $8.30
        finra_total_trade = min(math.ceil(finra_taf_per_share * 100) / 100, 8.30)
        finra_billed = finra_total_trade
    else:
        finra_billed = 0.0

    # CAT fee is per trade, not per share
    cat_billed = cat_fee_per_trade if side == "sell" else 0.0

    # Regulatory billed = SEC + FINRA + CAT
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


def _format_perf(df, trades, shares, buy_fee_per, sell_fee_per, sess_start):
    """
    Build performance lines for buy-and-hold and executed trades.

    Args:
        df: Price/quote dataframe with datetime index and 'ask'/'bid' columns.
        trades: List of trade tuples ((ask_ts, bid_ts), (ask_p, bid_p), buy_fee, sell_fee, shares_qty, position).
        shares: Requested shares per trade (used for buy-and-hold summary).
        buy_fee_per: Callable(price) -> per-share buy fee.
        sell_fee_per: Callable(price) -> per-share sell fee.
        sess_start: Session start time (datetime.time).

    Returns:
        dict with keys BUY&HOLD, TRADES, STRATEGY.
    """
    _r3 = lambda x: round(float(x), 3)

    # buy-and-hold over the session
    mask = (df.index.time >= sess_start) & (df.index.time <= params.sess_end)
    if mask.sum() == 0:
        bh_line = "BUY&HOLD: no bars in session range"
    else:
        open_ask = float(df.loc[mask, "ask"].iloc[0])
        close_bid = float(df.loc[mask, "bid"].iloc[-1])
        delta_val_bh = (close_bid - open_ask) * shares
        bh_buy_fee = buy_fee_per(open_ask) * shares
        bh_sell_fee = sell_fee_per(close_bid) * shares
        bh_pnl = _r3(delta_val_bh - (bh_buy_fee + bh_sell_fee))
        bh_line = (
            f"Shares({shares}): Bid(0) {_r3(close_bid)} - Ask(0) {_r3(open_ask)} "
            f"- Fee(0) = [{_r3(delta_val_bh)} - ({_r3(bh_buy_fee)}+{_r3(bh_sell_fee)})] = {bh_pnl}"
        )

    # trade summary
    trades_lines, realized_vals = [], []
    for i, trade in enumerate(trades, 1):
        (ask_ts, bid_ts), (ask_p, bid_p), buy_fee, sell_fee, shares_qty, position, action = trade
        delta_val = (bid_p - ask_p) * shares_qty
        pnl = _r3(delta_val - (buy_fee + sell_fee))
        # ret_pct = round(100 * (bid_p - ask_p) / ask_p, 3)
        realized_vals.append(pnl)
        bid_ts_str = bid_ts.strftime("%H:%M") if hasattr(bid_ts, "strftime") else str(bid_ts)
        ask_ts_str = ask_ts.strftime("%H:%M") if hasattr(ask_ts, "strftime") else str(ask_ts)
        trades_lines.append(
            f"Trade({i}): [Action({action}), Shares({shares_qty}), Position({position})]: "
            f"Bid({_r3(bid_p)})[{bid_ts_str}] - Ask({_r3(ask_p)})[{ask_ts_str}] - Fee = "
            f"[{_r3(delta_val)} - ({_r3(buy_fee)}+{_r3(sell_fee)})] = {pnl}"
        )

    strategy_line = (
        " + ".join([f"Trade({i}) {v}" for i, v in enumerate(realized_vals, 1)]) +
        f" = {_r3(sum(realized_vals))}"
    ) if realized_vals else "0.000"

    return {"BUY&HOLD": bh_line, "TRADES": trades_lines, "STRATEGY": strategy_line}


##################################
_last_trail = 0.0
_last_ask = 0.0
_last_position = 0.0
_last_cash = 100000.0
##################################


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


# def generate_tradact_elab(
#     df: pd.DataFrame,
#     col_signal: str,
#     sign_thresh,
#     trailstop_pct: float,
#     sellmin_idx: int,
#     sess_start: time,
#     col_close: str       = "close",
#     col_atr: str         = "atr_14",
#     col_rsi: str         = "rsi_6",
#     col_vwap: str        = "vwap_14",
#     rsi_thresh: int      = 50,
#     atr_mult: float      = 1.0,
#     vwap_atr_mult: float = 0.5,
# ) -> pd.DataFrame:
#     """
#     Generate discrete trade actions (1=Buy, -1=Sell, 0=NotInTrade, 2=InTrade), a trailing-stop series,
#     and an ATR-based stop series.
#     - Row-by-row generator: seeds entries, updates a running peak, computes a trailing
#       stop as `trail = peak * (1 - stop_frac)`, and issues sells when signal < threshold
#       and bid < trail (only during session).
#     - Also computes `atr_stop_price = peak - atr_mult * atr` for optional ATR-based exits.
#     - Supports scalar or per-row `sign_thresh`. Preserves carry-in via global _last_trail.
#     """
#     global _last_trail

#     df = df.copy()
#     df["action"] = 0  # not in trade by default

#     signal = df[col_signal].to_numpy(dtype=float)
#     close = df[col_close].to_numpy(dtype=float)
#     atr = df[col_atr].to_numpy(dtype=float)
#     rsi = df[col_rsi].to_numpy(dtype=float)
#     vwap = df[col_vwap].to_numpy(dtype=float)
#     bid = df["bid"].to_numpy(dtype=float)
    
#     sign_thresh = df[sign_thresh].to_numpy() if isinstance(sign_thresh, str) else sign_thresh # the threshold field can be string or numeric
#     is_series = np.ndim(sign_thresh) != 0 # the threshold can be a constant or a series

#     trail_arr = np.full(len(df), np.nan, dtype=float)
#     atr_arr = np.full(len(df), np.nan, dtype=float)
#     vwap_arr = np.full(len(df), np.nan, dtype=float)
#     stop_frac = float(trailstop_pct) / 100.0

#     # iterate rows and compute actions + trailing stop + atr stop
#     for i in range(len(df)):
#         sign_thr = float(sign_thresh[i]) if is_series else float(sign_thresh)
#         prev_action = df["action"].iat[i - 1] if i > 0 else (2 if _last_trail > 0 else 0)

#         # compute a single candidate peak and its trail value once per row
#         peak = close[i]
#         if i > 0 and prev_action in [1, 2]: 
#             peak = max(peak, trail_arr[i - 1] / (1.0 - stop_frac))
#         elif _last_trail > 0:
#             peak = max(peak, _last_trail / (1.0 - stop_frac))
#             df.at[df.index[i], "action"] = 2 # in trade
#             _last_trail = 0.0  # consume carried trail

#         trail_arr[i] = peak * (1.0 - stop_frac) # running trail
#         atr_arr[i]   = peak - atr_mult * atr[i] # running atr
#         vwap_arr[i] = vwap[i] + vwap_atr_mult * atr[i] # running vwap
        
#         # possible BUY                
#         if prev_action in [0,-1]:
#             if (signal[i] >= sign_thr 
#                  and (rsi[i] > rsi_thresh 
#                  or close[i] > vwap_arr[i])
#                  and df.index.time[i] >= sess_start): 
#                 df.at[df.index[i], "action"] = 1 # buy action
#             else: 
#                 df.at[df.index[i], "action"] = 0 # not in trade
#                 trail_arr[i] = atr_arr[i] = np.nan

#         # possible SELL 
#         if prev_action in [1,2]:
#             if (signal[i] < sign_thr
#                 and (bid[i] < trail_arr[i]
#                 or bid[i] < atr_arr[i])
#                 and df.index.time[i] >= sess_start):
#                 df.at[df.index[i], "action"] = -1 # sell action
#             else: 
#                 df.at[df.index[i], "action"] = 2 # in tradeo
                
#     df["trail_stop_price"] = pd.Series(trail_arr, index=df.index)
#     df["atr_stop_price"] = pd.Series(atr_arr, index=df.index)
#     df["vwap_stop_price"] = pd.Series(vwap_arr, index=df.index)

#     if sellmin_idx is None: # persist previous day trail value if still in trade
#         _last_trail = float(df["trail_stop_price"].iat[-1]) if int(df["action"].iat[-1]) in [1, 2] else 0.0
      
#     return df

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
    global _last_trail, _last_position

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
    buy_weight = np.zeros(len(df), dtype=float)
    sell_weight = np.zeros(len(df), dtype=float)
    stop_frac = float(trailstop_pct) / 100.0

    # iterate rows and compute actions + trailing stop + atr stop
    for i in range(len(df)):
        sign_thr = float(sign_thresh[i]) if is_series else float(sign_thresh)
        # prev_action = df["action"].iat[i - 1] if i > 0 else (2 if _last_trail > 0 else 0)

        # compute a single candidate peak and its trail value once per row
        peak = close[i]
        # if i > 0 and prev_action in [1, 2]: 
        if _last_position > 0: # if in trade, compare the peak with the previous value
            peak = max(peak, trail_arr[i - 1] / (1.0 - stop_frac))
        elif _last_trail > 0: # if carried trade, compare the peak with the previouos value of previous day
            peak = max(peak, _last_trail / (1.0 - stop_frac))
            # df.at[df.index[i], "action"] = 2 # in trade
            _last_trail = 0.0  # consume carried trail

        trail_arr[i] = peak * (1.0 - stop_frac) # running trail
        atr_arr[i]   = peak - atr_mult * atr[i] # running atr
        vwap_arr[i]  = vwap[i] + vwap_atr_mult * atr[i] # running vwap
        
        # # possible BUY                
        # if prev_action in [0,-1]:
        #     cond_buy = (
        #         signal[i] >= sign_thr 
        #         and (rsi[i] > rsi_thresh or close[i] > vwap_arr[i])
        #         and df.index.time[i] >= sess_start
        #     )
        #     if cond_buy:
        #         df.at[df.index[i], "action"] = 1  # buy action
        #         buy_weight[i] = (signal[i] - sign_thr) / sign_thr * 100

        #     else: 
        #         df.at[df.index[i], "action"] = 0 # not in trade
        #         trail_arr[i] = atr_arr[i] = np.nan

        # # possible SELL 
        # if prev_action in [1,2]:
        #     cond_sell = (
        #         signal[i] < sign_thr
        #         and (bid[i] < trail_arr[i] or bid[i] < atr_arr[i])
        #         and df.index.time[i] >= sess_start
        #     )
        #     if cond_sell:
        #         df.at[df.index[i], "action"] = -1  # sell action
        #         base = min(trail_arr[i], atr_arr[i])
        #         sell_weight[i] = (sign_thr - signal[i]) / sign_thr * 100
        #     else: 
        #         df.at[df.index[i], "action"] = 2 # in trade

        if df.index.time[i] >= sess_start:               
            if signal[i] >= sign_thr:   # possible BUY        
                if rsi[i] > rsi_thresh or close[i] > vwap_arr[i]:
                    df.at[df.index[i], "action"] = 1  # buy action
                    buy_weight[i] = (signal[i] - sign_thr) / sign_thr * 100
    
            else: # signal[i] < sign_thr   # possible SELL  
                if bid[i] < trail_arr[i] or bid[i] < atr_arr[i]:
                    df.at[df.index[i], "action"] = -1  # sell action
                    sell_weight[i] = (sign_thr - signal[i]) / sign_thr * 100
                
        # if not cond_sell and not cond_buy:
        #     if _last_position > 0:
        #         df.at[df.index[i], "action"] = 2 # in trade
        #     else: 
        #         df.at[df.index[i], "action"] = 0 # not in trade
        #         trail_arr[i] = atr_arr[i] = np.nan
                
                
    df["trail_stop_price"] = pd.Series(trail_arr, index=df.index)
    df["atr_stop_price"] = pd.Series(atr_arr, index=df.index)
    df["vwap_stop_price"] = pd.Series(vwap_arr, index=df.index)
    df["buy_weight"] = buy_weight
    df["sell_weight"] = sell_weight

    if sellmin_idx is None: # persist previous day trail value if still in trade
        # _last_trail = float(df["trail_stop_price"].iat[-1]) if int(df["action"].iat[-1]) in [1, 2] else 0.0
        _last_trail = float(df["trail_stop_price"].iat[-1]) if _last_position > 0 else 0.0
      
    return df
    
####################################################################################################### 


# def simulate_trading(
#     day,
#     df,
#     sellmin_idx: int,
#     sess_start: time,
#     shares: int = 1, # shares per trade
#     invest_frac: float = 0.1, # max percentage of owned cash to invest
# ) -> dict:
#     """
#     Minimal simulator that only executes generator actions.
#     - Trusts generator actions in "action" (1=Buy, -1=Sell, 0=Hold).
#     - Does not compute or enforce any trailing stop or make entry/exit decisions.
#     - Executes actions as-is, records trades and snapshots, and closes any open entry at the last bar.
#     """
#     global _last_ask, _last_trail, _last_position, _last_cash #, _last_ask_ts
#     updated = {}
#     buy_fee_per = lambda p: fees_for_one_share(price=p, side="buy")["total_per_share_billed"]
#     sell_fee_per = lambda p: fees_for_one_share(price=p, side="sell")["total_per_share_billed"]

#     # work on a sorted copy so we don't mutate caller's dataframe and iteration is stable
#     df = df.sort_index().copy()

#     # running state for the simulator
#     position, cash,  = float(_last_position), float(_last_cash)
#     ask_p, ask_ts, open_ask, close_bid = None, None, None, None
#     pos_buf, cash_buf, net_buf, act_buf, trades = [], [], [], [], []

#     # determine the timestamp that marks the day's final sell (based on sellmin_idx)
#     if sellmin_idx is None: 
#         last_bid_ts = df.index[-1] 
#     elif sellmin_idx >= 0: 
#         last_bid_ts = df[df.index.time >= sess_start].index[sellmin_idx] 
#     else: # negative
#         last_bid_ts = df.index[sellmin_idx] 

#     # iterate bars; execute only the actions written in "action"
#     for ts, row in df.iterrows():
#         bid, ask = float(row["bid"]), float(row["ask"])
#         action= "Hold"

#         # BUY branch
#         is_fake_buy  = row["action"] == 2 and _last_ask > 0 # carried trade with open position ('fake' buy)
#         if row["action"] == 1 and position == 0:  # new trade
#             max_affordable = int((cash * invest_frac) // (ask + buy_fee_per(ask)))
#             shares_buy = min(shares, max_affordable)
#             if shares_buy <= 0:
#                 action = "Hold"
#                 continue
#             position += shares_buy
#             action = "Buy"
#             ask_p = ask
#             ask_ts = ts
#             buy_fee = buy_fee_per(ask) * shares_buy
#             cash -= (ask_p * shares_buy) + buy_fee
#         elif is_fake_buy:  
#             ask_p = float(_last_ask)
#             ask_ts = df.index[0] # df.index[0] is a placeholder; if needed the true prior-day timestamp, persist and use _last_ask_ts instead
#             buy_fee = 0.0  # no additional buy fee
#             _last_ask = 0.0  # reset global variable

#         # SELL branch
#         is_fake_sell = _last_trail > 0 and ts == last_bid_ts # trade still open to carry to the next day ('fake' sell)
#         if position > 0 and (row["action"] == -1 or is_fake_sell):
#             shares_qty  = min(shares, position)  # shares to close (max: current position)
#             if is_fake_sell:  
#                 sell_fee = 0.0 # fee waived, charged later on real sell
#                 action = "Hold"  
#                 bid_p = ask_p # to simulate zero return for the trade not closed yet
#             else:  # regular sell
#                 sell_fee = sell_fee_per(bid) * shares_qty
#                 position -= shares_qty
#                 action = "Sell"
#                 bid_p = bid
#                 cash += (bid * shares_qty) - sell_fee
#             bid_ts = ts
#             pct = round(100 * (bid_p - ask_p) / ask_p, 3)
#             trades.append(((ask_ts, bid_ts), (ask_p, bid_p), pct, buy_fee, sell_fee, shares_qty))

#         # snapshot state for this bar (position, cash, net value, action, traded amount)
#         pos_buf.append(position)
#         cash_buf.append(cash)
#         net_buf.append(cash + position * bid)
#         act_buf.append(action)

#     # build df_sim from recorded snapshots
#     df_sim = df.assign(Position=pos_buf, Cash=cash_buf, NetValue=net_buf, Action=act_buf)
    
#     _last_position, _last_cash = float(position), float(cash)
#     # persist last bid for potential carry into next day (only when sellmin_idx is None)
#     if sellmin_idx is None and position > 0:
#         _last_ask = float(float(ask_p) if _last_trail > 0 else 0) # computed at the end of the day, to be reused the next day 
#         # _last_ask_ts = df.index[-1]

#     perf = _format_perf(
#         df=df,
#         trades=trades,
#         shares=shares,
#         buy_fee_per=buy_fee_per,
#         sell_fee_per=sell_fee_per,
#         sess_start=sess_start,
#     )
#     updated[day] = (df_sim, trades, perf)

#     return updated

def simulate_trading(
    day,
    df,
    sellmin_idx: int,
    sess_start: time,
    shares: int = 1, # shares per trade
    invest_frac: float = 0.1, # max percentage of owned cash to invest
) -> dict:
    """
    Minimal simulator that only executes generator actions.
    - Trusts generator actions in "action" (1=Buy, -1=Sell, 0=Hold).
    - Does not compute or enforce any trailing stop or make entry/exit decisions.
    - Executes actions as-is, records trades and snapshots, and closes any open entry at the last bar.
    """
    global _last_ask, _last_trail, _last_position, _last_cash, _past_bid, _past_ask #, _last_ask_ts
    updated = {}
    buy_fee_per = lambda p: fees_for_one_share(price=p, side="buy")["total_per_share_billed"]
    sell_fee_per = lambda p: fees_for_one_share(price=p, side="sell")["total_per_share_billed"]

    # work on a sorted copy so we don't mutate caller's dataframe and iteration is stable
    df = df.sort_index().copy()

    # running state for the simulator
    position, cash,  = float(_last_position), float(_last_cash)
    ask_p, bid_p, ask_ts, bid_ts, open_ask, close_bid = None, None, None, None, None, None
    pos_buf, cash_buf, net_buf, act_buf, shar_buf, trades = [], [], [], [], [], []

    # determine the timestamp that marks the day's final sell (based on sellmin_idx)
    if sellmin_idx is None: 
        last_bid_ts = df.index[-1] 
    elif sellmin_idx >= 0: 
        last_bid_ts = df[df.index.time >= sess_start].index[sellmin_idx] 
    else: # negative
        last_bid_ts = df.index[sellmin_idx] 

    # iterate bars; execute only the actions written in "action"
    for ts, row in df.iterrows():
        bid, ask = float(row["bid"]), float(row["ask"])
        action= "Hold"
        shares_qty = 0
        buy_fee = sell_fee = 0.0

        # BUY branch
        is_fake_buy = position > 0 and _last_ask > 0  and _last_trail > 0 # 
        # if row["action"] == 1 and position == 0:  # new trade
        if row["action"] == 1:  # buy
            # print('row["buy_weight"]',row["buy_weight"], ts) #################
            max_affordable = int((cash * invest_frac) // (ask + buy_fee_per(ask)))
            shares_qty = max(1, int(max_affordable * row["buy_weight"]))
            if shares_qty <= 0:
                action = "Hold"  # no buy; fall through to the bottom append
            else:
                position += shares_qty
                action = "Buy"
                ask_p = ask
                ask_ts = ts
                buy_fee = buy_fee_per(ask) * shares_qty
                cash -= (ask_p * shares_qty) + buy_fee
                _last_ask = ask_p if _last_ask == 0.0  and sellmin_idx is None else _last_ask # and _last_trail > 0
        elif is_fake_buy: # carried trade with open position ('fake' buy):  
            ask_p = float(_last_ask)
            ask_ts = df.index[0] # df.index[0] is a placeholder; if needed the true prior-day timestamp, persist and use _last_ask_ts instead
            # buy_fee = 0.0  # no additional buy fee
            _last_ask = 0.0  # reset global variable

        # SELL branch
        is_fake_sell = (ts == last_bid_ts) and _last_ask > 0 # trade still open to carry to the next day ('fake' sell) #### 
        if position > 0 and (row["action"] == -1 or is_fake_sell):
            # print('row["sell_weight"]',row["sell_weight"], ts) #################
            shares_qty = max(1, int(position * row["sell_weight"]))
            if is_fake_sell:  
                # sell_fee = 0.0 # fee waived, charged later on real sell
                action = "Carried"  #####
                bid_p = ask_p = _last_ask # to simulate zero return for the carried trade not closed yet
            else:  # regular sell
                sell_fee = sell_fee_per(bid) * shares_qty
                # position -= shares_qty
                position = max(0, position - shares_qty)
                action = "Sell"
                bid_p = bid
                cash += (bid * shares_qty) - sell_fee
                if position == 0:
                    _last_ask = 0.0
            bid_ts = ts

        if ask_ts and bid_ts and shares_qty > 0 and action!="Hold": # 
            trades.append(((ask_ts, bid_ts), (ask_p, bid_p), buy_fee, sell_fee, shares_qty, position, action))
            if action=="Sell":
                ask_p = bid_p
                ask_ts = bid_ts
                # _last_ask = 0
            else: # action=="Buy"
                bid_p = ask_p
                bid_ts = ask_ts
                # _last_ask = ask_p if _last_ask == 0 else _last_ask
            

        # snapshot state for this bar (position, cash, net value, action, traded amount)
        pos_buf.append(position)
        shar_buf.append(shares_qty)
        cash_buf.append(cash)
        net_buf.append(cash + position * bid)
        act_buf.append(action)

    # build df_sim from recorded snapshots
    df_sim = df.assign(Position=pos_buf, Cash=cash_buf, NetValue=net_buf, Action=act_buf, Shares=shar_buf)
    
    _last_position, _last_cash = float(position), float(cash)
    # persist last bid for potential carry into next day (only when sellmin_idx is None)
    # if sellmin_idx is None and position > 0:
    #     _last_ask = float(float(ask_p) if _last_trail > 0 else 0) # computed at the end of the day, to be reused the next day 
    #     # _last_ask_ts = df.index[-1]

    perf = _format_perf(
        df=df,
        trades=trades,
        shares=shares,
        buy_fee_per=buy_fee_per,
        sell_fee_per=sell_fee_per,
        sess_start=sess_start,
    )
    updated[day] = (df_sim, trades, perf)

    return updated