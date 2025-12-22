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
_sell_intr_posamnt = params.init_cash   # intraday pot start
_last_pnl          = params.init_cash   # strategy prior PnL baseline
_last_cash         = params.init_cash   # cash for trading loop
_last_position     = 0
_last_trail        = 0.0
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


def _format_perf(
    df,
    trades,
    shares,
    # buy_fee_per,
    # sell_fee_per,
    sess_start,
):
    """
    Build performance summary (intraday buy/sell, trades, strategy deltas).
    """
    global _sell_intr_posamnt, _last_pnl

    _round = lambda x: round(float(x), 3)
    mask = (df.index.time >= sess_start) & (df.index.time <= params.sess_end)
    
    # minimal intraday: buy all possible at open, sell all at close, persist proceeds
    ask_buy  = float(df.loc[mask, "ask"].iloc[0])
    bid_sell = float(df.loc[mask, "bid"].iloc[-1])
    # fee_buy  = buy_fee_per(ask_buy)
    # fee_sell = sell_fee_per(bid_sell)
    fee_buy = fees_for_one_share(price=ask_buy, side="buy")["total_per_share_billed"] 
    fee_sell = fees_for_one_share(price=bid_sell, side="sell")["total_per_share_billed"]
    
    shares     = int(_sell_intr_posamnt // (ask_buy + fee_buy))
    amt_buy    = shares * (ask_buy + fee_buy)
    amt_sell   = shares * (bid_sell - fee_sell)
    intr_pnl   = amt_sell - amt_buy
    
    _sell_intr_posamnt = amt_sell
    
    intraday_line = (
        f"shares({shares}); "
        f"[buy@{_round(ask_buy)}, fee_buy={_round(fee_buy)}, pos_amnt_buy {_round(amt_buy)}]; "
        f"[sell@{_round(bid_sell)}, fee_sell={_round(fee_sell)}, pos_amnt_sell {_round(amt_sell)}]; "
        f"PNL={_round(intr_pnl)}"
    )

    trades_lines, delta_totals = [], []
    trade_ids = pd.Series(np.nan, index=df.index)
    tr_idx = 0  # counts only buy/sell trades

    for trade in trades:
        (
            ts,
            (ask, bid),
            (buy_fee, sell_fee),
            (buy_cost, sell_cost),
            action, shares_qty, position, pos_amount, cash, tot_pnl
        ) = trade

        if action not in ("Sell", "Buy"):
            continue  # skip and don't increment

        tr_idx += 1
        trade_ids.loc[df.index == ts] = tr_idx

        if action == "Buy":
            cost = -buy_cost
            fee = -buy_fee
        else:  # Sell
            cost = sell_cost
            fee = -sell_fee
        cash_tr = cost + fee
                
        trades_lines.append(
            f"Tr[{tr_idx}]: Time({ts.strftime("%H:%M")})Ask({_round(ask)})Bid({_round(bid)})Act({action})"
            f"Shrs({shares_qty})Pos({position}); "
            f"CashTr({_round(cash_tr)})=Cost({_round(cost)})-Fee({_round(fee)}); "
            f"P&L({_round(tot_pnl)})=CashTot({_round(cash)})+PosAmt({_round(pos_amount)})"
        )

        delta = tot_pnl - _last_pnl
        delta_totals.append((tr_idx, _round(delta)))
        _last_pnl = tot_pnl

    strategy_line = (
        " + ".join([f"Δ[{idx}]:{val}" for idx, val in delta_totals]) +
        f" PNL={_round(sum(val for _, val in delta_totals))}"
    ) 
    
    df = df.assign(TradeID=trade_ids)
    perf = {"INTRADAY": intraday_line, "TRADES": trades_lines, "STRATEGY": strategy_line}
    return df, perf

    
#################################################################


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

        # compute a single candidate peak and its trail value once per row
        peak = max(close[i], trail_arr[i - 1] / (1.0 - stop_frac))
        trail_arr[i] = peak * (1.0 - stop_frac) # running trail
        atr_arr[i]   = peak - atr_mult * atr[i] # running atr
        vwap_arr[i]  = vwap[i] + vwap_atr_mult * atr[i] # running vwap

        if df.index.time[i] >= sess_start:               
            if signal[i] >= sign_thr:   # possible BUY        
                if rsi[i] > rsi_thresh or close[i] > vwap_arr[i]:
                    df.at[df.index[i], "action"] = 1  # buy action
                    buy_weight[i] = (signal[i] - sign_thr) / sign_thr * 100
    
            else: # signal[i] < sign_thr   # possible SELL  
                if bid[i] < trail_arr[i] or bid[i] < atr_arr[i]:
                    df.at[df.index[i], "action"] = -1  # sell action
                    sell_weight[i] = (sign_thr - signal[i]) / sign_thr * 100
                
    df["trail_stop_price"] = pd.Series(trail_arr, index=df.index)
    df["atr_stop_price"] = pd.Series(atr_arr, index=df.index)
    df["vwap_stop_price"] = pd.Series(vwap_arr, index=df.index)
    df["buy_weight"] = buy_weight
    df["sell_weight"] = sell_weight
      
    return df

    
####################################################################################################### 


def simulate_trading(
    day,
    df,
    sellmin_idx: int,
    sess_start: time,
    shares: int = 1,
    invest_frac: float = 0.1,
) -> dict:
    """
    Simulate intraday trading using provided actions; carry prior day state.
    - Uses globals for carried position/cash and last bid/ask markers.
    """
    global _last_position, _last_cash

    df = df.sort_index().copy()
    updated = {}
    # buy_fee_per = lambda p: fees_for_one_share(price=p, side="buy")["total_per_share_billed"]
    # sell_fee_per = lambda p: fees_for_one_share(price=p, side="sell")["total_per_share_billed"]
    
    position = int(_last_position)
    cash = _last_cash
    buy_cost = sell_cost = 0.0
    pos_buf, posamt_buf, cash_buf, pnl_buf, act_buf, shar_buf = [], [], [], [], [], []
    trades = []

    # Determine last bid timestamp based on sellmin_idx
    if sellmin_idx is None:
        last_bid_ts = df.index[-1]
    else:
        last_bid_ts = df.index[sellmin_idx]

    # Iterate bars; execute only given actions
    for ts, row in df.iterrows():
        bid = row["bid"]
        ask = row["ask"]
        action = "Hold"
        shares_qty = 0
        buy_fee = sell_fee = buy_cost = sell_cost = 0.0

        # BUY branch
        if row["action"] == 1:  # buy
            action = "Buy"
            # shares_max = int((cash * invest_frac) // (ask + buy_fee_per(ask)))
            per_share_buy_fee = fees_for_one_share(price=ask, side="buy")["total_per_share_billed"] 
            shares_max = int((cash * invest_frac) // (ask + per_share_buy_fee))
            shares_qty = max(1, int(shares_max * row["buy_weight"]))
            position += shares_qty
            buy_cost = ask * shares_qty
            # buy_fee = buy_fee_per(ask) * shares_qty
            buy_fee = per_share_buy_fee * shares_qty
            cash -= buy_cost + buy_fee

        # SELL branch
        if position > 0 and row["action"] == -1:
            action = "Sell"
            per_share_sell_fee = fees_for_one_share(price=bid, side="sell")["total_per_share_billed"]
            shares_qty = max(1, int(position * row["sell_weight"]))
            position = max(0, position - shares_qty)
            sell_cost = bid * shares_qty
            # sell_fee = sell_fee_per(bid) * shares_qty
            sell_fee = per_share_sell_fee * shares_qty
            cash += sell_cost - sell_fee

        pos_amount = position * bid
        tot_pnl = cash + pos_amount

        trades.append((
            ts,
            (ask, bid),
            (buy_fee, sell_fee),
            (buy_cost, sell_cost),
            action,
            shares_qty,
            position,
            pos_amount,
            cash,
            tot_pnl
        ))

        # snapshots: MUST run once per row
        pos_buf.append(position)
        posamt_buf.append(pos_amount)
        shar_buf.append(shares_qty)
        cash_buf.append(cash)
        pnl_buf.append(tot_pnl)
        act_buf.append(action)

    df_sim = df.assign(Position=pos_buf, Posamt=posamt_buf, Cash=cash_buf, Pnl=pnl_buf, Action=act_buf, Shares=shar_buf)
    
    df_sim, perf = _format_perf(
        df=df_sim,
        trades=trades,
        shares=shares,
        # buy_fee_per=buy_fee_per,
        # sell_fee_per=sell_fee_per,
        sess_start=sess_start,
    )

    # Persist end-of-day state
    _last_position = position
    _last_cash = cash
    
    updated[day] = (df_sim, trades, perf)
    return updated

