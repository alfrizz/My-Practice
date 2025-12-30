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


#######################################################################################################


def reset_globals():
    """Reset module-level global variables used by simulate_trading/_format_perf."""
    global _sell_intr_posamnt, _last_pnl, _last_cash, _last_position, _last_trail
    _sell_intr_posamnt = params.init_cash   # intraday pot start
    _last_pnl          = params.init_cash   # strategy prior PnL baseline
    _last_cash         = params.init_cash   # cash for trading loop
    _last_position     = 0
    _last_trail        = None

    
#######################################################################################################


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


#######################################################################################################


def _format_perf(
    df,
    trades,
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
    fee_buy = fees_for_one_share(price=ask_buy, side="buy")["total_per_share_billed"] 
    fee_sell = fees_for_one_share(price=bid_sell, side="sell")["total_per_share_billed"]
    
    shares     = int(_sell_intr_posamnt // (ask_buy + fee_buy))
    amt_buy    = shares * (ask_buy + fee_buy)
    amt_sell   = shares * (bid_sell - fee_sell)
    intr_pnl   = amt_sell - amt_buy
    
    _sell_intr_posamnt = amt_sell
    
    intraday_line = (
        f"shares({shares}); "
        f"[ask_buy={_round(ask_buy)}, fee_buy={_round(fee_buy)}, pos_amnt_buy={_round(amt_buy)}]; "
        f"[bid_sell={_round(bid_sell)}, fee_sell={_round(fee_sell)}, pos_amnt_sell={_round(amt_sell)}]; "
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
            f"Tr[{tr_idx}]: Time({ts.strftime('%H:%M')})Ask({_round(ask)})Bid({_round(bid)})Act({action})"
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

    



#######################################################################################################

    
def generate_actions(
    df: pd.DataFrame,
    col_signal: str,
    sign_thresh: str,
    col_atr: str,
    col_rsi: str,
    col_vwap: str,
    reset_peak: bool,
    rsi_thresh: int,
    trailstop_pct: float,
    atr_mult: float,
    vwap_atr_mult: float,    
    col_close: str   = "close",
    sess_start: time = params.sess_start_reg,
) -> pd.DataFrame:
    """
    Generate discrete trade actions (1=Buy, -1=Sell, 0=NotInTrade, 2=InTrade), a trailing-stop series,
    and an ATR-based stop series.
    - Row-by-row generator: seeds entries, updates a running peak, computes a trailing
      stop as `trail = peak * (1 - stop_frac)`, and issues sells when signal < threshold
      and bid < trail (only during session).
    - Also computes `atr_stop_price = peak - atr_mult * atr` for optional ATR-based exits.
    - Supports scalar or per-row `sign_thresh`. 
    """
    df = df.copy()
    df["action"] = 0  # not in trade by default

    signal = df[col_signal].to_numpy(dtype=float)
    close = df[col_close].to_numpy(dtype=float)
    atr = df[col_atr].to_numpy(dtype=float)
    rsi = df[col_rsi].to_numpy(dtype=float)
    vwap = df[col_vwap].to_numpy(dtype=float)
    
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
        peak = max(close[i], trail_arr[i - 1] / (1.0 - stop_frac) if i>0 else close[i])
        trail_arr[i] = peak * (1.0 - stop_frac) # running trail 
        atr_arr[i]   = peak - atr_mult * atr[i] # running atr (ATR decreses it: to exit a trade, require a smaller close price in volatile markets)
        vwap_arr[i]  = vwap[i] + vwap_atr_mult * atr[i] # running vwap (ATR increses it: to enter a trade, require a larger close price in volatile markets)

        if df.index.time[i] >= sess_start:               
            if signal[i] >= sign_thr:   # possible BUY        
                if rsi[i] > rsi_thresh or close[i] > vwap_arr[i]:
                    df.at[df.index[i], "action"] = 1  # buy action
                    buy_weight[i] = (signal[i] - sign_thr) / sign_thr
                    # buy_weight[i]  = 0.0 if (not np.isfinite(sign_thr) or sign_thr == 0) else (signal[i] - sign_thr) / sign_thr
    
            else: # signal[i] < sign_thr   # possible SELL  
                if close[i] < trail_arr[i] or close[i] < atr_arr[i]:
                    df.at[df.index[i], "action"] = -1  # sell action
                    trail_arr[i] = close[i] * (1.0 - stop_frac) if reset_peak else trail_arr[i]
                    sell_weight[i] = (sign_thr - signal[i]) / sign_thr
                    # sell_weight[i] = 0.0 if (not np.isfinite(sign_thr) or sign_thr == 0) else (sign_thr - signal[i]) / sign_thr
        
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
    invest_frac: float = 0.1,   # fraction of cash available to allocate per buy signal (0..1)
    buy_factor: float  = 0.0,    # interpolation factor for buys: 0 => use buy_weight as-is; 1 => use full shares_max (0..1)
    sell_factor: float = 0.0,   # interpolation factor for sells: 0 => use sell_weight as-is; 1 => sell full position (0..1)
    sess_start: time   = params.sess_start_reg,
) -> dict:
    """
    Simulate intraday trading using discrete actions produced by generate_tradact_elab.
    Behavior
    - Iterates rows (intraday bars) in chronological order and executes only actions present in `df["action"]`.
    - Maintains carried state across calls using globals `_last_position` and `_last_cash`.
    - For BUY actions: computes an affordable `shares_max` from `cash`, `invest_frac`, `ask`, and per-share buy fee,
      then sizes the order by interpolating between the generator's `buy_weight` and full allocation using `buy_factor`.
    - For SELL actions: sizes the order by interpolating between the generator's `sell_weight` and full position using `sell_factor`.
    - Ensures executed shares do not exceed affordability (`shares_max`) or holdings (`position`) and skips zero-sized trades.
    - Records each executed trade and snapshots Position, Cash, Pnl, Shares, Action for every bar.
    """
    global _last_position, _last_cash

    df = df.sort_index().copy()
    updated = {}
    
    position = int(_last_position)
    cash = _last_cash
    buy_cost = sell_cost = 0.0
    pos_buf, posamt_buf, cash_buf, pnl_buf, act_buf, shar_buf = [], [], [], [], [], []
    trades = []

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
            per_share_buy_fee = fees_for_one_share(price=ask, side="buy")["total_per_share_billed"] 
            # limit to allocation target and to the number of affordable shares
            shares_max = min(int(cash // (ask + per_share_buy_fee)), math.ceil((cash * invest_frac) / (ask + per_share_buy_fee)))
            # buy: interpolate weight toward full allocation, compute affordable shares, clamp to [0, shares_max]
            shares_qty = min(shares_max, max(0, math.ceil(shares_max * (row["buy_weight"] * (1.0 - buy_factor) + buy_factor))))
            position += shares_qty
            buy_cost = ask * shares_qty
            buy_fee = per_share_buy_fee * shares_qty
            cash -= buy_cost + buy_fee

        # SELL branch
        if position > 0 and row["action"] == -1:
            action = "Sell"
            per_share_sell_fee = fees_for_one_share(price=bid, side="sell")["total_per_share_billed"]
            # sell: interpolate weight toward full position, compute shares to sell, clamp to [0, position]
            shares_qty = min(position, max(0, math.ceil(position * (row["sell_weight"] * (1.0 - sell_factor) + sell_factor))))
            position -= shares_qty
            sell_cost = bid * shares_qty
            sell_fee = per_share_sell_fee * shares_qty
            cash += sell_cost - sell_fee

        pos_amount = position * bid
        tot_pnl = cash + pos_amount

        if shares_qty > 0:
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
        sess_start=sess_start,
    )

    # Persist end-of-day state
    _last_position = position
    _last_cash = cash
    
    updated[day] = (df_sim, trades, perf)
    return updated

