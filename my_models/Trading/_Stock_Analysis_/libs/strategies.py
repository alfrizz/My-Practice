from libs import params

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
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm


#######################################################################################################

    
def generate_actions(
    df: pd.DataFrame,
    col_signal: str,
    sign_thresh: str,
    col_atr: str,
    col_rsi: str,
    col_vwap: str,
    reset_peak: bool,
    rsi_min_thresh: int,
    trailstop_pct: float,
    atr_mult: float,
    vwap_atr_mult: float,    
    col_close: str   = "close",
    sess_start: time = params.sess_start_reg,
) -> pd.DataFrame:
    """
    Generate per-row discrete trade actions (1=Buy, -1=Sell, 0=Hold) 
    plus three stop series (trailing stop, ATR stop, VWAP-based stop) 
    and buy/sell weights.
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
    # stop_frac = float(trailstop_pct) / 100
    stop_frac = trailstop_pct/100 if trailstop_pct > 1 else trailstop_pct

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
                if rsi[i] > rsi_min_thresh or close[i] > vwap_arr[i]:
                    df.at[df.index[i], "action"] = 1  # buy action
                    buy_weight[i] = (signal[i] - sign_thr) / sign_thr
    
            else: # signal[i] < sign_thr   # possible SELL  
                if close[i] < trail_arr[i] or close[i] < atr_arr[i]:
                    df.at[df.index[i], "action"] = -1  # sell action
                    trail_arr[i] = close[i] * (1.0 - stop_frac) if reset_peak else trail_arr[i]
                    sell_weight[i] = (sign_thr - signal[i]) / sign_thr
        
    df["trail_stop_price"] = pd.Series(trail_arr, index=df.index)
    df["atr_stop_price"] = pd.Series(atr_arr, index=df.index)
    df["vwap_stop_price"] = pd.Series(vwap_arr, index=df.index)
    df["buy_weight"] = buy_weight
    df["sell_weight"] = sell_weight
      
    return df


##########################################


def generate_actions_slope(
    df: pd.DataFrame,
    col_signal: str,
    sign_thresh: str,
    col_atr: str,
    col_rsi: str,
    col_vwap: str,
    reset_peak: bool,
    rsi_min_thresh: int,
    rsi_max_thresh: float,
    trailstop_pct: float,
    atr_mult: float,
    vwap_atr_mult: float,    
    col_close: str   = "close",
    sess_start: time = params.sess_start_reg,
) -> pd.DataFrame:
    """
    Generate per-row discrete trade actions (1=Buy, -1=Sell, 0=Hold) 
    plus three stop series (trailing stop, ATR stop, VWAP-based stop) 
    and buy/sell weights.
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
    stop_frac = trailstop_pct/100 if trailstop_pct > 1 else trailstop_pct

    # iterate rows and compute actions + trailing stop + atr stop
    for i in range(len(df)):
        sign_thr = float(sign_thresh[i]) if is_series else float(sign_thresh)
        # compute a single candidate peak and its trail value once per row
        peak = max(close[i], trail_arr[i - 1] / (1.0 - stop_frac) if i>0 else close[i])
        trail_arr[i] = peak * (1.0 - stop_frac) # running trail 
        atr_arr[i]   = peak - atr_mult * atr[i] # running atr (ATR decreses it: to exit a trade, require a smaller close price in volatile markets)
        vwap_arr[i]  = vwap[i] + vwap_atr_mult * atr[i] # running vwap (ATR increses it: to enter a trade, require a larger close price in volatile markets)

        if df.index.time[i] >= sess_start:
            # BUY: require signal >= threshold AND either (RSI above rsi_min_thresh but below rsi_max_thresh)
            # or price above VWAP reference or positive signal slope
            if signal[i] >= sign_thr:
                if ((rsi[i] > rsi_min_thresh and rsi[i] < rsi_max_thresh) or \
                    close[i] > vwap_arr[i]) and \
                signal[i] - signal[i-1] > 0: # increasing slope
                    df.at[df.index[i], "action"] = 1
                    buy_weight[i] = (signal[i] - sign_thr) / sign_thr

            # SELL: hard ATR or trailing stop, or simple signal reversal as soft exit
            else:
                if (close[i] < atr_arr[i] or \
                    close[i] < trail_arr[i]) and \
                signal[i] - signal[i-1] < 0: # decreasing slope
                    df.at[df.index[i], "action"] = -1
                    trail_arr[i] = close[i] * (1.0 - stop_frac) if reset_peak else trail_arr[i]
                    sell_weight[i] = (sign_thr - signal[i]) / sign_thr

        
    df["trail_stop_price"] = pd.Series(trail_arr, index=df.index)
    df["atr_stop_price"] = pd.Series(atr_arr, index=df.index)
    df["vwap_stop_price"] = pd.Series(vwap_arr, index=df.index)
    df["buy_weight"] = buy_weight
    df["sell_weight"] = sell_weight
      
    return df


##########################################


# def generate_actions_alpaca(
#     df: pd.DataFrame,
#     col_signal: str,
#     sign_thresh: str,
#     col_atr: str,
#     col_adx: str,
#     col_rsi: str,
#     col_vwap: str,
#     col_vol_spike: str,
#     reset_peak: bool,
#     rsi_min_thresh: int,
#     rsi_max_thresh: float,
#     adx_thresh: float,
#     vol_thresh: float,
#     trailstop_pct: float,
#     atr_mult: float,
#     vwap_atr_mult: float,    
#     col_close: str   = "close",
#     sess_start: time = params.sess_start_reg,
# ) -> pd.DataFrame:
#     """
#     Generate per-row discrete trade actions (1=Buy, -1=Sell, 0=Hold) 
#     plus three stop series (trailing stop, ATR stop, VWAP-based stop) 
#     and buy/sell weights.
#     """
#     df = df.copy()
#     df["action"] = 0  # not in trade by default

#     signal      = df[col_signal].to_numpy(dtype=float)
#     close       = df[col_close].to_numpy(dtype=float)
#     atr         = df[col_atr].to_numpy(dtype=float)
#     adx         = df[col_adx].to_numpy(dtype=float)
#     rsi         = df[col_rsi].to_numpy(dtype=float)
#     vwap        = df[col_vwap].to_numpy(dtype=float)
#     vol_spike   = df[col_vol_spike].to_numpy(dtype=float)

#     sign_slope  = np.gradient(signal)
#     sign_thresh = df[sign_thresh].to_numpy() if isinstance(sign_thresh, str) else sign_thresh # the threshold field can be string or numeric
#     is_series   = np.ndim(sign_thresh) != 0 # the threshold can be a constant or a series

#     trail_arr   = np.full(len(df), np.nan, dtype=float)
#     atr_arr     = np.full(len(df), np.nan, dtype=float)
#     vwap_arr    = np.full(len(df), np.nan, dtype=float)
    
#     buy_weight  = np.zeros(len(df), dtype=float)
#     sell_weight = np.zeros(len(df), dtype=float)
#     stop_frac   = trailstop_pct/100 if trailstop_pct > 1 else trailstop_pct

#     # iterate rows and compute actions + trailing stop + atr stop
#     for i in range(len(df)):
#         sign_thr = float(sign_thresh[i]) if is_series else float(sign_thresh)
#         # compute a single candidate peak and its trail value once per row
#         peak = max(close[i], trail_arr[i - 1] / (1.0 - stop_frac) if i>0 else close[i])
#         trail_arr[i] = peak * (1.0 - stop_frac) # running trail 
#         atr_arr[i]   = peak - atr_mult * atr[i] # running atr (ATR decreses it: to exit a trade, require a smaller close price in volatile markets)
#         vwap_arr[i]  = vwap[i] + vwap_atr_mult * atr[i] # running vwap (ATR increses it: to enter a trade, require a larger close price in volatile markets)

#         if df.index.time[i] >= sess_start:
#             # BUY conditions
#             if signal[i] >= sign_thr \
#             and adx[i] >= adx_thresh \
#             and vol_spike[i] >= vol_thresh \
#             and slope[i] > 0 \
#             and ((rsi[i] > rsi_min_thresh and rsi[i] < rsi_max_thresh) \
#                  or close[i] > vwap_arr[i]): 
#             df.at[df.index[i], "action"] = 1 
#             buy_weight[i] = (signal[i] - sign_thr) / sign_thr

#             # SELL conditions
#             elif signal[i] < sign_thr \
#             and (close[i] < atr_arr[i] \
#                  or close[i] < trail_arr[i] \
#                  or slope[i] < 0): 
#                     df.at[df.index[i], "action"] = -1
#                     trail_arr[i] = close[i] * (1.0 - stop_frac) if reset_peak else trail_arr[i]
#                     sell_weight[i] = (sign_thr - signal[i]) / sign_thr

        
#     df["trail_stop_price"] = pd.Series(trail_arr, index=df.index)
#     df["atr_stop_price"] = pd.Series(atr_arr, index=df.index)
#     df["vwap_stop_price"] = pd.Series(vwap_arr, index=df.index)
#     df["buy_weight"] = buy_weight
#     df["sell_weight"] = sell_weight
      
#     return df



def generate_actions_alpaca(
    df: pd.DataFrame,
    col_signal: str,
    sign_thresh,                # float, int, or column name
    col_atr: str,
    col_adx: str,
    col_rsi: str,
    col_vwap: str,
    col_vol_spike: str,
    reset_peak: bool,
    rsi_min_thresh: float,
    rsi_max_thresh: float,
    adx_thresh: float,
    vol_thresh: float,
    trailstop_pct: float,  
    atr_mult: float,
    vwap_atr_mult: float,
    col_close: str   = "close",
    sess_start: time = None,    # pass a time; if None, no session filter
) -> pd.DataFrame:
    """
    Generate per-row discrete trade actions (1=Buy, -1=Sell, 0=Hold)
    plus three stop series (trailing stop, ATR stop, VWAP-based stop)
    and buy/sell weights.

    Notes:
    - sign_thresh can be a scalar or a column name in df.
    - trailstop_pct is always a percentage (e.g., 1.0 => 1%).
    - VWAP stop is treated as an entry filter (price above vwap + k*ATR).
    """
    df = df.copy()
    df["action"] = 0

    signal    = df[col_signal].to_numpy(dtype=float)
    close     = df[col_close].to_numpy(dtype=float)
    atr       = df[col_atr].to_numpy(dtype=float)
    adx       = df[col_adx].to_numpy(dtype=float)
    rsi       = df[col_rsi].to_numpy(dtype=float)
    vwap      = df[col_vwap].to_numpy(dtype=float)
    vol_spike = df[col_vol_spike].to_numpy(dtype=float)

    slope     = np.gradient(signal)
    if isinstance(sign_thresh, str):
        sign_thresh_arr = df[sign_thresh].to_numpy(dtype=float)
        is_series = True
    else:
        sign_thresh_arr = float(sign_thresh)
        is_series = False

    # clamp to [0, 100] and convert to fraction
    trailstop_pct = max(0.0, min(100.0, trailstop_pct))
    stop_frac = trailstop_pct / 100.0
    EPS = 1e-12

    trail_arr = np.full(len(df), np.nan, dtype=float)
    atr_arr   = np.full(len(df), np.nan, dtype=float)
    vwap_arr  = np.full(len(df), np.nan, dtype=float)
    buy_weight  = np.zeros(len(df), dtype=float)
    sell_weight = np.zeros(len(df), dtype=float)

    for i in range(len(df)):
        # session gate
        if sess_start and df.index[i].time() < sess_start:
            continue

        sign_thr = float(sign_thresh_arr[i]) if is_series else float(sign_thresh_arr)

        # compute peak from prior info
        prev_peak = np.nan if i == 0 else trail_arr[i-1] / (1.0 - stop_frac)
        base_peak = close[i] if np.isnan(prev_peak) else max(close[i], prev_peak)
        peak = base_peak

        trail_arr[i] = peak * (1.0 - stop_frac)                # trailing stop
        atr_arr[i]   = peak - atr_mult * atr[i]                # ATR-based stop
        vwap_arr[i]  = vwap[i] + vwap_atr_mult * atr[i]        # VWAP+ATR entry bar

        # BUY conditions
        if (
            signal[i] >= sign_thr and
            adx[i] >= adx_thresh and
            vol_spike[i] >= vol_thresh and
            slope[i] > 0 and
            ((rsi_min_thresh < rsi[i] < rsi_max_thresh) or (close[i] > vwap_arr[i]))
        ):
            df.at[df.index[i], "action"] = 1
            buy_weight[i] = max((signal[i] - sign_thr) / max(sign_thr, EPS), 0.0)

        # SELL conditions
        elif (
            signal[i] < sign_thr and
            (
                close[i] < atr_arr[i] or
                close[i] < trail_arr[i] or
                slope[i] < 0
            )
        ):
            df.at[df.index[i], "action"] = -1
            sell_weight[i] = max((sign_thr - signal[i]) / max(sign_thr, EPS), 0.0)
            if reset_peak:
                trail_arr[i] = close[i] * (1.0 - stop_frac)

    df["trail_stop_price"] = pd.Series(trail_arr, index=df.index)
    df["atr_stop_price"]   = pd.Series(atr_arr, index=df.index)
    df["vwap_stop_price"]  = pd.Series(vwap_arr, index=df.index)
    df["buy_weight"]       = buy_weight
    df["sell_weight"]      = sell_weight

    return df

    
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
    else: # 'buy'
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


def reset_globals(start_price: float = None):
    """
    Reset module-level simulation state.

    Reinitializes globals used by the simulator and optionally seeds the
    carried buy-and-hold position using a canonical start price.
    """
    global _sell_intr_posamnt, _last_pnl, _last_cash, _bh_leftover, _last_bh_final, _bh_shares, _last_position

    _sell_intr_posamnt = params.init_cash   # intraday pot start
    _last_pnl          = params.init_cash   # strategy prior PnL baseline
    _last_cash         = params.init_cash   # cash for trading loop
    _last_position     = 0

     # initialize B&H
    fee_open = fees_for_one_share(price=start_price, side="buy")["total_per_share_billed"]
    _bh_shares = int(params.init_cash // (start_price + fee_open))
    invested = _bh_shares * (start_price + fee_open)
    _bh_leftover = params.init_cash - invested
    _last_bh_final = params.init_cash


#######################################################################################################


def compute_intraday_bh(prev_cap: float, df: pd.DataFrame):
    """
    Compute intraday integer-share simulation and B&H totals for one day.

    Returns the updated intraday capital, a formatted intraday summary line,
    and a formatted buy-and-hold summary line.
    """
    global _bh_shares, _bh_leftover, _last_bh_final
    _round = lambda x: round(float(x), 3)
    mask = (df.index.time >= params.sess_start_reg) & (df.index.time <= params.sess_end)

    if not mask.any(): # use whatever bars exist (even if outside the mask window)
        ask_buy  = float(df["ask"].iloc[0])
        bid_sell = float(df["bid"].iloc[-1])
    else:
        ask_buy  = float(df.loc[mask, "ask"].iloc[0])
        bid_sell = float(df.loc[mask, "bid"].iloc[-1])

    fee_buy  = fees_for_one_share(price=ask_buy, side="buy")["total_per_share_billed"]
    fee_sell = fees_for_one_share(price=bid_sell, side="sell")["total_per_share_billed"]

    shares = int(prev_cap // (ask_buy + fee_buy))
    if shares > 0:
        amt_buy  = shares * (ask_buy + fee_buy)          # cash spent on shares + buy fees
        leftover = prev_cap - amt_buy                    # leftover cash not invested
        amt_sell = shares * (bid_sell - fee_sell)       # proceeds after sell fees
        new_cap  = leftover + amt_sell
        intr_pnl = new_cap - prev_cap
    else:
        amt_buy = 0.0
        leftover = prev_cap
        amt_sell = 0.0
        new_cap = prev_cap
        intr_pnl = 0.0

    intraday_line = (
        f"shares({shares}); "
        f"[ask_buy={_round(ask_buy)}, fee_buy={_round(fee_buy)}, pos_amnt_buy={_round(amt_buy)}]; "
        f"[bid_sell={_round(bid_sell)}, fee_sell={_round(fee_sell)}, pos_amnt_sell={_round(amt_sell)}]; "
        f"leftover={_round(leftover)}; PNL={_round(intr_pnl)}"
    )

    # compute proceeds/value using stored integer shares and leftover
    proceeds  = _bh_shares * (bid_sell - fee_sell)
    final_cap = _bh_leftover + proceeds
        
    # per-day B&H delta (today's total_cap minus yesterday's total_cap)
    per_day_bh_gain = final_cap - _last_bh_final
    _last_bh_final = final_cap

    # formatted line describing the daily B&H state
    buynhold_line = (
        f"shares({_bh_shares}); [start={_round(ask_buy)} end={_round(bid_sell)}]; "
        f"leftover={_round(_bh_leftover)}; total_cap={_round(final_cap)}; PNL={_round(per_day_bh_gain)}"
    )

    return new_cap, intraday_line, buynhold_line


############################################


def _format_perf(df, trades):
    """
    Build per-day performance summaries.

    Produces formatted strings for INTRADAY, TRADES, STRATEGY, and BUYNHOLD
    using helper functions to ensure consistent integer-share and fee logic.
    """
    global _sell_intr_posamnt, _last_pnl

    _round = lambda x: round(float(x), 3)

    # --- trades / strategy lines (unchanged) ---
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

    df = df.assign(TradeID=trade_ids)

    strategy_line = (
        " + ".join([f"Δ[{idx}]:{val}" for idx, val in delta_totals]) +
        f" PNL={_round(sum(val for _, val in delta_totals))}"
    )

   # --- one-time buy & hold for the day using same integer-share + leftover logic --- &  --- intraday using helper (carries leftover cash) ---
    new_cap, intraday_line, buynhold_line = compute_intraday_bh(
       prev_cap=_sell_intr_posamnt, 
        df=df, 
    )
    
    _sell_intr_posamnt = new_cap

    perf = {
        "INTRADAY": intraday_line,
        "TRADES": trades_lines,
        "STRATEGY": strategy_line,
        "BUYNHOLD": buynhold_line
    }
    return df, perf


####################################################################################################### 


def simulate_trading(
    day,
    df,
    buy_factor: float  = 0.0,    # interpolation factor for buys: 0 => use buy_weight as-is; 1 => use full shares_max (0..1)
    sell_factor: float = 0.0,   # interpolation factor for sells: 0 => use sell_weight as-is; 1 => sell full position (0..1)
) -> dict:
    """
    Simulate intraday trading using discrete actions produced by generate_trade function
    """
    global _last_position, _last_cash

    df = df.sort_index().copy()
    updated_results = {}
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
        
        per_share_buy_fee = fees_for_one_share(price=ask, side="buy")["total_per_share_billed"]
        shares_max = math.floor(cash / (ask + per_share_buy_fee))
        per_share_sell_fee = fees_for_one_share(price=bid, side="sell")["total_per_share_billed"]
        
        # BUY branch
        if shares_max > 0 and row["action"] == 1:  # buy
            action = "Buy" 
            # buy: interpolate weight toward full allocation, compute affordable shares, clamp to [0, shares_max]
            shares_qty = min(shares_max, max(0, math.ceil(shares_max * (row["buy_weight"] * (1.0 - buy_factor) + buy_factor))))
            position += shares_qty
            buy_cost = ask * shares_qty
            buy_fee = per_share_buy_fee * shares_qty
            cash -= buy_cost + buy_fee

        # SELL branch
        if position > 0 and row["action"] == -1:
            action = "Sell"
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
    )

    # Persist end-of-day state
    _last_position = position
    _last_cash = cash
    
    updated_results[day] = (df_sim, trades, perf)
    return updated_results


#######################################################################################################


def _parse_eq_value(s: str) -> float:
    """
    Parse a formatted performance line and return the numeric PNL value.

    Extracts the trailing 'PNL=' value from a performance string or returns 0.0.
    """
    return float(s.rsplit("PNL=", 1)[-1].strip()) if s else 0.0


###########################################


def rolling_monthly_summary(df, sim_results):
    """
    Produce a rolling monthly performance report.

    Aggregates per-day deltas for INTRADAY, STRATEGY, and BUYNHOLD, carries
    each metric forward independently, and prints monthly summaries.
    """
    month_map = defaultdict(list)
    for k in sorted(sim_results):
        ts = pd.to_datetime(k)
        month_map[ts.to_period("M")].append((ts.normalize(), sim_results[k][2]))

    def pct(gain: float, base: float) -> float:
        return (gain / base * 100.0) if base else 0.0

    rolling_bh = rolling_strategy = rolling_intraday = params.init_cash
    months = sorted(month_map.keys())

    # first-month start ask (for header printing)
    first_month = months[0]
    first_days = [d for d, _ in month_map[first_month]]
    df_first = df[df.index.normalize().isin(first_days)]
    start_m_approx = df_first.loc[df_first.index.normalize() == min(first_days), "ask"].iloc[0]

    print("Rolling monthly summary (each metric carries its own capital forward):")

    for m in months:
        items = month_map[m]
        days_m = [d for d, _ in items]
        df_m = df[df.index.normalize().isin(days_m)]

        start_m = df_m.loc[df_m.index.normalize() == min(days_m), "ask"].iloc[0]
        end_m   = df_m.loc[df_m.index.normalize() == max(days_m), "bid"].iloc[-1]

        # intraday, strategy and buynhold are per-day deltas (sum them)
        intraday_m = sum(_parse_eq_value(p["INTRADAY"]) for _, p in items)
        strategy_m = sum(_parse_eq_value(p["STRATEGY"]) for _, p in items)
        one_time_bh_m = sum(_parse_eq_value(p["BUYNHOLD"]) for _, p in items)

        trades_m = sum(len(p["TRADES"]) for _, p in items)
        ndays_m = len(set(days_m))

        # start and final absolute B&H capitals (carry forward rolling_bh)
        start_bh = rolling_bh
        final_bh = start_bh + one_time_bh_m

        start_strat = rolling_strategy
        start_intr = rolling_intraday
        final_strat = start_strat + strategy_m
        final_intr = start_intr + intraday_m

        print(f"\nMonthly Summary {m} ({min(days_m).date()} = {start_m_approx:.3f} → {max(days_m).date()} = {end_m:.3f})")
        print(f"Num. trading days: {ndays_m}  Trades Count: {trades_m}")
        print(f"One-Time B&H gain: {one_time_bh_m:.3f} | start: {start_bh:.3f} | final: {final_bh:.3f} | PnL%: {pct(one_time_bh_m, start_bh):.2f}%")
        print(f"Sum Strategy gain: {strategy_m:.3f} | start: {start_strat:.3f} | final: {final_strat:.3f} | PnL%: {pct(strategy_m, start_strat):.2f}%")
        print(f"Sum Intraday gain: {intraday_m:.3f} | start: {start_intr:.3f} | final: {final_intr:.3f} | PnL%: {pct(intraday_m, start_intr):.2f}%")

        # carry forward
        rolling_bh = final_bh
        rolling_strategy = final_strat
        rolling_intraday = final_intr

        start_m_approx = end_m

    return {"final_bh": rolling_bh, "final_strategy": rolling_strategy, "final_intraday": rolling_intraday}


##############################################


def aggregate_performance(df: pd.DataFrame,
                          perf_list: list = None,
                          sim_results: dict = None,
                          monthy_summary: bool = True) -> None:
    """
    Compute and display overall performance metrics.

    Aggregates per-day performance into totals, prints summary statistics,
    and optionally calls the rolling monthly summary.
    """
    dates_all = [d for d in sim_results]
    perf_list = [sim_results[date][2] for date in sorted(dates_all)]

    def pct(gain: float) -> float:
        return (gain / params.init_cash * 100.0)

    first_day = df.index.normalize().min()
    last_day  = df.index.normalize().max()
    start_ask = df.loc[df.index.normalize() == first_day, "ask"].iloc[0]
    end_bid   = df.loc[df.index.normalize() == last_day,  "bid"].iloc[-1]

    num_days = df.index.normalize().nunique()
    trades_count = sum(len(perf_day["TRADES"]) for perf_day in perf_list)

    # print("\n" + "=" * 115)
    print(f"Overall Summary ({first_day.date()} = {start_ask:.3f} → {last_day.date()} = {end_bid:.3f})")
    print(f"Num. trading days: {num_days}")
    print(f"Trades Count: {trades_count}")
    print(f"Initial capital: {params.init_cash:.3f}")

     # BUYNHOLD, STRATEGY and INTRADAY entries are per-day deltas. Sum them, always parsed from perf_list
    strategy_per_day = [_parse_eq_value(perf_day["STRATEGY"]) for perf_day in perf_list]
    strategy_sum = sum(strategy_per_day)
    strategy_final = params.init_cash + strategy_sum

    intraday_per_day = [_parse_eq_value(perf_day["INTRADAY"]) for perf_day in perf_list]
    intraday_sum = sum(intraday_per_day)
    intraday_final = params.init_cash + intraday_sum

    bh_per_day = [_parse_eq_value(perf_day["BUYNHOLD"]) for perf_day in perf_list]
    one_time_bh_gain = sum(bh_per_day)
    one_time_bh_final = params.init_cash + one_time_bh_gain

    # print totals (use canonical all-in B&H)
    print(f"\nOne-Time B&H gain: {one_time_bh_gain:.3f} | final: {one_time_bh_final:.3f} | PnL%: {pct(one_time_bh_gain):.2f}%")
    print(f"Sum Strategy gain: {strategy_sum:.3f} | final: {strategy_final:.3f} | PnL%: {pct(strategy_sum):.2f}%")
    print(f"Sum Intraday gain: {intraday_sum:.3f} | final: {intraday_final:.3f} | PnL%: {pct(intraday_sum):.2f}%")

    # per-day / per-trade metrics (use canonical B&H)
    one_time_bh_per_day_avg = one_time_bh_gain / num_days if num_days else 0.0
    strategy_per_day_avg    = strategy_sum / num_days if num_days else 0.0
    strategy_per_trade_avg  = strategy_sum / trades_count if trades_count else 0.0

    print(f"\nOne-Time B&H gain per day: {one_time_bh_per_day_avg:.4f}")
    print(f"Strategy gain per day: {strategy_per_day_avg:.4f}")
    print(f"Strategy gain per trade: {strategy_per_trade_avg:.4f}")

    # Bar plot: one-time B&H, strategy, intraday
    primary = {
        "One-Time B&H gain": one_time_bh_gain,
        "Sum Strategy gain": strategy_sum,
        "Sum Intraday gain": intraday_sum,
    }
    secondary = {
        "One-Time B&H per day": one_time_bh_per_day_avg,
        "Strategy gain per day": strategy_per_day_avg,
        "Strategy gain per trade": strategy_per_trade_avg,
    }

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    names1 = list(primary.keys())
    names2 = list(secondary.keys())
    x1 = np.arange(len(names1))
    x2 = np.arange(len(names2)) + len(names1)

    width = 0.6
    bars1 = ax1.bar(x1, list(primary.values()), width, color="#4C72B0", alpha=0.9, label="Absolute")
    bars2 = ax2.bar(x2, list(secondary.values()), width, color="#C44E52", alpha=0.9, label="Relative")

    all_names = names1 + names2
    ax1.set_xticks(np.concatenate([x1, x2]))
    ax1.set_xticklabels(all_names, rotation=30, ha="right")
    ax1.set_ylabel("USD (absolute)")
    ax2.set_ylabel("USD (per trade/day)")
    ax1.set_title(f"Performance Summary ({first_day.date()} → {last_day.date()})")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.show()

    if monthy_summary:
        rolling_monthly_summary(df, sim_results)

    
        