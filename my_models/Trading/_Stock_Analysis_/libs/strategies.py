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


##################################
_sell_intr_posamnt = params.init_cash   # intraday pot start
_last_pnl          = params.init_cash   # strategy prior PnL baseline
_last_cash         = params.init_cash   # cash for trading loop
_last_position     = 0
_last_trail        = 0.0
##################################

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




# def _format_perf(df, trades, shares, buy_fee_per, sell_fee_per, sess_start, start_tot):
#     """
#     Build performance summary (buy/hold, trades, strategy deltas).
#     start_tot seeds the first delta with prior-day ending total.
#     """
#     _round = lambda x: round(float(x), 3)
#     mask = (df.index.time >= sess_start) & (df.index.time <= params.sess_end)
#     # Buy-and-hold over the session
#     if mask.sum() == 0:
#         bh_line = "BUY&HOLD: no bars in session range"
#     else:
#         open_ask = float(df.loc[mask, "ask"].iloc[0])
#         close_bid = float(df.loc[mask, "bid"].iloc[-1])
#         fee_open = buy_fee_per(open_ask)
#         fee_close = sell_fee_per(close_bid)
#         # max shares purchasable with init_cash including buy fee
#         shares_bh = int(params.init_cash // (open_ask + fee_open))
#         invested = shares_bh * (open_ask + fee_open)
#         proceeds = shares_bh * (close_bid - fee_close)
#         bh_pnl = _round(proceeds - invested)
#         bh_line = (
#             f"BUY&HOLD(init cash {params.init_cash}): sh({shares_bh}) "
#             f"buy@{_round(open_ask)} fee={_round(fee_open)} "
#             f"sell@{_round(close_bid)} fee={_round(fee_close)} "
#             f"PNL={bh_pnl}"
#         )

#     trades_lines, realized_vals, delta_totals = [], [], []
#     prev_tot = start_tot  # seed with prior-day end total

#     for i, trade in enumerate(trades, 1):
#         (
#             (ask_ts, bid_ts), (ask_p, bid_p),
#             (buy_fee, sell_fee), (buy_loss, sell_profit),
#             action, shares_qty, position, pos_amount, cash, tot_pnl
#         ) = trade

#         ask_ts_str = ask_ts.strftime("%H:%M") if hasattr(ask_ts, "strftime") else str(ask_ts)

#         if action == "Sell":
#             cash_trade = float(sell_profit) - float(sell_fee)
#             realized_vals.append(_round(cash_trade))
#             trades_lines.append(
#                 f"[Tr:{i}] Time({ask_ts_str})Ask({_round(ask_p)})Bid({_round(bid_p)})Act({action})"
#                 f"Shrs({shares_qty})Pos({position}); "
#                 f"CashTr({_round(cash_trade)})=Prof({_round(sell_profit)})-Fee({_round(sell_fee)}); "
#                 f"P&L({_round(tot_pnl)})=CashTot({_round(cash)})+PosAmt({_round(pos_amount)})"
#             )
#         else:  # Buy / Carried
#             cash_trade = -float(buy_loss) - float(buy_fee)
#             realized_vals.append(_round(cash_trade))
#             trades_lines.append(
#                 f"[Tr:{i}] Time({ask_ts_str})Ask({_round(ask_p)})Bid({_round(bid_p)})Act({action})"
#                 f"Shrs({shares_qty})Pos({position}); "
#                 f"CashTr({_round(cash_trade)})=-Loss({_round(buy_loss)})-Fee({_round(buy_fee)}); "
#                 f"P&L({_round(tot_pnl)})=CashTot({_round(cash)})+PosAmt({_round(pos_amount)})"
#             )

#         # ΔTotP&L vs previous trade (first one vs prior-day close)
#         delta = tot_pnl - prev_tot
#         delta_totals.append(_round(delta))
#         prev_tot = tot_pnl

#     strategy_line = (
#         " + ".join([f"Δ{i}:{v}" for i, v in enumerate(delta_totals, 1)]) +
#         f" PNL={_round(sum(delta_totals))}"
#     ) if delta_totals else "PNL=0.000"

#     return {"BUY&HOLD": bh_line, "TRADES": trades_lines, "STRATEGY": strategy_line} 


def _format_perf(
    df,
    trades,
    shares,
    buy_fee_per,
    sell_fee_per,
    sess_start,
):
    """
    Build performance summary (intraday buy/sell, trades, strategy deltas).
    """
    global _sell_intr_posamnt, _last_pnl

    _round = lambda x: round(float(x), 3)
    mask = (df.index.time >= sess_start) & (df.index.time <= params.sess_end)
    
    # minimal intraday: buy all possible at open, sell all at close, persist proceeds
    open_ask  = float(df.loc[mask, "ask"].iloc[0])
    close_bid = float(df.loc[mask, "bid"].iloc[-1])
    fee_open  = buy_fee_per(open_ask)
    fee_close = sell_fee_per(close_bid)
    
    shares     = int(_sell_intr_posamnt // (open_ask + fee_open))
    buy_amt    = shares * (open_ask + fee_open)
    sell_amt   = shares * (close_bid - fee_close)
    intr_pnl   = sell_amt - buy_amt
    
    _sell_intr_posamnt = sell_amt
    
    intraday_line = (
        f"shares({shares}); "
        f"[buy@{_round(open_ask)}, fee={_round(fee_open)}, pos_amnt_buy {_round(buy_amt)}]; "
        f"[sell@{_round(close_bid)}, fee={_round(fee_close)}, pos_amnt_sell {_round(sell_amt)}]; "
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
#     global _last_trail, _last_position

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
#     buy_weight = np.zeros(len(df), dtype=float)
#     sell_weight = np.zeros(len(df), dtype=float)
#     stop_frac = float(trailstop_pct) / 100.0

#     # iterate rows and compute actions + trailing stop + atr stop
#     for i in range(len(df)):
#         sign_thr = float(sign_thresh[i]) if is_series else float(sign_thresh)
#         # prev_action = df["action"].iat[i - 1] if i > 0 else (2 if _last_trail > 0 else 0)

#         # compute a single candidate peak and its trail value once per row
#         peak = close[i]
#         # if i > 0 and prev_action in [1, 2]: 
#         if _last_position > 0: # if in trade, compare the peak with the previous value
#             peak = max(peak, trail_arr[i - 1] / (1.0 - stop_frac))
#         elif _last_trail > 0: # if carried trade, compare the peak with the previouos value of previous day
#             peak = max(peak, _last_trail / (1.0 - stop_frac))
#             # df.at[df.index[i], "action"] = 2 # in trade
#             _last_trail = 0.0  # consume carried trail

#         trail_arr[i] = peak * (1.0 - stop_frac) # running trail
#         atr_arr[i]   = peak - atr_mult * atr[i] # running atr
#         vwap_arr[i]  = vwap[i] + vwap_atr_mult * atr[i] # running vwap
        
#         # # possible BUY                
#         # if prev_action in [0,-1]:
#         #     cond_buy = (
#         #         signal[i] >= sign_thr 
#         #         and (rsi[i] > rsi_thresh or close[i] > vwap_arr[i])
#         #         and df.index.time[i] >= sess_start
#         #     )
#         #     if cond_buy:
#         #         df.at[df.index[i], "action"] = 1  # buy action
#         #         buy_weight[i] = (signal[i] - sign_thr) / sign_thr * 100

#         #     else: 
#         #         df.at[df.index[i], "action"] = 0 # not in trade
#         #         trail_arr[i] = atr_arr[i] = np.nan

#         # # possible SELL 
#         # if prev_action in [1,2]:
#         #     cond_sell = (
#         #         signal[i] < sign_thr
#         #         and (bid[i] < trail_arr[i] or bid[i] < atr_arr[i])
#         #         and df.index.time[i] >= sess_start
#         #     )
#         #     if cond_sell:
#         #         df.at[df.index[i], "action"] = -1  # sell action
#         #         base = min(trail_arr[i], atr_arr[i])
#         #         sell_weight[i] = (sign_thr - signal[i]) / sign_thr * 100
#         #     else: 
#         #         df.at[df.index[i], "action"] = 2 # in trade

#         if df.index.time[i] >= sess_start:               
#             if signal[i] >= sign_thr:   # possible BUY        
#                 if rsi[i] > rsi_thresh or close[i] > vwap_arr[i]:
#                     df.at[df.index[i], "action"] = 1  # buy action
#                     buy_weight[i] = (signal[i] - sign_thr) / sign_thr * 100
    
#             else: # signal[i] < sign_thr   # possible SELL  
#                 if bid[i] < trail_arr[i] or bid[i] < atr_arr[i]:
#                     df.at[df.index[i], "action"] = -1  # sell action
#                     sell_weight[i] = (sign_thr - signal[i]) / sign_thr * 100
                
#         # if not cond_sell and not cond_buy:
#         #     if _last_position > 0:
#         #         df.at[df.index[i], "action"] = 2 # in trade
#         #     else: 
#         #         df.at[df.index[i], "action"] = 0 # not in trade
#         #         trail_arr[i] = atr_arr[i] = np.nan
                
                
#     df["trail_stop_price"] = pd.Series(trail_arr, index=df.index)
#     df["atr_stop_price"] = pd.Series(atr_arr, index=df.index)
#     df["vwap_stop_price"] = pd.Series(vwap_arr, index=df.index)
#     df["buy_weight"] = buy_weight
#     df["sell_weight"] = sell_weight

#     if sellmin_idx is None: # persist previous day trail value if still in trade
#         # _last_trail = float(df["trail_stop_price"].iat[-1]) if int(df["action"].iat[-1]) in [1, 2] else 0.0
#         _last_trail = float(df["trail_stop_price"].iat[-1]) if _last_position > 0 else 0.0
      
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

        # compute a single candidate peak and its trail value once per row
        # if _last_position > 0:
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



# def simulate_trading(
#     day,
#     df,
#     sellmin_idx: int,
#     sess_start: time,
#     shares: int = 1,
#     invest_frac: float = 0.1,
# ) -> dict:
#     """
#     Simulate intraday trading using provided actions; carry prior day state.
#     - Uses globals for carried position/cash and last bid/ask markers.
#     """
#     global _last_ask, _last_bid, _last_trail, _last_position, _last_cash, _last_pnl

#     updated = {}
#     buy_fee_per = lambda p: fees_for_one_share(price=p, side="buy")["total_per_share_billed"]
#     sell_fee_per = lambda p: fees_for_one_share(price=p, side="sell")["total_per_share_billed"]

#     # Work on sorted copy
#     df = df.sort_index().copy()

#     # Seed prev_pnl from prior day's last bid if available; else today's first bid
#     # prior_bid = float(_last_bid) if _last_bid else float(df["bid"].iloc[0])
#     _last_pnl = _last_cash + _last_position * _last_bid #prior_bid

#     # Running state
#     position = _last_position
#     cash = _last_cash
#     buy_cost = sell_cost = 0.0
#     ask_p = bid_p = ask_ts = bid_ts = None
#     pos_buf = []
#     cash_buf = []
#     net_buf = []
#     act_buf = []
#     shar_buf = []
#     trades = []

#     # Determine last bid timestamp based on sellmin_idx
#     if sellmin_idx is None:
#         last_bid_ts = df.index[-1]
#     # elif sellmin_idx >= 0:
#     #     last_bid_ts = df[df.index.time >= sess_start].index[sellmin_idx]
#     else:
#         last_bid_ts = df.index[sellmin_idx]

#     # Iterate bars; execute only given actions
#     for ts, row in df.iterrows():
#         bid = row["bid"]
#         ask = row["ask"]
#         action = "Hold"
#         shares_qty = 0
#         buy_fee = sell_fee = buy_cost = sell_cost = 0.0

#         # BUY branch
#         is_fake_buy = position > 0 and _last_ask > 0 and _last_trail > 0
#         if row["action"] == 1:  # buy
#             max_affordable = int((cash * invest_frac) // (ask + buy_fee_per(ask)))
#             shares_qty = max(1, int(max_affordable * row["buy_weight"]))
#             # if shares_qty <= 0:
#             #     action = "Hold"
#             # else:
#             position += shares_qty
#             action = "Buy"
#             ask_p = ask
#             ask_ts = ts
#             buy_cost = ask_p * shares_qty
#             buy_fee = buy_fee_per(ask) * shares_qty
#             cash -= buy_cost + buy_fee
#             # Preserve existing _last_ask semantics
#             _last_ask = ask_p if _last_ask == 0.0 and sellmin_idx is None else _last_ask
#         elif is_fake_buy:  # carried trade with open position
#             ask_p = float(_last_ask)
#             ask_ts = df.index[0]  # placeholder; prior-day ts not persisted
#             _last_ask = 0.0  # reset

#         # SELL branch
#         is_fake_sell = (ts == last_bid_ts) and _last_ask > 0
#         if position > 0 and (row["action"] == -1 or is_fake_sell):
#             shares_qty = max(1, int(position * row["sell_weight"]))
#             if is_fake_sell:
#                 action = "Carried"
#                 bid_p = ask_p = _last_ask  # zero return for carried trade not closed yet
#             else:
#                 sell_cost = bid * shares_qty
#                 sell_fee = sell_fee_per(bid) * shares_qty
#                 position = max(0, position - shares_qty)
#                 action = "Sell"
#                 bid_p = bid
#                 cash += sell_cost - sell_fee
#                 if position == 0:
#                     _last_ask = 0.0
#             bid_ts = ts

#         pos_amount = position * bid
#         tot_pnl = cash + pos_amount

#         # Record trade if executed
#         if ask_ts and bid_ts and shares_qty > 0 and action != "Hold":
#             trades.append(
#                 (
#                     (ask_ts, bid_ts),
#                     (ask_p, bid_p),
#                     (buy_fee, sell_fee),
#                     (buy_cost, sell_cost),
#                     action,
#                     shares_qty,
#                     position,
#                     pos_amount,
#                     cash,
#                     tot_pnl,
#                 )
#             )
#             if action == "Sell":
#                 ask_p = bid_p
#                 ask_ts = bid_ts
#             else:  # Buy/Carried
#                 bid_p = ask_p
#                 bid_ts = ask_ts

#         # snapshots: MUST run once per row
#         pos_buf.append(position)
#         shar_buf.append(shares_qty)
#         cash_buf.append(cash)
#         net_buf.append(tot_pnl)
#         act_buf.append(action)

#     # Build sim dataframe
#     df_sim = df.assign(Position=pos_buf, Cash=cash_buf, NetValue=net_buf, Action=act_buf, Shares=shar_buf)

#     # Persist end-of-day state
#     _last_position, _last_cash = float(position), float(cash)
#     _last_bid = float(df["bid"].iloc[-1])  # save close bid for next day

#     perf = _format_perf(
#         df=df,
#         trades=trades,
#         shares=shares,
#         buy_fee_per=buy_fee_per,
#         sell_fee_per=sell_fee_per,
#         sess_start=sess_start,
#         # prev_pnl=prev_pnl,  # strategy prior-day pnl
#     )
#     updated[day] = (df_sim, trades, perf)
#     return updated


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
    buy_fee_per = lambda p: fees_for_one_share(price=p, side="buy")["total_per_share_billed"]
    sell_fee_per = lambda p: fees_for_one_share(price=p, side="sell")["total_per_share_billed"]
    
    position = int(_last_position)
    cash = _last_cash
    buy_cost = sell_cost = 0.0
    pos_buf, cash_buf, pnl_buf, act_buf, shar_buf = [], [], [], [], []
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
            shares_max = int((cash * invest_frac) // (ask + buy_fee_per(ask)))
            shares_qty = max(1, int(shares_max * row["buy_weight"]))
            position += shares_qty
            buy_cost = ask * shares_qty
            buy_fee = buy_fee_per(ask) * shares_qty
            cash -= buy_cost + buy_fee

        # SELL branch
        if position > 0 and row["action"] == -1:
            action = "Sell"
            shares_qty = max(1, int(position * row["sell_weight"]))
            position = max(0, position - shares_qty)
            sell_cost = bid * shares_qty
            sell_fee = sell_fee_per(bid) * shares_qty
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
        shar_buf.append(shares_qty)
        cash_buf.append(cash)
        pnl_buf.append(tot_pnl)
        act_buf.append(action)

    df_sim = df.assign(Position=pos_buf, Cash=cash_buf, Pnl=pnl_buf, Action=act_buf, Shares=shar_buf)
    
    df_sim, perf = _format_perf(
        df=df_sim,
        trades=trades,
        shares=shares,
        buy_fee_per=buy_fee_per,
        sell_fee_per=sell_fee_per,
        sess_start=sess_start,
    )

    # Persist end-of-day state
    _last_position = position
    _last_cash = cash
    
    updated[day] = (df_sim, trades, perf)
    return updated

