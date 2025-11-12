from libs import plots, params

from typing import Sequence, List, Tuple, Optional, Union, Dict
import gc 
import os
import io
import shutil
import tempfile
import atexit
import copy
import re
import math
import time

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output, display

from captum.attr import IntegratedGradients
from tqdm.auto import tqdm
import ta
from ta.volume import OnBalanceVolumeIndicator

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler, QuantileTransformer, OrdinalEncoder
from sklearn.decomposition import PCA

from scipy.stats import spearmanr, skew, kurtosis, ks_2samp


##########################################################################################################


# def create_features(
#     df: pd.DataFrame,
#     mult_feats_win:   float = 1.0,
#     sma_short:        int   = 14,
#     sma_long:         int   = 28,
#     rsi_window:       int   = 14,
#     macd_fast:        int   = 12,
#     macd_slow:        int   = 26,
#     macd_sig:         int   = 9,
#     atr_window:       int   = 14,
#     bb_window:        int   = 20,
#     obv_sma:          int   = 14,
#     vwap_window:      int   = 14,
#     vol_spike_window: int   = 14
# ) -> pd.DataFrame:
#     """
#     Compute raw OHLCV features and classic indicators on 1-min bars,
#     scaling every lookback window by mult_feats_win.

#     Steps:
#       1) Scale all indicator windows via mult_feats_win (including MACD).
#       2) Compute simple returns and log-returns.
#       3) Candlestick geometry: body, %body, upper/lower shadows, range_pct.
#       4) RSI, MACD line/signal/diff, SMA(short/long) + percent deviations.
#       5) ATR + atr_pct, Bollinger Bands + width, +DI/−DI/ADX.
#       6) OBV: level, diff, pct_change, rolling SMA.
#       7) VWAP + deviation from price.
#       8) vol_spike ratio on its own window.
#       9) Calendar columns: hour, day_of_week, month.
#      10) Drop only the initial rows until every indicator has data.
#     """
#     df = df.copy()
#     df.index = pd.to_datetime(df.index)
#     eps = 1e-8

#     # Helper to scale windows
#     def WM(x: int) -> int:
#         return max(1, int(round(x * mult_feats_win)))

#     # 1) scaled window lengths
#     w_sma_s     = WM(sma_short)
#     w_sma_l     = WM(sma_long)
#     w_rsi       = WM(rsi_window)
#     w_macd_f    = WM(macd_fast)
#     w_macd_s    = WM(macd_slow)
#     w_macd_sig  = WM(macd_sig)
#     w_atr       = WM(atr_window)
#     w_bb        = WM(bb_window)
#     w_obv       = WM(obv_sma)
#     w_vwap      = WM(vwap_window)
#     w_vol_spike = WM(vol_spike_window)

#     # Base columns
#     cols_in = ["open","high","low","close","volume", params.label_col]
#     out     = df[cols_in].copy()
#     c, o, h, l = out["close"], out["open"], out["high"], out["low"]

#     # 2) Returns
#     out["ret"]     = c.pct_change()
#     out["log_ret"] = np.log(c + eps).diff()
    
#     # 2.1) Rate‐of‐Change over sma_short window
#     roc_window = w_sma_s
#     out[f"roc_{roc_window}"] = c.diff(roc_window) / (c.shift(roc_window) + eps)

#     # 3) Candlestick geometry
#     out["body"]       = c - o
#     out["body_pct"]   = (c - o) / (o + eps)
#     out["upper_shad"] = h - out[["open","close"]].max(axis=1)
#     out["lower_shad"] = out[["open","close"]].min(axis=1) - l
#     out["range_pct"]  = (h - l) / (c + eps)

#     # 4) RSI
#     out[f"rsi_{w_rsi}"] = ta.momentum.RSIIndicator(close=c, window=w_rsi).rsi()

#     # 5) MACD (all windows scaled)
#     macd = ta.trend.MACD(
#         close=c,
#         window_fast=w_macd_f,
#         window_slow=w_macd_s,
#         window_sign=w_macd_sig
#     )
#     out[f"macd_line_{w_macd_f}_{w_macd_s}_{w_macd_sig}"]   = macd.macd()
#     out[f"macd_signal_{w_macd_f}_{w_macd_s}_{w_macd_sig}"] = macd.macd_signal()
#     out[f"macd_diff_{w_macd_f}_{w_macd_s}_{w_macd_sig}"]   = macd.macd_diff()

#     # 6) SMA + percent deviation
#     sma_s = c.rolling(w_sma_s, min_periods=1).mean()
#     sma_l = c.rolling(w_sma_l, min_periods=1).mean()
#     out[f"sma_{w_sma_s}"]     = sma_s
#     out[f"sma_{w_sma_l}"]     = sma_l
#     out[f"sma_pct_{w_sma_s}"] = (c - sma_s) / (sma_s + eps)
#     out[f"sma_pct_{w_sma_l}"] = (c - sma_l) / (sma_l + eps)

#     # 7) ATR + ATR percent
#     atr = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=w_atr)
#     out[f"atr_{w_atr}"]     = atr.average_true_range()
#     out[f"atr_pct_{w_atr}"] = atr.average_true_range() / (c + eps)

#     # 8) Bollinger Bands + width
#     bb    = ta.volatility.BollingerBands(close=c, window=w_bb, window_dev=2)
#     lband = bb.bollinger_lband()
#     hband = bb.bollinger_hband()
#     mavg  = bb.bollinger_mavg()
#     out[f"bb_lband_{w_bb}"] = lband
#     out[f"bb_hband_{w_bb}"] = hband
#     out[f"bb_w_{w_bb}"]     = (hband - lband) / (mavg + eps)

#     # 9) +DI, −DI, ADX
#     adx = ta.trend.ADXIndicator(high=h, low=l, close=c, window=w_atr)
#     out[f"plus_di_{w_atr}"]  = adx.adx_pos()
#     out[f"minus_di_{w_atr}"] = adx.adx_neg()
#     out[f"adx_{w_atr}"]      = adx.adx()

#     # 10) OBV: level, diff, pct_change, SMA
#     obv = ta.volume.OnBalanceVolumeIndicator(close=c, volume=out["volume"])
#     out["obv"]              = obv.on_balance_volume()
#     out[f"obv_diff_{w_obv}"] = out["obv"].diff()
#     out[f"obv_pct_{w_obv}"]  = out["obv"].pct_change()
#     out[f"obv_sma_{w_obv}"]  = out["obv"].rolling(w_obv, min_periods=1).mean()

#     # 11) VWAP + deviation
#     vwap = ta.volume.VolumeWeightedAveragePrice(
#         high=h, low=l, close=c, volume=out["volume"], window=w_vwap
#     )
#     out[f"vwap_{w_vwap}"]     = vwap.volume_weighted_average_price()
#     out[f"vwap_dev_{w_vwap}"] = (c - out[f"vwap_{w_vwap}"]) / (out[f"vwap_{w_vwap}"] + eps)

#     # 12) vol_spike ratio (dedicated window)
#     vol_roll = out["volume"].rolling(w_vol_spike, min_periods=1).mean()
#     out[f"vol_spike_{w_vol_spike}"] = out["volume"] / (vol_roll + eps)

#     # 13) calendar
#     out["hour"]        = df.index.hour
#     out["day_of_week"] = df.index.dayofweek
#     out["month"]       = df.index.month

#     # 14) drop only until every column is first valid
#     first_valid = out.apply(lambda series: series.first_valid_index()).max()
#     return out.loc[first_valid:].copy()


def standard_indicators(
    df: pd.DataFrame,
    mult_inds_win = None,   # float or list[float]; if None default [0.5,1.0,2.0]
    sma_short:        int   = 14,
    sma_long:         int   = 28,
    rsi_window:       int   = 14,
    macd_fast:        int   = 12,
    macd_slow:        int   = 26,
    macd_sig:         int   = 9,
    atr_window:       int   = 14,
    bb_window:        int   = 20,
    obv_sma:          int   = 14,
    vwap_window:      int   = 14,
    vol_spike_window: int   = 14,
    label_col:        str   = "signal"
) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    eps = 1e-9

    # multipliers
    if mult_inds_win is None:
        mults = [0.5, 1.0, 2.0]
    elif isinstance(mult_inds_win, (int, float)):
        mults = [float(mult_inds_win)]
    else:
        mults = list(mult_inds_win)

    bases = {
        "sma_short": sma_short,
        "sma_long":  sma_long,
        "rsi":       rsi_window,
        "macd_f":    macd_fast,
        "macd_s":    macd_slow,
        "macd_sig":  macd_sig,
        "atr":       atr_window,
        "bb":        bb_window,
        "obv":       obv_sma,
        "vwap":      vwap_window,
        "vol_spike": vol_spike_window
    }

    def W(base, m): return max(1, int(round(base * m)))

    # collect window set for each indicator type
    windows = set()
    for m in mults:
        for k, b in bases.items():
            windows.add((k, W(b, m)))
    for k, b in bases.items():
        windows.add((k, W(b, 1.0)))

    # inputs and basic safe casts
    cols_in = ["open", "high", "low", "close", "volume"]
    if label_col in df.columns:
        cols_in.append(label_col)
    base = df[cols_in].copy()
    o, h, l, c = base["open"], base["high"], base["low"], base["close"]

    new = {}

    # returns
    new["ret"] = c.pct_change()
    new["log_ret"] = np.log(c + eps).diff()

    # multi-horizon returns & lags
    horizons = sorted({1,5,15,60} | {W(bases["sma_short"], m) for m in mults})
    for H in horizons:
        new[f"ret_{H}"] = c.pct_change(H)
    new["lag1_ret"] = new["ret"].shift(1)
    new["lag2_ret"] = new["ret"].shift(2)
    new["lag3_ret"] = new["ret"].shift(3)

    # ROC (using sma windows as representative price windows)
    sma_windows = sorted({w for (k, w) in windows if k.startswith("sma")})
    for w in sma_windows:
        new[f"roc_{w}"] = c.diff(w) / (c.shift(w) + eps)

    # candlestick geometry
    new["body"] = c - o
    new["body_pct"] = (c - o) / (o + eps)
    new["upper_shad"] = h - np.maximum(o, c)
    new["lower_shad"] = np.minimum(o, c) - l
    new["range_pct"] = (h - l) / (c + eps)

    # SMA / EMA levels + percent deviations + ema_dev
    sma_set = sorted({w for (k, w) in windows if k.startswith("sma")})
    for w in sma_set:
        s = c.rolling(w, min_periods=1).mean()
        new[f"sma_{w}"] = s
        new[f"sma_pct_{w}"] = (c - s) / (s + eps)
        e = c.ewm(span=w, adjust=False).mean()
        new[f"ema_{w}"] = e
        new[f"ema_dev_{w}"] = (c - e) / (e + eps)

    # RSI
    rsi_windows = sorted({w for (k, w) in windows if k == "rsi"})
    for w in rsi_windows:
        new[f"rsi_{w}"] = ta.momentum.RSIIndicator(close=c, window=w).rsi()

    # MACD variants (ensure fast < slow)
    macd_specs = sorted({(W(bases["macd_f"], m), W(bases["macd_s"], m), W(bases["macd_sig"], m)) for m in mults} |
                        {(bases["macd_f"], bases["macd_s"], bases["macd_sig"])})
    for f_w, s_w, sig_w in macd_specs:
        if f_w >= s_w:
            s_w = f_w + 1
        macd = ta.trend.MACD(close=c, window_fast=f_w, window_slow=s_w, window_sign=sig_w)
        suf = f"{f_w}_{s_w}_{sig_w}"
        new[f"macd_line_{suf}"] = macd.macd()
        new[f"macd_signal_{suf}"] = macd.macd_signal()
        new[f"macd_diff_{suf}"] = macd.macd_diff()

    # ATR and ATR percent
    atr_windows = sorted({w for (k, w) in windows if k == "atr"})
    for w in atr_windows:
        atr = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=w)
        new[f"atr_{w}"] = atr.average_true_range()
        new[f"atr_pct_{w}"] = atr.average_true_range() / (c + eps)

    # Bollinger bands + width
    bb_windows = sorted({w for (k, w) in windows if k == "bb"})
    for w in bb_windows:
        bb = ta.volatility.BollingerBands(close=c, window=w, window_dev=2)
        lband = bb.bollinger_lband()
        hband = bb.bollinger_hband()
        mavg = bb.bollinger_mavg()
        new[f"bb_lband_{w}"] = lband
        new[f"bb_hband_{w}"] = hband
        new[f"bb_w_{w}"] = (hband - lband) / (mavg + eps)

    # DI / ADX (use ATR windows set)
    di_windows = sorted({w for (k, w) in windows if k == "atr"})
    for w in di_windows:
        adx = ta.trend.ADXIndicator(high=h, low=l, close=c, window=w)
        new[f"plus_di_{w}"] = adx.adx_pos()
        new[f"minus_di_{w}"] = adx.adx_neg()
        new[f"adx_{w}"] = adx.adx()

    # cumulative OBV (classic)
    new["obv"] = OnBalanceVolumeIndicator(close=c, volume=base["volume"]).on_balance_volume()
    
    # windows for obv-derived features (use diff for windowed change; keep sma and z)
    obv_windows = sorted({w for (k, w) in windows if k == "obv"})
    for w in obv_windows:
        # windowed change: diff over w steps (classic)
        new[f"obv_diff_{w}"] = new["obv"].diff(w)
        # normalize change by recent price level for comparability
        price_roll_mean = c.rolling(w, min_periods=1).mean().abs()
        new[f"obv_pct_{w}"] = new[f"obv_diff_{w}"] / (price_roll_mean + eps)
        # rolling simple moving average of cumulative OBV (classic) and a z-like dev
        new[f"obv_sma_{w}"] = new["obv"].rolling(w, min_periods=1).mean()
        new[f"obv_z_{w}"] = (new["obv"] - new[f"obv_sma_{w}"]) / (new[f"obv_sma_{w}"].rolling(w, min_periods=1).std().fillna(0.0) + eps)

    # VWAP and percent dev + rolling zscore on percent
    vwap_w = W(bases["vwap"], 1.0)
    vwap = ta.volume.VolumeWeightedAveragePrice(high=h, low=l, close=c, volume=base["volume"], window=vwap_w)
    new[f"vwap_{vwap_w}"] = vwap.volume_weighted_average_price()
    # percent deviation (×100) and local z on the percent series
    new[f"vwap_dev_pct_{vwap_w}"] = 100.0 * (c - new[f"vwap_{vwap_w}"]) / (new[f"vwap_{vwap_w}"] + eps)
    x_pct = new[f"vwap_dev_pct_{vwap_w}"]
    new[f"z_vwap_dev_{vwap_w}"] = (x_pct - x_pct.rolling(obv_sma, min_periods=1).mean()) / (x_pct.rolling(obv_sma, min_periods=1).std() + eps)

    # volume spike and z
    vol_w = W(bases["vol_spike"], 1.0)
    vol_roll = base["volume"].rolling(vol_w, min_periods=1).mean()
    new[f"vol_spike_{vol_w}"] = base["volume"] / (vol_roll + eps)
    v_mu = base["volume"].rolling(vol_w, min_periods=1).mean()
    v_sigma = base["volume"].rolling(vol_w, min_periods=1).std()
    new[f"vol_z_{vol_w}"] = (base["volume"] - v_mu) / (v_sigma + eps)

    # rolling realized volatility
    vol_windows = sorted({W(bases["sma_short"], m) for m in mults} | {W(bases["sma_long"], 1.0)})
    for w in vol_windows:
        new[f"ret_std_{w}"] = new["ret"].rolling(w, min_periods=1).std()

    # rolling extrema and normalized distances
    ext_windows = sorted({W(bases["sma_long"], m) for m in mults})
    for w in ext_windows:
        new[f"rolling_max_close_{w}"] = base["close"].rolling(w, min_periods=1).max()
        new[f"rolling_min_close_{w}"] = base["close"].rolling(w, min_periods=1).min()
        new[f"dist_high_{w}"] = (new[f"rolling_max_close_{w}"] - c) / (c + eps)
        new[f"dist_low_{w}"] = (c - new[f"rolling_min_close_{w}"]) / (c + eps)

    # z-scores for ATR and BB width
    for w in atr_windows:
        x = new[f"atr_pct_{w}"]
        new[f"z_atr_{w}"] = (x - x.rolling(w, min_periods=1).mean()) / (x.rolling(w, min_periods=1).std() + eps)
    for w in bb_windows:
        x = new[f"bb_w_{w}"]
        new[f"z_bbw_{w}"] = (x - x.rolling(w, min_periods=1).mean()) / (x.rolling(w, min_periods=1).std() + eps)

    # calendar/cyclical
    new["hour"] = df.index.hour
    new["hour_sin"] = np.sin(2 * np.pi * new["hour"] / 24.0)
    new["hour_cos"] = np.cos(2 * np.pi * new["hour"] / 24.0)
    new["day_of_week"] = df.index.dayofweek
    new["month"] = df.index.month

    out = pd.concat([base, pd.DataFrame(new, index=base.index)], axis=1)
    return out.copy()


##########################################################################################################


def engineered_indicators(
    df: pd.DataFrame,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
    adx_thr: float = 20.0,
    mult_w: int = 14,
    eps: float = 1e-9
) -> pd.DataFrame:
    """
    Build engineered, stationary signals from a base-indicators DataFrame.

    Behavior
    - Dynamically discovers available base columns (sma_, ema_, rsi_, macd_diff_, bb_*, adx_, plus_di_, minus_di_,
      obv_*, vwap_dev_, atr_pct_, ret_*, etc.).
    - Produces only engineered columns that can be computed from available bases.
    - Returns a DataFrame with only engineered features, dtype=float64, NaNs filled with 0.0.
    - Designed to be called after basic_indicators and before pruning/scaling.

    Engineered features produced (when bases exist)
    - eng_ma, eng_macd
    - eng_bb, eng_bb_mid
    - eng_rsi
    - eng_adx
    - eng_obv
    - eng_atr_div, z_eng_atr
    - eng_sma_short, eng_sma_long
    - eng_vwap, z_vwap_dev
    - z_bb_w, z_obv_diff
    - mom_sum_H, mom_std_H for H in [1,5,15,60] if ret_H present
    - eng_ema_cross_up, eng_ema_cross_down
    """
    out = pd.DataFrame(index=df.index)
    produced: List[str] = []

    # helpers
    def find_prefix(prefix: str):
        return next((c for c in df.columns if c.startswith(prefix)), None)

    def collect_sma_pair():
        sma_cols = [c for c in df.columns if re.match(r"^sma_\d+$", c)]
        if not sma_cols:
            return None, None
        # sort by window size
        def w(c):
            m = re.search(r"_(\d+)$", c)
            return int(m.group(1)) if m else 10**9
        sma_cols = sorted(sma_cols, key=w)
        if len(sma_cols) >= 2:
            return sma_cols[0], sma_cols[1]
        return sma_cols[0], sma_cols[0]

    close = df.get("close", pd.Series(index=df.index, dtype=float)).astype(float)

    # discover bases
    sma_s, sma_l = collect_sma_pair()
    # macd_diff: prefer any macd_diff_*
    macd_diff = find_prefix("macd_diff_")
    bb_l = find_prefix("bb_lband_"); bb_h = find_prefix("bb_hband_"); bb_w = find_prefix("bb_w_")
    rsi_col = find_prefix("rsi_")
    plus_di = find_prefix("plus_di_"); minus_di = find_prefix("minus_di_"); adx_col = find_prefix("adx_")
    obv_diff = find_prefix("obv_diff_"); obv_sma = find_prefix("obv_sma_")
    vwap_dev = find_prefix("vwap_dev_") or find_prefix("vwap_")
    atr_pct = find_prefix("atr_pct_")

    # MA ratio (short vs long)
    if sma_s and sma_l:
        out["eng_ma"] = (df[sma_s].astype(float) - df[sma_l].astype(float)) / (df[sma_l].astype(float) + eps)
        produced.append("eng_ma")

    # MACD normalized by long SMA (if available)
    if macd_diff and sma_l:
        out["eng_macd"] = df[macd_diff].astype(float) / (df[sma_l].astype(float) + eps)
        produced.append("eng_macd")

    # Bollinger deviation signals
    if bb_l and bb_h and bb_w:
        lo = df[bb_l].astype(float); hi = df[bb_h].astype(float); bw = df[bb_w].astype(float)
        dev = np.where(close < lo, lo - close, np.where(close > hi, close - hi, 0.0))
        out["eng_bb"] = pd.Series(dev, index=df.index) / (bw + eps)
        out["eng_bb_mid"] = ((lo + hi) / 2 - close) / (close + eps)
        produced.extend(["eng_bb", "eng_bb_mid"])

    # RSI signals (distance beyond thresholds)
    if rsi_col:
        rv = df[rsi_col].astype(float)
        low_d = np.clip((rsi_low - rv), 0, None) / 100.0
        high_d = np.clip((rv - rsi_high), 0, None) / 100.0
        out["eng_rsi"] = np.where(rv < rsi_low, low_d, np.where(rv > rsi_high, high_d, 0.0))
        produced.append("eng_rsi")

    # ADX directional signal (signed strength)
    if plus_di and minus_di and adx_col:
        di_diff = df[plus_di].astype(float) - df[minus_di].astype(float)
        diff_abs = di_diff.abs() / 100.0
        ex = np.clip((df[adx_col].astype(float) - adx_thr) / 100.0, 0, None)
        out["eng_adx"] = np.sign(di_diff) * diff_abs * ex
        produced.append("eng_adx")

    # OBV per-volume normalized
    if obv_diff and obv_sma:
        out["eng_obv"] = df[obv_diff].astype(float) / (df[obv_sma].astype(float) + eps)
        produced.append("eng_obv")

    # ATR-derived signals
    if atr_pct:
        ratio = df[atr_pct].astype(float)
        rm = ratio.rolling(mult_w, min_periods=1).mean()
        out["eng_atr_div"] = (ratio - rm) * 10_000
        out["z_eng_atr"] = (ratio - rm) / (ratio.rolling(mult_w, min_periods=1).std() + eps)
        produced.extend(["eng_atr_div", "z_eng_atr"])

    # simple percent-distance to moving averages
    if sma_s:
        out["eng_sma_short"] = (df[sma_s].astype(float) - close) / (close + eps)
        produced.append("eng_sma_short")
    if sma_l:
        out["eng_sma_long"] = (df[sma_l].astype(float) - close) / (close + eps)
        produced.append("eng_sma_long")

    # VWAP percent deviation and local zscore 
    if vwap_dev:
        vwap_base = df[vwap_dev].astype(float)
        eng_vwap_pct = 100.0 * (vwap_base - close) / (close + eps)   # percent dev
        out["eng_vwap"] = eng_vwap_pct
        out["z_vwap_dev"] = (eng_vwap_pct - eng_vwap_pct.rolling(mult_w, min_periods=1).mean()) / (
                             eng_vwap_pct.rolling(mult_w, min_periods=1).std().fillna(0.0) + eps)
        produced.extend(["eng_vwap", "z_vwap_dev"])

    # BB width z-score
    if bb_w:
        x = df[bb_w].astype(float)
        out["z_bb_w"] = (x - x.rolling(mult_w, min_periods=1).mean()) / (x.rolling(mult_w, min_periods=1).std() + eps)
        produced.append("z_bb_w")

    # OBV diff z-score vs its SMA
    if obv_diff and obv_sma:
        x = df[obv_diff].astype(float)
        base_sma = df[obv_sma].astype(float)
        out["z_obv_diff"] = (x - base_sma) / (base_sma.rolling(mult_w, min_periods=1).std() + eps)
        produced.append("z_obv_diff")

    # momentum aggregates (sum and std) for available horizons
    for H in [1, 5, 15, 60]:
        ret_col = f"ret_{H}"
        if ret_col in df.columns:
            out[f"mom_sum_{H}"] = df[ret_col].rolling(H, min_periods=1).sum()
            out[f"mom_std_{H}"] = df[ret_col].rolling(H, min_periods=1).std()
            produced.extend([f"mom_sum_{H}", f"mom_std_{H}"])

    # EMA crossover binary flags (use corresponding ema_N columns if present)
    try:
        s_w = int(re.search(r"_(\d+)$", sma_s).group(1)) if sma_s else None
        l_w = int(re.search(r"_(\d+)$", sma_l).group(1)) if sma_l else None
    except Exception:
        s_w = l_w = None
    s_ema_col = f"ema_{s_w}" if s_w else None
    l_ema_col = f"ema_{l_w}" if l_w else None
    if s_ema_col in df.columns and l_ema_col in df.columns:
        out["eng_ema_cross_up"] = (df[s_ema_col].astype(float) > df[l_ema_col].astype(float)).astype(float)
        out["eng_ema_cross_down"] = (df[s_ema_col].astype(float) < df[l_ema_col].astype(float)).astype(float)
        produced.extend(["eng_ema_cross_up", "eng_ema_cross_down"])

    # finalize: keep only produced features, ensure numeric dtype and fill NaNs
    produced = [p for p in produced if p in out.columns]
    for col in produced:
        out[col] = out[col].astype(float).fillna(0.0)

    return out[produced].copy()


##########################################################################################################



def prune_and_percentiles(
    df_unsc: pd.DataFrame,
    train_prop: float = 0.7,
    pct_shift_thresh: float = 0.20,
    abs_shift_thresh: float = 1e-4,
    frac_outside_thresh: float = 0.05,
    ks_pval_thresh: float = 0.001,
    min_train_samples: int = 30,
    fail_count_to_drop: int = 2
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Prune unstable numeric features and compute per-feature winsorize percentiles (pct_pair)
    derived only from bulk_ratio = IQR / (q99 - q01).

    Behavior and outputs (keeps the same names/logic as the previous version):
    - Splits df_unsc into contiguous TRAIN/VAL/TEST using train_prop (VAL and TEST share the remainder).
    - For each numeric feature computes TRAIN percentiles q01,q25,q75,q99, IQR, full_span and bulk_ratio.
    - Runs stability checks (median shifts, frac outside, KS, constant-on-train) and assigns status OK or DROP.
    - Adds a tail-count guard: if q01/q99 are supported by fewer than 3 TRAIN samples, they are replaced with q05/q95
      before computing full_span (prevents single extreme points from dominating full_span).
    - Maps bulk_ratio -> pct_pair with a simple clipped linear rule. Mapping constants are chosen to be slightly
      more forgiving than before: br_max=0.15 and wide endpoint = (0.1, 99.9).
    - Returns (df_pruned, to_drop, diag) where diag contains q01,q25,q75,q99,full_span,iqr,bulk_ratio,
      frac_val_out, frac_te_out, status, reason and pct_pair=(pct_low,pct_high).

    Notes:
    - All existing variable names and the overall structure are preserved for minimal, clear change.
    - The function uses tqdm for progress over features.
    """
    N = len(df_unsc)
    if N == 0:
        return df_unsc.copy(), [], pd.DataFrame()

    # contiguous TRAIN/VAL/TEST splits
    n_tr = int(N * train_prop)
    n_rem = N - n_tr
    n_val = n_rem // 2
    tr_slice = slice(0, n_tr)
    val_slice = slice(n_tr, n_tr + n_val)
    te_slice = slice(n_tr + n_val, N)

    raw_tr = df_unsc.iloc[tr_slice].replace([np.inf, -np.inf], np.nan)
    raw_val = df_unsc.iloc[val_slice].replace([np.inf, -np.inf], np.nan)
    raw_te = df_unsc.iloc[te_slice].replace([np.inf, -np.inf], np.nan)

    numeric_cols = [c for c in df_unsc.columns if pd.api.types.is_numeric_dtype(df_unsc[c])]
    rows = []

    def is_calendar(col: str) -> bool:
        return bool(re.search(r"^(hour|day_of_week|month)$", col))
    def is_binary_like(col: str, tr_series: pd.Series) -> bool:
        vals = tr_series.dropna().unique()
        if len(vals) > 0 and set(np.unique(vals)).issubset({0, 1}):
            return True
        if col.startswith("eng_") and (col.endswith("_up") or col.endswith("_down")):
            return True
        return False
    def is_zscore_like(col: str) -> bool:
        return bool(re.search(r"^(z_|z_.*|z_bbw_|z_atr_|z_obv_|z_eng_)", col)) or col.startswith("ema_dev_") or col.endswith("_pct")
    def safe_denominator(tr_arr: np.ndarray) -> float:
        if len(tr_arr) == 0:
            return 1e-6
        med = float(np.nanmedian(tr_arr))
        mad = float(np.nanmedian(np.abs(tr_arr - med)))
        std = float(np.nanstd(tr_arr))
        return max(abs(med), mad, std, 1e-6)

    for feat in tqdm(numeric_cols, desc="prune_and_percentiles", unit="feat"):
        tr = raw_tr[feat].dropna()
        vl = raw_val[feat].dropna()
        te = raw_te[feat].dropna()

        med_tr = float(np.nanmedian(tr)) if len(tr) else np.nan
        med_val = float(np.nanmedian(vl)) if len(vl) else np.nan
        med_te = float(np.nanmedian(te)) if len(te) else np.nan

        denom = safe_denominator(tr.to_numpy()) if len(tr) else 1e-6
        pct_shift_val = abs(med_val - med_tr) / denom if not np.isnan(med_tr) else 0.0
        pct_shift_te  = abs(med_te - med_tr) / denom if not np.isnan(med_tr) else 0.0
        abs_shift_val = abs(med_val - med_tr)
        abs_shift_te  = abs(med_te - med_tr)

        skip_pct = is_calendar(feat) or is_binary_like(feat, tr) or is_zscore_like(feat)
        if not skip_pct:
            pct_shift_flag = (pct_shift_val > pct_shift_thresh) or (pct_shift_te > pct_shift_thresh)
        else:
            pct_shift_flag = (abs_shift_val > abs_shift_thresh) or (abs_shift_te > abs_shift_thresh)

        # TRAIN percentiles (standard)
        q01 = np.nanpercentile(tr, 1) if len(tr) else np.nan
        q25 = np.nanpercentile(tr, 25) if len(tr) else np.nan
        q75 = np.nanpercentile(tr, 75) if len(tr) else np.nan
        q99 = np.nanpercentile(tr, 99) if len(tr) else np.nan

        # --- Minimal robustification: tail-count guard
        # If q01/q99 are supported by very few TRAIN points, use q05/q95 instead
        min_tail_count = 3
        if len(tr):
            try:
                if (tr <= q01).sum() < min_tail_count:
                    q01 = np.nanpercentile(tr, 5)
                if (tr >= q99).sum() < min_tail_count:
                    q99 = np.nanpercentile(tr, 95)
            except Exception:
                # keep original q01/q99 if unusual error occurs
                pass

        full_span = q99 - q01 if not (np.isnan(q99) or np.isnan(q01)) else np.nan
        iqr = q75 - q25 if not (np.isnan(q75) or np.isnan(q25)) else np.nan
        bulk_ratio = (iqr / (full_span + 1e-12)) if not np.isnan(iqr) and not np.isnan(full_span) else np.nan

        def frac_outside(series: pd.Series) -> float:
            if len(series) == 0 or np.isnan(q01) or np.isnan(q99):
                return 0.0
            s = series.dropna()
            if len(s) == 0:
                return 0.0
            return float(((s < q01).sum() + (s > q99).sum()) / len(s))
        frac_val_out = frac_outside(vl)
        frac_te_out = frac_outside(te)
        frac_out_flag = (frac_val_out > frac_outside_thresh) or (frac_te_out > frac_outside_thresh)

        ks_p = np.nan
        ks_flag = False
        try:
            if len(tr) >= min_train_samples and len(te) >= min_train_samples:
                _, ks_p = ks_2samp(tr, te)
                ks_flag = (ks_p < ks_pval_thresh)
        except Exception:
            ks_p = np.nan
            ks_flag = False

        const_on_train = False
        if len(tr) and np.isfinite(tr.min()) and np.isfinite(tr.max()):
            const_on_train = (abs(tr.max() - tr.min()) < 1e-12)
        const_flag = const_on_train

        fail_reasons = []
        fail_count = 0
        if pct_shift_flag:
            fail_count += 1
            fail_reasons.append(f"pct_shift_val={pct_shift_val:.3f};te={pct_shift_te:.3f}")
        if frac_out_flag:
            fail_count += 1
            fail_reasons.append(f"frac_out_val={frac_val_out:.3f};te={frac_te_out:.3f}")
        if ks_flag:
            fail_count += 1
            fail_reasons.append(f"ks_p={ks_p:.4g}")
        if const_flag:
            fail_count += 10
            fail_reasons.append("constant_on_train")

        status = "OK" if fail_count < fail_count_to_drop else "DROP"

        rows.append({
            "feature": feat,
            "q01": q01, "q25": q25, "q75": q75, "q99": q99,
            "full_span": full_span, "iqr": iqr, "bulk_ratio": bulk_ratio,
            "med_tr": med_tr, "med_val": med_val, "med_te": med_te,
            "pct_shift_val": pct_shift_val, "pct_shift_te": pct_shift_te,
            "abs_shift_val": abs_shift_val, "abs_shift_te": abs_shift_te,
            "frac_val_out": frac_val_out, "frac_te_out": frac_te_out,
            "ks_p": ks_p,
            "nan_mask_train": raw_tr[feat].isna().any(),
            "const_on_train": const_on_train,
            "fail_count": fail_count,
            "status": status,
            "reason": "; ".join(fail_reasons) if fail_reasons else ""
        })

    diag = pd.DataFrame(rows).set_index("feature").sort_values(["status", "fail_count"], ascending=[True, False])

    to_drop = diag[diag["status"] == "DROP"].index.tolist()
    df_pruned = df_unsc.drop(columns=to_drop, errors="ignore")

    # SIMPLE mapping from bulk_ratio only -> pct_pair (minimal, friendlier constants)
    br_min = 0.0
    br_max = 0.15               # slightly larger threshold: treat bulk_ratio >= 0.15 as "wide"
    pct_low_narrow, pct_high_narrow = 20.0, 80.0   # narrow endpoint (strong clipping)
    pct_low_wide, pct_high_wide = 0.1, 99.9        # wide endpoint (very loose clipping)

    def map_bulk_to_pcts_from_bulkratio(br, span):
        # safe fallback when span invalid
        if span is None or np.isnan(span) or span == 0:
            return (0.5, 99.5)
        br_val = float(br) if not np.isnan(br) else br_max
        br_clip = float(np.clip(br_val, br_min, br_max))
        t = (br_clip - br_min) / (br_max - br_min) if (br_max - br_min) > 0 else 1.0
        pct_low = pct_low_narrow + (pct_low_wide - pct_low_narrow) * t
        pct_high = pct_high_narrow + (pct_high_wide - pct_high_narrow) * t
        pct_low = float(np.clip(pct_low, 0.0, 49.9))
        pct_high = float(np.clip(pct_high, 50.1, 100.0))
        if pct_low >= pct_high:
            return (0.5, 99.5)
        return (round(pct_low, 3), round(pct_high, 3))

    pct_pairs = []
    for feat in diag.index:
        br = diag.at[feat, "bulk_ratio"] if "bulk_ratio" in diag.columns else np.nan
        span = diag.at[feat, "full_span"] if "full_span" in diag.columns else np.nan
        pct_pairs.append(map_bulk_to_pcts_from_bulkratio(br, span))

    diag = diag.copy()
    diag["pct_pair"] = pct_pairs

    return df_pruned, to_drop, diag



#########################################################################################################



def scaling_with_percentiles(
    df: pd.DataFrame,
    label_col: str,
    diag: pd.DataFrame,
    train_prop: float = 0.7,
    val_prop: float = 0.15,
    winsorize: bool = True
) -> pd.DataFrame:
    """
    Scale numeric features to [0,1] using TRAIN-based winsorize + MinMax with per-feature pct pairs.

    Inputs
      - df: full DataFrame (indexed by timestamp), numeric features + label_col
      - label_col: name of the label column to exclude from scaling
      - diag: diagnostics DataFrame produced by prune_and_percentiles; must contain a column
              "pct_pair" with (pct_low, pct_high) per feature (indexed by feature name).
      - train_prop, val_prop: contiguous TRAIN/VAL/TEST split fractions (train_prop + val_prop < 1)
      - winsorize: whether to apply TRAIN-based clipping before MinMax

    Behavior
      - For each numeric feature (excluding label_col) reads per-feature (pct_low,pct_high) from diag["pct_pair"].
      - If pct_pair missing or invalid, falls back to (0.5, 99.5).
      - Computes TRAIN percentiles per feature, clips TRAIN/VAL/TEST values to those cutpoints,
        computes TRAIN clipped min/max and then applies MinMax scaling (day-by-day).
      - Preserves NaNs and non-numeric columns unchanged.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.replace([np.inf, -np.inf], np.nan)

    # numeric feature columns (cast once) and exclude label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c != label_col]
    df[numeric_cols] = df[numeric_cols].astype(np.float64)

    if not feat_cols:
        return df

    # splits
    N = len(df)
    n_tr = int(N * train_prop)
    n_val = int(N * val_prop)
    if n_tr + n_val >= N:
        raise ValueError("train_prop + val_prop must sum < 1.0")
    tr_slice = slice(0, n_tr)
    val_slice = slice(n_tr, n_tr + n_val)
    te_slice = slice(n_tr + n_val, N)
    df_tr = df.iloc[tr_slice].copy()

    # build per-feature pct_low/pct_high from diag["pct_pair"] (with fallback)
    pct_low_map: Dict[str, float] = {}
    pct_high_map: Dict[str, float] = {}
    for c in feat_cols:
        try:
            pair = diag.at[c, "pct_pair"]
            if isinstance(pair, (tuple, list)) and len(pair) >= 2:
                lo, hi = float(pair[0]), float(pair[1])
            else:
                lo, hi = 0.5, 99.5
        except Exception:
            lo, hi = 0.5, 99.5
        # sanitize
        lo = float(np.clip(lo, 0.0, 49.9))
        hi = float(np.clip(hi, 50.1, 100.0))
        if lo >= hi:
            lo, hi = 0.5, 99.5
        pct_low_map[c] = lo
        pct_high_map[c] = hi

    # compute TRAIN percentiles per-feature for winsorize (with progress)
    clip_low: Dict[str, float] = {}
    clip_high: Dict[str, float] = {}
    if winsorize:
        for c in feat_cols:
            arr = df_tr[c].to_numpy(dtype=float)
            clip_low[c] = np.nanpercentile(arr, pct_low_map[c]) if arr.size else np.nan
            clip_high[c] = np.nanpercentile(arr, pct_high_map[c]) if arr.size else np.nan

    # helper to apply clipping on numpy array
    def apply_clip_array(arr: np.ndarray, c: str) -> np.ndarray:
        if not winsorize:
            return arr
        low = clip_low[c]; high = clip_high[c]
        a = np.copy(arr)
        mask = np.isnan(a)
        a = np.clip(a, low, high)
        a[mask] = np.nan
        return a

    # compute clipped TRAIN min/max and spans (with progress)
    col_min: Dict[str, float] = {}
    col_max: Dict[str, float] = {}
    span: Dict[str, float] = {}
    for c in feat_cols:
        col = df_tr[c].to_numpy(dtype=float)
        colc = apply_clip_array(col, c) if winsorize else col
        col_min[c] = np.nanmin(colc) if np.any(~np.isnan(colc)) else np.nan
        col_max[c] = np.nanmax(colc) if np.any(~np.isnan(colc)) else np.nan
        if np.isnan(col_min[c]) or np.isnan(col_max[c]) or (col_max[c] - col_min[c]) == 0.0:
            span[c] = 1.0
        else:
            span[c] = col_max[c] - col_min[c]

    # transform helper for a block (preserves NaNs)
    def transform_block(block: pd.DataFrame) -> pd.DataFrame:
        out = block.copy()
        for c in feat_cols:
            arr = out[c].to_numpy(dtype=float)
            if winsorize:
                low = clip_low[c]; high = clip_high[c]
                mask = np.isnan(arr)
                arr = np.clip(arr, low, high)
                arr[mask] = np.nan
            mn = col_min[c]; sp = span[c]
            res = (arr - mn) / sp
            res = np.where(np.isnan(arr), np.nan, np.clip(res, 0.0, 1.0))
            out[c] = res
        return out[feat_cols].astype(np.float64)

    # apply transforms day-by-day for each split (with progress over days)
    def transform_split(split_df: pd.DataFrame, desc: str) -> pd.DataFrame:
        parts = []
        days = split_df.index.normalize().unique()
        for _, block in tqdm(split_df.groupby(split_df.index.normalize()),
                              desc=desc, unit="day", total=len(days)):
            parts.append(transform_block(block))
        if not parts:
            return pd.DataFrame(columns=feat_cols, index=split_df.index)
        return pd.concat(parts).reindex(split_df.index)

    tr_scaled = transform_split(df.iloc[tr_slice], "scale train days")
    v_scaled = transform_split(df.iloc[val_slice], "scale val days")
    te_scaled = transform_split(df.iloc[te_slice], "scale test days")

    # reassemble final DataFrame
    df_tr.loc[:, feat_cols] = tr_scaled
    df_v = df.iloc[n_tr:n_tr + n_val].copy()
    df_te = df.iloc[n_tr + n_val:].copy()
    df_v.loc[:, feat_cols] = v_scaled
    df_te.loc[:, feat_cols] = te_scaled

    df_all = pd.concat([df_tr, df_v, df_te]).sort_index()

    return df_all


#########################################################################################################


def scaling_diagnostics(
    df_unscaled: pd.DataFrame,
    df_scaled: pd.DataFrame,
    train_prop: float = 0.7,
    clip_thresh: float = 0.01
) -> pd.DataFrame:
    """
    Compact per-feature scaling diagnostics (run after scaling).
    Returns a DataFrame indexed by feature with medians, IQRs (unscaled),
    clip rates on VAL/TEST (scaled), NaN-mask preservation check, and a suggested action.
    """
    df_u = df_unscaled.copy()
    df_s = df_scaled.copy()
    common = [c for c in df_u.columns if c in df_s.columns and pd.api.types.is_numeric_dtype(df_u[c])]
    if not common:
        return pd.DataFrame()

    N = len(df_u)
    n_tr = int(N * train_prop)
    n_rem = N - n_tr
    n_val = n_rem // 2
    tr_slice = slice(0, n_tr)
    val_slice = slice(n_tr, n_tr + n_val)
    te_slice = slice(n_tr + n_val, N)

    tr_u = df_u.iloc[tr_slice]; val_u = df_u.iloc[val_slice]; te_u = df_u.iloc[te_slice]
    tr_s = df_s.iloc[tr_slice]; val_s = df_s.iloc[val_slice]; te_s = df_s.iloc[te_slice]

    def clip_rate(series):
        if series is None or len(series) == 0:
            return 0.0
        s = series.dropna()
        if len(s) == 0:
            return 0.0
        return float(((s == 0.0).sum() + (s == 1.0).sum()) / len(s))

    rows = []
    for feat in common:
        trx = tr_u[feat].dropna()
        vx  = val_u[feat].dropna()
        tex = te_u[feat].dropna()

        # basic raw stats (unscaled)
        med_tr = float(trx.median()) if len(trx) else np.nan
        med_val = float(vx.median()) if len(vx) else np.nan
        med_te = float(tex.median()) if len(tex) else np.nan
        iqr_tr = float(trx.quantile(0.75) - trx.quantile(0.25)) if len(trx) else np.nan
        iqr_val = float(vx.quantile(0.75) - vx.quantile(0.25)) if len(vx) else np.nan
        iqr_te = float(tex.quantile(0.75) - tex.quantile(0.25)) if len(tex) else np.nan

        # scaled clip rates (exact 0 or 1) on VAL/TEST
        clip_val = clip_rate(val_s.get(feat))
        clip_te  = clip_rate(te_s.get(feat))

        # NaN-mask preservation on TRAIN
        nan_mask_ok = True
        if feat in tr_s.columns:
            raw_mask = tr_u[feat].isna()
            scaled_mask = tr_s[feat].isna()
            nan_mask_ok = raw_mask.equals(scaled_mask)

        # scaled-train constant check
        const_on_train_scaled = False
        s_tr = tr_s[feat].dropna()
        if len(s_tr) > 0:
            const_on_train_scaled = np.isclose(s_tr.max(), s_tr.min())

        # suggested action (simple ordered rules)
        action = "ok"
        reasons = []

        if not nan_mask_ok:
            action = "inspect_feature"
            reasons.append("nan_mask_changed")

        if (clip_val > clip_thresh) or (clip_te > clip_thresh):
            if action == "ok":
                action = "tune_winsorize"
            reasons.append(f"clip_val={clip_val:.3f};clip_te={clip_te:.3f}")

        if (not np.isnan(iqr_tr) and iqr_tr == 0.0) and ((not np.isnan(iqr_val) and iqr_val > 0.0) or (not np.isnan(iqr_te) and iqr_te > 0.0)):
            if action == "ok":
                action = "try_robust_scaler"
            reasons.append("iqr_tr_zero")

        # IQR inflation: VAL or TEST >> TRAIN
        if (not np.isnan(iqr_tr) and iqr_tr > 0.0):
            if (not np.isnan(iqr_val) and (iqr_val / (iqr_tr + 1e-12) > 5.0)) or (not np.isnan(iqr_te) and (iqr_te / (iqr_tr + 1e-12) > 5.0)):
                if action == "ok":
                    action = "try_robust_scaler"
                reasons.append("iqr_inflation")

        if const_on_train_scaled and (not (len(trx) and np.nanstd(trx) == 0.0)):
            if action == "ok":
                action = "inspect_feature"
            reasons.append("scaled_train_constant")

        rows.append({
            "feature": feat,
            "med_tr": med_tr, "med_val": med_val, "med_te": med_te,
            "iqr_tr": iqr_tr, "iqr_val": iqr_val, "iqr_te": iqr_te,
            "clip_val": clip_val, "clip_te": clip_te,
            "nan_mask_ok": nan_mask_ok,
            "const_on_train_scaled": const_on_train_scaled,
            "suggested_action": action,
            "reason": "; ".join(reasons)
        })

    return pd.DataFrame(rows).set_index("feature")




#########################################################################################################


def prune_features_by_variance_and_correlation(
    X_all: pd.DataFrame,
    y: pd.Series,
    min_std: float = 1e-6,
    max_corr: float = 0.9
) -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame]:
    """
    1) Remove features with std < min_std
    2) From the remaining features, drop members of high-correlation groups
       (absolute correlation > max_corr). For each correlated group keep the
       feature that has the highest absolute correlation with the target y.

    Returns:
      kept_after_std      : list of feature names kept after the std filter
      kept_after_correlation : list of feature names kept after correlation pruning
      corr_full           : correlation matrix of features after std filter (DataFrame)
      corr_pruned         : correlation matrix of features after correlation pruning (DataFrame)
    """
    # 1) Filter by standard deviation
    feature_stds = X_all.std(axis=0, ddof=0)
    kept_after_std = feature_stds[feature_stds >= min_std].index.tolist()
    dropped_low_variance = feature_stds[feature_stds < min_std].index.tolist()

    X_var = X_all.loc[:, kept_after_std].copy()

    # 2) Correlation matrix (absolute) for the features kept after std filter
    corr_before = X_var.corr().abs()

    # 3) Upper triangle mask (exclude diagonal)
    mask_upper = np.triu(np.ones(corr_before.shape), k=1).astype(bool)
    upper_tri = corr_before.where(mask_upper)

    # 4) Prune highly correlated features
    to_drop: Set[str] = set()
    for col in upper_tri.columns:
        # find features (rows) correlated above threshold with this column
        high_corr = upper_tri.index[upper_tri[col] > max_corr].tolist()
        if high_corr:
            group = [col] + high_corr
            # choose best feature in this group by absolute correlation with target y
            # align indices to be safe
            corr_with_target = X_var.loc[:, group].corrwith(y).abs()
            best_feat = corr_with_target.idxmax()
            to_drop.update(set(group) - {best_feat})

    kept_after_corr = [f for f in kept_after_std if f not in to_drop]

    # 5) Correlation matrix after pruning (absolute)
    corr_after = X_var.loc[:, kept_after_corr].corr().abs()

    # Logging summary
    print("Dropped low-variance features:", dropped_low_variance)
    print("Dropped high-correlation features:", sorted(list(to_drop)))
    print("Kept after std filter (count):", len(kept_after_std))
    print("Kept after correlation pruning (count):", len(kept_after_corr))

    return kept_after_std, kept_after_corr, corr_before, corr_after

##########################


def plot_correlation_before_after(
    corr_full: pd.DataFrame,
    corr_pruned: pd.DataFrame,
    figsize: Tuple[int, int] = (18, 8),
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm"
) -> None:
    """
    Plot two side-by-side heatmaps:
      - corr_full  : correlation matrix before pruning
      - corr_pruned: correlation matrix after pruning

    Both inputs are absolute-valued correlation DataFrames (0..1).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(
        corr_full,
        ax=axes[0],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar_kws={"shrink": 0.7}
    )
    axes[0].set_title("Correlation Before Pruning")

    sns.heatmap(
        corr_pruned,
        ax=axes[1],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar_kws={"shrink": 0.7}
    )
    axes[1].set_title("Correlation After Pruning")

    plt.tight_layout()
    plt.show()


#########################################################################################################


def predict_windows(model, X_np, batch_size=1024, device=params.device):
    """
    Run the trained model on a numpy array of windows and return 1D predictions.

    Parameters
    - model: trained torch.nn.Module already moved to device and in eval mode
    - X_np: numpy array of shape (N, L, F) with dtype convertible to float32
    - batch_size: inference batch size

    Returns
    - preds: numpy array shape (N,) with model predictions
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[i:i + batch_size].astype("float32")).to(device)
            out = model(xb)
            p = out[0].detach().cpu().numpy().reshape(-1)
            preds.append(p)
    return np.concatenate(preds, axis=0)


###################################### 


def live_display_importances(imp_series, features, label, method,
                             batch=1, pause=0.02, threshold=None):
    """
    Incrementally display importances from a completed pd.Series.

    - imp_series: pd.Series indexed by feature name (values = importance)
    - features: ordered list of features to reveal (must match index names)
    - label: target name for title
    - method: text for title/legend
    - batch: how many features to reveal per UI update
    - pause: sleep between updates (UI breathing)
    - threshold: optional vertical line value
    """
    imp_series = pd.Series(imp_series).reindex(features).fillna(0.0)
    revealed = {}
    for i in range(0, len(features), batch):
        for f in features[i:i+batch]:
            revealed[f] = float(imp_series[f])
        clear_output(wait=True)
        s = pd.Series(revealed).sort_values()
        plt.figure(figsize=(6, max(3, len(s)*0.18)))
        colors = sns.color_palette("vlag", len(s))
        plt.barh(s.index, s.values, color=colors)
        if threshold is not None:
            plt.axvline(threshold, color="gray", linestyle="--")
            if method.lower().startswith("corr"):
                plt.axvline(-threshold, color="gray", linestyle="--")
        plt.title(f"{method} Importance (partial) for {label}")
        plt.xlabel("Importance")
        plt.tight_layout()
        display(plt.gcf())
        plt.close()
        time.sleep(pause)
    return imp_series.sort_values(ascending=False)


##################################


def update_feature_importances(fi_dict, importance_type, values: pd.Series):
    """
    fi_dict: master dict
    importance_type: one of "corr","mi","perm","shap","lasso"
    values: pd.Series indexed by feature name
    """
    for feat, val in values.items():
        if feat in fi_dict:
            fi_dict[feat][importance_type] = val


