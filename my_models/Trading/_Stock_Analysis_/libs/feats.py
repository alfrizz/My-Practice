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
from sklearn.neighbors import NearestNeighbors

from scipy.stats import spearmanr, skew, kurtosis, ks_2samp, pearsonr


##########################################################################################################



# def standard_indicators(
#     df: pd.DataFrame,
#     mult_inds_win = None,   # float or list[float]; if None default [0.5,1.0,2.0]
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
#     vol_spike_window: int   = 14,
#     label_col:        str   = "signal"
# ) -> pd.DataFrame:
#     df = df.copy()
#     df.index = pd.to_datetime(df.index)
#     eps = 1e-9

#     # multipliers
#     if mult_inds_win is None:
#         mults = [0.5, 1.0, 2.0]
#     elif isinstance(mult_inds_win, (int, float)):
#         mults = [float(mult_inds_win)]
#     else:
#         mults = list(mult_inds_win)

#     bases = {
#         "sma_short": sma_short,
#         "sma_long":  sma_long,
#         "rsi":       rsi_window,
#         "macd_f":    macd_fast,
#         "macd_s":    macd_slow,
#         "macd_sig":  macd_sig,
#         "atr":       atr_window,
#         "bb":        bb_window,
#         "obv":       obv_sma,
#         "vwap":      vwap_window,
#         "vol_spike": vol_spike_window
#     }

#     def W(base, m): return max(1, int(round(base * m)))

#     # collect window set for each indicator type
#     windows = set()
#     for m in mults:
#         for k, b in bases.items():
#             windows.add((k, W(b, m)))
#     for k, b in bases.items():
#         windows.add((k, W(b, 1.0)))

#     # inputs and basic safe casts
#     cols_in = ["open", "high", "low", "close", "volume"]
#     if label_col in df.columns:
#         cols_in.append(label_col)
#     base = df[cols_in].copy()
#     o, h, l, c = base["open"], base["high"], base["low"], base["close"]

#     new = {}

#     # returns
#     new["ret"] = c.pct_change()
#     new["log_ret"] = np.log(c + eps).diff()

#     # multi-horizon returns & lags
#     horizons = sorted({1,5,15,60} | {W(bases["sma_short"], m) for m in mults})
#     for H in horizons:
#         new[f"ret_{H}"] = c.pct_change(H)
#     new["lag1_ret"] = new["ret"].shift(1)
#     new["lag2_ret"] = new["ret"].shift(2)
#     new["lag3_ret"] = new["ret"].shift(3)

#     # ROC (using sma windows as representative price windows)
#     sma_windows = sorted({w for (k, w) in windows if k.startswith("sma")})
#     for w in sma_windows:
#         new[f"roc_{w}"] = c.diff(w) / (c.shift(w) + eps)

#     # candlestick geometry
#     new["body"] = c - o
#     new["body_pct"] = (c - o) / (o + eps)
#     new["upper_shad"] = h - np.maximum(o, c)
#     new["lower_shad"] = np.minimum(o, c) - l
#     new["range_pct"] = (h - l) / (c + eps)

#     # SMA / EMA levels + percent deviations + ema_dev
#     sma_set = sorted({w for (k, w) in windows if k.startswith("sma")})
#     for w in sma_set:
#         s = c.rolling(w, min_periods=1).mean()
#         new[f"sma_{w}"] = s
#         new[f"sma_pct_{w}"] = (c - s) / (s + eps)
#         e = c.ewm(span=w, adjust=False).mean()
#         new[f"ema_{w}"] = e
#         new[f"ema_dev_{w}"] = (c - e) / (e + eps)

#     # RSI
#     rsi_windows = sorted({w for (k, w) in windows if k == "rsi"})
#     for w in rsi_windows:
#         new[f"rsi_{w}"] = ta.momentum.RSIIndicator(close=c, window=w).rsi()

#     # MACD variants (ensure fast < slow)
#     macd_specs = sorted({(W(bases["macd_f"], m), W(bases["macd_s"], m), W(bases["macd_sig"], m)) for m in mults} |
#                         {(bases["macd_f"], bases["macd_s"], bases["macd_sig"])})
#     for f_w, s_w, sig_w in macd_specs:
#         if f_w >= s_w:
#             s_w = f_w + 1
#         macd = ta.trend.MACD(close=c, window_fast=f_w, window_slow=s_w, window_sign=sig_w)
#         suf = f"{f_w}_{s_w}_{sig_w}"
#         new[f"macd_line_{suf}"] = macd.macd()
#         new[f"macd_signal_{suf}"] = macd.macd_signal()
#         new[f"macd_diff_{suf}"] = macd.macd_diff()

#     # ATR and ATR percent
#     atr_windows = sorted({w for (k, w) in windows if k == "atr"})
#     for w in atr_windows:
#         atr = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=w)
#         new[f"atr_{w}"] = atr.average_true_range()
#         new[f"atr_pct_{w}"] = atr.average_true_range() / (c + eps)

#     # Bollinger bands + width
#     bb_windows = sorted({w for (k, w) in windows if k == "bb"})
#     for w in bb_windows:
#         bb = ta.volatility.BollingerBands(close=c, window=w, window_dev=2)
#         lband = bb.bollinger_lband()
#         hband = bb.bollinger_hband()
#         mavg = bb.bollinger_mavg()
#         new[f"bb_lband_{w}"] = lband
#         new[f"bb_hband_{w}"] = hband
#         new[f"bb_w_{w}"] = (hband - lband) / (mavg + eps)

#     # DI / ADX (use ATR windows set)
#     di_windows = sorted({w for (k, w) in windows if k == "atr"})
#     for w in di_windows:
#         adx = ta.trend.ADXIndicator(high=h, low=l, close=c, window=w)
#         new[f"plus_di_{w}"] = adx.adx_pos()
#         new[f"minus_di_{w}"] = adx.adx_neg()
#         new[f"adx_{w}"] = adx.adx()

#     # cumulative OBV (classic)
#     new["obv"] = OnBalanceVolumeIndicator(close=c, volume=base["volume"]).on_balance_volume()
    
#     # windows for obv-derived features (use diff for windowed change; keep sma and z)
#     obv_windows = sorted({w for (k, w) in windows if k == "obv"})
#     for w in obv_windows:
#         # windowed change: diff over w steps (classic)
#         new[f"obv_diff_{w}"] = new["obv"].diff(w)
#         # normalize change by recent price level for comparability
#         price_roll_mean = c.rolling(w, min_periods=1).mean().abs()
#         new[f"obv_pct_{w}"] = new[f"obv_diff_{w}"] / (price_roll_mean + eps)
#         # rolling simple moving average of cumulative OBV (classic) and a z-like dev
#         new[f"obv_sma_{w}"] = new["obv"].rolling(w, min_periods=1).mean()
#         new[f"obv_z_{w}"] = (new["obv"] - new[f"obv_sma_{w}"]) / (new[f"obv_sma_{w}"].rolling(w, min_periods=1).std().fillna(0.0) + eps)

#     # VWAP and percent dev + rolling zscore on percent
#     vwap_w = W(bases["vwap"], 1.0)
#     vwap = ta.volume.VolumeWeightedAveragePrice(high=h, low=l, close=c, volume=base["volume"], window=vwap_w)
#     new[f"vwap_{vwap_w}"] = vwap.volume_weighted_average_price()
#     # percent deviation (×100) and local z on the percent series
#     new[f"vwap_dev_pct_{vwap_w}"] = 100.0 * (c - new[f"vwap_{vwap_w}"]) / (new[f"vwap_{vwap_w}"] + eps)
#     x_pct = new[f"vwap_dev_pct_{vwap_w}"]
#     new[f"z_vwap_dev_{vwap_w}"] = (x_pct - x_pct.rolling(obv_sma, min_periods=1).mean()) / (x_pct.rolling(obv_sma, min_periods=1).std() + eps)

#     # volume spike and z
#     vol_w = W(bases["vol_spike"], 1.0)
#     vol_roll = base["volume"].rolling(vol_w, min_periods=1).mean()
#     new[f"vol_spike_{vol_w}"] = base["volume"] / (vol_roll + eps)
#     v_mu = base["volume"].rolling(vol_w, min_periods=1).mean()
#     v_sigma = base["volume"].rolling(vol_w, min_periods=1).std()
#     new[f"vol_z_{vol_w}"] = (base["volume"] - v_mu) / (v_sigma + eps)

#     # rolling realized volatility
#     vol_windows = sorted({W(bases["sma_short"], m) for m in mults} | {W(bases["sma_long"], 1.0)})
#     for w in vol_windows:
#         new[f"ret_std_{w}"] = new["ret"].rolling(w, min_periods=1).std()

#     # rolling extrema and normalized distances
#     ext_windows = sorted({W(bases["sma_long"], m) for m in mults})
#     for w in ext_windows:
#         new[f"rolling_max_close_{w}"] = base["close"].rolling(w, min_periods=1).max()
#         new[f"rolling_min_close_{w}"] = base["close"].rolling(w, min_periods=1).min()
#         new[f"dist_high_{w}"] = (new[f"rolling_max_close_{w}"] - c) / (c + eps)
#         new[f"dist_low_{w}"] = (c - new[f"rolling_min_close_{w}"]) / (c + eps)

#     # z-scores for ATR and BB width
#     for w in atr_windows:
#         x = new[f"atr_pct_{w}"]
#         new[f"z_atr_{w}"] = (x - x.rolling(w, min_periods=1).mean()) / (x.rolling(w, min_periods=1).std() + eps)
#     for w in bb_windows:
#         x = new[f"bb_w_{w}"]
#         new[f"z_bbw_{w}"] = (x - x.rolling(w, min_periods=1).mean()) / (x.rolling(w, min_periods=1).std() + eps)

#     # calendar/cyclical
#     new["hour"] = df.index.hour
#     new["hour_sin"] = np.sin(2 * np.pi * new["hour"] / 24.0)
#     new["hour_cos"] = np.cos(2 * np.pi * new["hour"] / 24.0)
#     new["day_of_week"] = df.index.dayofweek
#     new["month"] = df.index.month

#     out = pd.concat([base, pd.DataFrame(new, index=base.index)], axis=1)
#     return out.copy()



def standard_indicators(
    df: pd.DataFrame,
    mult_inds_win = None,
    sma_short: int = 14,
    sma_long: int = 28,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_sig: int = 9,
    atr_window: int = 14,
    bb_window: int = 20,
    obv_sma: int = 14,
    vwap_window: int = 14,
    vol_spike_window: int = 14,
    label_col: str = "signal",
    eps: float = 1e-9,
    small_factor: float = 1e-3,
    z_std_floor_factor: float = 1e-3
) -> pd.DataFrame:
    """
    Compute base technical indicators (returns a DataFrame with original price/volume
    columns plus indicators). Minimal, causal, and numerically safe:
    - All rolling ops use past + present by default; if you need purely past-only windows
      use .shift(1).rolling(...) upstream before calling this function.
    - Denominators and rolling stds are floored to avoid divide-by-zero or extreme ratios.
    - No hidden fallback percentiles or silent rewrites of diagnostics here.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if mult_inds_win is None:
        mults = [0.5, 1.0, 2.0]
    elif isinstance(mult_inds_win, (int, float)):
        mults = [float(mult_inds_win)]
    else:
        mults = list(mult_inds_win)

    bases = {
        "sma_short": sma_short, "sma_long": sma_long, "rsi": rsi_window,
        "macd_f": macd_fast, "macd_s": macd_slow, "macd_sig": macd_sig,
        "atr": atr_window, "bb": bb_window, "obv": obv_sma,
        "vwap": vwap_window, "vol_spike": vol_spike_window
    }

    def W(base, m): return max(1, int(round(base * m)))

    windows = set()
    for m in mults:
        for k, b in bases.items():
            windows.add((k, W(b, m)))
    for k, b in bases.items():
        windows.add((k, W(b, 1.0)))

    cols_in = ["open", "high", "low", "close", "volume"]
    if label_col in df.columns:
        cols_in.append(label_col)
    base = df[cols_in].copy()
    o, h, l, c = base["open"], base["high"], base["low"], base["close"]

    # stable helpers
    def _safe_denom_series(divisor: pd.Series, window: Optional[int] = None) -> np.ndarray:
        s = divisor.astype(float)
        s_arr = s.to_numpy(dtype=float)
        if window:
            med = s.rolling(window, min_periods=1).median().abs().ffill().fillna(0.0).to_numpy(dtype=float)
            floor_arr = np.maximum(med * small_factor, eps)
        else:
            floor_val = max(eps, abs(np.nanmedian(s_arr)) * small_factor)
            floor_arr = np.full_like(s_arr, floor_val, dtype=float)
        finite_nonzero = np.isfinite(s_arr) & (np.abs(s_arr) > 0.0)
        denom_safe = np.where(finite_nonzero, np.sign(s_arr) * np.maximum(np.abs(s_arr), floor_arr), floor_arr)
        return denom_safe

    def _std_floor(series: pd.Series, window: int) -> pd.Series:
        rstd = series.rolling(window, min_periods=1).std()
        global_std = np.nanstd(series.to_numpy(dtype=float))
        floor = max(eps, abs(global_std) * z_std_floor_factor)
        return rstd.fillna(floor).replace(0.0, floor)

    new = {}

    # returns
    new["ret"] = c.pct_change()
    new["log_ret"] = np.log(c + eps).diff()

    # multi-horizon returns & lags
    horizons = sorted({1, 5, 15, 60} | {W(bases["sma_short"], m) for m in mults})
    for H in horizons:
        new[f"ret_{H}"] = c.pct_change(H)
    new["lag1_ret"] = new["ret"].shift(1)
    new["lag2_ret"] = new["ret"].shift(2)
    new["lag3_ret"] = new["ret"].shift(3)

    # SMA / EMA
    sma_set = sorted({w for (k, w) in windows if k.startswith("sma")})
    for w in sma_set:
        s = c.rolling(w, min_periods=1).mean()
        new[f"sma_{w}"] = s
        new[f"sma_pct_{w}"] = (c - s) / (s + eps)
        e = c.ewm(span=w, adjust=False).mean()
        new[f"ema_{w}"] = e
        new[f"ema_dev_{w}"] = (c - e) / (e + eps)

    # ROC as percent-change (skip w==1 because new["ret"] already holds the 1-period pct change)
    sma_windows = sorted({w for (k, w) in windows if k.startswith("sma")})
    for w in sma_windows:
        if w == 1:
            continue
        new[f"roc_{w}"] = c.pct_change(w)

    # candlestick geometry
    new["body"] = c - o
    new["body_pct"] = (c - o) / (o + eps)
    new["upper_shad"] = h - np.maximum(o, c)
    new["lower_shad"] = np.minimum(o, c) - l
    new["range_pct"] = (h - l) / (c + eps)

    # RSI
    rsi_windows = sorted({w for (k, w) in windows if k == "rsi"})
    for w in rsi_windows:
        new[f"rsi_{w}"] = ta.momentum.RSIIndicator(close=c, window=w).rsi()

    # MACD
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

    # ATR
    atr_windows = sorted({w for (k, w) in windows if k == "atr"})
    for w in atr_windows:
        atr = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=w)
        new[f"atr_{w}"] = atr.average_true_range()
        new[f"atr_pct_{w}"] = atr.average_true_range() / (c + eps)

    # Bollinger bands
    bb_windows = sorted({w for (k, w) in windows if k == "bb"})
    for w in bb_windows:
        bb = ta.volatility.BollingerBands(close=c, window=w, window_dev=2)
        lband = bb.bollinger_lband()
        hband = bb.bollinger_hband()
        mavg = bb.bollinger_mavg()
        new[f"bb_lband_{w}"] = lband
        new[f"bb_hband_{w}"] = hband
        new[f"bb_w_{w}"] = (hband - lband) / (mavg + eps)

    # DI / ADX
    di_windows = sorted({w for (k, w) in windows if k == "atr"})
    for w in di_windows:
        adx = ta.trend.ADXIndicator(high=h, low=l, close=c, window=w)
        new[f"plus_di_{w}"] = adx.adx_pos()
        new[f"minus_di_{w}"] = adx.adx_neg()
        new[f"adx_{w}"] = adx.adx()

    # OBV and derived features
    new["obv"] = ta.volume.OnBalanceVolumeIndicator(close=c, volume=base["volume"]).on_balance_volume()
    obv_windows = sorted({w for (k, w) in windows if k == "obv"})
    for w in obv_windows:
        new[f"obv_diff_{w}"] = new["obv"].diff(w)
        price_roll_mean = c.rolling(w, min_periods=1).mean().abs()
        denom_arr = _safe_denom_series(price_roll_mean, window=w)
        # elementwise division safe (denom_arr length equals index length)
        num = new[f"obv_diff_{w}"].to_numpy(dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = num / denom_arr
        new[f"obv_pct_{w}"] = pd.Series(np.where(np.isnan(num), np.nan, pct), index=base.index)
        new[f"obv_sma_{w}"] = new["obv"].rolling(w, min_periods=1).mean()
        base_sma = new[f"obv_sma_{w}"]
        std_eff = _std_floor(base_sma, w)
        new[f"obv_z_{w}"] = (new["obv"] - base_sma) / (std_eff + eps)

    # VWAP
    vwap_w = W(bases["vwap"], 1.0)
    vwap = ta.volume.VolumeWeightedAveragePrice(high=h, low=l, close=c, volume=base["volume"], window=vwap_w)
    new[f"vwap_{vwap_w}"] = vwap.volume_weighted_average_price()
    new[f"vwap_dev_pct_{vwap_w}"] = 100.0 * (c - new[f"vwap_{vwap_w}"]) / (new[f"vwap_{vwap_w}"] + eps)
    x_pct = new[f"vwap_dev_pct_{vwap_w}"]
    new[f"z_vwap_dev_{vwap_w}"] = (x_pct - x_pct.rolling(obv_sma, min_periods=1).mean()) / (x_pct.rolling(obv_sma, min_periods=1).std().fillna(eps) + eps)

    # volume spike
    vol_w = W(bases["vol_spike"], 1.0)
    vol_roll = base["volume"].rolling(vol_w, min_periods=1).mean()
    new[f"vol_spike_{vol_w}"] = base["volume"] / (vol_roll + eps)
    v_mu = base["volume"].rolling(vol_w, min_periods=1).mean()
    v_sigma = base["volume"].rolling(vol_w, min_periods=1).std().fillna(eps).replace(0.0, eps)
    new[f"vol_z_{vol_w}"] = (base["volume"] - v_mu) / (v_sigma + eps)

    # rolling volatility
    vol_windows = sorted({W(bases["sma_short"], m) for m in mults} | {W(bases["sma_long"], 1.0)})
    for w in vol_windows:
        new[f"ret_std_{w}"] = new["ret"].rolling(w, min_periods=1).std()

    # rolling extrema and distances
    ext_windows = sorted({W(bases["sma_long"], m) for m in mults})
    for w in ext_windows:
        new[f"rolling_max_close_{w}"] = base["close"].rolling(w, min_periods=1).max()
        new[f"rolling_min_close_{w}"] = base["close"].rolling(w, min_periods=1).min()
        new[f"dist_high_{w}"] = (new[f"rolling_max_close_{w}"] - c) / (c + eps)
        new[f"dist_low_{w}"] = (c - new[f"rolling_min_close_{w}"]) / (c + eps)

    # z-scores for ATR and BB width
    for w in atr_windows:
        x = new[f"atr_{w}"]
        std_eff = _std_floor(x, w)
        new[f"z_atr_{w}"] = (x - x.rolling(w, min_periods=1).mean()) / (std_eff + eps)
    for w in bb_windows:
        x = new[f"bb_w_{w}"]
        std_eff = _std_floor(x, w)
        new[f"z_bbw_{w}"] = (x - x.rolling(w, min_periods=1).mean()) / (std_eff + eps)

    # calendar
    new["hour"] = df.index.hour
    new["hour_sin"] = np.sin(2 * np.pi * new["hour"] / 24.0)
    new["hour_cos"] = np.cos(2 * np.pi * new["hour"] / 24.0)
    new["day_of_week"] = df.index.dayofweek
    new["month"] = df.index.month

    out = pd.concat([base, pd.DataFrame(new, index=base.index)], axis=1)
    return out.copy()


##########################################################################################################


# def engineered_indicators(
#     df: pd.DataFrame,
#     rsi_low: float = 30.0,
#     rsi_high: float = 70.0,
#     adx_thr: float = 20.0,
#     mult_w: int = 14,
#     eps: float = 1e-9
# ) -> pd.DataFrame:
#     """
#     Build engineered, stationary signals from a base-indicators DataFrame.

#     Behavior
#     - Dynamically discovers available base columns (sma_, ema_, rsi_, macd_diff_, bb_*, adx_, plus_di_, minus_di_,
#       obv_*, vwap_dev_, atr_pct_, ret_*, etc.).
#     - Produces only engineered columns that can be computed from available bases.
#     - Returns a DataFrame with only engineered features, dtype=float64, NaNs filled with 0.0.
#     - Designed to be called after basic_indicators and before pruning/scaling.

#     Engineered features produced (when bases exist)
#     - eng_ma, eng_macd
#     - eng_bb, eng_bb_mid
#     - eng_rsi
#     - eng_adx
#     - eng_obv
#     - eng_atr_div, z_eng_atr
#     - eng_sma_short, eng_sma_long
#     - eng_vwap, z_vwap_dev
#     - z_bb_w, z_obv_diff
#     - mom_sum_H, mom_std_H for H in [1,5,15,60] if ret_H present
#     - eng_ema_cross_up, eng_ema_cross_down
#     """
#     out = pd.DataFrame(index=df.index)
#     produced: List[str] = []

#     # helpers
#     def find_prefix(prefix: str):
#         return next((c for c in df.columns if c.startswith(prefix)), None)

#     def collect_sma_pair():
#         sma_cols = [c for c in df.columns if re.match(r"^sma_\d+$", c)]
#         if not sma_cols:
#             return None, None
#         # sort by window size
#         def w(c):
#             m = re.search(r"_(\d+)$", c)
#             return int(m.group(1)) if m else 10**9
#         sma_cols = sorted(sma_cols, key=w)
#         if len(sma_cols) >= 2:
#             return sma_cols[0], sma_cols[1]
#         return sma_cols[0], sma_cols[0]

#     close = df.get("close", pd.Series(index=df.index, dtype=float)).astype(float)

#     # discover bases
#     sma_s, sma_l = collect_sma_pair()
#     # macd_diff: prefer any macd_diff_*
#     macd_diff = find_prefix("macd_diff_")
#     bb_l = find_prefix("bb_lband_"); bb_h = find_prefix("bb_hband_"); bb_w = find_prefix("bb_w_")
#     rsi_col = find_prefix("rsi_")
#     plus_di = find_prefix("plus_di_"); minus_di = find_prefix("minus_di_"); adx_col = find_prefix("adx_")
#     obv_diff = find_prefix("obv_diff_"); obv_sma = find_prefix("obv_sma_")
#     vwap_dev = find_prefix("vwap_dev_") or find_prefix("vwap_")
#     atr_pct = find_prefix("atr_pct_")

#     # MA ratio (short vs long)
#     if sma_s and sma_l:
#         out["eng_ma"] = (df[sma_s].astype(float) - df[sma_l].astype(float)) / (df[sma_l].astype(float) + eps)
#         produced.append("eng_ma")

#     # MACD normalized by long SMA (if available)
#     if macd_diff and sma_l:
#         out["eng_macd"] = df[macd_diff].astype(float) / (df[sma_l].astype(float) + eps)
#         produced.append("eng_macd")

#     # Bollinger deviation signals
#     if bb_l and bb_h and bb_w:
#         lo = df[bb_l].astype(float); hi = df[bb_h].astype(float); bw = df[bb_w].astype(float)
#         dev = np.where(close < lo, lo - close, np.where(close > hi, close - hi, 0.0))
#         out["eng_bb"] = pd.Series(dev, index=df.index) / (bw + eps)
#         out["eng_bb_mid"] = ((lo + hi) / 2 - close) / (close + eps)
#         produced.extend(["eng_bb", "eng_bb_mid"])

#     # RSI signals (distance beyond thresholds)
#     if rsi_col:
#         rv = df[rsi_col].astype(float)
#         low_d = np.clip((rsi_low - rv), 0, None) / 100.0
#         high_d = np.clip((rv - rsi_high), 0, None) / 100.0
#         out["eng_rsi"] = np.where(rv < rsi_low, low_d, np.where(rv > rsi_high, high_d, 0.0))
#         produced.append("eng_rsi")

#     # ADX directional signal (signed strength)
#     if plus_di and minus_di and adx_col:
#         di_diff = df[plus_di].astype(float) - df[minus_di].astype(float)
#         diff_abs = di_diff.abs() / 100.0
#         ex = np.clip((df[adx_col].astype(float) - adx_thr) / 100.0, 0, None)
#         out["eng_adx"] = np.sign(di_diff) * diff_abs * ex
#         produced.append("eng_adx")

#     # OBV per-volume normalized
#     if obv_diff and obv_sma:
#         out["eng_obv"] = df[obv_diff].astype(float) / (df[obv_sma].astype(float) + eps)
#         produced.append("eng_obv")

#     # ATR-derived signals
#     if atr_pct:
#         ratio = df[atr_pct].astype(float)
#         rm = ratio.rolling(mult_w, min_periods=1).mean()
#         out["eng_atr_div"] = (ratio - rm) * 10_000
#         out["z_eng_atr"] = (ratio - rm) / (ratio.rolling(mult_w, min_periods=1).std() + eps)
#         produced.extend(["eng_atr_div", "z_eng_atr"])

#     # simple percent-distance to moving averages
#     if sma_s:
#         out["eng_sma_short"] = (df[sma_s].astype(float) - close) / (close + eps)
#         produced.append("eng_sma_short")
#     if sma_l:
#         out["eng_sma_long"] = (df[sma_l].astype(float) - close) / (close + eps)
#         produced.append("eng_sma_long")

#     # VWAP percent deviation and local zscore 
#     if vwap_dev:
#         vwap_base = df[vwap_dev].astype(float)
#         eng_vwap_pct = 100.0 * (vwap_base - close) / (close + eps)   # percent dev
#         out["eng_vwap"] = eng_vwap_pct
#         out["z_vwap_dev"] = (eng_vwap_pct - eng_vwap_pct.rolling(mult_w, min_periods=1).mean()) / (
#                              eng_vwap_pct.rolling(mult_w, min_periods=1).std().fillna(0.0) + eps)
#         produced.extend(["eng_vwap", "z_vwap_dev"])

#     # BB width z-score
#     if bb_w:
#         x = df[bb_w].astype(float)
#         out["z_bb_w"] = (x - x.rolling(mult_w, min_periods=1).mean()) / (x.rolling(mult_w, min_periods=1).std() + eps)
#         produced.append("z_bb_w")

#     # OBV diff z-score vs its SMA
#     if obv_diff and obv_sma:
#         x = df[obv_diff].astype(float)
#         base_sma = df[obv_sma].astype(float)
#         out["z_obv_diff"] = (x - base_sma) / (base_sma.rolling(mult_w, min_periods=1).std() + eps)
#         produced.append("z_obv_diff")

#     # momentum aggregates (sum and std) for available horizons
#     for H in [1, 5, 15, 60]:
#         ret_col = f"ret_{H}"
#         if ret_col in df.columns:
#             out[f"mom_sum_{H}"] = df[ret_col].rolling(H, min_periods=1).sum()
#             out[f"mom_std_{H}"] = df[ret_col].rolling(H, min_periods=1).std()
#             produced.extend([f"mom_sum_{H}", f"mom_std_{H}"])

#     # EMA crossover binary flags (use corresponding ema_N columns if present)
#     try:
#         s_w = int(re.search(r"_(\d+)$", sma_s).group(1)) if sma_s else None
#         l_w = int(re.search(r"_(\d+)$", sma_l).group(1)) if sma_l else None
#     except Exception:
#         s_w = l_w = None
#     s_ema_col = f"ema_{s_w}" if s_w else None
#     l_ema_col = f"ema_{l_w}" if l_w else None
#     if s_ema_col in df.columns and l_ema_col in df.columns:
#         out["eng_ema_cross_up"] = (df[s_ema_col].astype(float) > df[l_ema_col].astype(float)).astype(float)
#         out["eng_ema_cross_down"] = (df[s_ema_col].astype(float) < df[l_ema_col].astype(float)).astype(float)
#         produced.extend(["eng_ema_cross_up", "eng_ema_cross_down"])

#     # finalize: keep only produced features, ensure numeric dtype and fill NaNs
#     produced = [p for p in produced if p in out.columns]
#     for col in produced:
#         out[col] = out[col].astype(float).fillna(0.0)

#     return out[produced].copy()



def engineered_indicators(
    df: pd.DataFrame,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
    adx_thr: float = 20.0,
    mult_w: int = 14,
    eps: float = 1e-9,
    fillna_zero: bool = True,
    small_factor: float = 1e-3,
    ratio_clip_abs: float = 1e6,
    z_std_floor_factor: float = 1e-3
) -> pd.DataFrame:
    """
    Build engineered signals from base indicators.
    - Uses only past/present values (no negative shifts, no bfill).
    - Safe denominators and z-score floors guarantee finite outputs.
    - Keeps original names and minimal changes to logic.
    """
    out = pd.DataFrame(index=df.index)
    produced: List[str] = []

    def _find(pref: str) -> Optional[str]:
        return next((c for c in df.columns if c.startswith(pref)), None)

    def _std_floor(series: pd.Series, window: int):
        rstd = series.rolling(window, min_periods=1).std()
        global_std = np.nanstd(series.to_numpy(dtype=float))
        floor = max(eps, abs(global_std) * z_std_floor_factor)
        return rstd.fillna(floor).replace(0.0, floor)

    # helpers and base series
    close = df.get("close", pd.Series(index=df.index, dtype=float)).astype(float)
    sma_s = _find("sma_"); sma_s, sma_l = None, None
    # collect sma pair robustly
    sma_cols = [c for c in df.columns if re.match(r"^sma_\d+$", c)]
    if sma_cols:
        def _w(c): m = re.search(r"_(\d+)$", c); return int(m.group(1)) if m else 10**9
        sma_cols = sorted(sma_cols, key=_w)
        sma_s = sma_cols[0]
        sma_l = sma_cols[1] if len(sma_cols) > 1 else sma_cols[0]

    macd_diff = _find("macd_diff_")
    bb_l = _find("bb_lband_"); bb_h = _find("bb_hband_"); bb_w = _find("bb_w_")
    rsi_col = _find("rsi_")
    plus_di = _find("plus_di_"); minus_di = _find("minus_di_"); adx_col = _find("adx_")
    obv_diff = _find("obv_diff_"); obv_sma = _find("obv_sma_")
    vwap_dev = _find("vwap_dev_") or _find("vwap_")
    atr_pct = _find("atr_pct_")

    # eng_ma
    if sma_s and sma_l:
        out["eng_ma"] = (df[sma_s].astype(float) - df[sma_l].astype(float)) / (df[sma_l].astype(float) + eps)
        produced.append("eng_ma")

    # eng_macd
    if macd_diff and sma_l:
        out["eng_macd"] = df[macd_diff].astype(float) / (df[sma_l].astype(float) + eps)
        produced.append("eng_macd")

    # eng_bb / mid
    if bb_l and bb_h and bb_w:
        lo = df[bb_l].astype(float); hi = df[bb_h].astype(float); bw = df[bb_w].astype(float)
        dev = np.where(close < lo, lo - close, np.where(close > hi, close - hi, 0.0))
        out["eng_bb"] = pd.Series(dev, index=df.index) / (bw + eps)
        out["eng_bb_mid"] = ((lo + hi) / 2 - close) / (close + eps)
        produced.extend(["eng_bb", "eng_bb_mid"])

    # eng_rsi
    if rsi_col:
        rv = df[rsi_col].astype(float)
        low_d = np.clip((rsi_low - rv), 0, None) / 100.0
        high_d = np.clip((rv - rsi_high), 0, None) / 100.0
        out["eng_rsi"] = np.where(rv < rsi_low, low_d, np.where(rv > rsi_high, high_d, 0.0))
        produced.append("eng_rsi")

    # eng_adx
    if plus_di and minus_di and adx_col:
        di_diff = df[plus_di].astype(float) - df[minus_di].astype(float)
        diff_abs = di_diff.abs() / 100.0
        ex = np.clip((df[adx_col].astype(float) - adx_thr) / 100.0, 0, None)
        out["eng_adx"] = np.sign(di_diff) * diff_abs * ex
        produced.append("eng_adx")

    # eng_obv (safe denom + winsorized numerator)
    if obv_diff and obv_sma:
        num = df[obv_diff].astype(float).to_numpy(dtype=float)
        # winsorize numerator
        try:
            lo_cut = np.nanpercentile(num, 0.5) if num.size else np.nan
            hi_cut = np.nanpercentile(num, 99.5) if num.size else np.nan
            num_clipped = np.copy(num)
            mask_num = np.isnan(num_clipped)
            num_clipped = np.clip(num_clipped, lo_cut, hi_cut)
            num_clipped[mask_num] = np.nan
        except Exception:
            num_clipped = num

        den_s = df[obv_sma].astype(float).rolling(window=mult_w, min_periods=1).median().ffill().fillna(0.0)
        den_arr = den_s.to_numpy(dtype=float)
        den_floor = np.maximum(np.abs(den_arr) * small_factor, eps)
        finite_nonzero = np.isfinite(den_arr) & (np.abs(den_arr) > 0.0)
        den_safe = np.where(finite_nonzero, np.sign(den_arr) * np.maximum(np.abs(den_arr), den_floor), den_floor)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = num_clipped / den_safe
        ratio = np.where(np.isnan(num_clipped), np.nan, np.clip(ratio, -ratio_clip_abs, ratio_clip_abs))
        out["eng_obv"] = pd.Series(ratio, index=df.index).astype(float)
        produced.append("eng_obv")

    # eng_atr_div, z_eng_atr
    if atr_pct:
        ratio = df[atr_pct].astype(float)
        rm = ratio.rolling(mult_w, min_periods=1).mean()
        out["eng_atr_div"] = (ratio - rm) * 10_000
        std_r = ratio.rolling(mult_w, min_periods=1).std()
        global_std = np.nanstd(ratio.to_numpy(dtype=float))
        std_floor = max(eps, abs(global_std) * z_std_floor_factor)
        std_eff = std_r.fillna(std_floor).replace(0.0, std_floor)
        out["z_eng_atr"] = (ratio - rm) / (std_eff + eps)
        produced.extend(["eng_atr_div", "z_eng_atr"])

    # eng_sma distances
    if sma_s:
        out["eng_sma_short"] = (df[sma_s].astype(float) - close) / (close + eps)
        produced.append("eng_sma_short")
    if sma_l:
        out["eng_sma_long"] = (df[sma_l].astype(float) - close) / (close + eps)
        produced.append("eng_sma_long")

    # eng_vwap and z
    if vwap_dev:
        vwap_base = df[vwap_dev].astype(float)
        eng_vwap_pct = 100.0 * (vwap_base - close) / (close + eps)
        out["eng_vwap"] = eng_vwap_pct
        znum = eng_vwap_pct
        zm = znum.rolling(mult_w, min_periods=1).mean()
        zs = znum.rolling(mult_w, min_periods=1).std()
        global_std = np.nanstd(znum.to_numpy(dtype=float))
        std_floor = max(eps, abs(global_std) * z_std_floor_factor)
        zs_eff = zs.fillna(std_floor).replace(0.0, std_floor)
        out["z_vwap_dev"] = (znum - zm) / (zs_eff + eps)
        produced.extend(["eng_vwap", "z_vwap_dev"])

    # z_bb_w
    if bb_w:
        x = df[bb_w].astype(float)
        xm = x.rolling(mult_w, min_periods=1).mean()
        xs = x.rolling(mult_w, min_periods=1).std()
        global_std = np.nanstd(x.to_numpy(dtype=float))
        std_floor = max(eps, abs(global_std) * z_std_floor_factor)
        xs_eff = xs.fillna(std_floor).replace(0.0, std_floor)
        out["z_bb_w"] = (x - xm) / (xs_eff + eps)
        produced.append("z_bb_w")

    # z_obv_diff
    if obv_diff and obv_sma:
        x = df[obv_diff].astype(float)
        base_sma = df[obv_sma].astype(float)
        bs_std = base_sma.rolling(mult_w, min_periods=1).std()
        global_std = np.nanstd(base_sma.to_numpy(dtype=float))
        std_floor = max(eps, abs(global_std) * z_std_floor_factor)
        bs_eff = bs_std.fillna(std_floor).replace(0.0, std_floor)
        out["z_obv_diff"] = (x - base_sma) / (bs_eff + eps)
        produced.append("z_obv_diff")

    # momentum aggregates
    for H in [1, 5, 15, 60]:
        ret_col = f"ret_{H}"
        if ret_col in df.columns:
            out[f"mom_sum_{H}"] = df[ret_col].rolling(H, min_periods=1).sum()
            out[f"mom_std_{H}"] = df[ret_col].rolling(H, min_periods=1).std()
            produced.extend([f"mom_sum_{H}", f"mom_std_{H}"])

    # ema cross flags
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

    # finalize
    produced = [p for p in produced if p in out.columns]
    for col in produced:
        out[col] = out[col].astype(float)
        if fillna_zero:
            out[col] = out[col].fillna(0.0)

    return out[produced].copy()



##########################################################################################################


# def prune_and_diag(
#     df_unsc: pd.DataFrame,
#     train_prop: float = 0.7,
#     pct_shift_thresh: float = 0.20,
#     frac_outside_thresh: float = 0.05,
#     ks_pval_thresh: float = 0.001,
#     min_train_samples: int = 30,
#     fail_count_to_drop: int = 2,
#     mad_mul: float = 1.5,
#     rel_width_scale: float = 0.10,
#     gamma: float = 1.0,
#     conc_bins: int = 80,
#     mode_frac_thresh: float = 0.75,
#     zero_frac_thresh: float = 0.90,
#     frac_in_min: float = 0.02,
# ):
#     """
#     Prune features and compute diagnostics used for percentile assignment.

#     - Pruning: runs simple drift/quality checks (median shifts, fraction outside validation/test,
#       KS two-sample, constant-on-train) and sets status DROP/OK.
#     - Diagnostics: computes per-feature diagnostics needed by percentile assignment:
#       score, mode_frac, zero_frac, frac_in, top_bin_share, plus shift/outlier/ks info.
#     - Returns: (df_pruned, to_drop, diag) where diag is a DataFrame indexed by feature.
#     """
#     # internal stability constants
#     zero_tol = 1e-8
#     denom_eps = 1e-6
#     hw_eps = 1e-12
#     const_eps = 1e-12
#     tail_min_count = 3

#     N = len(df_unsc)
#     if N == 0:
#         return df_unsc.copy(), [], pd.DataFrame()

#     n_tr = int(N * train_prop)
#     n_rem = N - n_tr
#     n_val = n_rem // 2
#     tr = df_unsc.iloc[:n_tr].replace([np.inf, -np.inf], np.nan)
#     vl = df_unsc.iloc[n_tr:n_tr + n_val].replace([np.inf, -np.inf], np.nan)
#     te = df_unsc.iloc[n_tr + n_val:].replace([np.inf, -np.inf], np.nan)

#     numeric_cols = [c for c in df_unsc.columns if pd.api.types.is_numeric_dtype(df_unsc[c])]
#     rows = []

#     def _mad(arr: np.ndarray) -> float:
#         if len(arr) == 0:
#             return 0.0
#         med = float(np.nanmedian(arr))
#         return float(np.nanmedian(np.abs(arr - med)))

#     def safe_denominator(arr: np.ndarray) -> float:
#         if len(arr) == 0:
#             return denom_eps
#         med = float(np.nanmedian(arr))
#         mad = float(np.nanmedian(np.abs(arr - med)))
#         std = float(np.nanstd(arr))
#         return max(abs(med), mad, std, denom_eps)

#     def frac_outside(s: pd.Series, q01, q99) -> float:
#         if len(s) == 0 or np.isnan(q01) or np.isnan(q99):
#             return 0.0
#         s2 = s.dropna()
#         if len(s2) == 0:
#             return 0.0
#         return float(((s2 < q01).sum() + (s2 > q99).sum()) / len(s2))

#     for feat in tqdm(numeric_cols, desc="prune_and_diag", unit="feat"):
#         tr_s = tr[feat].dropna()
#         vl_s = vl[feat].dropna()
#         te_s = te[feat].dropna()

#         med_tr = float(np.nanmedian(tr_s)) if len(tr_s) else np.nan
#         med_val = float(np.nanmedian(vl_s)) if len(vl_s) else np.nan
#         med_te = float(np.nanmedian(te_s)) if len(te_s) else np.nan

#         denom = safe_denominator(tr_s.to_numpy()) if len(tr_s) else denom_eps
#         pct_shift_val = abs(med_val - med_tr) / denom if not np.isnan(med_tr) else 0.0
#         pct_shift_te = abs(med_te - med_tr) / denom if not np.isnan(med_tr) else 0.0
#         abs_shift_val = abs(med_val - med_tr)
#         abs_shift_te = abs(med_te - med_tr)

#         # simplified: always compute relative pct_shift_flag
#         pct_shift_flag = (pct_shift_val > pct_shift_thresh) or (pct_shift_te > pct_shift_thresh)

#         q01 = np.nanpercentile(tr_s, 1) if len(tr_s) else np.nan
#         q25 = np.nanpercentile(tr_s, 25) if len(tr_s) else np.nan
#         q75 = np.nanpercentile(tr_s, 75) if len(tr_s) else np.nan
#         q99 = np.nanpercentile(tr_s, 99) if len(tr_s) else np.nan

#         if len(tr_s):
#             if (tr_s <= q01).sum() < tail_min_count:
#                 q01 = np.nanpercentile(tr_s, 5)
#             if (tr_s >= q99).sum() < tail_min_count:
#                 q99 = np.nanpercentile(tr_s, 95)

#         full_span = q99 - q01 if not (np.isnan(q99) or np.isnan(q01)) else np.nan
#         iqr = q75 - q25 if not (np.isnan(q75) or np.isnan(q25)) else np.nan
#         bulk_ratio = (iqr / (full_span + const_eps)) if not np.isnan(iqr) and not np.isnan(full_span) else np.nan

#         # compute the score components (same as original logic)
#         score = 0.0
#         mode_frac = 0.0
#         zero_frac = 0.0
#         frac_in = 0.0
#         top_bin_share = 0.0
#         if len(tr_s) > 0:
#             mode_frac = float(tr_s.value_counts(normalize=True).iloc[0])
#             zero_frac = float(((tr_s.abs() <= zero_tol)).sum()) / len(tr_s)
#             if (tr_s.nunique() <= 5 and mode_frac >= 0.90) or (zero_frac >= 0.90):
#                 score = 0.0
#             else:
#                 if np.isnan(full_span) or full_span == 0 or np.isnan(bulk_ratio):
#                     score = 0.0
#                 else:
#                     if len(tr_s) >= min_train_samples:
#                         m = _mad(tr_s.to_numpy())
#                         hw = float(max(hw_eps, mad_mul * m))
#                         inside = tr_s[(tr_s >= (med_tr - hw)) & (tr_s <= (med_tr + hw))]
#                         frac_in = float(len(inside)) / len(tr_s) if len(tr_s) > 0 else 0.0

#                         rel_width = 1.0 if (np.isnan(full_span) or full_span <= 0) else min(1.0, (2.0 * hw) / float(full_span))
#                         rel_factor = 1.0 - min(1.0, rel_width / float(rel_width_scale))
#                         base_score = float(np.clip(frac_in * rel_factor, 0.0, 1.0))

#                         # histogram concentration
#                         if not (np.isnan(q01) or np.isnan(q99) or q99 <= q01):
#                             edges = np.linspace(q01, q99, conc_bins + 1)
#                             h, _ = np.histogram(tr_s, bins=edges)
#                             hist_max = int(h.max()) if h.size else 0
#                             p = h / h.sum() if h.sum() > 0 else np.zeros_like(h)
#                             top_bin_share = float(p.max()) if p.size else 0.0
#                             conc = float(hist_max) / len(tr_s) if len(tr_s) else 0.0
#                         else:
#                             top_bin_share = 1.0 if tr_s.nunique() == 1 else 0.0
#                             conc = 1.0 if tr_s.nunique() == 1 else 0.0

#                         allow_conc = (
#                             (mode_frac < mode_frac_thresh)
#                             and (zero_frac < zero_frac_thresh)
#                             and (rel_factor > 0.0)
#                             and (frac_in > frac_in_min)
#                         )
#                         conc_term = float(gamma * conc * (1.0 - mode_frac)) if allow_conc else 0.0
#                         score = float(np.clip(base_score + conc_term, 0.0, 1.0))

#         # compute other failure flags (unchanged)
#         frac_val_out = frac_outside(vl_s, q01, q99)
#         frac_te_out = frac_outside(te_s, q01, q99)
#         frac_out_flag = (frac_val_out > frac_outside_thresh) or (frac_te_out > frac_outside_thresh)

#         ks_p = np.nan; ks_flag = False
#         try:
#             if len(tr_s) >= min_train_samples and len(te_s) >= min_train_samples:
#                 _, ks_p = ks_2samp(tr_s, te_s)
#                 ks_flag = (ks_p < ks_pval_thresh)
#         except Exception:
#             ks_p = np.nan; ks_flag = False

#         const_on_train = False
#         if len(tr_s) and np.isfinite(tr_s.min()) and np.isfinite(tr_s.max()):
#             const_tol = max(const_eps, denom * 1e-12)
#             const_on_train = (abs(tr_s.max() - tr_s.min()) < const_tol)
#         const_flag = const_on_train

#         fail_reasons = []
#         fail_count = 0
#         if pct_shift_flag:
#             fail_count += 1; fail_reasons.append(f"pct_shift_val={pct_shift_val:.3f};te={pct_shift_te:.3f}")
#         if frac_out_flag:
#             fail_count += 1; fail_reasons.append(f"frac_out_val={frac_val_out:.3f};te={frac_te_out:.3f}")
#         if ks_flag:
#             fail_count += 1; fail_reasons.append(f"ks_p={ks_p:.4g}")
#         if const_flag:
#             fail_count += 10; fail_reasons.append("constant_on_train")

#         status = "OK" if fail_count < fail_count_to_drop else "DROP"

#         rows.append({
#             "feature": feat,
#             "q01": q01, "q25": q25, "q75": q75, "q99": q99,
#             "full_span": full_span, "iqr": iqr, "bulk_ratio": bulk_ratio,
#             "med_tr": med_tr, "med_val": med_val, "med_te": med_te,
#             "pct_shift_val": pct_shift_val, "pct_shift_te": pct_shift_te,
#             "abs_shift_val": abs_shift_val, "abs_shift_te": abs_shift_te,
#             "frac_val_out": frac_val_out, "frac_te_out": frac_te_out,
#             "ks_p": ks_p,
#             "nan_mask_train": tr[feat].isna().any(),
#             "const_on_train": const_on_train,
#             "fail_count": fail_count,
#             "status": status,
#             "reason": "; ".join(fail_reasons) if fail_reasons else "",
#             "score": float(score),
#             "mode_frac": float(mode_frac),
#             "zero_frac": float(zero_frac),
#             "frac_in": float(frac_in),
#             "top_bin_share": float(top_bin_share)
#         })

#     diag = pd.DataFrame(rows).set_index("feature").sort_values(["status", "fail_count"], ascending=[True, False])
#     to_drop = diag[diag["status"] == "DROP"].index.tolist()
#     df_pruned = df_unsc.drop(columns=to_drop, errors="ignore")
#     return df_pruned, to_drop, diag



def prune_and_diag(
    df_unsc: pd.DataFrame,
    train_prop: float = 0.7,
    # per-test thresholds (conservative, data-driven defaults)
    pct_shift_thresh: float = 0.16,       # relative median shift threshold (recommended data-driven)
    frac_outside_thresh: float = 0.06,    # fraction outside TRAIN q01-q99 (recommended data-driven)
    ks_pval_thresh: float = 1e-6,         # KS p-value threshold (KS flagged but only effective with other tests)
    min_train_samples: int = 50,          # min samples to run KS test
    mad_mul: float = 1.5,
    rel_width_scale: float = 0.10,
    gamma: float = 1.0,
    conc_bins: int = 80,
    mode_frac_thresh: float = 0.75,
    zero_frac_thresh: float = 0.90,
    frac_in_min: float = 0.02,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Prune features and compute diagnostics used for percentile assignment.

    Behavior and intent
    - Tests (each produces a boolean flag):
        * median shift: VAL or TEST median vs TRAIN median (relative to a robust denominator)
        * fraction outside TRAIN tails: share of VAL/TEST outside TRAIN [q01,q99]
        * KS two-sample: TRAIN vs TEST (p < ks_pval_thresh) — recorded but only counts when
          coincident with shift or frac-out to avoid KS alone dropping many features
        * constant-on-train: near-constant TRAIN values
    - Drop rule (keeps "no fail-count" policy): DROP if ANY of
        pct_shift_fail OR frac_out_fail OR const_fail OR (ks_fail AND (pct_shift_fail OR frac_out_fail))
      This keeps the simple OR logic while preventing KS alone from removing most features.
    - Diagnostics: returns per-feature stats required later (percentiles, medians, frac_out, ks_p,
      score, mode_frac, zero_frac, frac_in, top_bin_share) for auditing and tuning.
    - Returns: (df_pruned, to_drop, diag) where diag is indexed by feature.
    """
    # stability constants
    zero_tol = 1e-8
    denom_eps = 1e-6
    hw_eps = 1e-12
    const_eps = 1e-12
    tail_min_count = 3

    N = len(df_unsc)
    if N == 0:
        return df_unsc.copy(), [], pd.DataFrame()

    # contiguous split: TRAIN / VAL / TEST
    n_tr = int(N * train_prop)
    n_rem = N - n_tr
    n_val = n_rem // 2
    tr = df_unsc.iloc[:n_tr].replace([np.inf, -np.inf], np.nan)
    vl = df_unsc.iloc[n_tr:n_tr + n_val].replace([np.inf, -np.inf], np.nan)
    te = df_unsc.iloc[n_tr + n_val:].replace([np.inf, -np.inf], np.nan)

    numeric_cols = [c for c in df_unsc.columns if pd.api.types.is_numeric_dtype(df_unsc[c])]
    rows = []

    def _mad(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return 0.0
        med = float(np.nanmedian(arr))
        return float(np.nanmedian(np.abs(arr - med)))

    def safe_denominator(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return denom_eps
        med = float(np.nanmedian(arr))
        mad = float(np.nanmedian(np.abs(arr - med)))
        std = float(np.nanstd(arr))
        return max(abs(med), mad, std, denom_eps)

    def frac_outside(s: pd.Series, q01, q99) -> float:
        if len(s) == 0 or np.isnan(q01) or np.isnan(q99):
            return 0.0
        s2 = s.dropna()
        if len(s2) == 0:
            return 0.0
        return float(((s2 < q01).sum() + (s2 > q99).sum()) / len(s2))

    for feat in tqdm(numeric_cols, desc="prune_and_diag", unit="feat"):
        tr_s = tr[feat].dropna()
        vl_s = vl[feat].dropna()
        te_s = te[feat].dropna()

        med_tr = float(np.nanmedian(tr_s)) if len(tr_s) else np.nan
        med_val = float(np.nanmedian(vl_s)) if len(vl_s) else np.nan
        med_te = float(np.nanmedian(te_s)) if len(te_s) else np.nan

        denom = safe_denominator(tr_s.to_numpy()) if len(tr_s) else denom_eps
        pct_shift_val = abs(med_val - med_tr) / denom if not np.isnan(med_tr) else 0.0
        pct_shift_te = abs(med_te - med_tr) / denom if not np.isnan(med_tr) else 0.0
        abs_shift_val = abs(med_val - med_tr)
        abs_shift_te = abs(med_te - med_tr)

        # TRAIN percentiles with tiny-tail fallback
        q01 = np.nanpercentile(tr_s, 1) if len(tr_s) else np.nan
        q25 = np.nanpercentile(tr_s, 25) if len(tr_s) else np.nan
        q75 = np.nanpercentile(tr_s, 75) if len(tr_s) else np.nan
        q99 = np.nanpercentile(tr_s, 99) if len(tr_s) else np.nan

        if len(tr_s):
            if (tr_s <= q01).sum() < tail_min_count:
                q01 = np.nanpercentile(tr_s, 5)
            if (tr_s >= q99).sum() < tail_min_count:
                q99 = np.nanpercentile(tr_s, 95)

        full_span = q99 - q01 if not (np.isnan(q99) or np.isnan(q01)) else np.nan
        iqr = q75 - q25 if not (np.isnan(q75) or np.isnan(q25)) else np.nan
        bulk_ratio = (iqr / (full_span + const_eps)) if not np.isnan(iqr) and not np.isnan(full_span) else np.nan

        # scoring components (kept for diagnostics only)
        score = 0.0
        mode_frac = 0.0
        zero_frac = 0.0
        frac_in = 0.0
        top_bin_share = 0.0
        if len(tr_s) > 0:
            mode_frac = float(tr_s.value_counts(normalize=True).iloc[0])
            zero_frac = float(((tr_s.abs() <= zero_tol)).sum()) / len(tr_s)
            if (tr_s.nunique() <= 5 and mode_frac >= 0.90) or (zero_frac >= 0.90):
                score = 0.0
            else:
                if np.isnan(full_span) or full_span == 0 or np.isnan(bulk_ratio):
                    score = 0.0
                else:
                    if len(tr_s) >= min_train_samples:
                        m = _mad(tr_s.to_numpy())
                        hw = float(max(hw_eps, mad_mul * m))
                        inside = tr_s[(tr_s >= (med_tr - hw)) & (tr_s <= (med_tr + hw))]
                        frac_in = float(len(inside)) / len(tr_s) if len(tr_s) > 0 else 0.0

                        rel_width = 1.0 if (np.isnan(full_span) or full_span <= 0) else min(1.0, (2.0 * hw) / float(full_span))
                        rel_factor = 1.0 - min(1.0, rel_width / float(rel_width_scale))
                        base_score = float(np.clip(frac_in * rel_factor, 0.0, 1.0))

                        if not (np.isnan(q01) or np.isnan(q99) or q99 <= q01):
                            edges = np.linspace(q01, q99, conc_bins + 1)
                            h, _ = np.histogram(tr_s, bins=edges)
                            hist_max = int(h.max()) if h.size else 0
                            p = h / h.sum() if h.sum() > 0 else np.zeros_like(h)
                            top_bin_share = float(p.max()) if p.size else 0.0
                            conc = float(hist_max) / len(tr_s) if len(tr_s) else 0.0
                        else:
                            top_bin_share = 1.0 if tr_s.nunique() == 1 else 0.0
                            conc = 1.0 if tr_s.nunique() == 1 else 0.0

                        allow_conc = (
                            (mode_frac < mode_frac_thresh)
                            and (zero_frac < zero_frac_thresh)
                            and (rel_factor > 0.0)
                            and (frac_in > frac_in_min)
                        )
                        conc_term = float(gamma * conc * (1.0 - mode_frac)) if allow_conc else 0.0
                        score = float(np.clip(base_score + conc_term, 0.0, 1.0))

        # failure tests (each controlled by a threshold)
        pct_shift_fail = (pct_shift_val > pct_shift_thresh) or (pct_shift_te > pct_shift_thresh)
        frac_val_out = frac_outside(vl_s, q01, q99)
        frac_te_out = frac_outside(te_s, q01, q99)
        frac_out_fail = (frac_val_out > frac_outside_thresh) or (frac_te_out > frac_outside_thresh)

        ks_p = np.nan
        ks_fail = False
        try:
            if len(tr_s) >= min_train_samples and len(te_s) >= min_train_samples:
                _, ks_p = ks_2samp(tr_s, te_s)
                ks_fail = (ks_p < ks_pval_thresh)
        except Exception:
            ks_p = np.nan; ks_fail = False

        const_on_train = False
        if len(tr_s) and np.isfinite(tr_s.min()) and np.isfinite(tr_s.max()):
            const_tol = max(const_eps, denom * 1e-12)
            const_on_train = (abs(tr_s.max() - tr_s.min()) < const_tol)
        const_fail = const_on_train

        # KS only counts when coincident with shift or frac-out to avoid KS-alone drops
        ks_effective = ks_fail and (pct_shift_fail or frac_out_fail)

        # DROP if ANY of the core tests is true (OR), but KS must be coincident
        drop_reasons: List[str] = []
        if pct_shift_fail:
            drop_reasons.append(f"pct_shift_val={pct_shift_val:.3f};te={pct_shift_te:.3f}")
        if frac_out_fail:
            drop_reasons.append(f"frac_out_val={frac_val_out:.3f};te={frac_te_out:.3f}")
        if ks_effective:
            drop_reasons.append(f"ks_p={ks_p:.4g}")
        if const_fail:
            drop_reasons.append("constant_on_train")

        status = "DROP" if drop_reasons else "OK"
        reason = "; ".join(drop_reasons)

        rows.append({
            "feature": feat,
            "q01": q01, "q25": q25, "q75": q75, "q99": q99,
            "full_span": full_span, "iqr": iqr, "bulk_ratio": bulk_ratio,
            "med_tr": med_tr, "med_val": med_val, "med_te": med_te,
            "pct_shift_val": pct_shift_val, "pct_shift_te": pct_shift_te,
            "abs_shift_val": abs_shift_val, "abs_shift_te": abs_shift_te,
            "frac_val_out": frac_val_out, "frac_te_out": frac_te_out,
            "ks_p": ks_p,
            "nan_mask_train": tr[feat].isna().any(),
            "const_on_train": const_on_train,
            "status": status,
            "reason": reason,
            "score": float(score),
            "mode_frac": float(mode_frac),
            "zero_frac": float(zero_frac),
            "frac_in": float(frac_in),
            "top_bin_share": float(top_bin_share)
        })

    diag = pd.DataFrame(rows).set_index("feature").sort_values(["status"], ascending=[True])
    to_drop = diag[diag["status"] == "DROP"].index.tolist()
    df_pruned = df_unsc.drop(columns=to_drop, errors="ignore")
    return df_pruned, to_drop, diag


############################################################################ 


# def assign_percentiles_from_diag(diag: pd.DataFrame, flag_range: tuple = (30.0, 70.0)):
#     """
#     Assign final percentile pairs from diagnostics.

#     Input:
#       - diag: DataFrame indexed by feature containing at least these columns:
#           score, zero_frac, mode_frac, frac_in, top_bin_share
#       - flag_range: tuple (low_pct, high_pct) to assign to detected kept_narrow features

#     Behavior:
#       - zero_mass := (zero_frac >= 0.05) and (mode_frac >= 0.05)
#       - center_dom := (frac_in >= 0.60) and (top_bin_share >= 0.35)
#       - excluded_local := zero_mass OR center_dom  -> assigned "excluded"
#       - kept_narrow := not excluded_local AND score > 0 -> assigned flag_range
#       - otherwise -> assigned (0.0, 100.0)
#     Output:
#       - DataFrame copy of diag with added columns: low_pct, high_pct, assigned_reason
#     """
#     out = diag.copy()
#     low_list, high_list, reason_list = [], [], []

#     for feat, r in out.iterrows():
#         mode_frac = r.get("mode_frac", 0.0)
#         zero_frac = r.get("zero_frac", 0.0)
#         frac_in = r.get("frac_in", 0.0)
#         top_bin = r.get("top_bin_share", 0.0)
#         score = r.get("score", 0.0)

#         zero_mass_flag = (zero_frac >= 0.05) and (mode_frac >= 0.05)
#         center_dom_flag = (frac_in >= 0.60) and (top_bin >= 0.35)
#         excluded_local = bool(zero_mass_flag or center_dom_flag)

#         if excluded_local:
#             low, high = 0.0, 100.0
#             assigned = "excluded"
#         elif score > 0.0:
#             low, high = float(flag_range[0]), float(flag_range[1])
#             assigned = "kept_narrow"
#         else:
#             low, high = 0.0, 100.0
#             assigned = ""

#         low_list.append(low)
#         high_list.append(high)
#         reason_list.append(assigned)

#     out["low_pct"] = low_list
#     out["high_pct"] = high_list
#     out["assigned_reason"] = reason_list
#     return out


def assign_percentiles_from_diag(
    diag: pd.DataFrame,
    custom_range: Tuple[float, float] = (30.0, 70.0),
    standard_range: Tuple[float, float] = (0.0, 100.0),
) -> pd.DataFrame:
    """
    Assign final percentile pairs from diagnostics.

    Inputs
      - diag: DataFrame indexed by feature containing at least these columns:
          score, zero_frac, mode_frac, frac_in, top_bin_share
      - custom_range: percentile pair assigned to "kept_narrow" features (e.g. (30,70))
      - standard_range: percentile pair assigned to all other features by default
                        (e.g. (0,100) to mean "no winsorize requested", or a trimmed
                        pair like (0.1,99.9) if you want automatic trimming)

    Behavior
      - zero_mass := (zero_frac >= 0.05) and (mode_frac >= 0.05)
      - center_dom := (frac_in >= 0.60) and (top_bin_share >= 0.35)
      - excluded_local := zero_mass OR center_dom  -> assigned "excluded"
      - kept_narrow := not excluded_local AND score > 0 -> assigned custom_range
      - otherwise -> assigned standard_range
    Output
      - copy of diag with added columns: low_pct, high_pct, assigned_reason
    """
    out = diag.copy()

    # ensure ranges are valid numeric tuples
    def _norm_range(r):
        if not (isinstance(r, (tuple, list)) and len(r) >= 2):
            raise ValueError(f"percentile range must be a tuple (low, high), got {r!r}")
        lo, hi = float(r[0]), float(r[1])
        if not (0.0 <= lo <= 100.0 and 0.0 <= hi <= 100.0 and lo < hi):
            raise ValueError(f"invalid percentile range {r!r}, must satisfy 0 <= low < high <= 100")
        return lo, hi

    custom_lo, custom_hi = _norm_range(custom_range)
    standard_lo, standard_hi = _norm_range(standard_range)

    low_list, high_list, reason_list = [], [], []

    for feat, r in out.iterrows():
        mode_frac = r.get("mode_frac", 0.0)
        zero_frac = r.get("zero_frac", 0.0)
        frac_in = r.get("frac_in", 0.0)
        top_bin = r.get("top_bin_share", 0.0)
        score = r.get("score", 0.0)

        zero_mass_flag = (zero_frac >= 0.05) and (mode_frac >= 0.05)
        center_dom_flag = (frac_in >= 0.60) and (top_bin >= 0.35)
        excluded_local = bool(zero_mass_flag or center_dom_flag)

        if excluded_local:
            lo, hi = standard_lo, standard_hi
            assigned = "excluded"
        elif score > 0.0:
            lo, hi = custom_lo, custom_hi
            assigned = "kept_narrow"
        else:
            lo, hi = standard_lo, standard_hi
            assigned = ""

        low_list.append(lo)
        high_list.append(hi)
        reason_list.append(assigned)

    out["low_pct"] = low_list
    out["high_pct"] = high_list
    out["assigned_reason"] = reason_list
    return out

#########################################################################################################


# def scaling_with_percentiles(
#     df: pd.DataFrame,
#     label_col: str,
#     diag: pd.DataFrame,
#     train_prop: float = 0.7,
#     val_prop: float = 0.15,
#     winsorize: bool = True,
# ) -> pd.DataFrame:
#     """
#     Scale numeric features to [0,1] using TRAIN-based winsorize + MinMax with per-feature percentiles.

#     Inputs
#       - df: full DataFrame (indexed by timestamp), numeric features + label_col
#       - label_col: name of the label column to exclude from scaling
#       - diag: diagnostics DataFrame produced by prune_and_diag / assign_percentiles_from_diag.
#               Expected: indexed by feature name and containing either:
#                 * "pct_pair" column with (low_pct, high_pct) tuples; OR
#                 * "low_pct" and "high_pct" columns (used preferentially).
#       - train_prop, val_prop: contiguous TRAIN/VAL/TEST split fractions (train_prop + val_prop < 1)
#       - winsorize: whether to apply TRAIN-based clipping before MinMax

#     Behavior
#       - For each numeric feature (excluding label_col) reads per-feature percentiles from diag.
#       - If pct info missing/invalid, falls back to (0.5, 99.5).
#       - Computes TRAIN percentiles per feature, clips TRAIN/VAL/TEST values to those cutpoints,
#         computes TRAIN clipped min/max and then applies MinMax scaling.
#       - Preserves NaNs and non-numeric columns unchanged.
#     """
#     df = df.copy()
#     df.index = pd.to_datetime(df.index)
#     df = df.replace([np.inf, -np.inf], np.nan)

#     # numeric feature columns (cast once) and exclude label
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     feat_cols = [c for c in numeric_cols if c != label_col]
#     df[numeric_cols] = df[numeric_cols].astype(np.float64)

#     if not feat_cols:
#         return df

#     # splits
#     N = len(df)
#     n_tr = int(N * train_prop)
#     n_val = int(N * val_prop)
#     if n_tr + n_val >= N:
#         raise ValueError("train_prop + val_prop must sum < 1.0")
#     tr_slice = slice(0, n_tr)
#     val_slice = slice(n_tr, n_tr + n_val)
#     te_slice = slice(n_tr + n_val, N)
#     df_tr = df.iloc[tr_slice].copy()

#     # Prepare diag index alignment
#     diag_local = diag.copy()
#     if "low_pct" in diag_local.columns and "high_pct" in diag_local.columns:
#         diag_local["pct_pair"] = list(zip(diag_local["low_pct"].astype(float), diag_local["high_pct"].astype(float)))
#     # ensure pct_pair exists (may be missing for some features)
#     # Build per-feature pct maps with fallback
#     pct_low_map = {}
#     pct_high_map = {}
#     for c in feat_cols:
#         lo, hi = 0.5, 99.5  # default fallback
#         try:
#             pair = diag_local.at[c, "pct_pair"]
#             if isinstance(pair, (tuple, list)) and len(pair) >= 2:
#                 lo, hi = float(pair[0]), float(pair[1])
#         except Exception:
#             # missing feature in diag or invalid pair => fallback
#             lo, hi = 0.5, 99.5
#         # sanitize
#         lo = float(np.clip(lo, 0.0, 49.9))
#         hi = float(np.clip(hi, 50.1, 100.0))
#         if lo >= hi:
#             lo, hi = 0.5, 99.5
#         pct_low_map[c] = lo
#         pct_high_map[c] = hi

#     # compute TRAIN percentiles per-feature for winsorize (with progress)
#     clip_low = {}
#     clip_high = {}
#     if winsorize:
#         for c in feat_cols:
#             arr = df_tr[c].to_numpy(dtype=float)
#             clip_low[c] = np.nanpercentile(arr, pct_low_map[c]) if arr.size else np.nan
#             clip_high[c] = np.nanpercentile(arr, pct_high_map[c]) if arr.size else np.nan

#     # helper to apply clipping on numpy array
#     def apply_clip_array(arr: np.ndarray, c: str) -> np.ndarray:
#         if not winsorize:
#             return arr
#         low = clip_low[c]; high = clip_high[c]
#         a = np.copy(arr)
#         mask = np.isnan(a)
#         a = np.clip(a, low, high)
#         a[mask] = np.nan
#         return a

#     # compute clipped TRAIN min/max and spans (with progress)
#     col_min = {}
#     col_max = {}
#     span = {}
#     for c in feat_cols:
#         col = df_tr[c].to_numpy(dtype=float)
#         colc = apply_clip_array(col, c) if winsorize else col
#         col_min[c] = np.nanmin(colc) if np.any(~np.isnan(colc)) else np.nan
#         col_max[c] = np.nanmax(colc) if np.any(~np.isnan(colc)) else np.nan
#         if np.isnan(col_min[c]) or np.isnan(col_max[c]) or (col_max[c] - col_min[c]) == 0.0:
#             span[c] = 1.0
#         else:
#             span[c] = col_max[c] - col_min[c]

#     # transform helper for a block (preserves NaNs)
#     def transform_block(block: pd.DataFrame) -> pd.DataFrame:
#         out = block.copy()
#         for c in feat_cols:
#             arr = out[c].to_numpy(dtype=float)
#             if winsorize:
#                 low = clip_low[c]; high = clip_high[c]
#                 mask = np.isnan(arr)
#                 arr = np.clip(arr, low, high)
#                 arr[mask] = np.nan
#             mn = col_min[c]; sp = span[c]
#             res = (arr - mn) / sp
#             res = np.where(np.isnan(arr), np.nan, np.clip(res, 0.0, 1.0))
#             out[c] = res
#         return out[feat_cols].astype(np.float64)

#     # apply transforms day-by-day for each split (with progress over days)
#     def transform_split(split_df: pd.DataFrame, desc: str) -> pd.DataFrame:
#         parts = []
#         days = split_df.index.normalize().unique()
#         for _, block in tqdm(split_df.groupby(split_df.index.normalize()),
#                               desc=desc, unit="day", total=len(days)):
#             parts.append(transform_block(block))
#         if not parts:
#             return pd.DataFrame(columns=feat_cols, index=split_df.index)
#         return pd.concat(parts).reindex(split_df.index)

#     tr_scaled = transform_split(df.iloc[tr_slice], "scale train days")
#     v_scaled = transform_split(df.iloc[val_slice], "scale val days")
#     te_scaled = transform_split(df.iloc[te_slice], "scale test days")

#     # reassemble final DataFrame
#     df_tr.loc[:, feat_cols] = tr_scaled
#     df_v = df.iloc[n_tr:n_tr + n_val].copy()
#     df_te = df.iloc[n_tr + n_val:].copy()
#     df_v.loc[:, feat_cols] = v_scaled
#     df_te.loc[:, feat_cols] = te_scaled

#     df_all = pd.concat([df_tr, df_v, df_te]).sort_index()

#     return df_all


def scaling_with_percentiles(
    df: pd.DataFrame,
    label_col: str,
    diag: pd.DataFrame,
    train_prop: float = 0.7,
    val_prop: float = 0.15,
    winsorize: bool = True,
    max_abs_cutpoint: float = 1e12,
) -> pd.DataFrame:
    """
    Scale numeric features to [0,1] using TRAIN-based winsorize + MinMax with per-feature percentiles.

    Key rules (strict)
    - Percentile pairs must be provided by `diag` (as low_pct/high_pct or pct_pair). Missing pairs cause
      a ValueError to surface (fail-fast) so the caller is forced to supply diagnostics for all features.
    - The scaler uses the exact percentiles from diag. There is NO implicit fallback to other percentiles.
    - If diag gives (0.0, 100.0) exactly for a feature, that is interpreted as "no winsorize requested" and
      the scaler will skip clipping for that feature.
    - If TRAIN-computed cutpoints for a requested (non-0/100) pair are invalid (non-finite, hi <= lo, or
      absurd magnitude), the scaler will silently treat that feature as "no clipping" (safety), but it will
      not substitute alternate percentiles. The feature remains scaled by MinMax from the unclipped TRAIN data.
    - NaNs and non-numeric columns (including label_col) are preserved unchanged.
    - Splitting is contiguous: first rows = TRAIN, next = VAL, final = TEST.

    Inputs
      - df: full DataFrame (indexed by timestamp), numeric features + label_col
      - label_col: name of the label column to exclude from scaling
      - diag: diagnostics DataFrame indexed by feature and containing either:
          * "low_pct" and "high_pct" columns; or
          * "pct_pair" column with (low_pct, high_pct) tuples.
      - train_prop, val_prop: contiguous TRAIN/VAL/TEST split fractions (train_prop + val_prop < 1)
      - winsorize: whether to apply TRAIN-based clipping before MinMax (subject to diag)
      - max_abs_cutpoint: numeric threshold to consider a cutpoint insane (safety)

    Output
      - DataFrame with same index and columns, numeric features scaled to [0,1]
    """
    # copy and basic clean
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.replace([np.inf, -np.inf], np.nan)

    # select numeric features and exclude label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c != label_col]
    df[numeric_cols] = df[numeric_cols].astype(np.float64, copy=False)

    if not feat_cols:
        return df

    # contiguous train/val/test split indices
    N = len(df)
    n_tr = int(N * train_prop)
    n_val = int(N * val_prop)
    if n_tr + n_val >= N:
        raise ValueError("train_prop + val_prop must sum < 1.0")
    tr_slice = slice(0, n_tr)
    val_slice = slice(n_tr, n_tr + n_val)
    te_slice = slice(n_tr + n_val, N)
    df_tr = df.iloc[tr_slice].copy()

    # normalize diag to ensure pct_pair exists
    if not isinstance(diag, pd.DataFrame):
        raise ValueError("diag must be a pandas DataFrame indexed by feature")
    diag_local = diag.copy()
    if "low_pct" in diag_local.columns and "high_pct" in diag_local.columns:
        diag_local["pct_pair"] = list(zip(diag_local["low_pct"].astype(float), diag_local["high_pct"].astype(float)))

    # strict sourcing of percentiles: require them for every feature
    pct_low_map: Dict[str, float] = {}
    pct_high_map: Dict[str, float] = {}
    missing = []
    for c in feat_cols:
        if c not in diag_local.index:
            missing.append(c)
            continue
        pair = diag_local.at[c, "pct_pair"] if "pct_pair" in diag_local.columns else None
        if not (isinstance(pair, (tuple, list)) and len(pair) >= 2):
            missing.append(c)
            continue
        lo, hi = float(pair[0]), float(pair[1])
        pct_low_map[c] = lo
        pct_high_map[c] = hi

    if missing:
        raise ValueError(f"scaling_with_percentiles requires percentiles in diag for all features; missing: {missing}")

    # detect exact full-range request -> skip winsorize for those features
    skip_wins: Dict[str, bool] = {c: (pct_low_map[c] <= 0.0 and pct_high_map[c] >= 100.0) for c in feat_cols}

    # compute TRAIN cutpoints per-feature exactly as requested; validate but DO NOT fallback
    clip_low: Dict[str, float] = {}
    clip_high: Dict[str, float] = {}
    if winsorize:
        for c in feat_cols:
            if skip_wins.get(c, False):
                clip_low[c] = np.nan
                clip_high[c] = np.nan
                continue
            arr = df_tr[c].to_numpy(dtype=float)
            lo_cut = np.nanpercentile(arr, pct_low_map[c]) if arr.size else np.nan
            hi_cut = np.nanpercentile(arr, pct_high_map[c]) if arr.size else np.nan
            valid = (
                np.isfinite(lo_cut)
                and np.isfinite(hi_cut)
                and (hi_cut > lo_cut)
                and (abs(lo_cut) < max_abs_cutpoint)
                and (abs(hi_cut) < max_abs_cutpoint)
            )
            if not valid:
                # safety: mark as no clipping (do NOT substitute alternate percentiles)
                clip_low[c] = np.nan
                clip_high[c] = np.nan
                skip_wins[c] = True
            else:
                clip_low[c] = float(lo_cut)
                clip_high[c] = float(hi_cut)

    # helper to apply clipping on numpy array (respects skip_wins and invalid cutpoints)
    def apply_clip_array(arr: np.ndarray, c: str) -> np.ndarray:
        if not winsorize or skip_wins.get(c, False):
            return arr
        low = clip_low.get(c, np.nan); high = clip_high.get(c, np.nan)
        if (not np.isfinite(low)) or (not np.isfinite(high)):
            return arr
        a = np.copy(arr)
        mask = np.isnan(a)
        a = np.clip(a, low, high)
        a[mask] = np.nan
        return a

    # compute clipped TRAIN min/max and spans (safe defaults if data missing)
    col_min: Dict[str, float] = {}
    col_max: Dict[str, float] = {}
    span: Dict[str, float] = {}
    for c in feat_cols:
        col = df_tr[c].to_numpy(dtype=float)
        colc = apply_clip_array(col, c) if winsorize else col
        if np.any(~np.isnan(colc)):
            mn = np.nanmin(colc); mx = np.nanmax(colc)
            col_min[c] = float(mn) if np.isfinite(mn) else np.nan
            col_max[c] = float(mx) if np.isfinite(mx) else np.nan
            sp = (col_max[c] - col_min[c]) if (np.isfinite(col_min[c]) and np.isfinite(col_max[c])) else 0.0
            span[c] = float(sp) if sp != 0.0 else 1.0
        else:
            col_min[c] = np.nan
            col_max[c] = np.nan
            span[c] = 1.0

    # transform helper for a block (preserves NaNs)
    def transform_block(block: pd.DataFrame) -> pd.DataFrame:
        out = block.copy()
        for c in feat_cols:
            arr = out[c].to_numpy(dtype=float)
            arrc = apply_clip_array(arr, c) if winsorize else arr
            mn = col_min[c]; sp = span[c]
            res = (arrc - mn) / sp
            res = np.where(np.isnan(arrc), np.nan, np.clip(res, 0.0, 1.0))
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



# def scaling_diagnostics(
#     df_unscaled: pd.DataFrame,
#     df_scaled: pd.DataFrame,
#     train_prop: float = 0.7,
#     only_problem: bool = False,
#     eps: float = 1e-9,
#     conc_bins: int = 80,
#     conc_min_count: int = 50,
#     tiny_bulk_thresh: float = 0.02,   # if iqr_over_full <= this, mark tiny_bulk_only
#     tail_support_thresh: int = 30     # existing tail support threshold (kept for reference)
# ) -> pd.DataFrame:
#     """
#     Post-scaling diagnostics with automatic action suggestions.

#     Returns per-feature diagnostics; added columns:
#       - iqr_over_full, tail_low_cnt, tail_high_cnt
#       - tiny_bulk_only (iqr_over_full <= tiny_bulk_thresh)
#       - suggested_action_auto: one of ok, widen, relax_moderate, relax_strong, leave_binary
#     Decision logic (conservative):
#       * leave_binary when frac_clipped == 1.0
#       * widen when tiny_bulk_only is True
#       * relax_strong when frac_clipped >= 0.75
#       * relax_moderate when frac_clipped >= 0.5
#       * otherwise keep original suggested_action or ok
#     """
#     if df_unscaled is None or df_scaled is None:
#         return pd.DataFrame()
#     N = len(df_unscaled)
#     if N == 0:
#         return pd.DataFrame()

#     n_tr = int(N * train_prop)
#     n_rem = N - n_tr
#     n_val = n_rem // 2
#     tr_slice, val_slice, te_slice = slice(0, n_tr), slice(n_tr, n_tr + n_val), slice(n_tr + n_val, N)

#     common = [c for c in df_unscaled.columns if c in df_scaled.columns and pd.api.types.is_numeric_dtype(df_unscaled[c])]
#     if not common:
#         return pd.DataFrame()

#     tr_u = df_unscaled.iloc[tr_slice][common]
#     val_s = df_scaled.iloc[val_slice][common]
#     te_s  = df_scaled.iloc[te_slice][common]
#     full_s = df_scaled[common]

#     # basic stats
#     med_tr = tr_u.median(axis=0, skipna=True)
#     q01_tr = tr_u.quantile(0.01)
#     q05_tr = tr_u.quantile(0.05)
#     q10_tr = tr_u.quantile(0.10)
#     q25_tr = tr_u.quantile(0.25)
#     q75_tr = tr_u.quantile(0.75)
#     q95_tr = tr_u.quantile(0.95)
#     q99_tr = tr_u.quantile(0.99)
#     iqr_tr = (q75_tr - q25_tr)

#     scaled_range = (full_s.max(axis=0, skipna=True) - full_s.min(axis=0, skipna=True))

#     # clip counts
#     def clip_counts(df_block):
#         arr = df_block.to_numpy(dtype=float, na_value=np.nan)
#         mask = ~np.isnan(arr)
#         is0 = np.isclose(arr, 0.0, atol=eps) & mask
#         is1 = np.isclose(arr, 1.0, atol=eps) & mask
#         return np.sum(is0, axis=0), np.sum(is1, axis=0), np.sum(mask, axis=0)
#     v0, v1, vt = clip_counts(val_s)
#     t0, t1, tt = clip_counts(te_s)
#     clip_val = np.where(vt>0, (v0+v1)/vt, 0.0)
#     clip_te  = np.where(tt>0, (t0+t1)/tt, 0.0)
#     frac_clipped = (clip_val + clip_te) / 2.0

#     # adaptive clip threshold info for attrs
#     median_frac = float(np.nanmedian(frac_clipped)) if frac_clipped.size else 0.0
#     iqr_frac = float(np.nanpercentile(frac_clipped, 75) - np.nanpercentile(frac_clipped, 25)) if frac_clipped.size else 0.0
#     clip_thresh_adaptive = max(0.02, median_frac + 1.5 * iqr_frac)

#     # conc_score for TRAIN (compute where enough samples)
#     conc_scores = np.full(len(common), np.nan, dtype=float)
#     tr_counts = tr_u.count(axis=0).to_numpy(dtype=int)
#     for i, col in enumerate(tqdm(common, desc="conc_score (train)", unit="feat")):
#         if tr_counts[i] < conc_min_count:
#             continue
#         s = tr_u[col].dropna().to_numpy(dtype=float)
#         if s.size == 0:
#             continue
#         q01, q99 = np.nanpercentile(s, [1, 99])
#         if q99 <= q01:
#             conc_scores[i] = 1.0 if np.unique(s).size == 1 else np.nan
#             continue
#         edges = np.linspace(q01, q99, conc_bins + 1)
#         h, _ = np.histogram(s, bins=edges)
#         conc_scores[i] = float(h.max()) / float(s.size)

#     # narrowness metrics
#     iqr_arr = iqr_tr.to_numpy(dtype=float)
#     full_span_arr = (q99_tr - q01_tr).to_numpy(dtype=float)
#     iqr_over_full = np.where(~np.isnan(iqr_arr), iqr_arr / (np.abs(full_span_arr) + 1e-12), np.nan)
#     iqr_over_med = np.where(~np.isnan(iqr_arr), np.abs(iqr_arr) / (np.abs(med_tr.to_numpy(dtype=float)) + 1e-12), np.nan)
#     narrowness_score = np.nanmin(np.vstack([
#         np.nan_to_num(iqr_over_full, nan=1e6),
#         np.nan_to_num(iqr_over_med, nan=1e6)
#     ]), axis=0)

#     # tail support counts (TRAIN)
#     tail_low_cnt = np.array([(tr_u[col] <= q01_tr[col]).sum() if col in tr_u else 0 for col in common])
#     tail_high_cnt= np.array([(tr_u[col] >= q99_tr[col]).sum() if col in tr_u else 0 for col in common])

#     # assemble rows
#     rows = []
#     for i, col in enumerate(tqdm(common, desc="finalize diagnostics", unit="feat")):
#         m = float(med_tr.get(col, np.nan))
#         iqrv = float(iqr_tr.get(col, np.nan))
#         srange = float(scaled_range.get(col, np.nan))
#         fc = float(frac_clipped[i])
#         conc = float(conc_scores[i]) if not np.isnan(conc_scores[i]) else np.nan
#         iqrf = float(iqr_over_full[i]) if not np.isnan(iqr_over_full[i]) else np.nan
#         iqrmed = float(iqr_over_med[i]) if not np.isnan(iqr_over_med[i]) else np.nan
#         narrow = float(narrowness_score[i]) if not np.isnan(narrowness_score[i]) else np.nan
#         low_cnt = int(tail_low_cnt[i])
#         high_cnt= int(tail_high_cnt[i])

#         # classify clipping level
#         if np.isclose(fc, 1.0, atol=1e-12):
#             clip_class = "all_clipped"
#         elif fc > clip_thresh_adaptive:
#             clip_class = "high_clipped"
#         else:
#             clip_class = "ok"

#         # suggested_action existing heuristics (conservative)
#         action = "ok"; reasons=[]
#         if clip_class == "all_clipped":
#             action = "all_clipped"
#             reasons.append(f"frac_clipped={fc:.3f}")
#         elif clip_class == "high_clipped":
#             action = "tune_winsorize"
#             reasons.append(f"frac_clipped={fc:.3f}")
#         # tiny_bulk detection
#         tiny_bulk = (not np.isnan(iqrf) and iqrf <= tiny_bulk_thresh) or (not np.isnan(iqrmed) and iqrmed <= 0.02)
#         if tiny_bulk:
#             if action == "ok": action = "inspect_bulk"
#             reasons.append(f"iqr/full={iqrf:.4g}; iqr/|med|={iqrmed:.4g}")
#         rows.append({
#             "feature": col,
#             "med_tr": m, "iqr_tr": iqrv,
#             "scaled_range": srange,
#             "clip_val": float(clip_val[i]), "clip_te": float(clip_te[i]), "frac_clipped": fc,
#             "clip_class": clip_class,
#             "tail_low_cnt": low_cnt, "tail_high_cnt": high_cnt,
#             "iqr_over_full": iqrf, "iqr_over_med": iqrmed, "narrowness_score": narrow,
#             "nan_mask_ok": tr_u[col].isna().equals(df_scaled[col].iloc[:n_tr].isna()),
#             "conc_score": conc,
#             "suggested_action": action,
#             "reason": ";".join(reasons)
#         })

#     out = pd.DataFrame(rows).set_index("feature").sort_values(["suggested_action","frac_clipped"], ascending=[True, False])

#     # auto-suggest using tiny-bulk and clipping
#     def decide_row_auto(r):
#         frac = r["frac_clipped"]
#         iqrf = r["iqr_over_full"]
#         iqrm = r["iqr_over_med"]
#         tl = int(r["tail_low_cnt"]); th = int(r["tail_high_cnt"])
#         if np.isclose(frac, 1.0):
#             return "leave_binary"
#         if (not pd.isna(iqrf) and iqrf <= tiny_bulk_thresh) or (not pd.isna(iqrm) and iqrm <= 0.02):
#             return "widen"
#         if frac >= 0.75:
#             return "relax_strong"
#         if frac >= 0.5:
#             return "relax_moderate"
#         return r.get("suggested_action", "ok")

#     out["suggested_action_auto"] = out.apply(decide_row_auto, axis=1)

#     # prefix-level medians for feedback
#     def _pref(n): return n.split("_",1)[0] if "_" in n else n
#     pref_map = {}
#     for f, r in out.iterrows():
#         p = _pref(f)
#         pref_map.setdefault(p, []).append(r["frac_clipped"])
#     prefix_frac_med = {p: float(np.nanmedian([v for v in vals if not np.isnan(v)])) if any(~np.isnan(vals)) else np.nan
#                        for p, vals in pref_map.items()}

#     out.attrs["clip_thresh_adaptive"] = clip_thresh_adaptive
#     out.attrs["prefix_frac_med"] = prefix_frac_med
#     valid_conc = conc_scores[~np.isnan(conc_scores)]
#     out.attrs["conc_group_thresh"] = float(np.nanpercentile(valid_conc, 90)) if valid_conc.size else np.nan
#     out.attrs["tail_count_warning"] = tail_support_thresh
#     if only_problem:
#         return out[out["suggested_action_auto"] != "ok"]
#     return out



#########################################################################################################


# def prune_features_by_variance_and_correlation(
#     X_all: pd.DataFrame,
#     y: pd.Series,
#     min_std: float = 1e-6,
#     max_corr: float = 0.9
# ) -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame]:
#     """
#     1) Remove features with std < min_std
#     2) From the remaining features, drop members of high-correlation groups
#        (absolute correlation > max_corr). For each correlated group keep the
#        feature that has the highest absolute correlation with the target y.

#     Returns:
#       kept_after_std      : list of feature names kept after the std filter
#       kept_after_correlation : list of feature names kept after correlation pruning
#       corr_full           : correlation matrix of features after std filter (DataFrame)
#       corr_pruned         : correlation matrix of features after correlation pruning (DataFrame)
#     """
#     # 1) Filter by standard deviation
#     feature_stds = X_all.std(axis=0, ddof=0)
#     kept_after_std = feature_stds[feature_stds >= min_std].index.tolist()
#     dropped_low_variance = feature_stds[feature_stds < min_std].index.tolist()

#     X_var = X_all.loc[:, kept_after_std].copy()

#     # 2) Correlation matrix (absolute) for the features kept after std filter
#     corr_before = X_var.corr().abs()

#     # 3) Upper triangle mask (exclude diagonal)
#     mask_upper = np.triu(np.ones(corr_before.shape), k=1).astype(bool)
#     upper_tri = corr_before.where(mask_upper)

#     # 4) Prune highly correlated features
#     to_drop: Set[str] = set()
#     for col in upper_tri.columns:
#         # find features (rows) correlated above threshold with this column
#         high_corr = upper_tri.index[upper_tri[col] > max_corr].tolist()
#         if high_corr:
#             group = [col] + high_corr
#             # choose best feature in this group by absolute correlation with target y
#             # align indices to be safe
#             corr_with_target = X_var.loc[:, group].corrwith(y).abs()
#             best_feat = corr_with_target.idxmax()
#             to_drop.update(set(group) - {best_feat})

#     kept_after_corr = [f for f in kept_after_std if f not in to_drop]

#     # 5) Correlation matrix after pruning (absolute)
#     corr_after = X_var.loc[:, kept_after_corr].corr().abs()

#     # Logging summary
#     print("Dropped low-variance features:", dropped_low_variance)
#     print("Dropped high-correlation features:", sorted(list(to_drop)))
#     print("Kept after std filter (count):", len(kept_after_std))
#     print("Kept after correlation pruning (count):", len(kept_after_corr))

#     return kept_after_std, kept_after_corr, corr_before, corr_after


def prune_features_by_variance_and_correlation(
    X_all: pd.DataFrame,
    y: pd.Series,
    min_std: float = 1e-6,
    max_corr: float = 0.9,
) -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame]:
    """
    Prune numeric features by variance and pairwise Pearson correlation.
    
    Keeps numeric cols, drops std < min_std, computes one-shot abs Pearson corr,
    groups features with abs(corr) > max_corr, and keeps the member with highest
    abs(corr, y) (fallback: highest std). Prints brief summaries.
    """
    # numeric-only and align y
    X = X_all.select_dtypes(include=[np.number]).copy()
    y = y.reindex(X.index)

    # 1) variance filter
    stds = X.std(axis=0, ddof=0)
    kept_after_std = stds[stds >= min_std].index.tolist()
    dropped_low_variance = stds[stds < min_std].index.tolist()
    X_var = X.loc[:, kept_after_std].copy()
    p = X_var.shape[1]
    if p == 0:
        print("No numeric features after std filter.")
        print("Kept features: []")
        return [], [], pd.DataFrame(), pd.DataFrame()

    # 2) one-shot Pearson correlation via NumPy (handles NaNs by filling col mean)
    arr = X_var.to_numpy(dtype=float)               # shape (n, p)
    col_mean = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    if inds[0].size > 0:
        arr[inds] = np.take(col_mean, inds[1])

    n = arr.shape[0]
    arr_centered = arr - arr.mean(axis=0)
    stdm = arr_centered.std(axis=0, ddof=0)
    stdm_safe = np.where(stdm == 0, 1.0, stdm)
    normed = arr_centered / stdm_safe
    corr = (normed.T @ normed) / max(1, n - 1)
    corr_full = pd.DataFrame(np.clip(np.abs(corr), 0.0, 1.0), index=X_var.columns, columns=X_var.columns)

    # 3) upper triangle mask (exclude diagonal) and greedy grouping
    mask_upper = np.triu(np.ones(corr_full.shape), k=1).astype(bool)
    upper = corr_full.where(mask_upper)
    cols = list(upper.columns)

    # precompute corr with target (abs)
    try:
        corr_with_y = X_var.corrwith(y).abs()
    except Exception:
        corr_with_y = pd.Series(index=X_var.columns, data=np.nan)

    to_drop: Set[str] = set()
    kept_map: Dict[str, List[str]] = {}
    dropped_corr_info: Dict[str, Tuple[str, float]] = {}

    # pruning loop with tqdm (progress visible)
    for col in cols:
        if col in to_drop:
            continue
        high_corr = upper.index[upper[col] > max_corr].tolist()
        high_corr = [h for h in high_corr if h not in to_drop]
        if not high_corr:
            continue
        group = [col] + high_corr
        # choose representative: prefer corr with y, else std
        if corr_with_y.loc[group].notna().any():
            best = corr_with_y.loc[group].idxmax()
        else:
            best = stds.loc[group].idxmax()
        for member in group:
            if member == best:
                continue
            to_drop.add(member)
            kept_map.setdefault(best, []).append(member)
            corr_val = corr_full.loc[member, best] if (member in corr_full.index and best in corr_full.columns) else np.nan
            dropped_corr_info[member] = (best, float(corr_val) if not pd.isna(corr_val) else np.nan)

    pruned_feats = sorted(list(to_drop))
    kept_final_feats = [f for f in kept_after_std if f not in to_drop]
    corr_pruned = corr_full.loc[kept_final_feats, kept_final_feats] if kept_final_feats else pd.DataFrame()

    # 4) prints
    print(f"\nDropped low-variance features (n={len(dropped_low_variance)}):")
    if dropped_low_variance:
        for f in sorted(dropped_low_variance):
            print(f"  Dropped: {f}  (std={stds.loc[f]:.6g})")
    else:
        print("  None")

    if kept_map:
        print(f"\nDropped by correlation (n={len(pruned_feats)}), mapping Dropped <-- Kept (corr):")
        for kept_feat, dropped_list in kept_map.items():
            for dropped_feat in dropped_list:
                corr_val = dropped_corr_info.get(dropped_feat, (kept_feat, np.nan))[1]
                corr_str = f"{corr_val:.4f}" if not np.isnan(corr_val) else "nan"
                print(f"  Dropped: {dropped_feat}  (corr={corr_str})  <-- Kept: {kept_feat}")
    else:
        print("\nDropped by correlation: None")

    print(f"\nKept after std filter (n={len(kept_after_std)}).")
    print(f"Kept after correlation pruning (n={len(kept_final_feats)}).")
    print("\nFinal kept features:")
    print(" ", kept_final_feats if kept_final_feats else "None")

    return kept_final_feats, pruned_feats, corr_full, corr_pruned



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


