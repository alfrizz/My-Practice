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

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler, QuantileTransformer, OrdinalEncoder
from sklearn.decomposition import PCA

from scipy.stats import spearmanr, skew, kurtosis


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

def create_features(
    df: pd.DataFrame,
    mult_feats_win = None,   # float or list[float]; if None default [0.5,1.0,2.0]
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
    vol_spike_window: int   = 14
) -> pd.DataFrame:
    """
    Compute causal OHLCV indicators and multi-scale variants. Returns a DataFrame
    containing original OHLCV + label (params.label_col) plus many generated
    columns named with integer windows (e.g. sma_14, ema_28, macd_diff_12_26_9).

    Key behavior (preserved):
      - mult_feats_win: float or list; default [0.5,1.0,2.0] if None.
      - canonical windows always included; multi-scale versions produced per multiplier.
      - deterministic rounding: W(base,m) = max(1, int(round(base*m))).
      - MACD safety: enforce fast < slow after scaling.
      - All features are computed causally (no lookahead).
    Implementation note (minimal change): to avoid dataframe fragmentation we
    collect computed Series into a dict and concat once; this function returns
    the full per-call series (no internal warmup trimming). Caller should
    perform any global warmup trimming if desired.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    eps = 1e-8

    # multipliers list
    if mult_feats_win is None:
        mults = [0.5, 1.0, 2.0]
    elif isinstance(mult_feats_win, (int, float)):
        mults = [float(mult_feats_win)]
    else:
        mults = list(mult_feats_win)

    # canonical bases
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

    # deterministic rounding helper
    def W(base: int, m: float) -> int:
        return max(1, int(round(base * m)))

    # windows to compute (always include canonical)
    windows = set()
    for m in mults:
        for k, b in bases.items():
            windows.add((k, W(b, m)))
    for k, b in bases.items():
        windows.add((k, W(b, 1.0)))

    def name(pref: str, w: int) -> str:
        return f"{pref}_{w}"

    # base input columns
    cols_in = ["open", "high", "low", "close", "volume", params.label_col]
    base = df[cols_in].copy()
    c = base["close"]; o = base["open"]; h = base["high"]; l = base["low"]

    # collect computed Series here (single concat at end)
    new_cols = {}

    # simple returns
    new_cols["ret"]     = c.pct_change()
    new_cols["log_ret"] = np.log(c + eps).diff()

    # multi-horizon returns & lags
    horizons = sorted({1, 5, 15, 60} | {W(bases["sma_short"], m) for m in mults})
    for H in horizons:
        new_cols[f"ret_{H}"] = c.pct_change(H)
    new_cols["lag1_ret"] = new_cols["ret"].shift(1)
    new_cols["lag2_ret"] = new_cols["ret"].shift(2)
    new_cols["lag3_ret"] = new_cols["ret"].shift(3)

    # ROC
    sma_windows = sorted({w for (k, w) in windows if k.startswith("sma")})
    for w in sma_windows:
        new_cols[f"roc_{w}"] = c.diff(w) / (c.shift(w) + eps)

    # candlestick geometry
    new_cols["body"] = c - o
    new_cols["body_pct"] = (c - o) / (o + eps)
    new_cols["upper_shad"] = h - np.maximum(base["open"], base["close"])
    new_cols["lower_shad"] = np.minimum(base["open"], base["close"]) - l
    new_cols["range_pct"] = (h - l) / (c + eps)

    # SMA / EMA (min_periods=1 for early values)
    sma_set = sorted({w for (k, w) in windows if k in ("sma_short","sma_long")})
    for w in sma_set:
        s = c.rolling(w, min_periods=1).mean()
        new_cols[name("sma", w)] = s
        new_cols[name("sma_pct", w)] = (c - s) / (s + eps)
        e = c.ewm(span=w, adjust=False).mean()
        new_cols[name("ema", w)] = e
        new_cols[name("ema_dev", w)] = (c - e) / (e + eps)

    # RSI
    rsi_windows = sorted({w for (k, w) in windows if k == "rsi"})
    for w in rsi_windows:
        new_cols[name("rsi", w)] = ta.momentum.RSIIndicator(close=c, window=w).rsi()

    # MACD variants (ensure fast < slow)
    macd_windows = sorted({(W(bases["macd_f"], m), W(bases["macd_s"], m), W(bases["macd_sig"], m))
                           for m in mults} | {(bases["macd_f"], bases["macd_s"], bases["macd_sig"])})
    for f_w, s_w, sig_w in macd_windows:
        if f_w >= s_w:
            s_w = f_w + 1
        macd = ta.trend.MACD(close=c, window_fast=f_w, window_slow=s_w, window_sign=sig_w)
        suffix = f"{f_w}_{s_w}_{sig_w}"
        new_cols[f"macd_line_{suffix}"] = macd.macd()
        new_cols[f"macd_signal_{suffix}"] = macd.macd_signal()
        new_cols[f"macd_diff_{suffix}"] = macd.macd_diff()

    # ATR
    atr_windows = sorted({w for (k, w) in windows if k == "atr"})
    for w in atr_windows:
        atr = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=w)
        new_cols[name("atr", w)] = atr.average_true_range()
        new_cols[name("atr_pct", w)] = atr.average_true_range() / (c + eps)

    # Bollinger Bands
    bb_windows = sorted({w for (k, w) in windows if k == "bb"})
    for w in bb_windows:
        bb = ta.volatility.BollingerBands(close=c, window=w, window_dev=2)
        lband = bb.bollinger_lband()
        hband = bb.bollinger_hband()
        mavg = bb.bollinger_mavg()
        new_cols[name("bb_lband", w)] = lband
        new_cols[name("bb_hband", w)] = hband
        new_cols[name("bb_w", w)] = (hband - lband) / (mavg + eps)

    # DI / ADX
    di_windows = sorted({w for (k, w) in windows if k == "atr"})
    for w in di_windows:
        adx = ta.trend.ADXIndicator(high=h, low=l, close=c, window=w)
        new_cols[name("plus_di", w)] = adx.adx_pos()
        new_cols[name("minus_di", w)] = adx.adx_neg()
        new_cols[name("adx", w)] = adx.adx()

    # OBV and stats
    new_cols["obv"] = ta.volume.OnBalanceVolumeIndicator(close=c, volume=base["volume"]).on_balance_volume()
    obv_windows = sorted({w for (k, w) in windows if k == "obv"})
    for w in obv_windows:
        new_cols[name("obv_diff", w)] = new_cols["obv"].diff()
        new_cols[name("obv_pct", w)] = new_cols["obv"].diff(w) / (new_cols["obv"].rolling(w, min_periods=1).mean() + eps)
        new_cols[name("obv_sma", w)] = new_cols["obv"].rolling(w, min_periods=1).mean()
        new_cols[name("obv_z", w)] = (new_cols["obv"] - new_cols[name("obv_sma", w)]) / (new_cols[name("obv_sma", w)].rolling(w, min_periods=1).std() + eps)

    # VWAP and dev
    vwap_w = W(bases["vwap"], 1.0)
    vwap = ta.volume.VolumeWeightedAveragePrice(high=h, low=l, close=c, volume=base["volume"], window=vwap_w)
    new_cols[f"vwap_{vwap_w}"] = vwap.volume_weighted_average_price()
    new_cols[f"vwap_dev_{vwap_w}"] = (c - new_cols[f"vwap_{vwap_w}"]) / (new_cols[f"vwap_{vwap_w}"] + eps)

    # vol_spike
    vol_w = W(bases["vol_spike"], 1.0)
    vol_roll = base["volume"].rolling(vol_w, min_periods=1).mean()
    new_cols[f"vol_spike_{vol_w}"] = base["volume"] / (vol_roll + eps)

    # rolling realized volatility (use min_periods=1)
    vol_windows = sorted({W(bases["sma_short"], m) for m in mults} | {W(bases["sma_long"], 1.0)})
    for w in vol_windows:
        new_cols[f"ret_std_{w}"] = new_cols["ret"].rolling(w, min_periods=1).std()

    # extrema distances
    ext_windows = sorted({W(bases["sma_long"], m) for m in mults})
    for w in ext_windows:
        new_cols[f"rolling_max_close_{w}"] = base["close"].rolling(w, min_periods=1).max()
        new_cols[f"rolling_min_close_{w}"] = base["close"].rolling(w, min_periods=1).min()
        new_cols[f"dist_high_{w}"] = (new_cols[f"rolling_max_close_{w}"] - c) / (c + eps)
        new_cols[f"dist_low_{w}"] = (c - new_cols[f"rolling_min_close_{w}"]) / (c + eps)

    # volume z-score
    v_mu = base["volume"].rolling(vwap_w, min_periods=1).mean()
    v_sigma = base["volume"].rolling(vwap_w, min_periods=1).std()
    new_cols[f"vol_z_{vwap_w}"] = (base["volume"] - v_mu) / (v_sigma + eps)

    # roll z-scores for ATR and BB width (min_periods=1)
    for w in atr_windows:
        x = new_cols[name("atr_pct", w)]
        new_cols[f"z_atr_{w}"] = (x - x.rolling(w, min_periods=1).mean()) / (x.rolling(w, min_periods=1).std() + eps)
    for w in bb_windows:
        x = new_cols[name("bb_w", w)]
        new_cols[f"z_bbw_{w}"] = (x - x.rolling(w, min_periods=1).mean()) / (x.rolling(w, min_periods=1).std() + eps)

    # calendar / cyclical
    new_cols["hour"] = df.index.hour
    new_cols["hour_sin"] = np.sin(2 * np.pi * new_cols["hour"] / 24.0)
    new_cols["hour_cos"] = np.cos(2 * np.pi * new_cols["hour"] / 24.0)
    new_cols["day_of_week"] = df.index.dayofweek
    new_cols["month"] = df.index.month

    # final concat and defragment
    out = pd.concat([base, pd.DataFrame(new_cols, index=base.index)], axis=1)
    out = out.copy()

    return out.copy()



##########################################################################################################


# def features_engineering(
#     df: pd.DataFrame,
#     rsi_low:   float = 30.0,
#     rsi_high:  float = 70.0,
#     adx_thr:   float = 20.0,
#     mult_w:    int   = 14,
#     eps:       float = 1e-8
# ) -> pd.DataFrame:
#     """
#     Derive continuous “eng_” signals comparing indicators to baselines:

#       1) eng_ma       = (SMA_short – SMA_long) / SMA_long
#       2) eng_macd     = MACD_diff / SMA_long
#       3) eng_bb       = distance outside BBands / band width
#       4) eng_bb_mid   = (BB_mid – price) / price
#       5) eng_rsi      = deviation beyond [rsi_low, rsi_high]
#       6) eng_adx      = sign(DI+–DI–) × (|DI+–DI–|/100) × max(ADX–adx_thr,0)/100
#       7) eng_obv      = OBV_diff / OBV_SMA
#       8) eng_atr_div  = (ATR_pct – rolling_mean(ATR_pct, mult_w)) × 10 000
#       9) eng_sma_short= (SMA_short – price) / price
#      10) eng_sma_long = (SMA_long  – price) / price
#      11) eng_vwap     = (VWAP – price) / price
#     """
#     out   = pd.DataFrame(index=df.index)
#     close = df["close"]

#     # discover true column names dynamically
#     sma_cols      = sorted(
#         [c for c in df if c.startswith("sma_") and "_pct" not in c],
#         key=lambda c: int(c.split("_")[1])
#     )
#     sma_s, sma_l  = sma_cols[:2]
#     macd_diff_col = next(c for c in df if c.startswith("macd_diff_"))
#     bb_l_col      = next(c for c in df if c.startswith("bb_lband_"))
#     bb_h_col      = next(c for c in df if c.startswith("bb_hband_"))
#     bb_w_col      = next(c for c in df if c.startswith("bb_w_"))
#     rsi_col       = next(c for c in df if c.startswith("rsi_"))
#     plus_di_col   = next(c for c in df if c.startswith("plus_di_"))
#     minus_di_col  = next(c for c in df if c.startswith("minus_di_"))
#     adx_col       = next(c for c in df if c.startswith("adx_"))
#     obv_diff_col  = next(c for c in df if c.startswith("obv_diff_"))
#     obv_sma_col   = next(c for c in df if c.startswith("obv_sma_"))
#     vwap_col      = next(c for c in df if c.startswith("vwap_") and not c.endswith("_dev"))
#     atr_pct_col   = next(c for c in df if c.startswith("atr_pct_"))

#     # 1) MA spread ratio
#     out["eng_ma"] = (df[sma_s] - df[sma_l]) / (df[sma_l] + eps)

#     # 2) MACD diff over SMA_long
#     out["eng_macd"] = df[macd_diff_col] / (df[sma_l] + eps)

#     # 3) BB distance outside bands
#     lo, hi, bw = df[bb_l_col], df[bb_h_col], df[bb_w_col]
#     dev = np.where(close<lo, lo-close, np.where(close>hi, close-hi, 0.0))
#     out["eng_bb"] = dev / (bw + eps)

#     # 4) BB mid-band offset
#     out["eng_bb_mid"] = ((lo + hi)/2 - close) / (close + eps)

#     # 5) RSI threshold deviation
#     rv     = df[rsi_col]
#     low_d  = np.clip((rsi_low - rv), 0, None) / 100.0
#     high_d = np.clip((rv - rsi_high), 0, None) / 100.0
#     out["eng_rsi"] = np.where(rv<rsi_low, low_d, np.where(rv>rsi_high, high_d, 0.0))

#     # 6) ADX-weighted DI spread
#     di_diff   = df[plus_di_col] - df[minus_di_col]
#     diff_abs  = di_diff.abs() / 100.0
#     ex        = np.clip((df[adx_col]-adx_thr)/100.0, 0, None)
#     out["eng_adx"] = np.sign(di_diff) * diff_abs * ex

#     # 7) OBV divergence ratio
#     out["eng_obv"] = df[obv_diff_col] / (df[obv_sma_col] + eps)

#     # 8) ATR stationary deviation
#     ratio = df[atr_pct_col]
#     rm    = ratio.rolling(mult_w, min_periods=1).mean()
#     out["eng_atr_div"] = (ratio - rm) * 10_000

#     # 9) SMA vs price offsets
#     out["eng_sma_short"] = (df[sma_s] - close) / (close + eps)
#     out["eng_sma_long"]  = (df[sma_l] - close) / (close + eps)

#     # 10) VWAP vs price offset
#     out["eng_vwap"] = (df[vwap_col] - close) / (close + eps)

#     return out.dropna()

import re
import numpy as np
import pandas as pd
from typing import List

def features_engineering(
    df: pd.DataFrame,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
    adx_thr: float = 20.0,
    mult_w: int = 14,
    eps: float = 1e-8
) -> pd.DataFrame:
    """
    Build engineered, stationary signals from an indicator DataFrame.

    - Dynamically discovers available base indicator columns (sma_, rsi_, macd_diff_, bb_*, adx_*, obv_*, vwap_*, atr_pct_, ret_*, etc.).
    - Computes each engineered column only when its bases exist.
    - Returns only the engineered columns that were actually produced, with NaNs filled as 0.0 and dtype=float64.
    """
    out = pd.DataFrame(index=df.index)
    produced: List[str] = []

    # safe helpers
    def find(prefix: str):
        return next((c for c in df.columns if c.startswith(prefix)), None)

    def find_sma_pairs():
        sma_cols = [c for c in df.columns if c.startswith("sma_") and "_pct" not in c]
        def keyfn(c):
            m = re.search(r"_(\d+)$", c)
            return int(m.group(1)) if m else 10**9
        sma_cols = sorted(sma_cols, key=keyfn)
        if len(sma_cols) >= 2:
            return sma_cols[0], sma_cols[1]
        if len(sma_cols) == 1:
            return sma_cols[0], sma_cols[0]
        return None, None

    close = df.get("close", pd.Series(index=df.index, dtype=float)).astype(float)

    # discover bases
    sma_s, sma_l = find_sma_pairs()
    macd_diff = find("macd_diff_")
    bb_l = find("bb_lband_"); bb_h = find("bb_hband_"); bb_w = find("bb_w_")
    rsi_col = find("rsi_")
    plus_di = find("plus_di_"); minus_di = find("minus_di_"); adx_col = find("adx_")
    obv_diff = find("obv_diff_"); obv_sma = find("obv_sma_")
    vwap_dev = find("vwap_dev_") or find("vwap_")
    atr_pct = find("atr_pct_")

    # compute engineered features (append name to produced each time)
    if sma_s and sma_l:
        out["eng_ma"] = (df[sma_s].astype(float) - df[sma_l].astype(float)) / (df[sma_l].astype(float) + eps)
        produced.append("eng_ma")

    if macd_diff and sma_l:
        out["eng_macd"] = df[macd_diff].astype(float) / (df[sma_l].astype(float) + eps)
        produced.append("eng_macd")

    if bb_l and bb_h and bb_w:
        lo = df[bb_l].astype(float); hi = df[bb_h].astype(float); bw = df[bb_w].astype(float)
        dev = np.where(close < lo, lo - close, np.where(close > hi, close - hi, 0.0))
        out["eng_bb"] = pd.Series(dev, index=df.index) / (bw + eps)
        out["eng_bb_mid"] = ((lo + hi) / 2 - close) / (close + eps)
        produced.extend(["eng_bb", "eng_bb_mid"])

    if rsi_col:
        rv = df[rsi_col].astype(float)
        low_d = np.clip((rsi_low - rv), 0, None) / 100.0
        high_d = np.clip((rv - rsi_high), 0, None) / 100.0
        out["eng_rsi"] = np.where(rv < rsi_low, low_d, np.where(rv > rsi_high, high_d, 0.0))
        produced.append("eng_rsi")

    if plus_di and minus_di and adx_col:
        di_diff = df[plus_di].astype(float) - df[minus_di].astype(float)
        diff_abs = di_diff.abs() / 100.0
        ex = np.clip((df[adx_col].astype(float) - adx_thr) / 100.0, 0, None)
        out["eng_adx"] = np.sign(di_diff) * diff_abs * ex
        produced.append("eng_adx")

    if obv_diff and obv_sma:
        out["eng_obv"] = df[obv_diff].astype(float) / (df[obv_sma].astype(float) + eps)
        produced.append("eng_obv")

    if atr_pct:
        ratio = df[atr_pct].astype(float)
        rm = ratio.rolling(mult_w, min_periods=1).mean()
        out["eng_atr_div"] = (ratio - rm) * 10_000
        out["z_eng_atr"] = (ratio - rm) / (ratio.rolling(mult_w, min_periods=1).std() + eps)
        produced.extend(["eng_atr_div", "z_eng_atr"])

    if sma_s:
        out["eng_sma_short"] = (df[sma_s].astype(float) - close) / (close + eps)
        produced.append("eng_sma_short")
    if sma_l:
        out["eng_sma_long"] = (df[sma_l].astype(float) - close) / (close + eps)
        produced.append("eng_sma_long")

    if vwap_dev:
        out["eng_vwap"] = (df[vwap_dev].astype(float) - close) / (close + eps)
        x = df[vwap_dev].astype(float)
        out["z_vwap_dev"] = (x - x.rolling(mult_w, min_periods=1).mean()) / (x.rolling(mult_w, min_periods=1).std() + eps)
        produced.extend(["eng_vwap", "z_vwap_dev"])

    if bb_w:
        x = df[bb_w].astype(float)
        out["z_bb_w"] = (x - x.rolling(mult_w, min_periods=1).mean()) / (x.rolling(mult_w, min_periods=1).std() + eps)
        produced.append("z_bb_w")

    if obv_diff and obv_sma:
        x = df[obv_diff].astype(float)
        out["z_obv_diff"] = (x - df[obv_sma].astype(float)) / (df[obv_sma].astype(float).rolling(mult_w, min_periods=1).std() + eps)
        produced.append("z_obv_diff")

    # momentum aggregates (only if ret_* exist)
    for H in [1, 5, 15, 60]:
        ret_col = f"ret_{H}"
        if ret_col in df:
            out[f"mom_sum_{H}"] = df[ret_col].rolling(H, min_periods=1).sum()
            out[f"mom_std_{H}"] = df[ret_col].rolling(H, min_periods=1).std()
            produced.extend([f"mom_sum_{H}", f"mom_std_{H}"])

    # ema crossover flags
    try:
        s_w = int(sma_s.split("_")[1]) if sma_s else None
        l_w = int(sma_l.split("_")[1]) if sma_l else None
    except Exception:
        s_w = l_w = None
    s_ema_col = f"ema_{s_w}" if s_w else None
    l_ema_col = f"ema_{l_w}" if l_w else None
    if s_ema_col in df and l_ema_col in df:
        out["eng_ema_cross_up"] = (df[s_ema_col] > df[l_ema_col]).astype(float)
        out["eng_ema_cross_down"] = (df[s_ema_col] < df[l_ema_col]).astype(float)
        produced.extend(["eng_ema_cross_up", "eng_ema_cross_down"])

    # finalize: ensure produced ordering, numeric dtype, no NaNs
    produced = [p for p in produced if p in out.columns]  # dedupe/filter
    for col in produced:
        out[col] = out[col].astype(float).fillna(0.0)

    return out[produced].copy()


##########################################################################################################



def scale_minmax_all(
    df: pd.DataFrame,
    label_col: str,
    train_prop: float = 0.7,
    val_prop: float = 0.15,
    winsorize: bool = True,
    pct_low: float = 1.0,
    pct_high: float = 99.0
) -> pd.DataFrame:
    """
    Fit TRAIN-based per-feature winsorization (optional) and MinMax [0,1] scaling,
    then transform TRAIN / VAL / TEST day-by-day preserving NaNs and index.

    Key points
      - Fits percentiles and min/max on the contiguous TRAIN slice only.
      - Optional winsorize (pct_low, pct_high) clips extreme spikes before min/max.
      - All numeric columns except label_col are mapped to [0,1]; non-numeric columns
        are left unchanged and label_col is preserved.
      - Transformation is applied per-day (groupby index.normalize()) to keep
        intra-day NaN structure and avoid cross-day leakage.
      - Constant features on TRAIN are mapped to 0.0; features all-NaN on TRAIN
        remain NaN after transform.
    """

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.replace([np.inf, -np.inf], np.nan)

    # cast numeric to float64 early
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].astype(np.float64)

    # contiguous splits
    N = len(df)
    n_tr = int(N * train_prop)
    n_val = int(N * val_prop)
    if n_tr + n_val >= N:
        raise ValueError("train_prop + val_prop must sum < 1.0")
    df_tr = df.iloc[:n_tr].copy()
    df_v  = df.iloc[n_tr:n_tr + n_val].copy()
    df_te = df.iloc[n_tr + n_val:].copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c != label_col]
    if len(feat_cols) == 0:
        return df

    # ---------- SHOW progress for percentile computation ----------
    if winsorize:
        # show a short progress bar to indicate percentiles are being computed
        # we compute per-column percentiles; wrap in tqdm for visibility
        cols = feat_cols
        clip_low = np.empty(len(cols), dtype=float)
        clip_high = np.empty(len(cols), dtype=float)
        for i, c in enumerate(tqdm(cols, desc="compute pct for cols", unit="col")):
            col_vals = df_tr[c].to_numpy(dtype=float)
            clip_low[i] = np.nanpercentile(col_vals, pct_low)
            clip_high[i] = np.nanpercentile(col_vals, pct_high)
    else:
        clip_low = None
        clip_high = None

    # prepare training matrix after winsorize for min/max calculation (show progress)
    tr_mat = df_tr[feat_cols].to_numpy(dtype=float)
    if winsorize:
        mask_tr = np.isnan(tr_mat)
        tr_clip = np.clip(tr_mat, clip_low, clip_high)
        tr_clip[mask_tr] = np.nan
    else:
        tr_clip = tr_mat

    # compute min / max ignoring NaNs. show per-column progress if many columns
    col_min = np.empty(len(feat_cols), dtype=float)
    col_max = np.empty(len(feat_cols), dtype=float)
    for i, c in enumerate(tqdm(feat_cols, desc="train min/max per-col", unit="col")):
        col = tr_clip[:, i]
        col_min[i] = np.nanmin(col)
        col_max[i] = np.nanmax(col)

    span = col_max - col_min
    const_mask = (np.isnan(span) | (span == 0.0))
    span[const_mask] = 1.0

    def transform_block(block: pd.DataFrame) -> pd.DataFrame:
        A = block[feat_cols].to_numpy(dtype=float)
        if winsorize:
            A = np.clip(A, clip_low, clip_high)
        mask = np.isnan(A)
        out = (A - col_min) / span
        out = np.clip(out, 0.0, 1.0)
        out[mask] = np.nan
        return pd.DataFrame(out, index=block.index, columns=feat_cols, dtype=np.float64)

    def transform_split(split_df: pd.DataFrame, desc: str) -> pd.DataFrame:
        parts = []
        days = split_df.index.normalize().unique()
        total_days = len(days)
        for day, block in tqdm(split_df.groupby(split_df.index.normalize()),
                               desc=desc, unit="day", total=total_days):
            parts.append(transform_block(block))
        if len(parts) == 0:
            return pd.DataFrame(columns=feat_cols, index=split_df.index)
        return pd.concat(parts).reindex(split_df.index)

    # ---------- per-day transforms (existing progress bars) ----------
    tr_scaled = transform_split(df_tr, desc="transform train days")
    v_scaled  = transform_split(df_v,  desc="transform val days")
    te_scaled = transform_split(df_te, desc="transform test days")

    # ---------- show progress for reassembling / final concat ----------
    # assign scaled columns back into splits with a small progress indicator
    def reassemble(orig_split, scaled_feat_df, desc: str):
        out = orig_split.copy()
        # assign per-column to make progress visible when many columns
        for c in tqdm(feat_cols, desc=desc, unit="col"):
            if c in out.columns:
                out.loc[:, c] = scaled_feat_df[c].astype(np.float64)
        return out

    df_tr_s = reassemble(df_tr, tr_scaled, desc="reassemble train cols")
    df_v_s  = reassemble(df_v,  v_scaled,  desc="reassemble val cols")
    df_te_s = reassemble(df_te, te_scaled, desc="reassemble test cols")

    # final concat with a tiny status update
    for _ in tqdm([0], desc="final concat", unit="step"):
        df_all = pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()

    return df_all



# def assign_feature_groups(
#     df: pd.DataFrame,
#     cols: List[str],
#     *,
#     ratio_range:     float = 0.15,
#     heavy_thresh:    float = 1e7,
#     skew_thresh:     float = 3.0,
#     kurtosis_thresh: float = 5.0,
#     discrete_thresh: int   = 10,
#     overrides:       Dict[str, str] = None
# ) -> pd.DataFrame:
#     """
#     Inspect each feature’s raw distribution and bucket it into one of six
#     scaling groups, excluding any pure price‐level columns:

#       • EXCLUDE up front: calendar fields, OHLCV, raw SMA/VWAP/BBands.
#       • Compute min, max, 1/5/95/99-percentiles, skew, kurtosis, unique_count, zero_ratio.
#       • In order assign raw_group:
#          a) discrete
#          b) log_skewed
#          c) ratio
#          d) bounded
#          e) robust_tails
#          f) unbounded
#       • Apply any user overrides → group_final.
#       • Return DataFrame of stats + group_raw/group_final.
#     """
#     # 1) Build reserved set (drop from grouping)
#     reserved = {
#         "hour","day_of_week","month",
#         "open","high","low","close"
#     }
#     # add raw‐level BBands/SMA/VWAP
#     for c in cols:
#         if (c.startswith("bb_lband_")
#             or c.startswith("bb_hband_")
#             or (c.startswith("sma_") and "_pct" not in c)
#             or (c.startswith("vwap_") and not c.endswith("_dev"))):
#             reserved.add(c)

#     feats = [c for c in cols if c not in reserved]

#     # 2) Replace infinities, compute descriptive stats
#     data = df[feats].replace([np.inf, -np.inf], np.nan)
#     descr = (
#         data.describe(percentiles=[0.01,0.05,0.95,0.99])
#             .T
#             .rename(columns={"1%":"1%","5%":"5%","95%":"95%","99%":"99%"})
#     )
#     descr["skew"]         = data.skew().values
#     descr["kurtosis"]     = data.kurtosis().values
#     descr["unique_count"] = data.nunique().values
#     descr["zero_ratio"]   = (data==0).mean().values

#     # 3) Assign raw_group by priority rules
#     raw_group: Dict[str,str] = {}
#     for feat in feats:
#         mn, mx   = descr.at[feat,"min"], descr.at[feat,"max"]
#         p5, p95  = descr.at[feat,"5%"], descr.at[feat,"95%"]
#         sk       = descr.at[feat,"skew"]
#         kt       = descr.at[feat,"kurtosis"]
#         uc       = descr.at[feat,"unique_count"]

#         if uc <= discrete_thresh:
#             grp = "discrete"
#         elif mn >= 0 and sk > skew_thresh:
#             grp = "log_skewed"
#         elif p5 >= -ratio_range and p95 <= ratio_range:
#             grp = "ratio"
#         elif mn >= 0 and mx <= 100:
#             grp = "bounded"
#         elif abs(mn) >= heavy_thresh or abs(mx) >= heavy_thresh or kt >= kurtosis_thresh:
#             grp = "robust_tails"
#         else:
#             grp = "unbounded"

#         raw_group[feat] = grp

#     # 4) Apply overrides
#     overrides   = overrides or {}
#     final_group = {f: overrides.get(f, raw_group[f]) for f in feats}

#     # 5) Assemble assignment DataFrame
#     df_assign = descr.copy()
#     df_assign["group_raw"]   = df_assign.index.map(raw_group)
#     df_assign["group_final"] = df_assign.index.map(final_group)
#     return df_assign


  
# ##########################################################################################################


# def scale_with_splits(
#     df: pd.DataFrame,
#     assignment: pd.DataFrame,
#     train_prop: float = params.train_prop,
#     val_prop:   float = params.val_prop
# ) -> pd.DataFrame:
#     """
#     1) Copy & datetime‐parse; mask ±∞ → NaN.
#     2) Build cyclical calendar features, cast them to float.
#     3) Split CONTIGUOUSLY into TRAIN/VAL/TEST.
#     4) PCA‐compress each sin/cos pair → single calendar dimension.
#     5) Define reserved = raw OHLCV, label, calendar, plus
#        raw‐level BBands/SMA/VWAP.
#     6) From assignment.group_final, build six pipelines: bounded, ratio,
#        log_skewed, robust_tails, discrete, unbounded.
#     7) Fit on TRAIN features.
#     8) Transform each split day‐by‐day (tqdm) to preserve NaNs.
#     9) Drop all raw OHLCV & raw‐level columns; return final scaled DataFrame.
#     """
#     df = df.copy()
#     df.index = pd.to_datetime(df.index)
#     df = df.replace([np.inf, -np.inf], np.nan)

#     # 2) calendar features + sin/cos
#     df["hour"], df["day_of_week"], df["month"] = (
#         df.index.hour, df.index.dayofweek, df.index.month
#     )
#     for name, period in [("hour", 24), ("day_of_week", 7), ("month", 12)]:
#         vals = df[name]
#         df[f"{name}_sin"] = np.sin(2 * np.pi * vals / period)
#         df[f"{name}_cos"] = np.cos(2 * np.pi * vals / period)

#     # 2.1) cast raw calendar cols to float to avoid dtype‐incompatible assignment
#     for name in ("hour","day_of_week","month"):
#         df[name] = df[name].astype(np.float64)

#     # 3) CONTIGUOUS splits
#     N    = len(df)
#     n_tr = int(N * train_prop)
#     n_val= int(N * val_prop)
#     if n_tr + n_val >= N:
#         raise ValueError("train_prop + val_prop must sum < 1.0")
#     df_tr = df.iloc[:n_tr].copy()
#     df_v  = df.iloc[n_tr:n_tr + n_val].copy()
#     df_te = df.iloc[n_tr + n_val:].copy()

#     # 4) PCA compress sin/cos
#     for name in ("hour","day_of_week","month"):
#         pair = [f"{name}_sin", f"{name}_cos"]
#         pca  = PCA(n_components=1).fit(df_tr[pair])
#         for split in (df_tr, df_v, df_te):
#             # use .loc to avoid SettingWithCopyWarning
#             split.loc[:, name] = pca.transform(split[pair]).ravel()
#             split.drop(columns=pair, inplace=True)

#     # 5) reserved vs feature columns
#     reserved = {
#         "open","high","low","close",
#         params.label_col,
#         "hour","day_of_week","month"
#     }
#     # also drop raw‐level BBands/SMA/VWAP
#     for c in df.columns:
#         if (
#             c.startswith("bb_lband_")
#             or c.startswith("bb_hband_")
#             or (c.startswith("sma_") and "_pct" not in c)
#             or (c.startswith("vwap_") and not c.endswith("_dev"))
#         ):
#             reserved.add(c)

#     feat_cols = [c for c in df_tr.columns if c not in reserved]

#     # 6) build pipelines
#     mapping = assignment["group_final"].to_dict()
#     groups  = {
#         grp: [f for f in feat_cols if mapping.get(f) == grp]
#         for grp in ["bounded","ratio","log_skewed","robust_tails","robust_tails_light","robust_tails_heavy","discrete","unbounded"]
#     }

#     def clip01(X: np.ndarray) -> np.ndarray:
#         mask = np.isnan(X)
#         out  = np.clip(X, 0.0, 1.0)
#         out[mask] = np.nan
#         return out

#     class Winsorizer(FunctionTransformer):
#         def __init__(self, lower_pct=1.0, upper_pct=99.0):
#             super().__init__(func=None, inverse_func=None, validate=False)
#             self.lower_pct = lower_pct
#             self.upper_pct = upper_pct
    
#         def fit(self, X, y=None):
#             self.low_, self.high_ = (
#                 np.nanpercentile(X, self.lower_pct, axis=0),
#                 np.nanpercentile(X, self.upper_pct, axis=0)
#             )
#             return self
    
#         def transform(self, X):
#             mask = np.isnan(X)
#             out  = np.clip(X, self.low_, self.high_)
#             out[mask] = np.nan
#             return out

#     def signed_log(X: np.ndarray) -> np.ndarray:
#         return np.sign(X) * np.log1p(np.abs(X))

#     pipelines = [
#         ("bnd", Pipeline([
#             ("clip100", FunctionTransformer(lambda X: np.clip(X, 0, 100), validate=False)),
#             ("mm",      MinMaxScaler()),
#             ("c01",     FunctionTransformer(clip01, validate=False)),
#         ]), groups["bounded"]),

#         ("rat", Pipeline([
#             ("pt",  PowerTransformer(method="yeo-johnson", standardize=False)),
#             ("mm",  MinMaxScaler()),
#             ("c01", FunctionTransformer(clip01, validate=False)),
#         ]), groups["ratio"]),

#         ("lgs", Pipeline([
#             ("slog", FunctionTransformer(signed_log, validate=False)),
#             ("mm",   MinMaxScaler()),
#             ("c01",  FunctionTransformer(clip01, validate=False)),
#         ]), groups["log_skewed"]),

#         ("robust_tails_light", Pipeline([
#             ("win", Winsorizer(lower_pct=0.005, upper_pct=99.995)),
#             ("mm",  MinMaxScaler()),
#             ("c01", FunctionTransformer(clip01, validate=False)),
#         ]), groups["robust_tails_light"]),
    
#         ("robust_tails", Pipeline([
#             ("win", Winsorizer(lower_pct=1.0, upper_pct=99.0)),
#             ("mm",  MinMaxScaler()),
#             ("c01", FunctionTransformer(clip01, validate=False)),
#         ]), groups["robust_tails"]),
    
#         ("robust_tails_heavy", Pipeline([
#             ("win", Winsorizer(lower_pct=20, upper_pct=80)),
#             ("mm",  MinMaxScaler()),
#             ("c01", FunctionTransformer(clip01, validate=False)),
#         ]), groups["robust_tails_heavy"]),

#         ("dis", Pipeline([
#             ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
#             ("mm",  MinMaxScaler()),
#             ("c01", FunctionTransformer(clip01, validate=False)),
#         ]), groups["discrete"]),

#         ("unb", Pipeline([
#             ("std", StandardScaler()),
#             ("c3",  FunctionTransformer(lambda X: np.clip(X, -3, 3), validate=False)),
#             ("mm",  MinMaxScaler()),
#             ("c01", FunctionTransformer(clip01, validate=False)),
#         ]), groups["unbounded"]),
#     ]
#     ct = ColumnTransformer(transformers=pipelines, remainder="drop")

#     # 7) fit on TRAIN
#     ct.fit(df_tr[feat_cols])

#     # 8) flatten names & widths
#     flat_feats = [f for _,_,cols in pipelines for f in cols]
#     n_feats    = len(flat_feats)

#     # 9) transform day‐by‐day
#     def transform_by_day(split_df: pd.DataFrame, label: str) -> pd.DataFrame:
#         arr = np.empty((len(split_df), n_feats), dtype=float)
#         for day, block in tqdm(
#             split_df.groupby(split_df.index.normalize()),
#             desc=f"{label} days", unit="day"
#         ):
#             mask = split_df.index.normalize() == day
#             arr[mask] = ct.transform(block[feat_cols])

#         scaled = pd.DataFrame(arr, index=split_df.index, columns=flat_feats)
#         return pd.concat([scaled, split_df[list(reserved)]], axis=1)[split_df.columns]

#     df_tr_s = transform_by_day(df_tr, "train")
#     df_v_s  = transform_by_day(df_v,  "val")
#     df_te_s = transform_by_day(df_te, "test")

#     # 10) reassemble & drop raw columns
#     df_all = pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()
#     to_drop = ["open","high","low","close"] + [
#         c for c in df_all.columns
#         if c.startswith("bb_lband_")
#         or c.startswith("bb_hband_")
#         or (c.startswith("sma_") and "_pct" not in c)
#         or (c.startswith("vwap_") and not c.endswith("_dev"))
#     ]
#     return df_all.drop(columns=to_drop, errors="ignore")


# #########################################################################################################


# def compare_raw_vs_scaled(
#     df_raw: pd.DataFrame,
#     df_scaled: pd.DataFrame,
#     assignment: pd.DataFrame,
#     feat_cols: Optional[List[str]] = None,
#     train_prop: float = params.train_prop,
#     tol_range:  float = 1e-6
# ) -> pd.DataFrame:
#     """
#     On the TRAIN slice only, verify that each scaled feature
#     preserves its core invariants under our per-day [0,1] pipelines.

#     Functionality:
#       1) Split off the first train_prop fraction; replace ±inf with NaN.
#       2) Select features common to raw, scaled, optional feat_cols,
#          and present in assignment.index.
#       3) Compute TRAIN‐only percentiles (min,1%,5%,50%,95%,99%,max).
#       4) For each feature on TRAIN:
#            a) NaN‐mask unchanged.
#            b) All scaled values ∈ [0,1] ± tol_range.
#            c) If 'discrete': unique_count unchanged.
#            d) If 'bounded': exact linear clip(raw,0,100)/100 mapping.
#            e) If constant raw: scaled is constant.
#       5) Return a summary DataFrame with pass/fail flags and reasons.
#     """
#     # 1) TRAIN split and NaN‐clean
#     N    = len(df_raw)
#     n_tr = int(N * train_prop)
#     raw_tr = df_raw.iloc[:n_tr].replace([np.inf, -np.inf], np.nan)
#     sca_tr = df_scaled.iloc[:n_tr].replace([np.inf, -np.inf], np.nan)

#     # 2) Determine features to test
#     common = set(raw_tr.columns) & set(sca_tr.columns)
#     if feat_cols is not None:
#         common &= set(feat_cols)
#     features = [f for f in common if f in assignment.index]

#     # 3) Compute TRAIN‐only percentiles
#     qs = [0.01, 0.05, 0.50, 0.95, 0.99]
#     raw_q = (
#         raw_tr[features]
#         .describe(percentiles=qs).T
#         .loc[:, ['min','1%','5%','50%','95%','99%','max']]
#         .add_prefix('raw_')
#     )
#     sca_q = (
#         sca_tr[features]
#         .describe(percentiles=qs).T
#         .loc[:, ['min','1%','5%','50%','95%','99%','max']]
#         .add_prefix('scaled_')
#     )
#     cmp_df = pd.concat([raw_q, sca_q], axis=1)
#     cmp_df['group_final'] = assignment['group_final']

#     # 4) Prepare result containers
#     nan_ok       = {}
#     range_ok     = {}
#     discrete_ok  = {}
#     bounded_ok   = {}
#     const_ok     = {}

#     # 5) Per‐feature checks on TRAIN
#     for feat in tqdm(features, desc="Validating train-split"):
#         grp = assignment.at[feat, 'group_final']

#         # a) NaN‐mask unchanged
#         nan_ok[feat] = (raw_tr[feat].isna() == sca_tr[feat].isna()).all()

#         # align non‐NaN pairs
#         x = raw_tr[feat].dropna()
#         y = sca_tr[feat].dropna()
#         x, y = x.align(y, join='inner')

#         # b) Range containment [0,1]
#         if len(y):
#             ymin, ymax = y.min(), y.max()
#             range_ok[feat] = (ymin >= -tol_range) and (ymax <= 1 + tol_range)
#         else:
#             range_ok[feat] = True

#         # c) Discrete cardinality
#         if grp == 'discrete':
#             discrete_ok[feat] = x.nunique() == y.nunique()
#         else:
#             discrete_ok[feat] = True

#         # d) Bounded linear mapping
#         if grp == 'bounded' and len(x):
#             # clip raw to [0,100], then compute train-min/max
#             x_clip = np.clip(x, 0, 100)
#             lo, hi = x_clip.min(), x_clip.max()
#             # map into [0,1]
#             target = (x_clip - lo) / (hi - lo) if hi > lo else 0
#             bounded_ok[feat] = np.allclose(y, target, atol=tol_range)
#         else:
#             bounded_ok[feat] = True

#         # e) Constant‐feature behavior
#         rmin, rmax = raw_q.at[feat, 'raw_min'], raw_q.at[feat, 'raw_max']
#         if abs(rmax - rmin) < tol_range:
#             const_ok[feat] = y.nunique() == 1
#         else:
#             const_ok[feat] = True

#     # 6) Compile pass/fail status & reasons
#     status, reason = [], []
#     for feat in features:
#         errs = []
#         if not nan_ok[feat]:
#             errs.append("nan_mask_changed")
#         if not range_ok[feat]:
#             mn, mx = sca_q.at[feat, 'scaled_min'], sca_q.at[feat, 'scaled_max']
#             errs.append(f"range[{mn:.3f},{mx:.3f}]")
#         if not discrete_ok[feat]:
#             errs.append("cardinality_changed")
#         if not bounded_ok[feat]:
#             errs.append("non-linear_bounded")
#         if not const_ok[feat]:
#             errs.append("constant_not_const")

#         if errs:
#             status.append("FAIL")
#             reason.append("; ".join(errs))
#         else:
#             status.append("OK")
#             reason.append("all checks passed")

#     cmp_df['nan_mask_ok']       = pd.Series(nan_ok)
#     cmp_df['range_ok']          = pd.Series(range_ok)
#     cmp_df['discrete_ok']       = pd.Series(discrete_ok)
#     cmp_df['bounded_linear_ok'] = pd.Series(bounded_ok)
#     cmp_df['constant_ok']       = pd.Series(const_ok)
#     cmp_df['status']            = status
#     cmp_df['reason']            = reason

#     return cmp_df




def compare_raw_vs_scaled(
    df_raw: pd.DataFrame,
    df_scaled: pd.DataFrame,
    feat_cols: Optional[List[str]] = None,
    train_prop: float = 0.7,
    tol_range: float = 1e-6,
    return_failures: bool = True,
    clip_thresh: float = 0.01
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    TRAIN-only validation of [0,1] MinMax scaling plus simple VAL/TEST clip checks.

    Behavior:
      - Uses contiguous TRAIN slice (first train_prop fraction) to run invariants:
         * nan_mask_ok: missing-data mask unchanged on TRAIN
         * range_ok: scaled values on TRAIN lie in [0,1] +/- tol_range
         * const_ok: if raw is constant on TRAIN, scaled is constant
      - If feat_cols is not given, automatically selects numeric features but
        excludes known raw-level drifting columns (open/high/low/close and patterns).
      - Adds simple clip-rate checks on VAL and TEST: fraction of values == 0 or == 1;
        flags features where clip rate in VAL or TEST exceeds clip_thresh.
      - Returns (cmp_df, failures_df) where failures_df = rows with status FAIL.
    """
    # splits
    N = len(df_raw)
    n_tr = int(N * train_prop)
    raw_tr = df_raw.iloc[:n_tr].replace([np.inf, -np.inf], np.nan)
    sca_tr = df_scaled.iloc[:n_tr].replace([np.inf, -np.inf], np.nan)
    sca_v  = df_scaled.iloc[n_tr:int(n_tr + N * (1 - train_prop) * 0.0) + 0]  # placeholder, not used
    # define val/test splits explicitly
    n_val = int(N * (1 - train_prop) / 2)  # best-effort equal split of remainder
    df_val = df_scaled.iloc[n_tr:n_tr + n_val] if n_val > 0 else df_scaled.iloc[0:0]
    df_test = df_scaled.iloc[n_tr + n_val:] if (n_tr + n_val) < N else df_scaled.iloc[0:0]

    # 2) determine features to test
    common = set(raw_tr.columns) & set(sca_tr.columns)
    if feat_cols is not None:
        common &= set(feat_cols)

    # exclude raw-level drifting patterns by default
    raw_patterns = [
        r"^open$", r"^high$", r"^low$", r"^close$",
        r"^bb_lband_", r"^bb_hband_", r"^sma_\d+$", r"^vwap_(?!.*_dev)",
        r"^rolling_max_close_", r"^rolling_min_close_"
    ]
    def is_raw_level(col: str) -> bool:
        for p in raw_patterns:
            if re.match(p, col):
                return True
        return False

    numeric = [c for c in sorted(common)
               if pd.api.types.is_numeric_dtype(raw_tr[c]) and pd.api.types.is_numeric_dtype(sca_tr[c]) and not is_raw_level(c)]
    features = numeric

    if len(features) == 0:
        empty = pd.DataFrame(columns=[
            "raw_min","raw_1%","raw_5%","raw_50%","raw_95%","raw_99%","raw_max",
            "scaled_min","scaled_1%","scaled_5%","scaled_50%","scaled_95%","scaled_99%","scaled_max",
            "nan_mask_ok","range_ok","const_ok",
            "val_clip_rate","test_clip_rate","clip_ok","status","reason"
        ])
        return (empty, empty) if return_failures else (empty, None)

    # percentiles on TRAIN
    qs = [0.01, 0.05, 0.50, 0.95, 0.99]
    for _ in tqdm([0], desc="precomputing TRAIN percentiles", unit="step"):
        raw_desc = raw_tr[features].describe(percentiles=qs).T
        sca_desc = sca_tr[features].describe(percentiles=qs).T

    # per-feature checks
    nan_ok = {}
    range_ok = {}
    const_ok = {}
    val_clip_rate = {}
    test_clip_rate = {}
    reasons = {}

    for feat in tqdm(features, desc="Validating train-split", unit="feat"):
        # a) NaN-mask unchanged
        nan_ok[feat] = (raw_tr[feat].isna() == sca_tr[feat].isna()).all()

        # align non-NaN pairs on TRAIN
        x = raw_tr[feat].dropna()
        y = sca_tr[feat].dropna()
        x, y = x.align(y, join='inner')

        # b) Range containment [0,1] on TRAIN
        if len(y):
            ymin, ymax = float(y.min()), float(y.max())
            range_ok[feat] = (ymin >= -tol_range) and (ymax <= 1.0 + tol_range)
        else:
            range_ok[feat] = True

        # c) Constant feature behavior on TRAIN
        rmin = raw_desc.at[feat, "min"]
        rmax = raw_desc.at[feat, "max"]
        if np.isfinite(rmin) and np.isfinite(rmax) and abs(rmax - rmin) < tol_range:
            const_ok[feat] = (y.nunique() <= 1)
        else:
            const_ok[feat] = True

        # d) simple VAL/TEST clip rates (fraction exactly 0 or 1)
        def clip_rate(series: pd.Series) -> float:
            if series is None or len(series) == 0:
                return 0.0
            s = series.dropna()
            if len(s) == 0:
                return 0.0
            return float(((s == 0).sum() + (s == 1).sum()) / len(s))

        val_clip_rate[feat] = clip_rate(df_val[feat] if feat in df_val.columns else pd.Series(dtype=float))
        test_clip_rate[feat] = clip_rate(df_test[feat] if feat in df_test.columns else pd.Series(dtype=float))

        # build reason list and final reason
        err = []
        if not nan_ok[feat]:
            err.append("nan_mask_changed")
        if not range_ok[feat]:
            mn, mx = sca_desc.at[feat, "min"], sca_desc.at[feat, "max"]
            err.append(f"range[{mn:.6g},{mx:.6g}]")
        if not const_ok[feat]:
            err.append("constant_not_const")
        if val_clip_rate[feat] > clip_thresh:
            err.append(f"val_clip={val_clip_rate[feat]:.3f}")
        if test_clip_rate[feat] > clip_thresh:
            err.append(f"test_clip={test_clip_rate[feat]:.3f}")

        reasons[feat] = "all checks passed" if not err else "; ".join(err)

    # assemble result rows
    rows = {}
    for feat in features:
        rows[feat] = {
            "raw_min": raw_desc.at[feat, "min"],
            "raw_1%": raw_desc.at[feat, "1%"],
            "raw_5%": raw_desc.at[feat, "5%"],
            "raw_50%": raw_desc.at[feat, "50%"],
            "raw_95%": raw_desc.at[feat, "95%"],
            "raw_99%": raw_desc.at[feat, "99%"],
            "raw_max": raw_desc.at[feat, "max"],
            "scaled_min": sca_desc.at[feat, "min"],
            "scaled_1%": sca_desc.at[feat, "1%"],
            "scaled_5%": sca_desc.at[feat, "5%"],
            "scaled_50%": sca_desc.at[feat, "50%"],
            "scaled_95%": sca_desc.at[feat, "95%"],
            "scaled_99%": sca_desc.at[feat, "99%"],
            "scaled_max": sca_desc.at[feat, "max"],
            "nan_mask_ok": nan_ok[feat],
            "range_ok": range_ok[feat],
            "const_ok": const_ok[feat],
            "val_clip_rate": val_clip_rate[feat],
            "test_clip_rate": test_clip_rate[feat],
            "clip_ok": (val_clip_rate[feat] <= clip_thresh and test_clip_rate[feat] <= clip_thresh),
            "status": ("OK" if (nan_ok[feat] and range_ok[feat] and const_ok[feat] and (val_clip_rate[feat] <= clip_thresh) and (test_clip_rate[feat] <= clip_thresh)) else "FAIL"),
            "reason": reasons[feat],
        }

    cmp_df = pd.DataFrame.from_dict(rows, orient="index")

    failures_df = cmp_df[cmp_df["status"] == "FAIL"].copy() if return_failures else None
    return cmp_df, failures_df



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


