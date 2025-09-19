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

import pandas as pd
import numpy  as np
import math

from captum.attr import IntegratedGradients
from tqdm.auto import tqdm
import ta

import torch
import torch.backends.cudnn as cudnn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler, QuantileTransformer, OrdinalEncoder
from sklearn.decomposition import PCA

from scipy.stats import spearmanr, skew, kurtosis


##########################################################################################################


# def create_features(
#     df: pd.DataFrame,
#     window_multiplier: float = 1.0,
#     sma_short:   int         = 14,
#     sma_long:    int         = 28,
#     rsi_window:  int         = 14,
#     macd_fast:   int         = 12,
#     macd_slow:   int         = 26,
#     macd_sig:    int         = 9,
#     atr_window:  int         = 14,
#     bb_window:   int         = 20,
#     obv_sma:     int         = 14,
#     vwap_window: int         = 14
# ) -> pd.DataFrame:
#     """
#     Vectorized generation of OHLCV‐derived features, including returns:

#     1) Scale all window sizes by window_multiplier.
#     2) Compute price‐change channels:
#          • ret     = simple return
#          • log_ret = log‐return
#     3) Candlestick geometry: body, body_pct, upper_shad, lower_shad, range_pct.
#     4) Popular indicators:
#          • RSI(rsi_window)
#          • MACD(line, signal, diff)
#          • SMA(short/long) + pct deviations
#          • ATR(atr_window) + atr_pct
#          • Bollinger Bands(bb_window) + bb_width
#          • +DI, –DI, ADX(atr_window)
#          • OBV + obv_sma(obv_sma) + obv_pct
#          • VWAP(vwap_window) + vwap_dev
#          • vol_spike ratio
#     5) Calendar: hour, day_of_week, month.
#     6) Drop NaNs and return full DataFrame.
#     """
#     df = df.copy()
#     df.index = pd.to_datetime(df.index)

#     def WM(x):
#         return max(1, int(round(x * window_multiplier)))

#     # derive window lengths
#     w_sma_s = WM(sma_short)
#     w_sma_l = WM(sma_long)
#     w_rsi   = WM(rsi_window)
#     w_atr   = WM(atr_window)
#     w_bb    = WM(bb_window)
#     w_obv   = WM(obv_sma)
#     w_vwap  = WM(vwap_window)

#     # pick base columns
#     cols_in = ["open","high","low","close","volume","bid","ask", params.label_col]
#     out     = df[cols_in].copy()
#     c       = out["close"]

#     # 2) price‐change channels
#     out["ret"]     = c.pct_change()
#     out["log_ret"] = np.log(c).diff()

#     # 3) candlestick geometry
#     o, h, l = out.open, out.high, out.low
#     out["body"]       = c - o
#     out["body_pct"]   = (c - o) / (o + 1e-8)
#     out["upper_shad"] = h - out[["open","close"]].max(axis=1)
#     out["lower_shad"] = out[["open","close"]].min(axis=1) - l
#     out["range_pct"]  = (h - l) / (c + 1e-8)

#     # 4) RSI
#     out[f"rsi_{w_rsi}"] = ta.momentum.RSIIndicator(close=c, window=w_rsi).rsi()

#     # 5) MACD
#     macd = ta.trend.MACD(
#         close=c,
#         window_fast=macd_fast,
#         window_slow=macd_slow,
#         window_sign=macd_sig
#     )
#     out[f"macd_line_{macd_fast}_{macd_slow}_{macd_sig}"]   = macd.macd()
#     out[f"macd_signal_{macd_fast}_{macd_slow}_{macd_sig}"] = macd.macd_signal()
#     out[f"macd_diff_{macd_fast}_{macd_slow}_{macd_sig}"]   = macd.macd_diff()

#     # 6) SMA + pct deviation
#     sma_s = c.rolling(w_sma_s, min_periods=1).mean()
#     sma_l = c.rolling(w_sma_l, min_periods=1).mean()
#     out[f"sma_{w_sma_s}"]     = sma_s
#     out[f"sma_{w_sma_l}"]     = sma_l
#     out[f"sma_pct_{w_sma_s}"] = (c - sma_s) / (sma_s + 1e-8)
#     out[f"sma_pct_{w_sma_l}"] = (c - sma_l) / (sma_l + 1e-8)

#     # 7) ATR + pct
#     atr = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=w_atr).average_true_range()
#     out[f"atr_{w_atr}"]     = atr
#     out[f"atr_pct_{w_atr}"] = atr / (c + 1e-8)

#     # 8) Bollinger Bands + width
#     bb    = ta.volatility.BollingerBands(close=c, window=w_bb, window_dev=2)
#     lband = bb.bollinger_lband()
#     hband = bb.bollinger_hband()
#     mavg  = bb.bollinger_mavg()
#     out[f"bb_lband_{w_bb}"] = lband
#     out[f"bb_hband_{w_bb}"] = hband
#     out[f"bb_w_{w_bb}"]     = (hband - lband) / (mavg + 1e-8)

#     # 9) +DI, –DI, ADX
#     adx = ta.trend.ADXIndicator(high=h, low=l, close=c, window=w_atr)
#     out[f"plus_di_{w_atr}"]  = adx.adx_pos()
#     out[f"minus_di_{w_atr}"] = adx.adx_neg()
#     out[f"adx_{w_atr}"]      = adx.adx()

#     # 10) OBV + SMA + pct
#     obv = ta.volume.OnBalanceVolumeIndicator(close=c, volume=out.volume).on_balance_volume()
#     out["obv"]             = obv
#     out[f"obv_sma_{w_obv}"] = obv.rolling(w_obv, min_periods=1).mean()
#     out[f"obv_pct_{w_obv}"] = obv / (out.volume + 1e-8)

#     # 11) VWAP + deviation
#     vwap = ta.volume.VolumeWeightedAveragePrice(
#         high=h, low=l, close=c, volume=out.volume, window=w_vwap
#     ).volume_weighted_average_price()
#     out[f"vwap_{w_vwap}"]     = vwap
#     out[f"vwap_dev_{w_vwap}"] = (c - vwap) / (vwap + 1e-8)

#     # 12) vol_spike ratio
#     vol_roll = out.volume.rolling(w_obv, min_periods=1).mean()
#     out[f"vol_spike_{w_obv}"] = out.volume / (vol_roll + 1e-8)

#     # 13) calendar columns
#     out["hour"]        = out.index.hour
#     out["day_of_week"] = out.index.dayofweek
#     out["month"]       = out.index.month

#     # drop any NaNs from initial windows
#     return out.dropna()


def create_features(
    df: pd.DataFrame,
    window_multiplier: float = 1.0,
    sma_short:   int   = 14,
    sma_long:    int   = 28,
    rsi_window:  int   = 14,
    macd_fast:   int   = 12,
    macd_slow:   int   = 26,
    macd_sig:    int   = 9,
    atr_window:  int   = 14,
    bb_window:   int   = 20,
    obv_sma:     int   = 14,
    vwap_window: int   = 14,
    vol_spike_window: int = 14
) -> pd.DataFrame:
    """
    Compute raw OHLCV features and classic indicators on 1-min bars,
    scaling every lookback window by window_multiplier.

    Steps:
      1) Scale all indicator windows via window_multiplier (including MACD).
      2) Compute simple returns and log-returns.
      3) Candlestick geometry: body, %body, upper/lower shadows, range_pct.
      4) RSI, MACD line/signal/diff, SMA(short/long) + percent deviations.
      5) ATR + atr_pct, Bollinger Bands + width, +DI/−DI/ADX.
      6) OBV: level, diff, pct_change, rolling SMA.
      7) VWAP + deviation from price.
      8) vol_spike ratio on its own window.
      9) Calendar columns: hour, day_of_week, month.
     10) Drop only the initial rows until every indicator has data.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    eps = 1e-8

    # Helper to scale windows
    def WM(x: int) -> int:
        return max(1, int(round(x * window_multiplier)))

    # 1) scaled window lengths
    w_sma_s     = WM(sma_short)
    w_sma_l     = WM(sma_long)
    w_rsi       = WM(rsi_window)
    w_macd_f    = WM(macd_fast)
    w_macd_s    = WM(macd_slow)
    w_macd_sig  = WM(macd_sig)
    w_atr       = WM(atr_window)
    w_bb        = WM(bb_window)
    w_obv       = WM(obv_sma)
    w_vwap      = WM(vwap_window)
    w_vol_spike = WM(vol_spike_window)

    # Base columns
    cols_in = ["open","high","low","close","volume","bid","ask", params.label_col]
    out     = df[cols_in].copy()
    c, o, h, l = out["close"], out["open"], out["high"], out["low"]

    # 2) Returns
    out["ret"]     = c.pct_change()
    out["log_ret"] = np.log(c + eps).diff()

    # 3) Candlestick geometry
    out["body"]       = c - o
    out["body_pct"]   = (c - o) / (o + eps)
    out["upper_shad"] = h - out[["open","close"]].max(axis=1)
    out["lower_shad"] = out[["open","close"]].min(axis=1) - l
    out["range_pct"]  = (h - l) / (c + eps)

    # 4) RSI
    out[f"rsi_{w_rsi}"] = ta.momentum.RSIIndicator(close=c, window=w_rsi).rsi()

    # 5) MACD (all windows scaled)
    macd = ta.trend.MACD(
        close=c,
        window_fast=w_macd_f,
        window_slow=w_macd_s,
        window_sign=w_macd_sig
    )
    out[f"macd_line_{w_macd_f}_{w_macd_s}_{w_macd_sig}"]   = macd.macd()
    out[f"macd_signal_{w_macd_f}_{w_macd_s}_{w_macd_sig}"] = macd.macd_signal()
    out[f"macd_diff_{w_macd_f}_{w_macd_s}_{w_macd_sig}"]   = macd.macd_diff()

    # 6) SMA + percent deviation
    sma_s = c.rolling(w_sma_s, min_periods=1).mean()
    sma_l = c.rolling(w_sma_l, min_periods=1).mean()
    out[f"sma_{w_sma_s}"]     = sma_s
    out[f"sma_{w_sma_l}"]     = sma_l
    out[f"sma_pct_{w_sma_s}"] = (c - sma_s) / (sma_s + eps)
    out[f"sma_pct_{w_sma_l}"] = (c - sma_l) / (sma_l + eps)

    # 7) ATR + ATR percent
    atr = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=w_atr)
    out[f"atr_{w_atr}"]     = atr.average_true_range()
    out[f"atr_pct_{w_atr}"] = atr.average_true_range() / (c + eps)

    # 8) Bollinger Bands + width
    bb    = ta.volatility.BollingerBands(close=c, window=w_bb, window_dev=2)
    lband = bb.bollinger_lband()
    hband = bb.bollinger_hband()
    mavg  = bb.bollinger_mavg()
    out[f"bb_lband_{w_bb}"] = lband
    out[f"bb_hband_{w_bb}"] = hband
    out[f"bb_w_{w_bb}"]     = (hband - lband) / (mavg + eps)

    # 9) +DI, −DI, ADX
    adx = ta.trend.ADXIndicator(high=h, low=l, close=c, window=w_atr)
    out[f"plus_di_{w_atr}"]  = adx.adx_pos()
    out[f"minus_di_{w_atr}"] = adx.adx_neg()
    out[f"adx_{w_atr}"]      = adx.adx()

    # 10) OBV: level, diff, pct_change, SMA
    obv = ta.volume.OnBalanceVolumeIndicator(close=c, volume=out["volume"])
    out["obv"]              = obv.on_balance_volume()
    out[f"obv_diff_{w_obv}"] = out["obv"].diff()
    out[f"obv_pct_{w_obv}"]  = out["obv"].pct_change()
    out[f"obv_sma_{w_obv}"]  = out["obv"].rolling(w_obv, min_periods=1).mean()

    # 11) VWAP + deviation
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=h, low=l, close=c, volume=out["volume"], window=w_vwap
    )
    out[f"vwap_{w_vwap}"]     = vwap.volume_weighted_average_price()
    out[f"vwap_dev_{w_vwap}"] = (c - out[f"vwap_{w_vwap}"]) / (out[f"vwap_{w_vwap}"] + eps)

    # 12) vol_spike ratio (dedicated window)
    vol_roll = out["volume"].rolling(w_vol_spike, min_periods=1).mean()
    out[f"vol_spike_{w_vol_spike}"] = out["volume"] / (vol_roll + eps)

    # 13) calendar
    out["hour"]        = df.index.hour
    out["day_of_week"] = df.index.dayofweek
    out["month"]       = df.index.month

    # 14) drop only until every column is first valid
    first_valid = out.apply(lambda series: series.first_valid_index()).max()
    return out.loc[first_valid:].copy()


def features_engineering(
    df: pd.DataFrame,
    rsi_low:   float = 30.0,
    rsi_high:  float = 70.0,
    adx_thr:   float = 20.0,
    mult_w:    int   = 14,
    eps:       float = 1e-8
) -> pd.DataFrame:
    """
    Derive continuous “eng_” signals comparing indicators to baselines:

      1) eng_ma       = (SMA_short – SMA_long) / SMA_long
      2) eng_macd     = MACD_diff / SMA_long
      3) eng_bb       = distance outside BBands / band width
      4) eng_bb_mid   = (BB_mid – price) / price
      5) eng_rsi      = deviation beyond [rsi_low, rsi_high]
      6) eng_adx      = sign(DI+–DI–) × (|DI+–DI–|/100) × max(ADX–adx_thr,0)/100
      7) eng_obv      = OBV_diff / OBV_SMA
      8) eng_atr_div  = (ATR_pct – rolling_mean(ATR_pct, mult_w)) × 10 000
      9) eng_sma_short= (SMA_short – price) / price
     10) eng_sma_long = (SMA_long  – price) / price
     11) eng_vwap     = (VWAP – price) / price
    """
    out   = pd.DataFrame(index=df.index)
    close = df["close"]

    # discover true column names dynamically
    sma_cols      = sorted(
        [c for c in df if c.startswith("sma_") and "_pct" not in c],
        key=lambda c: int(c.split("_")[1])
    )
    sma_s, sma_l  = sma_cols[:2]
    macd_diff_col = next(c for c in df if c.startswith("macd_diff_"))
    bb_l_col      = next(c for c in df if c.startswith("bb_lband_"))
    bb_h_col      = next(c for c in df if c.startswith("bb_hband_"))
    bb_w_col      = next(c for c in df if c.startswith("bb_w_"))
    rsi_col       = next(c for c in df if c.startswith("rsi_"))
    plus_di_col   = next(c for c in df if c.startswith("plus_di_"))
    minus_di_col  = next(c for c in df if c.startswith("minus_di_"))
    adx_col       = next(c for c in df if c.startswith("adx_"))
    obv_diff_col  = next(c for c in df if c.startswith("obv_diff_"))
    obv_sma_col   = next(c for c in df if c.startswith("obv_sma_"))
    vwap_col      = next(c for c in df if c.startswith("vwap_") and not c.endswith("_dev"))
    atr_pct_col   = next(c for c in df if c.startswith("atr_pct_"))

    # 1) MA spread ratio
    out["eng_ma"] = (df[sma_s] - df[sma_l]) / (df[sma_l] + eps)

    # 2) MACD diff over SMA_long
    out["eng_macd"] = df[macd_diff_col] / (df[sma_l] + eps)

    # 3) BB distance outside bands
    lo, hi, bw = df[bb_l_col], df[bb_h_col], df[bb_w_col]
    dev = np.where(close<lo, lo-close, np.where(close>hi, close-hi, 0.0))
    out["eng_bb"] = dev / (bw + eps)

    # 4) BB mid-band offset
    out["eng_bb_mid"] = ((lo + hi)/2 - close) / (close + eps)

    # 5) RSI threshold deviation
    rv     = df[rsi_col]
    low_d  = np.clip((rsi_low - rv), 0, None) / 100.0
    high_d = np.clip((rv - rsi_high), 0, None) / 100.0
    out["eng_rsi"] = np.where(rv<rsi_low, low_d, np.where(rv>rsi_high, high_d, 0.0))

    # 6) ADX-weighted DI spread
    di_diff   = df[plus_di_col] - df[minus_di_col]
    diff_abs  = di_diff.abs() / 100.0
    ex        = np.clip((df[adx_col]-adx_thr)/100.0, 0, None)
    out["eng_adx"] = np.sign(di_diff) * diff_abs * ex

    # 7) OBV divergence ratio
    out["eng_obv"] = df[obv_diff_col] / (df[obv_sma_col] + eps)

    # 8) ATR stationary deviation
    ratio = df[atr_pct_col]
    rm    = ratio.rolling(mult_w, min_periods=1).mean()
    out["eng_atr_div"] = (ratio - rm) * 10_000

    # 9) SMA vs price offsets
    out["eng_sma_short"] = (df[sma_s] - close) / (close + eps)
    out["eng_sma_long"]  = (df[sma_l] - close) / (close + eps)

    # 10) VWAP vs price offset
    out["eng_vwap"] = (df[vwap_col] - close) / (close + eps)

    return out.dropna()


# ##########################################################################################################


# def features_engineering(
#     df: pd.DataFrame,
#     rsi_low:   float = 30.0,
#     rsi_high:  float = 70.0,
#     adx_thr:   float = 20.0,
#     mult_w:    int   = 14,
#     eps:       float = 1e-8
# ) -> pd.DataFrame:
#     """
#     Build continuous “eng_” signals from raw indicators + relative‐price bands.

#     1) eng_ma        = (SMA_short – SMA_long) / SMA_long
#     2) eng_macd      = MACD_diff / SMA_long
#     3) eng_bb        = distance outside BBands / BB_width
#     4) eng_bb_mid    = (BB_mid – close) / close
#     5) eng_rsi       = distance beyond [rsi_low, rsi_high] / 100
#     6) eng_adx       = sign(DI+–DI–) × (|DI+–DI–|/100) × max(ADX–adx_thr, 0)/100
#     7) eng_obv       = (OBV – OBV_SMA) / OBV_SMA
#     8) eng_atr_div   = 10 000 × [(ATR/close) – rolling_mean(ATR/close, mult_w)]
#     9) eng_sma_short = (SMA_short – close) / close
#    10) eng_sma_long  = (SMA_long  – close) / close
#    11) eng_vwap      = (VWAP – close) / close

#     Returns a DataFrame of these engineered features, indexed same as `df`.
#     """
#     out   = pd.DataFrame(index=df.index)
#     close = df["close"]

#     # 1) Find true SMA columns (exclude sma_pct_*)
#     sma_cols = [
#         c for c in df.columns
#         if c.startswith("sma_") and c.split("_")[1].isdigit()
#     ]
#     sma_cols = sorted(sma_cols, key=lambda c: int(c.split("_")[1]))
#     sma_s, sma_l = sma_cols[:2]

#     # 2) Locate MACD diff
#     macd_diff_col = next(c for c in df.columns if c.startswith("macd_diff_"))

#     # 3) Locate Bollinger lband/hband/width
#     bb_l_col = next(c for c in df.columns if c.startswith("bb_lband_"))
#     bb_h_col = next(c for c in df.columns if c.startswith("bb_hband_"))
#     bb_w_col = next(c for c in df.columns if c.startswith("bb_w_"))

#     # 4) Locate RSI
#     rsi_col = next(c for c in df.columns if c.startswith("rsi_"))

#     # 5) Locate +DI, –DI, ADX
#     plus_di_col  = next(c for c in df.columns if c.startswith("plus_di_"))
#     minus_di_col = next(c for c in df.columns if c.startswith("minus_di_"))
#     adx_col      = next(c for c in df.columns if c.startswith("adx_"))

#     # 6) Locate OBV & its SMA
#     obv_col     = "obv"
#     obv_sma_col = next(c for c in df.columns if c.startswith("obv_sma_"))

#     # 7) Locate ATR/close pct
#     atr_pct_col = next(c for c in df.columns if c.startswith("atr_pct_"))

#     # 1) MA spread ratio
#     out["eng_ma"] = ((df[sma_s] - df[sma_l]) / (df[sma_l] + eps)).round(3)

#     # 2) MACD diff ratio
#     out["eng_macd"] = (df[macd_diff_col] / (df[sma_l] + eps)).round(3)

#     # 3) Bollinger deviation ratio
#     lo, hi, bw = df[bb_l_col], df[bb_h_col], df[bb_w_col]
#     dev = np.where(close < lo, lo - close,
#           np.where(close > hi, close - hi, 0.0))
#     out["eng_bb"] = (dev / (bw + eps)).round(3)

#     # 4) Bollinger mid‐band relative
#     bb_mid = (lo + hi) * 0.5
#     out["eng_bb_mid"] = ((bb_mid - close) / (close + eps)).round(4)

#     # 5) RSI threshold ratio
#     rsi_vals = df[rsi_col]
#     low_dev  = np.clip((rsi_low  - rsi_vals), 0, None) / 100.0
#     high_dev = np.clip((rsi_vals - rsi_high), 0, None) / 100.0
#     out["eng_rsi"] = np.where(
#         rsi_vals < rsi_low, low_dev,
#         np.where(rsi_vals > rsi_high, high_dev, 0.0)
#     ).round(3)

#     # 6) ADX‐weighted DI spread
#     di_diff    = df[plus_di_col] - df[minus_di_col]
#     diff_abs   = di_diff.abs() / 100.0
#     ex         = np.clip((df[adx_col] - adx_thr) / 100.0, 0, None)
#     out["eng_adx"] = (np.sign(di_diff) * diff_abs * ex).round(3)

#     # 7) OBV divergence ratio
#     out["eng_obv"] = (
#         (df[obv_col] - df[obv_sma_col]) / (df[obv_sma_col] + eps)
#     ).round(3)

#     # 8) ATR/close stationary deviation
#     ratio = df[atr_pct_col]
#     rm    = ratio.rolling(mult_w, min_periods=1).mean()
#     out["eng_atr_div"] = ((ratio - rm) * 10_000).round(1)

#     # 9) SMA short/long relative to price
#     out["eng_sma_short"] = ((df[sma_s] - close) / (close + eps)).round(4)
#     out["eng_sma_long"]  = ((df[sma_l] - close) / (close + eps)).round(4)

#     # 10) VWAP relative to price
#     vwap_col = next(c for c in df.columns if c.startswith("vwap_") and not c.endswith("_dev"))
#     out["eng_vwap"] = ((df[vwap_col] - close) / (close + eps)).round(4)

#     return out


##########################################################################################################


# def assign_feature_groups(
#     df: pd.DataFrame,
#     cols: List[str],
#     *,
#     ratio_range:       float = 0.15,
#     heavy_thresh:      float = 1e7,
#     skew_thresh:       float = 3.0,
#     kurtosis_thresh:   float = 5.0,
#     discrete_thresh:   int   = 10,
#     overrides:         Dict[str, str] = None
# ) -> pd.DataFrame:
#     """
#     Analyze each feature’s distribution and assign it into one of six
#     shape-preserving scaling groups, then apply any manual overrides.

#     Steps:
#       1) Vectorized describe() to get min, max, 1%, 5%, 50%, 95%, 99%.
#          Append skew, kurtosis, unique_count, zero_ratio.
#       2) For each feature, decide a raw group in this priority:
#          a) discrete      – unique_count ≤ discrete_thresh
#          b) ratio         – central bulk ∈ ±ratio_range
#          c) bounded       – all values ∈ [0,100]
#          d) log_skewed    – positive-only & skew > skew_thresh
#          e) robust_tails  – heavy extremes or kurtosis ≥ kurtosis_thresh
#          f) unbounded     – everything else
#       3) Apply only the user-provided overrides to raw assignments.
#       4) Return a DataFrame with stats plus 'group_raw' and 'group_final'.
#     """
#     # 1) summary stats via describe()
#     descr = df[cols].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).T.rename(
#         columns={'1%':'1%', '5%':'5%', '50%':'50%', '95%':'95%', '99%':'99%'}
#     )
#     # append distributional metrics
#     descr['skew']         = df[cols].skew().values
#     descr['kurtosis']     = df[cols].kurtosis().values
#     descr['unique_count'] = df[cols].nunique().values
#     descr['zero_ratio']   = (df[cols] == 0).mean().values

#     # 2) raw grouping logic
#     raw_group: Dict[str, str] = {}
#     for feat in cols:
#         mn, mx   = descr.at[feat, 'min'], descr.at[feat, 'max']
#         p5, p95  = descr.at[feat, '5%'],  descr.at[feat, '95%']
#         sk       = descr.at[feat, 'skew']
#         kt       = descr.at[feat, 'kurtosis']
#         uc       = descr.at[feat, 'unique_count']

#         if uc <= discrete_thresh:
#             grp = 'discrete'
#         elif p5 >= -ratio_range and p95 <= ratio_range:
#             grp = 'ratio'
#         elif mn >= 0 and mx <= 100:
#             grp = 'bounded'
#         elif mn >= 0 and sk > skew_thresh:
#             grp = 'log_skewed'
#         elif abs(mn) >= heavy_thresh or abs(mx) >= heavy_thresh or kt >= kurtosis_thresh:
#             grp = 'robust_tails'
#         else:
#             grp = 'unbounded'

#         raw_group[feat] = grp

#     # 3) apply only user-provided overrides
#     overrides   = overrides or {}
#     final_group = {f: overrides.get(f, raw_group[f]) for f in cols}

#     # 4) assemble and return assignment table
#     df_assign = descr.copy()
#     df_assign['group_raw']   = df_assign.index.map(raw_group)
#     df_assign['group_final'] = df_assign.index.map(final_group)
#     return df_assign



def assign_feature_groups(
    df: pd.DataFrame,
    cols: List[str],
    *,
    ratio_range:       float = 0.15,
    heavy_thresh:      float = 1e7,
    skew_thresh:       float = 3.0,
    kurtosis_thresh:   float = 5.0,
    discrete_thresh:   int   = 10,
    overrides:         Dict[str, str] = None
) -> pd.DataFrame:
    """
    Analyze each feature’s distribution to assign it to one of six
    shape-preserving scaling groups, then apply any user overrides.

    1) Replace infinities with NaN and gather vectorized stats
       (min, max, 1%,5%,50%,95%,99%) via describe().
       Append skewness, kurtosis, unique_count, zero_ratio.
    2) For each feature, apply these rules in this order:
         a) discrete      — unique_count ≤ discrete_thresh
         b) log_skewed    — strictly ≥0 & skew > skew_thresh
         c) ratio         — central bulk ∈ ±ratio_range
         d) bounded       — all values ∈ [0,100]
         e) robust_tails  — extreme spikes or fat tails
                              (|min|max| ≥ heavy_thresh OR kurtosis ≥ kurtosis_thresh)
         f) unbounded     — fallback
    3) Layer on the exact overrides you pass in (no hidden defaults).
    4) Return DataFrame of stats with 'group_raw' and 'group_final'.
    """
    # 1) cleanse infinities and compute stats
    data = df[cols].replace([np.inf, -np.inf], np.nan)
    descr = (
        data.describe(percentiles=[0.01,0.05,0.95,0.99])
            .T
            .rename(columns={'1%':'1%','5%':'5%','50%':'50%','95%':'95%','99%':'99%'})
    )
    descr['skew']         = data.skew().values
    descr['kurtosis']     = data.kurtosis().values
    descr['unique_count'] = data.nunique().values
    descr['zero_ratio']   = (data == 0).mean().values

    # 2) raw grouping
    raw_group: Dict[str,str] = {}
    for feat in cols:
        mn, mx   = descr.at[feat,'min'], descr.at[feat,'max']
        p5, p95  = descr.at[feat,'5%'], descr.at[feat,'95%']
        sk       = descr.at[feat,'skew']
        kt       = descr.at[feat,'kurtosis']
        uc       = descr.at[feat,'unique_count']

        # a) discrete cardinality
        if uc <= discrete_thresh:
            grp = 'discrete'

        # b) heavy positive skew (log‐transform)
        elif mn >= 0 and sk > skew_thresh:
            grp = 'log_skewed'

        # c) small ±bulk (Yeo–Johnson → MinMax)
        elif p5 >= -ratio_range and p95 <= ratio_range:
            grp = 'ratio'

        # d) natural [0–100] oscillators
        elif mn >= 0 and mx <= 100:
            grp = 'bounded'

        # e) two‐sided extremes or fat tails (winsorize)
        elif abs(mn) >= heavy_thresh or abs(mx) >= heavy_thresh or kt >= kurtosis_thresh:
            grp = 'robust_tails'

        # f) everything else
        else:
            grp = 'unbounded'

        raw_group[feat] = grp

    # 3) apply only user overrides
    overrides   = overrides or {}
    final_group = {f: overrides.get(f, raw_group[f]) for f in cols}

    # 4) assemble assignment table
    df_assign = descr.copy()
    df_assign['group_raw']   = df_assign.index.map(raw_group)
    df_assign['group_final'] = df_assign.index.map(final_group)
    return df_assign


##########################################################################################################


class Winsorizer(FunctionTransformer):
    """Clips each column to its 1st and 99th percentiles computed on fit."""
    def __init__(self):
        super().__init__(func=None, inverse_func=None, validate=False)

    def fit(self, X, y=None):
        self.lower_ = np.percentile(X, 1, axis=0)
        self.upper_ = np.percentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)

############ 

def scale_with_splits(
    df: pd.DataFrame,
    assignment: pd.DataFrame,
    train_prop: float = params.train_prop,
    val_prop:   float = params.val_prop
) -> pd.DataFrame:
    """
    Split a time‐series into train/val/test, scale each feature by its
    assigned pipeline, and reassemble a unified [0,1] dataset.

    1) Copy and timestamp the index.
    2) Generate cyclical calendar features (hour/day_of_week/month) as sin/cos.
    3) Split contiguously: train first M, then val next N, remainder → test.
    4) Reduce each sin/cos pair into a single PCA component.
    5) Identify feature columns vs reserved (OHLCV, bid/ask, label, calendar).
    6) Build lists of features per group using assignment["group_final"].
    7) Define six pipelines for:
         • bounded      → linear divide by 100  
         • ratio        → Yeo–Johnson → MinMax  
         • log_skewed   → signed log1p → MinMax  
         • robust_tails → Winsorizer → MinMax  
         • discrete     → Ordinal → MinMax  
         • unbounded    → StandardScaler → clip ±3σ → MinMax  
    8) Fit ColumnTransformer on train features.
    9) Transform each split _per day_ with a tqdm bar to preserve day‐block scaling.
   10) Concatenate splits, restore reserved cols, drop raw OHLCV.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # 2) Add sin/cos calendar features
    df["hour"], df["day_of_week"], df["month"] = (
        df.index.hour, df.index.dayofweek, df.index.month
    )
    for name, period in [("hour", 24), ("day_of_week", 7), ("month", 12)]:
        vals = df[name]
        df[f"{name}_sin"] = np.sin(2 * np.pi * vals / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * vals / period)

    # 3) Contiguous split into train/val/test
    N    = len(df)
    n_tr = int(N * train_prop)
    n_val = int(N * val_prop)
    if n_tr + n_val >= N:
        raise ValueError("train_prop + val_prop must sum < 1.0")
    df_tr = df.iloc[:n_tr].copy()
    df_v  = df.iloc[n_tr : n_tr + n_val].copy()
    df_te = df.iloc[n_tr + n_val :].copy()

    # 4) PCA‐compress each sin/cos pair
    for cal in ("hour", "day_of_week", "month"):
        sincos = [f"{cal}_sin", f"{cal}_cos"]
        pca    = PCA(n_components=1)
        pca.fit(df_tr[sincos])
        for split in (df_tr, df_v, df_te):
            split[cal] = pca.transform(split[sincos])
            split.drop(columns=sincos, inplace=True)

    # 5) Carve out feature vs reserved columns
    reserved = {
        "open","high","low","close","volume","bid","ask",
        params.label_col, "hour","day_of_week","month"
    }
    feat_cols = [c for c in df_tr.columns if c not in reserved]

    # 6) Build per‐group feature lists
    mapping = assignment["group_final"].to_dict()
    groups  = {grp: [f for f in feat_cols if mapping.get(f)==grp]
               for grp in ["bounded","ratio","log_skewed","robust_tails","discrete","unbounded"]}

    # 7) Define pipelines for each group
    def signed_log1p(X): 
        return np.sign(X) * np.log1p(np.abs(X))

    pipelines = [
        ("bnd", FunctionTransformer(lambda X: X/100.0, validate=False), groups["bounded"]),
        ("rat", Pipeline([
            ("pt", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("mm", MinMaxScaler())
        ]), groups["ratio"]),
        ("lgs", Pipeline([
            ("slog", FunctionTransformer(signed_log1p, validate=False)),
            ("mm",   MinMaxScaler())
        ]), groups["log_skewed"]),
        ("rbt", Pipeline([
            ("win", Winsorizer()),
            ("mm",  MinMaxScaler())
        ]), groups["robust_tails"]),
        ("dis", Pipeline([
            ("ord", OrdinalEncoder()),
            ("mm",  MinMaxScaler())
        ]), groups["discrete"]),
        ("unb", Pipeline([
            ("std",   StandardScaler()),
            ("clip",  FunctionTransformer(lambda X: np.clip(X, -3, 3), validate=False)),
            ("mm",    MinMaxScaler())
        ]), groups["unbounded"]),
    ]
    ct = ColumnTransformer(transformers=pipelines, remainder="drop")

    # 8) Fit on training features
    ct.fit(df_tr[feat_cols])

    # 9) Transform _per day_ with progress bar
    def transform_by_day(split_df, label):
        arr = np.empty((len(split_df), len(feat_cols)))
        for day, block in tqdm(
            split_df.groupby(split_df.index.normalize()),
            desc=f"{label} days", unit="day"
        ):
            mask      = split_df.index.normalize() == day
            arr[mask] = ct.transform(block[feat_cols])
        scaled = pd.DataFrame(arr, index=split_df.index, columns=feat_cols)
        return pd.concat([scaled, split_df[list(reserved)]], axis=1)[split_df.columns]

    df_tr_s = transform_by_day(df_tr,  "train")
    df_v_s  = transform_by_day(df_v,   "val")
    df_te_s = transform_by_day(df_te,  "test")

    # 10) Reassemble and drop raw OHLCV
    df_all = pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()
    return df_all.drop(columns=["open","high","low","close","volume"], errors="ignore")


#########################################################################################################


# def compare_raw_vs_scaled(
#     df_raw:     pd.DataFrame,
#     df_scaled:  pd.DataFrame,
#     assignment: pd.DataFrame,
#     feat_cols:  list[str] | None = None,
#     tol:        float = 1e-6
# ) -> pd.DataFrame:
#     """
#     Compare raw vs scaled features, then verify:

#       • min/max in [0,1] per group rules
#       • Spearman ρ ≈ 1 (monotonic transform)
#       • per-day monotonicity (scaled values preserve raw ordering)
#     """
#     if feat_cols is None:
#         feat_cols = [c for c in df_raw.columns if c in df_scaled.columns]

#     qs = [0.01, 0.05, 0.50, 0.95, 0.99]
#     raw = (
#         df_raw[feat_cols]
#         .describe(percentiles=qs).T
#         .loc[:, ['min','1%','5%','50%','95%','99%','max']]
#     )
#     raw.columns = [f"raw_{c}" for c in raw.columns]

#     scaled = (
#         df_scaled[feat_cols]
#         .describe(percentiles=qs).T
#         .loc[:, ['min','1%','5%','50%','95%','99%','max']]
#     )
#     scaled.columns = [f"scaled_{c}" for c in scaled.columns]

#     cmp_df = pd.concat([raw, scaled], axis=1)
#     cmp_df['group_final'] = assignment['group_final']

#     # precompute stats for spearman & monotonicity
#     spearman_rho = {}
#     is_mono      = {}
#     for feat in tqdm(feat_cols, desc="Checking shape"):
#         x = df_raw[feat].dropna()
#         y = df_scaled.loc[x.index, feat]

#         rho = spearmanr(x, y).correlation
#         order   = np.argsort(x.values)
#         y_sorted= y.values[order]
#         mono    = np.all(np.diff(y_sorted) >= -tol)

#         spearman_rho[feat] = rho
#         is_mono[feat]      = mono

#     # final checks
#     statuses, reasons = [], []
#     for feat, row in cmp_df.iterrows():
#         grp = row.group_final
#         mn, mx = row.scaled_min, row.scaled_max
#         rho    = spearman_rho[feat]
#         mono   = is_mono[feat]

#         errs = []
#         # range check
#         if grp == "bounded":
#             if not (mn >= -tol and mx <= 1+tol):
#                 errs.append(f"range[{mn:.3f},{mx:.3f}]")
#         else:
#             if not (abs(mn) <= tol and abs(mx-1) <= tol):
#                 errs.append(f"range[{mn:.3f},{mx:.3f}]")

#         # spearman check
#         if abs(rho - 1) > 1e-3:
#             errs.append(f"rho={rho:.3f}")

#         # monotonic check
#         if not mono:
#             errs.append("non-monotonic")

#         if errs:
#             statuses.append("FAIL")
#             reasons.append("; ".join(errs))
#         else:
#             statuses.append("OK")
#             reasons.append("all checks passed")

#     cmp_df['status']       = statuses
#     cmp_df['reason']       = reasons
#     cmp_df['spearman_rho'] = pd.Series(spearman_rho)
#     cmp_df['is_monotonic'] = pd.Series(is_mono)

#     return cmp_df


def compare_raw_vs_scaled(
    df_raw: pd.DataFrame,
    df_scaled: pd.DataFrame,
    assignment: pd.DataFrame,
    feat_cols: Optional[List[str]] = None,
    tol_range: float = 1e-6,
    tol_spearman: float = 1e-4,
    check_per_day: bool = True
) -> pd.DataFrame:
    """
    Compare raw vs scaled features and verify shape preservation.

    Steps:
      1) Replace inf→NaN in both raw & scaled.
      2) Determine feature list = intersection of:
           - columns common to raw & scaled
           - optionally your provided feat_cols
           - the assignment.index (so we skip 'signal', etc.)
      3) Compute quantiles (min,1%,5%,50%,95%,99%,max) for raw & scaled.
      4) For each feature:
           a) Build mask = (raw.notna & scaled.notna)
           b) Spearman ρ on aligned pairs
           c) Global monotonic: sort by raw, check scaled non-decreasing
           d) Per-day monotonic if enabled
           e) Cardinality preserved for discrete group
      5) Range check: all scaled values ∈ [0,1] ± tol_range
      6) Collate status ('OK'/'FAIL') and reasons for any failing test.
    """
    # 1) clean infinities
    raw = df_raw.replace([np.inf, -np.inf], np.nan)
    sca = df_scaled.replace([np.inf, -np.inf], np.nan)

    # 2) build feature list
    all_cols = set(raw.columns) & set(sca.columns)
    if feat_cols is not None:
        all_cols &= set(feat_cols)
    # restrict to only those indexed in your assignment table
    feat_cols = [f for f in all_cols if f in assignment.index]

    # 3) compute quantiles
    qs = [0.01, 0.05, 0.50, 0.95, 0.99]
    raw_q = ( raw[feat_cols]
               .describe(percentiles=qs).T
               .loc[:, ['min','1%','5%','50%','95%','99%','max']]
               .add_prefix('raw_') )
    sca_q = ( sca[feat_cols]
               .describe(percentiles=qs).T
               .loc[:, ['min','1%','5%','50%','95%','99%','max']]
               .add_prefix('scaled_') )
    cmp_df = pd.concat([raw_q, sca_q], axis=1)
    cmp_df['group_final'] = assignment['group_final']

    # prepare maps
    rho_map = {}
    mono_glob = {}
    mono_day  = {}
    card_ok   = {}

    # 4) per‐feature checks
    for feat in tqdm(feat_cols, desc="Shape checks"):
        x = raw[feat]
        y = sca[feat]

        # a) align pairs where neither is NaN
        mask = x.notna() & y.notna()
        x2, y2 = x[mask], y[mask]

        # b) spearman (or 1.0 if too few points)
        if len(x2) < 2:
            rho = 1.0
        else:
            rho = spearmanr(x2, y2).correlation
            if rho is None:
                rho = 1.0
        rho_map[feat] = rho

        # c) global monotonic
        idx = np.argsort(x2.values)
        y_sorted = y2.values[idx]
        mono_glob[feat] = np.all(np.diff(y_sorted) >= -tol_range)

        # d) daily monotonic
        if check_per_day:
            ok = True
            for day, block in x2.groupby(x2.index.normalize()):
                yb = y2.loc[block.index].values
                if not np.all(np.diff(yb) >= -tol_range):
                    ok = False
                    break
            mono_day[feat] = ok
        else:
            mono_day[feat] = True

        # e) cardinality for 'discrete'
        grp = assignment.at[feat, 'group_final']
        if grp == 'discrete':
            card_ok[feat] = (x2.nunique() == y2.nunique())
        else:
            card_ok[feat] = True

    # 5) range, status & reasons
    statuses, reasons = [], []
    for feat, row in cmp_df.iterrows():
        mn, mx = row['scaled_min'], row['scaled_max']
        errs = []

        # range containment
        if not (mn >= -tol_range and mx <= 1 + tol_range):
            errs.append(f"range[{mn:.3f},{mx:.3f}]")

        # correlation
        if abs(rho_map[feat] - 1) > tol_spearman:
            errs.append(f"rho={rho_map[feat]:.3f}")

        # monotonic checks
        if not mono_glob[feat]:
            errs.append("non-monotonic_global")
        if check_per_day and not mono_day[feat]:
            errs.append("non-monotonic_daily")

        # discrete cardinality
        if not card_ok[feat]:
            errs.append("cardinality_changed")

        if errs:
            statuses.append("FAIL")
            reasons.append("; ".join(errs))
        else:
            statuses.append("OK")
            reasons.append("all checks passed")

    cmp_df['spearman_rho']   = pd.Series(rho_map)
    cmp_df['mono_global']    = pd.Series(mono_glob)
    cmp_df['mono_daily']     = pd.Series(mono_day)
    cmp_df['cardinality_ok'] = pd.Series(card_ok)
    cmp_df['status']         = statuses
    cmp_df['reason']         = reasons

    return cmp_df

#########################################################################################################


def ig_feature_importance(
    model, loader, feature_names, device,
    n_samples=100, n_steps=50
):
    """
    Integrated-Gradients (Captum)
    Post-training features impact: it provides the per-feature attributions summed over time.
    It digs into the trained PyTorch model and attributes how much each input feature drove the final prediction in each window. 
    We’ll sum attributions over the time axis and average across some test windows.
    Runs Integrated Gradients on up to n_samples windows from loader.
    Returns a DataFrame of mean |IG| per feature.
    """
    # 1) disable cuDNN so RNN backward works in eval mode
    cudnn_enabled = cudnn.enabled
    cudnn.enabled = False

    model.eval()

    # 2) wrap model to return only the final regression output
    def forward_reg(x):
        pr, _, _ = model(x)
        return torch.sigmoid(pr[..., -1, 0])

    ig = IntegratedGradients(forward_reg)

    # 3) accumulator for feature‐wise attributions
    total_attr = np.zeros(len(feature_names), dtype=float)
    count = 0

    # 4) loop over windows
    for batch in tqdm(loader, desc="IG windows", total=n_samples):
        xb, y_sig, *_, lengths = batch
        W = lengths[0]
        x = xb[0, :W].unsqueeze(0).to(device)   # shape: (1, W, F)

        # 5) compute IG attributions
        atts, delta = ig.attribute(
            inputs=x,
            baselines=torch.zeros_like(x),
            n_steps=n_steps,
            internal_batch_size=1,
            return_convergence_delta=True
        )

        # 6) collapse attributions across all dims except the last (features)
        attr_np  = atts.detach().abs().cpu().numpy()  
        abs_sum  = attr_np.reshape(-1, attr_np.shape[-1]).sum(axis=0)

        total_attr += abs_sum
        count += 1
        if count >= n_samples:
            break

        # 7) free GPU memory
        del atts, delta, x
        torch.cuda.empty_cache()

    # 8) restore cuDNN
    cudnn.enabled = cudnn_enabled

    # 9) average and return as DataFrame
    avg_attr = total_attr / count
    imp_df   = pd.DataFrame({
        "feature":    feature_names,
        "importance": avg_attr
    }).sort_values("importance", ascending=False)

    return imp_df


