from libs import plots, params

from typing import Sequence, List, Tuple, Optional, Union
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
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.decomposition import PCA


#########################################################################################################


# def features_creation(
#     df: pd.DataFrame,
#     base_window: int = params.smooth_sign_win_tick
# ) -> pd.DataFrame:
#     """
#     Build a unified feature set for OHLCV bars using a single base_window.

#     1) Ensure df has a DateTimeIndex.
#     2) Derive all rolling/window parameters from base_window:
#        • short_w        = max(2, base_window//2)
#        • long_w         = base_window * 2
#        • macd_sig       = max(1, base_window//4)
#        • return_periods = [1, base_window, long_w]
#     3) Compute indicators:
#        - Trend: EMA(short_w), SMA(short_w, base_window, long_w), MACD(short_w, base_window, macd_sig)
#        - Volatility: ATR(base_window, long_w) + rolling ATR(base_window), Bollinger Bands(base_window)
#        - Momentum: RSI(base_window), Stochastic %K(base_window), %D(macd_sig)
#        - Directional: +DI(base_window), –DI(base_window), ADX(base_window)
#        - Volume: OBV + SMA(base_window), VWAP deviation(base_window), volume_spike(base_window)
#        - Calendar: hour, day_of_week, month
#     4) Copy raw open/high/low/close/volume/bid/ask/label through.
#     5) Drop rows with NaNs and return only the selected columns.
#     """
#     # 1) Ensure datetime index
#     if not isinstance(df.index, pd.DatetimeIndex):
#         df.index = pd.to_datetime(df.index)
#     out = df.copy()

#     # 2) Derive window sizes
#     short_w        = max(2, base_window // 2)
#     long_w         = base_window * 2
#     macd_sig       = max(1, base_window // 4)
#     return_periods = [1, base_window, long_w]

#     # 3a) Trend indicators
#     out[f"ema_{short_w}"] = ta.trend.EMAIndicator(
#         close=out["close"], window=short_w
#     ).ema_indicator()
#     for w in (short_w, base_window, long_w):
#         out[f"sma_{w}"] = ta.trend.SMAIndicator(
#             close=out["close"], window=w
#         ).sma_indicator()

#     macd = ta.trend.MACD(
#         close      = out["close"],
#         window_slow= base_window,
#         window_fast= short_w,
#         window_sign= macd_sig
#     )
#     out["macd_line"]   = macd.macd()
#     out["macd_signal"] = macd.macd_signal()
#     out["macd_diff"]   = macd.macd_diff()

#     # 3b) Volatility indicators
#     for w in (base_window, long_w):
#         atr_col = f"atr_{w}"
#         out[atr_col] = ta.volatility.AverageTrueRange(
#             high  = out["high"],
#             low   = out["low"],
#             close = out["close"],
#             window= w
#         ).average_true_range()
#         out[f"{atr_col}_sma_{base_window}"] = out[atr_col].rolling(
#             base_window
#         ).mean()

#     bb = ta.volatility.BollingerBands(
#         close      = out["close"],
#         window     = base_window,
#         window_dev = 2
#     )
#     out[f"bb_lband_{base_window}"] = bb.bollinger_lband()
#     out[f"bb_hband_{base_window}"] = bb.bollinger_hband()
#     mid = bb.bollinger_mavg()
#     out[f"bb_width_{base_window}"] = (
#         out[f"bb_hband_{base_window}"] - out[f"bb_lband_{base_window}"]
#     ) / mid

#     # 3c) Momentum indicators
#     out[f"rsi_{base_window}"] = ta.momentum.RSIIndicator(
#         close=out["close"], window=base_window
#     ).rsi()

#     stoch = ta.momentum.StochasticOscillator(
#         high          = out["high"],
#         low           = out["low"],
#         close         = out["close"],
#         window        = base_window,
#         smooth_window = macd_sig
#     )
#     out[f"stoch_k_{base_window}"] = stoch.stoch()
#     out[f"stoch_d_{macd_sig}"]    = stoch.stoch_signal()

#     # 3d) Directional indicators
#     adx = ta.trend.ADXIndicator(
#         high  = out["high"],
#         low   = out["low"],
#         close = out["close"],
#         window= base_window
#     )
#     out[f"plus_di_{base_window}"]  = adx.adx_pos()
#     out[f"minus_di_{base_window}"] = adx.adx_neg()
#     out[f"adx_{base_window}"]      = adx.adx()

#     # 3e) Volume indicators
#     out["obv"] = ta.volume.OnBalanceVolumeIndicator(
#         close = out["close"],
#         volume= out["volume"]
#     ).on_balance_volume()
#     out[f"obv_sma_{base_window}"] = out["obv"].rolling(
#         base_window
#     ).mean()

#     vwap = ta.volume.VolumeWeightedAveragePrice(
#         high   = out["high"],
#         low    = out["low"],
#         close  = out["close"],
#         volume = out["volume"],
#         window = base_window
#     ).volume_weighted_average_price()
#     out[f"vwap_dev_{base_window}"] = (out["close"] - vwap) / vwap

#     out[f"volume_spike_{base_window}"] = (
#         out["volume"] /
#         out["volume"].rolling(base_window).mean()
#     )

#     # 3f) Returns & short-term volatility
#     for rp in return_periods:
#         out[f"r_{rp}"] = np.log(out["close"] / out["close"].shift(rp))
#     out[f"vol_{base_window}"] = out["r_1"].rolling(
#         base_window
#     ).std()

#     # 3g) Calendar features
#     out["hour"]        = out.index.hour
#     # fix: Pandas uses `.dayofweek`, not `.day_of_week`
#     out["day_of_week"] = out.index.dayofweek
#     out["month"]       = out.index.month

#     # 4) Copy raw OHLCV + bid/ask/label through
#     for col in ["open", "high", "low", "close", "volume",
#                 "bid", "ask", params.label_col]:
#         out[col] = df[col]

#     # 5) Select final columns & drop rows with NaNs
#     keep = []
#     keep.append(f"ema_{short_w}")
#     keep += [f"sma_{w}" for w in (short_w, base_window, long_w)]
#     keep += ["macd_line", "macd_signal", "macd_diff"]
#     keep += [f"atr_{w}" for w in (base_window, long_w)]
#     keep += [f"atr_{w}_sma_{base_window}" for w in (base_window, long_w)]
#     keep += [
#         f"bb_lband_{base_window}", f"bb_hband_{base_window}",
#         f"bb_width_{base_window}"
#     ]
#     keep += [f"rsi_{base_window}", f"stoch_k_{base_window}", f"stoch_d_{macd_sig}"]
#     keep += [
#         f"plus_di_{base_window}", f"minus_di_{base_window}",
#         f"adx_{base_window}"
#     ]
#     keep += ["obv", f"obv_sma_{base_window}", f"vwap_dev_{base_window}",
#              f"volume_spike_{base_window}"]
#     keep += [f"r_{rp}" for rp in return_periods] + [f"vol_{base_window}"]
#     keep += ["hour", "day_of_week", "month"]
#     keep += ["open", "high", "low", "close", "volume",
#              "bid", "ask", params.label_col]

#     return out.loc[:, keep].dropna()


#########################################################################################################


# def features_engineering(
#     df: pd.DataFrame,
#     base_window:             int = params.smooth_sign_win_tick,
#     low_rsi:                 float = 30.0,
#     high_rsi:                float = 70.0,
#     adx_thresh:              float = 20.0,
#     adx_multiplier_window:   int   = None,
#     eps:                     float = 1e-8
# ) -> pd.DataFrame:
#     """
#     Build seven engineered signals (eng_*) from the base indicators in df.

#     1) eng_ma      = EMA(short_w) – SMA(long_w)
#     2) eng_macd    = MACD line – MACD signal
#     3) eng_bb      = price distance outside Bollinger Bands
#     4) eng_rsi     = distance beyond RSI thresholds
#     5) eng_adx     = ADX-weighted |+DI – –DI| spread, normalized
#     6) eng_obv     = OBV divergence from its SMA
#     7) eng_atr_div = divergence of ATR/price ratio, scaled

#     This function assumes df already contains columns named:
#        ema_{short_w}, sma_{long_w}, macd_line, macd_signal,
#        bb_lband_{aw}, bb_hband_{aw}, rsi_{aw},
#        plus_di_{aw}, minus_di_{aw}, adx_{aw},
#        obv, obv_sma_{aw},
#        atr_{aw}
#     """
#     out = df.copy()

#     # derive the same windows used upstream in features_creation
#     short_w = max(2, base_window // 2)
#     long_w  = base_window * 2
#     aw      = base_window
#     sig_w   = max(1, aw // 4)

#     # 1) MA spread: EMA(short) minus SMA(long)
#     out["eng_ma"] = (
#         out[f"ema_{short_w}"] - out[f"sma_{long_w}"]
#     ).round(3)

#     # 2) MACD histogram
#     out["eng_macd"] = (out["macd_line"] - out["macd_signal"]).round(3)

#     # 3) Bollinger Bands distance
#     lo    = out[f"bb_lband_{aw}"]
#     hi    = out[f"bb_hband_{aw}"]
#     price = out["close"]
#     out["eng_bb"] = np.where(
#         price < lo,   (lo - price),
#         np.where(price > hi, (price - hi), 0.0)
#     ).round(3)

#     # 4) RSI threshold distance
#     rsi = out[f"rsi_{aw}"]
#     out["eng_rsi"] = np.where(
#         rsi < low_rsi,  (low_rsi  - rsi),
#         np.where(rsi > high_rsi, (rsi - high_rsi), 0.0)
#     ).round(3)

#     # 5) ADX-weighted DI spread normalized over a window
#     plus  = out[f"plus_di_{aw}"]
#     minus = out[f"minus_di_{aw}"]
#     adx   = out[f"adx_{aw}"]
#     di    = (plus - minus).abs()
#     excess= (adx - adx_thresh).clip(lower=0.0)
#     raw   = di * excess

#     mult_w = adx_multiplier_window or aw
#     max_excess = (adx.rolling(mult_w).max() - adx_thresh).clip(lower=0.0)
#     max_raw    = raw.rolling(mult_w).max().clip(lower=eps)
#     scale      = (max_excess / max_raw).fillna(0.0)
#     sign       = np.sign(plus - minus)

#     out["eng_adx"] = np.where(
#         adx > adx_thresh,
#         (sign * raw * scale).round(3),
#         0.0
#     )

#     # 6) OBV divergence
#     out["eng_obv"] = (
#         out["obv"] - out[f"obv_sma_{aw}"]
#     ).round(3)

#     # 7) ATR/price-ratio divergence
#     # compute ratio on the fly instead of expecting missing columns
#     atr_col    = out[f"atr_{aw}"]
#     atr_ratio  = atr_col / out["close"]
#     atr_ratio_sma = atr_ratio.rolling(aw).mean()
#     out["eng_atr_div"] = ((atr_ratio - atr_ratio_sma) * 10_000).round(1)

#     return out


def create_standard_features(
    df: pd.DataFrame,
    sma_short: int = 20,
    sma_long:  int = 100,
    rsi_window: int = 14,
    macd_fast:  int = 12,
    macd_slow:  int = 26,
    macd_sig:   int = 9,
    atr_window: int = 14,
    bb_window:  int = 20,
    obv_sma:    int = 14,
    vwap_window:int = 20
) -> pd.DataFrame:
    """
    1) Ensure df.index is DateTimeIndex.
    2) Pass through raw OHLCV: open, high, low, close, volume.
    3) Compute 1m‐bar indicators with trading‐typical windows:
       • Trend: RSI(rsi_window), SMA(sma_short), SMA(sma_long), MACD(macd_fast,macd_slow,macd_sig)
       • Volatility: ATR(atr_window), BollingerBands(bb_window) lower, upper, width
       • Directional: +DI(atr_window), ‑DI(atr_window), ADX(atr_window)
       • Volume: OBV; OBV_SMA(obv_sma); VWAP(vwap_window); volume_spike(obv_sma)
       • Calendar: hour, day_of_week, month
    4) Use min_periods=1 on rolling EMAs/SMAs to avoid dropping first N rows.
    5) Drop any remaining NaNs and return DataFrame of raw + standard features.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    out = df[["open","high","low","close","volume","bid","ask",params.label_col]].copy()

    # Trend indicators
    out["rsi"] = ta.momentum.RSIIndicator(
        close=out["close"], window=rsi_window
    ).rsi().round(3)

    macd = ta.trend.MACD(
        close=out["close"],
        window_fast=macd_fast,
        window_slow=macd_slow,
        window_sign=macd_sig
    )
    out["macd_line"]   = macd.macd().round(3)
    out["macd_signal"] = macd.macd_signal().round(3)
    out["macd_diff"]   = macd.macd_diff().round(3)

    out[f"sma_{sma_short}"] = out["close"].rolling(
        sma_short, min_periods=1
    ).mean().round(3)
    out[f"sma_{sma_long}"]  = out["close"].rolling(
        sma_long, min_periods=1
    ).mean().round(3)

    # Volatility indicators
    out["atr"] = ta.volatility.AverageTrueRange(
        high=out["high"], low=out["low"],
        close=out["close"], window=atr_window
    ).average_true_range().round(3)

    bb = ta.volatility.BollingerBands(
        close=out["close"], window=bb_window, window_dev=2
    )
    out["bb_lband"] = bb.bollinger_lband().round(3)
    out["bb_hband"] = bb.bollinger_hband().round(3)
    mavg = bb.bollinger_mavg()
    out["bb_width"] = ((out["bb_hband"] - out["bb_lband"]) / mavg).round(3)

    # Directional Movement
    adx = ta.trend.ADXIndicator(
        high=out["high"], low=out["low"],
        close=out["close"], window=atr_window
    )
    out["plus_di"]  = adx.adx_pos().round(3)
    out["minus_di"] = adx.adx_neg().round(3)
    out["adx"]      = adx.adx().round(3)

    # Volume indicators
    out["obv"]        = ta.volume.OnBalanceVolumeIndicator(
        close=out["close"], volume=out["volume"]
    ).on_balance_volume().round(3)
    out["obv_sma"]    = out["obv"].rolling(
        obv_sma, min_periods=1
    ).mean().round(3)
    out["vwap"]       = ta.volume.VolumeWeightedAveragePrice(
        high=out["high"], low=out["low"],
        close=out["close"], volume=out["volume"],
        window=vwap_window
    ).volume_weighted_average_price().round(3)
    out["vol_spike"] = (
        out["volume"] / out["volume"].rolling(
            obv_sma, min_periods=1
        ).mean()
    ).round(3)

    # Calendar features
    out["hour"]        = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["month"]       = out.index.month

    return out.dropna()


#########################################################################################################


def features_engineering(
    df: pd.DataFrame,
    rsi_low:  float = 30.0,
    rsi_high: float = 70.0,
    adx_thr:  float = 20.0,
    mult_w:   int   = 14,
    eps:      float = 1e-8
) -> pd.DataFrame:
    """
    Build seven continuous “eng_*” signals from standard indicators:
      1) eng_ma      = sma_short – sma_long
      2) eng_macd    = macd_diff
      3) eng_bb      = how far price is outside BBands
      4) eng_rsi     = how far RSI is beyond thresholds [rsi_low,rsi_high]
      5) eng_adx     = signed DI spread × (ADX–adx_thr), scaled by rolling max
      6) eng_obv     = obv – obv_sma
      7) eng_atr_div = 10k × ((atr/close) – rolling_mean(atr/close, mult_w))
    Returns only the eng_* columns, no NaNs dropped here.
    """
    out = pd.DataFrame(index=df.index)

    # 1) MA spread
    sma_cols = [c for c in df.columns if c.startswith("sma_")]
    out["eng_ma"] = (
        df[sma_cols[0]] - df[sma_cols[1]]
    ).round(3)

    # 2) MACD histogram
    out["eng_macd"] = df["macd_diff"].round(3)

    # 3) BB distance
    price = df["close"]; lo = df["bb_lband"]; hi = df["bb_hband"]
    out["eng_bb"] = np.where(
        price < lo, lo - price,
        np.where(price > hi, price - hi, 0.0)
    ).round(3)

    # 4) RSI threshold
    rsi = df["rsi"]
    out["eng_rsi"] = np.where(
        rsi < rsi_low,  rsi_low - rsi,
        np.where(rsi > rsi_high, rsi - rsi_high, 0.0)
    ).round(3)

    # 5) ADX‐weighted DI spread
    plus = df["plus_di"]; minus = df["minus_di"]; adx = df["adx"]
    diff = (plus - minus).abs()
    excess = (adx - adx_thr).clip(lower=0.0)
    raw = diff * excess
    max_exc = excess.rolling(mult_w, min_periods=1).max()
    max_raw = raw.rolling(mult_w, min_periods=1).max().clip(lower=eps)
    scale = (max_exc / max_raw).fillna(0.0)
    sign = np.sign(plus - minus)
    out["eng_adx"] = np.where(
        adx > adx_thr,
        (sign * raw * scale).round(3),
        0.0
    )

    # 6) OBV divergence
    out["eng_obv"] = (df["obv"] - df["obv_sma"]).round(3)

    # 7) ATR/price divergence
    ratio = df["atr"] / df["close"]
    rm = ratio.rolling(mult_w, min_periods=1).mean()
    out["eng_atr_div"] = ((ratio - rm) * 10_000).round(1)

    return out


#########################################################################################################


def create_custom_window_features(
    df: pd.DataFrame,
    base_w: int | None = None
) -> pd.DataFrame:
    """
    1) If base_w is None or ≤1, return empty DataFrame.
    2) Else compute “cust_” indicators keyed to base_w:
       • EMAs & SMAs at [base_w/2, base_w, 2×base_w]
       • MACD(short=base_w/2, slow=base_w, sig=base_w//4)
       • ATR & BBands(base_w) if base_w≠14/20
       • RSI(base_w) & Stoch(base_w) if base_w≠14
       • DI/ADX(base_w) if base_w≠14
       • OBV & OBV_SMA(base_w)
       • VWAP_dev(base_w)
       • vol_spike(base_w)
       • log‐returns over [1,base_w,2×base_w] & rolling vol(base_w)
    3) Drop any NaNs and return only cust_* columns.
    """
    if not base_w or base_w < 2:
        return pd.DataFrame(index=df.index)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    out = pd.DataFrame(index=df.index)
    half, double = max(2, base_w//2), base_w*2
    sig = max(1, base_w//4)

    # Helper: skip when matching standard windows
    std = dict(rsi=14, atr=14, bb=20, obv_sma=14)

    # 1) EMA & SMAs
    out[f"cust_ema_{half}"] = ta.trend.EMAIndicator(
        close=df["close"], window=half
    ).ema_indicator().round(3)
    for w in (half, base_w, double):
        out[f"cust_sma_{w}"] = df["close"].rolling(
            w, min_periods=1
        ).mean().round(3)

    # 2) MACD
    macd = ta.trend.MACD(
        close=df["close"],
        window_fast=half,
        window_slow=base_w,
        window_sign=sig
    )
    out["cust_macd_diff"] = macd.macd_diff().round(3)

    # 3) ATR
    if base_w != std["atr"]:
        out[f"cust_atr_{base_w}"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"],
            close=df["close"], window=base_w
        ).average_true_range().round(3)
    if double != std["atr"]:
        out[f"cust_atr_{double}"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"],
            close=df["close"], window=double
        ).average_true_range().round(3)

    # 4) Bollinger Bands
    if base_w != std["bb"]:
        bb = ta.volatility.BollingerBands(
            close=df["close"], window=base_w, window_dev=2
        )
        out[f"cust_bb_lband_{base_w}"] = bb.bollinger_lband().round(3)
        out[f"cust_bb_hband_{base_w}"] = bb.bollinger_hband().round(3)
        m = bb.bollinger_mavg()
        out[f"cust_bb_width_{base_w}"] = (
            (out[f"cust_bb_hband_{base_w}"] - out[f"cust_bb_lband_{base_w}"])
            / m
        ).round(3)

    # 5) RSI & Stochastic
    if base_w != std["rsi"]:
        out[f"cust_rsi_{base_w}"] = ta.momentum.RSIIndicator(
            close=df["close"], window=base_w
        ).rsi().round(3)
    st = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"],
        window=base_w, smooth_window=sig
    )
    out[f"cust_stoch_k_{base_w}"] = st.stoch().round(3)
    out[f"cust_stoch_d_{sig}"]    = st.stoch_signal().round(3)

    # 6) DI/ADX
    if base_w != std["atr"]:
        adx = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"],
            close=df["close"], window=base_w
        )
        out[f"cust_plus_di_{base_w}"]  = adx.adx_pos().round(3)
        out[f"cust_minus_di_{base_w}"] = adx.adx_neg().round(3)
        out[f"cust_adx_{base_w}"]      = adx.adx().round(3)

    # 7) OBV & related
    out["cust_obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume().round(3)
    if base_w != std["obv_sma"]:
        out[f"cust_obv_sma_{base_w}"] = out["cust_obv"].rolling(
            base_w, min_periods=1
        ).mean().round(3)

    # 8) VWAP deviation & vol_spike
    out[f"cust_vwap_dev_{base_w}"] = (
        (df["close"] - ta.volume.VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"],
            close=df["close"], volume=df["volume"],
            window=base_w
        ).volume_weighted_average_price())
        / df["close"]
    ).round(6)

    if base_w != std["obv_sma"]:
        out[f"cust_vol_spike_{base_w}"] = (
            df["volume"] / df["volume"].rolling(
                base_w, min_periods=1
            ).mean()
        ).round(3)

    # 9) Returns & rolling vol
    for p in (1, base_w, double):
        out[f"cust_r_{p}"] = np.log(df["close"] / df["close"].shift(p)).round(6)
    out[f"cust_vol_{base_w}"] = out["cust_r_1"].rolling(
        base_w, min_periods=1
    ).std().round(6)

    return out.dropna()

#########################################################################################################


    # def scale_with_splits(
    #     df: pd.DataFrame,
    #     train_prop: float = params.train_prop,
    #     val_prop:   float = params.val_prop
    # ) -> pd.DataFrame:
    #     """
    #     Chronologically split, encode cyclic dates, scale features, and preserve day-level progress.
    
    #     1) Deep-copy and split df into train / val / test by chronological index.
    #     2) Encode hour, day_of_week, month as sine & cosine columns on each split.
    #     3) Dynamically group columns into:
    #        - price_feats     (OHLCV, ATR, OBV, Bollinger bands)
    #        - ratio_feats     (returns, vol-ratios, RSI, Stochastics, ATR/price, engineered)
    #        - indicator_feats (EMA, SMA, MACD, DI/ADX)
    #     4) Fit a ColumnTransformer on TRAIN only, with:
    #        • RobustScaler      → price_feats  
    #        • Yeo‐Johnson + Std  → ratio_feats  
    #        • StandardScaler     → indicator_feats  
    #        remainder kept for cyclic, bid/ask, label
    #     5) For each split (train/val/test):
    #          a) Pre-allocate one NumPy array sized (n_rows, n_output_feats).
    #          b) Loop per calendar day with tqdm, transform only that day’s slice,
    #             and write into the array at the proper row positions.
    #          c) Wrap the array into a DataFrame preserving index & column order.
    #     6) Compress each sin/cos pair back to one column via 1-component PCA (fit on TRAIN).
    #     7) Reattach label + raw bid/ask, concatenate splits, sort by index, and return.
    #     """
    #     # 1) split
    #     df = df.copy()
    #     N    = len(df)
    #     n_tr = int(N * train_prop)
    #     n_val= int(N * val_prop)
    #     if n_tr + n_val >= N:
    #         raise ValueError("train_prop + val_prop must sum to < 1.0")
    
    #     df_tr = df.iloc[:n_tr].copy()
    #     df_v  = df.iloc[n_tr:n_tr + n_val].copy()
    #     df_te = df.iloc[n_tr + n_val:].copy()
    
    #     # 2) encode cyclic date parts
    #     for part in (df_tr, df_v, df_te):
    #         h = part["hour"]
    #         part["hour_sin"], part["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
    
    #         dow = part["day_of_week"]
    #         part["dow_sin"], part["dow_cos"] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
    
    #         m = part["month"]
    #         part["mo_sin"], part["mo_cos"] = np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)
    
    #     # 3) dynamic feature grouping
    #     cols = df.columns
    
    #     price_feats = [
    #         c for c in cols
    #         if c in ("open","high","low","close","volume")
    #            or (c.startswith("atr_") and "ratio" not in c)
    #            or (c.startswith("obv") and "_sma" not in c)
    #            or c.startswith("bb_lband_")
    #            or c.startswith("bb_hband_")
    #     ]
    
    #     ratio_feats = [
    #         c for c in cols
    #         if c.startswith("r_")
    #            or c.startswith("vol_")
    #            or "spike" in c
    #            or "vwap_dev" in c
    #            or c.startswith("rsi_")
    #            or c.startswith("bb_width_")
    #            or c.startswith("stoch_k_")
    #            or c.startswith("stoch_d_")
    #            or "atr_ratio" in c
    #            or c.startswith("obv_sma_")
    #            or c.startswith("atr_sma_")
    #            or c.startswith("eng_")
    #     ]
    
    #     indicator_feats = [
    #         c for c in cols
    #         if c.startswith("ema_")
    #            or c.startswith("sma_")
    #            or c.startswith("macd_")
    #            or c.startswith("plus_di_")
    #            or c.startswith("minus_di_")
    #            or c.startswith("adx_")
    #     ]
    
    #     # 4) build & fit transformer on TRAIN
    #     ct = ColumnTransformer([
    #         ("price",     RobustScaler(), price_feats),
    #         ("ratio",     Pipeline([
    #             ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
    #             ("std",   StandardScaler())
    #         ]), ratio_feats),
    #         ("indicator", StandardScaler(), indicator_feats),
    #     ], remainder="passthrough")
    #     ct.fit(df_tr)
    
    #     def _transform_with_progress(split_df, split_name):
    #         """
    #         Transform split_df in day-batches into one large array, showing tqdm.
    #         """
    #         # determine output columns
    #         rest = [c for c in split_df.columns
    #                 if c not in price_feats + ratio_feats + indicator_feats]
    #         out_cols = price_feats + ratio_feats + indicator_feats + rest
    
    #         n_rows, n_feats = len(split_df), len(out_cols)
    #         out_arr = np.empty((n_rows, n_feats), dtype=float)
    
    #         # iterate by calendar day
    #         for day, day_df in tqdm(
    #             split_df.groupby(split_df.index.normalize()),
    #             desc=f"{split_name.capitalize()} days", unit="day"
    #         ):
    #             mask = split_df.index.normalize() == day
    #             arr  = ct.transform(day_df)
    #             out_arr[mask, :] = arr
    
    #         return pd.DataFrame(out_arr, columns=out_cols, index=split_df.index)
    
    #     # 5) transform each split
    #     df_tr_s = _transform_with_progress(df_tr, "train")
    #     df_v_s  = _transform_with_progress(df_v,  "val")
    #     df_te_s = _transform_with_progress(df_te, "test")
    
    #     # 6) PCA compress cyclic sin/cos → single feature
    #     cyclic_pairs = [
    #         ("hour",       ["hour_sin","hour_cos"]),
    #         ("day_of_week",["dow_sin","dow_cos"]),
    #         ("month",      ["mo_sin","mo_cos"]),
    #     ]
    #     for feat_name, cols_pair in tqdm(cyclic_pairs, desc="PCA compress", unit="feat"):
    #         pca = PCA(n_components=1)
    #         vals_train = df_tr_s[cols_pair].values
    #         df_tr_s[feat_name] = pca.fit_transform(vals_train).ravel().round(3)
    #         df_v_s [feat_name] = pca.transform(df_v_s [cols_pair].values).ravel().round(3)
    #         df_te_s[feat_name] = pca.transform(df_te_s[cols_pair].values).ravel().round(3)
    
    #         df_tr_s.drop(cols_pair, axis=1, inplace=True)
    #         df_v_s .drop(cols_pair, axis=1, inplace=True)
    #         df_te_s.drop(cols_pair, axis=1, inplace=True)
    
    #     # 7) reattach label + raw bid/ask, concat splits
    #     for scaled, orig in ((df_tr_s, df_tr), (df_v_s, df_v), (df_te_s, df_te)):
    #         scaled[params.label_col] = orig[params.label_col].values
    #         if "bid" in orig.columns:
    #             scaled["bid"] = orig["bid"].values
    #         if "ask" in orig.columns:
    #             scaled["ask"] = orig["ask"].values
    
    #     return pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()
    


def scale_with_splits(
    df: pd.DataFrame,
    train_prop: float = params.train_prop,
    val_prop:   float = params.val_prop
) -> pd.DataFrame:
    """
    1) Split chronologically (train/val/test) to prevent leakage.
    2) In each split, encode hour, day_of_week, month as sine & cosine.
    3) Partition features into four groups for domain‐aware scaling:
       • bounded_inds   : RSI, Stochastics, +DI/–DI, ADX (native 0–100).  
       • price_feats    : OHLCV, ATR, OBV, VWAP, BBands.  
       • ratio_feats    : log-returns, vol_spike, vwap_dev, BB width, eng_*.  
       • unbounded_inds : EMA, SMA, MACD drifts.  
       Everything else (sines/cosines, bid, ask, label) is passed through unchanged.
    4) Fit a ColumnTransformer on TRAIN only:
       – bounded_inds   → x / 100  
       – price_feats    → RobustScaler(5–95%) → StandardScaler() → MinMaxScaler(0,1)  
       – ratio_feats    → PowerTransformer(yeo-johnson) → StandardScaler()  
                          → clip ±3σ → MinMaxScaler(0,1)  
       – unbounded_inds → StandardScaler() → MinMaxScaler(0,1)  
    5) For each split (“train”, “val”, “test”), and for each calendar day within it,
       transform the scaled feature groups (tqdm progress per split & per day).
    6) PCA‐compress each sin/cos pair back into “hour”, “day_of_week”, “month”.
    7) Concatenate train/val/test, sort by index, and return.
    """
    df = df.copy()
    N     = len(df)
    n_tr  = int(N * train_prop)
    n_val = int(N * val_prop)
    if n_tr + n_val >= N:
        raise ValueError("train_prop + val_prop must sum to < 1.0")

    # 1) Chronological splits
    df_tr = df.iloc[:n_tr].copy()
    df_v  = df.iloc[n_tr : n_tr + n_val].copy()
    df_te = df.iloc[n_tr + n_val :].copy()

    # 2) Encode cyclic time features
    for split in (df_tr, df_v, df_te):
        h = split["hour"]
        split["hour_sin"], split["hour_cos"] = np.sin(2*np.pi*h/24),   np.cos(2*np.pi*h/24)
        dow = split["day_of_week"]
        split["dow_sin"], split["dow_cos"]   = np.sin(2*np.pi*dow/7),  np.cos(2*np.pi*dow/7)
        m = split["month"]
        split["mo_sin"],  split["mo_cos"]    = np.sin(2*np.pi*m/12),   np.cos(2*np.pi*m/12)

    # 3) Identify feature groups
    cols = df.columns
    bounded_inds = [c for c in cols if c.startswith(("rsi_","stoch_","plus_di_","minus_di_","adx_"))]
    price_feats  = [c for c in cols
                    if c in ("open","high","low","close","volume")
                    or c.startswith(("atr_","obv","vwap","bb20_lband","bb20_hband"))]
    ratio_feats  = [c for c in cols
                    if c.startswith("r_")
                    or "vol_spike" in c
                    or "vwap_dev" in c
                    or c.endswith("_width")
                    or c.startswith("eng_")]
    unbounded_inds = [c for c in cols
                      if c.startswith(("ema_","sma_","macd_"))
                      and c not in bounded_inds]

    scaled_feats = bounded_inds + price_feats + ratio_feats + unbounded_inds

    # 4) Build & fit transformer on TRAIN[scaled_feats]
    ct = ColumnTransformer([
        ("bounded",
         FunctionTransformer(lambda X: X / 100.0),
         bounded_inds),
        ("price",
         Pipeline([
             ("robust", RobustScaler(quantile_range=(5,95))),
             ("std",    StandardScaler()),
             ("mm",     MinMaxScaler(feature_range=(0,1))),
         ]),
         price_feats),
        ("ratio",
         Pipeline([
             ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
             ("std",   StandardScaler()),
             ("clip",  FunctionTransformer(lambda X: np.clip(X, -3, 3))),
             ("mm",    MinMaxScaler(feature_range=(0,1))),
         ]),
         ratio_feats),
        ("unbound",
         Pipeline([
             ("std", StandardScaler()),
             ("mm",  MinMaxScaler(feature_range=(0,1))),
         ]),
         unbounded_inds),
    ], remainder="drop")

    ct.fit(df_tr[scaled_feats])

    # 5) Transform per split and per day
    def transform_per_day(split_df, split_name):
        # We will fill this array day-by-day
        out_arr = np.empty((len(split_df), len(scaled_feats)), dtype=float)

        for day, day_df in tqdm(
            split_df.groupby(split_df.index.normalize()),
            desc=f"{split_name} days", unit="day"
        ):
            mask = split_df.index.normalize() == day
            out_arr[mask] = ct.transform(day_df[scaled_feats])

        # Reassemble scaled features
        df_scaled = pd.DataFrame(out_arr,
                                 index=split_df.index,
                                 columns=scaled_feats)

        # 5c) Reattach untouched columns
        for c in split_df.columns:
            if c not in scaled_feats:
                df_scaled[c] = split_df[c]

        return df_scaled

    df_tr_s, df_v_s, df_te_s = [], [], []
    for split_name, split_df in tqdm(
        [("train", df_tr), ("val", df_v), ("test", df_te)],
        desc="Scaling splits", unit="split"
    ):
        scaled = transform_per_day(split_df, split_name)
        if split_name == "train":
            df_tr_s = scaled
        elif split_name == "val":
            df_v_s = scaled
        else:
            df_te_s = scaled

    # 6) PCA‐compress each sin/cos pair back to single fields
    for feat, (c1, c2) in zip(
        ["hour","day_of_week","month"],
        [("hour_sin","hour_cos"), ("dow_sin","dow_cos"), ("mo_sin","mo_cos")]
    ):
        pca = PCA(n_components=1)
        vals_tr = df_tr_s[[c1, c2]].values
        df_tr_s[feat] = pca.fit_transform(vals_tr).ravel().round(3)
        df_v_s [feat] = pca.transform(df_v_s [[c1, c2]].values).ravel().round(3)
        df_te_s[feat] = pca.transform(df_te_s[[c1, c2]].values).ravel().round(3)

        df_tr_s.drop([c1, c2], axis=1, inplace=True)
        df_v_s .drop([c1, c2], axis=1, inplace=True)
        df_te_s.drop([c1, c2], axis=1, inplace=True)

    # 7) Concatenate and sort
    return pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()



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


