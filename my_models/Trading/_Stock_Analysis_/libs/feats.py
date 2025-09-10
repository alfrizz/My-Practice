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

from sklearn.compose   import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    PowerTransformer,
    OneHotEncoder
)
from sklearn.decomposition import PCA


#########################################################################################################


# def features_creation(
#     df: pd.DataFrame,
#     ma_window: int = 20
# ) -> pd.DataFrame:
#     """
#     Build a rich feature set for time-series bars in three stages:

#       1) Enforce a DateTimeIndex on df.
#       2) Compute core technical indicators on OHLCV:
#          • Trend: EMA(12), SMA(26), MACD line/signal/diff
#          • Volatility: Bollinger Bands (lower, upper, relative width),
#            ATR(14) + rolling ATR(14)
#          • Momentum: RSI(14), Stochastic %K(14,3) and %D(14,3)
#          • Directional: +DI(14), –DI(14), ADX(14)
#          • Volume: OBV + rolling OBV(20), VWAP deviation, volume spike
#          • Returns & vol: log-returns r_1, r_5, r_15; vol_15
#          • Calendar flags: hour, day_of_week, month

#       3) Copy raw open/high/low/close/volume/bid/ask/label through.
#       4) Drop any rows with NaNs and return.

#     Returns:
#         A DataFrame of selected features ready for modeling.
#     """
#     # 1) Ensure datetime index
#     if not isinstance(df.index, pd.DatetimeIndex):
#         df.index = pd.to_datetime(df.index)
#     out = df.copy()

#     # 2) Trend indicators
#     out["ema"] = ta.trend.EMAIndicator(
#         close=df["close"], window=12
#     ).ema_indicator()

#     out["sma"] = ta.trend.SMAIndicator(
#         close=df["close"], window=26
#     ).sma_indicator()

#     macd = ta.trend.MACD(
#         close=df["close"], window_slow=26, window_fast=12, window_sign=9
#     )
#     out["macd_line"]   = macd.macd()
#     out["macd_signal"] = macd.macd_signal()
#     out["macd_diff"]   = macd.macd_diff()

#     # Bollinger Bands + relative width
#     bb    = ta.volatility.BollingerBands(
#         close=df["close"], window=20, window_dev=2
#     )
#     lower = bb.bollinger_lband()
#     upper = bb.bollinger_hband()
#     mid   = bb.bollinger_mavg()
#     out["bb_lband"]    = lower
#     out["bb_hband"]    = upper
#     out["bb_width_20"] = (upper - lower) / mid

#     # Momentum: RSI
#     out["rsi"] = ta.momentum.RSIIndicator(
#         close=df["close"], window=14
#     ).rsi()

#     # Directional: +DI, –DI, ADX
#     adx = ta.trend.ADXIndicator(
#         high=df["high"], low=df["low"], close=df["close"], window=14
#     )
#     out["plus_di"]  = adx.adx_pos()
#     out["minus_di"] = adx.adx_neg()
#     out["adx"]      = adx.adx()

#     # Volatility: ATR + rolling ATR
#     out["atr_14"]  = ta.volatility.AverageTrueRange(
#         high=df["high"], low=df["low"], close=df["close"], window=14
#     ).average_true_range()
#     out["atr_sma"] = out["atr_14"].rolling(ma_window).mean()

#     # ATR ratio features
#     out["atr_ratio"]     = out["atr_14"] / df["close"]
#     out["atr_ratio_sma"] = out["atr_ratio"].rolling(ma_window).mean()

#     # Volume: OBV + rolling OBV
#     out["obv"]     = ta.volume.OnBalanceVolumeIndicator(
#         close=df["close"], volume=df["volume"]
#     ).on_balance_volume()
#     out["obv_sma"] = out["obv"].rolling(ma_window).mean()

#     # VWAP deviation
#     vwap = ta.volume.VolumeWeightedAveragePrice(
#         high=df["high"], low=df["low"], close=df["close"],
#         volume=df["volume"], window=ma_window
#     ).volume_weighted_average_price()
#     out["vwap_dev"] = (df["close"] - vwap) / vwap

#     # Returns and short-term volatility
#     for n in (1, 5, 15):
#         out[f"r_{n}"] = np.log(df["close"] / df["close"].shift(n))
#     out["vol_15"] = out["r_1"].rolling(ma_window).std()

#     # Volume spike
#     out["volume_spike"] = df["volume"] / df["volume"].rolling(ma_window).mean()

#     # Stochastic oscillator %K, %D
#     stoch = ta.momentum.StochasticOscillator(
#         high=df["high"], low=df["low"], close=df["close"],
#         window=14, smooth_window=3
#     )
#     out["stoch_k_14"] = stoch.stoch()
#     out["stoch_d_3"]  = stoch.stoch_signal()

#     # Calendar features
#     out["hour"]        = df.index.hour
#     out["day_of_week"] = df.index.dayofweek
#     out["month"]       = df.index.month

#     # 3) Copy raw OHLCV + bid/ask/label
#     for col in [
#         "open", "high", "low", "close", "volume",
#         "bid", "ask", params.label_col
#     ]:
#         out[col] = df[col]

#     # 4) Select features, drop NaNs, return
#     keep = [
#         "ema", "sma",
#         "macd_line", "macd_signal", "macd_diff",
#         "bb_lband", "bb_hband", "bb_width_20",
#         "rsi", "plus_di", "minus_di", "adx",
#         "atr_14", "atr_sma", "atr_ratio", "atr_ratio_sma",
#         "obv", "obv_sma", "vwap_dev",
#         "r_1", "r_5", "r_15", "vol_15", "volume_spike",
#         "stoch_k_14", "stoch_d_3",
#         "hour", "day_of_week", "month",
#         "open", "high", "low", "close", "volume",
#         "bid", "ask", params.label_col
#     ]
#     return out.loc[:, keep].dropna()



def features_creation(
    df: pd.DataFrame,
    base_window: int = params.smooth_sign_win_tick
) -> pd.DataFrame:
    """
    Build a unified feature set for OHLCV bars using a single base_window.

    1) Ensure df has a DateTimeIndex.
    2) Derive all rolling/window parameters from base_window:
       • short_w        = max(2, base_window//2)
       • long_w         = base_window * 2
       • macd_sig       = max(1, base_window//4)
       • return_periods = [1, base_window, long_w]
    3) Compute indicators:
       - Trend: EMA(short_w), SMA(short_w, base_window, long_w), MACD(short_w, base_window, macd_sig)
       - Volatility: ATR(base_window, long_w) + rolling ATR(base_window), Bollinger Bands(base_window)
       - Momentum: RSI(base_window), Stochastic %K(base_window), %D(macd_sig)
       - Directional: +DI(base_window), –DI(base_window), ADX(base_window)
       - Volume: OBV + SMA(base_window), VWAP deviation(base_window), volume_spike(base_window)
       - Returns & Volatility: log-returns at return_periods, vol of r_1 over base_window
       - Calendar: hour, day_of_week, month
    4) Copy raw open/high/low/close/volume/bid/ask/label through.
    5) Drop rows with NaNs and return only the selected columns.
    """
    # 1) Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    out = df.copy()

    # 2) Derive window sizes
    short_w        = max(2, base_window // 2)
    long_w         = base_window * 2
    macd_sig       = max(1, base_window // 4)
    return_periods = [1, base_window, long_w]

    # 3a) Trend indicators
    out[f"ema_{short_w}"] = ta.trend.EMAIndicator(
        close=out["close"], window=short_w
    ).ema_indicator()
    for w in (short_w, base_window, long_w):
        out[f"sma_{w}"] = ta.trend.SMAIndicator(
            close=out["close"], window=w
        ).sma_indicator()

    macd = ta.trend.MACD(
        close      = out["close"],
        window_slow= base_window,
        window_fast= short_w,
        window_sign= macd_sig
    )
    out["macd_line"]   = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"]   = macd.macd_diff()

    # 3b) Volatility indicators
    for w in (base_window, long_w):
        atr_col = f"atr_{w}"
        out[atr_col] = ta.volatility.AverageTrueRange(
            high  = out["high"],
            low   = out["low"],
            close = out["close"],
            window= w
        ).average_true_range()
        out[f"{atr_col}_sma_{base_window}"] = out[atr_col].rolling(
            base_window
        ).mean()

    bb = ta.volatility.BollingerBands(
        close      = out["close"],
        window     = base_window,
        window_dev = 2
    )
    out[f"bb_lband_{base_window}"] = bb.bollinger_lband()
    out[f"bb_hband_{base_window}"] = bb.bollinger_hband()
    mid = bb.bollinger_mavg()
    out[f"bb_width_{base_window}"] = (
        out[f"bb_hband_{base_window}"] - out[f"bb_lband_{base_window}"]
    ) / mid

    # 3c) Momentum indicators
    out[f"rsi_{base_window}"] = ta.momentum.RSIIndicator(
        close=out["close"], window=base_window
    ).rsi()

    stoch = ta.momentum.StochasticOscillator(
        high          = out["high"],
        low           = out["low"],
        close         = out["close"],
        window        = base_window,
        smooth_window = macd_sig
    )
    out[f"stoch_k_{base_window}"] = stoch.stoch()
    out[f"stoch_d_{macd_sig}"]    = stoch.stoch_signal()

    # 3d) Directional indicators
    adx = ta.trend.ADXIndicator(
        high  = out["high"],
        low   = out["low"],
        close = out["close"],
        window= base_window
    )
    out[f"plus_di_{base_window}"]  = adx.adx_pos()
    out[f"minus_di_{base_window}"] = adx.adx_neg()
    out[f"adx_{base_window}"]      = adx.adx()

    # 3e) Volume indicators
    out["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close = out["close"],
        volume= out["volume"]
    ).on_balance_volume()
    out[f"obv_sma_{base_window}"] = out["obv"].rolling(
        base_window
    ).mean()

    vwap = ta.volume.VolumeWeightedAveragePrice(
        high   = out["high"],
        low    = out["low"],
        close  = out["close"],
        volume = out["volume"],
        window = base_window
    ).volume_weighted_average_price()
    out[f"vwap_dev_{base_window}"] = (out["close"] - vwap) / vwap

    out[f"volume_spike_{base_window}"] = (
        out["volume"] /
        out["volume"].rolling(base_window).mean()
    )

    # 3f) Returns & short-term volatility
    for rp in return_periods:
        out[f"r_{rp}"] = np.log(out["close"] / out["close"].shift(rp))
    out[f"vol_{base_window}"] = out["r_1"].rolling(
        base_window
    ).std()

    # 3g) Calendar features
    out["hour"]        = out.index.hour
    # fix: Pandas uses `.dayofweek`, not `.day_of_week`
    out["day_of_week"] = out.index.dayofweek
    out["month"]       = out.index.month

    # 4) Copy raw OHLCV + bid/ask/label through
    for col in ["open", "high", "low", "close", "volume",
                "bid", "ask", params.label_col]:
        out[col] = df[col]

    # 5) Select final columns & drop rows with NaNs
    keep = []
    keep.append(f"ema_{short_w}")
    keep += [f"sma_{w}" for w in (short_w, base_window, long_w)]
    keep += ["macd_line", "macd_signal", "macd_diff"]
    keep += [f"atr_{w}" for w in (base_window, long_w)]
    keep += [f"atr_{w}_sma_{base_window}" for w in (base_window, long_w)]
    keep += [
        f"bb_lband_{base_window}", f"bb_hband_{base_window}",
        f"bb_width_{base_window}"
    ]
    keep += [f"rsi_{base_window}", f"stoch_k_{base_window}", f"stoch_d_{macd_sig}"]
    keep += [
        f"plus_di_{base_window}", f"minus_di_{base_window}",
        f"adx_{base_window}"
    ]
    keep += ["obv", f"obv_sma_{base_window}", f"vwap_dev_{base_window}",
             f"volume_spike_{base_window}"]
    keep += [f"r_{rp}" for rp in return_periods] + [f"vol_{base_window}"]
    keep += ["hour", "day_of_week", "month"]
    keep += ["open", "high", "low", "close", "volume",
             "bid", "ask", params.label_col]

    return out.loc[:, keep].dropna()




#########################################################################################################


# def features_engineering(
#     df: pd.DataFrame,
#     low_rsi: int = 30,
#     high_rsi: int = 70,
#     adx_thresh: int = 20,
#     adx_window: int = 7,
#     eps: float = 1e-8
# ) -> pd.DataFrame:
#     """
#     Build higher‐level, continuous trading signals from base OHLCV indicators:

#       1) eng_ma       : difference between EMA(12) and SMA(26)
#       2) eng_macd     : MACD histogram (MACD line – signal)
#       3) eng_bb       : distance outside Bollinger Bands (no stateful hold)
#       4) eng_rsi      : distance beyond RSI thresholds (no stateful hold)
#       5) eng_adx      : ADX-weighted |+DI – –DI| spread, normalized over window
#       6) eng_obv      : on-balance volume divergence from its SMA
#       7) eng_atr_div  : divergence of ATR/price ratio, scaled

#     Returns:
#         df with seven new columns: 
#         ['eng_ma','eng_macd','eng_bb','eng_rsi','eng_adx','eng_obv','eng_atr_div']
#     """
#     price = df["close"]

#     # 1) MA spread: short-minus-long trend
#     df["eng_ma"] = (df["ema"] - df["sma"]).round(3)

#     # 2) MACD histogram
#     df["eng_macd"] = df["macd_diff"].round(3)

#     # 3) Bollinger Bands distance (zero inside bands, instant reset)
#     lo = df["bb_lband"]
#     hi = df["bb_hband"]
#     below = price < lo
#     above = price > hi
#     df["eng_bb"] = np.where(
#         below, (lo - price),
#         np.where(above, (hi - price), 0.0)
#     ).round(3)

#     # 4) RSI threshold distance (zero within [low_rsi, high_rsi])
#     rsi = df["rsi"]
#     below = rsi < low_rsi
#     above = rsi > high_rsi
#     df["eng_rsi"] = np.where(
#         below,  (low_rsi  - rsi),
#         np.where(above, (high_rsi - rsi), 0.0)
#     ).round(3)

#     # 5) ADX-weighted DI spread, normalized to recent window
#     plus, minus, adx = df["plus_di"], df["minus_di"], df["adx"]
#     di      = (plus - minus).abs()
#     excess  = (adx - adx_thresh).clip(lower=0.0)
#     raw     = di * excess
#     # rolling max of adx above threshold
#     max_adx = (adx.rolling(adx_window).max() - adx_thresh).clip(lower=0.0)
#     # rolling max of raw spread
#     max_raw = raw.rolling(adx_window).max().clip(lower=eps)
#     scale   = (max_adx / max_raw).fillna(0.0)
#     sign    = np.sign(plus - minus)
#     df["eng_adx"] = np.where(
#         adx > adx_thresh,
#         (sign * raw * scale).round(3),
#         0.0
#     )

#     # 6) OBV divergence
#     df["eng_obv"] = (df["obv"] - df["obv_sma"]).round(3)

#     # 7) ATR/price-ratio divergence scaled
#     df["eng_atr_div"] = ((df["atr_ratio"] - df["atr_ratio_sma"]) * 10_000).round(1)

#     return df


def features_engineering(
    df: pd.DataFrame,
    base_window:             int = params.smooth_sign_win_tick,
    low_rsi:                 float = 30.0,
    high_rsi:                float = 70.0,
    adx_thresh:              float = 20.0,
    adx_multiplier_window:   int   = None,
    eps:                     float = 1e-8
) -> pd.DataFrame:
    """
    Build seven engineered signals (eng_*) from the base indicators in df.

    1) eng_ma      = EMA(short_w) – SMA(long_w)
    2) eng_macd    = MACD line – MACD signal
    3) eng_bb      = price distance outside Bollinger Bands
    4) eng_rsi     = distance beyond RSI thresholds
    5) eng_adx     = ADX-weighted |+DI – –DI| spread, normalized
    6) eng_obv     = OBV divergence from its SMA
    7) eng_atr_div = divergence of ATR/price ratio, scaled

    This function assumes df already contains columns named:
       ema_{short_w}, sma_{long_w}, macd_line, macd_signal,
       bb_lband_{aw}, bb_hband_{aw}, rsi_{aw},
       plus_di_{aw}, minus_di_{aw}, adx_{aw},
       obv, obv_sma_{aw},
       atr_{aw}
    """
    out = df.copy()

    # derive the same windows used upstream in features_creation
    short_w = max(2, base_window // 2)
    long_w  = base_window * 2
    aw      = base_window
    sig_w   = max(1, aw // 4)

    # 1) MA spread: EMA(short) minus SMA(long)
    out["eng_ma"] = (
        out[f"ema_{short_w}"] - out[f"sma_{long_w}"]
    ).round(3)

    # 2) MACD histogram
    out["eng_macd"] = (out["macd_line"] - out["macd_signal"]).round(3)

    # 3) Bollinger Bands distance
    lo    = out[f"bb_lband_{aw}"]
    hi    = out[f"bb_hband_{aw}"]
    price = out["close"]
    out["eng_bb"] = np.where(
        price < lo,   (lo - price),
        np.where(price > hi, (price - hi), 0.0)
    ).round(3)

    # 4) RSI threshold distance
    rsi = out[f"rsi_{aw}"]
    out["eng_rsi"] = np.where(
        rsi < low_rsi,  (low_rsi  - rsi),
        np.where(rsi > high_rsi, (rsi - high_rsi), 0.0)
    ).round(3)

    # 5) ADX-weighted DI spread normalized over a window
    plus  = out[f"plus_di_{aw}"]
    minus = out[f"minus_di_{aw}"]
    adx   = out[f"adx_{aw}"]
    di    = (plus - minus).abs()
    excess= (adx - adx_thresh).clip(lower=0.0)
    raw   = di * excess

    mult_w = adx_multiplier_window or aw
    max_excess = (adx.rolling(mult_w).max() - adx_thresh).clip(lower=0.0)
    max_raw    = raw.rolling(mult_w).max().clip(lower=eps)
    scale      = (max_excess / max_raw).fillna(0.0)
    sign       = np.sign(plus - minus)

    out["eng_adx"] = np.where(
        adx > adx_thresh,
        (sign * raw * scale).round(3),
        0.0
    )

    # 6) OBV divergence
    out["eng_obv"] = (
        out["obv"] - out[f"obv_sma_{aw}"]
    ).round(3)

    # 7) ATR/price-ratio divergence
    # compute ratio on the fly instead of expecting missing columns
    atr_col    = out[f"atr_{aw}"]
    atr_ratio  = atr_col / out["close"]
    atr_ratio_sma = atr_ratio.rolling(aw).mean()
    out["eng_atr_div"] = ((atr_ratio - atr_ratio_sma) * 10_000).round(1)

    return out

#########################################################################################################


# def scale_with_splits(
#     df: pd.DataFrame,
#     train_prop: float = params.train_prop,
#     val_prop:   float = params.val_prop
# ) -> pd.DataFrame:
#     """
#     Chronologically split, encode cyclic date parts, and robustly scale all numeric features.

#     Steps:
#       1) Deep-copy and split df into train / val / test by row index.
#       2) Encode hour, day_of_week, month as sine & cosine pairs.
#       3) Define three feature groups:
#          • price_feats     : raw price & volume indicators
#          • ratio_feats     : returns, volume‐spikes, relative ratios, RSI, stochastics, eng_*
#          • indicator_feats : EMA, SMA, MACD line/signal/diff
#       4) Build ColumnTransformer with three pipelines:
#          - RobustScaler()                 on price_feats
#          - Pipeline(Yeo‐Johnson → StandardScaler) on ratio_feats
#          - StandardScaler()               on indicator_feats
#          remainder='passthrough' (cyclic, bid, ask, label)
#       5) Fit the transformers on TRAIN only; transform train/val/test in a tqdm loop.
#       6) Reassemble DataFrames preserving original index & columns.
#       7) Compress each cyclic sin/cos pair back to a single value via PCA in a tqdm loop.
#       8) Reattach label + raw bid/ask, concatenate splits, and return.

#     Returns:
#         A single DataFrame with all numeric features zero‐mean/unit‐var
#         (or IQR‐scaled for price), cyclic dates collapsed, plus label, bid, ask.
#     """
#     # 1) Deep-copy + chronological split
#     df = df.copy()
#     N = len(df)
#     n_tr = int(N * train_prop)
#     n_val = int(N * val_prop)
#     if n_tr + n_val >= N:
#         raise ValueError("train_prop + val_prop must sum to < 1.0")

#     df_tr = df.iloc[:n_tr].copy()
#     df_v  = df.iloc[n_tr:n_tr + n_val].copy()
#     df_te = df.iloc[n_tr + n_val:].copy()

#     # 2) Encode cyclic date parts
#     for part in (df_tr, df_v, df_te):
#         h = part["hour"]
#         part.loc[:, "hour_sin"] = np.sin(2 * np.pi * h / 24)
#         part.loc[:, "hour_cos"] = np.cos(2 * np.pi * h / 24)

#         dow = part["day_of_week"]
#         part.loc[:, "dow_sin"]  = np.sin(2 * np.pi * dow / 7)
#         part.loc[:, "dow_cos"]  = np.cos(2 * np.pi * dow / 7)

#         m = part["month"]
#         part.loc[:, "mo_sin"]   = np.sin(2 * np.pi * m / 12)
#         part.loc[:, "mo_cos"]   = np.cos(2 * np.pi * m / 12)

#     # 3) Define feature groups based on original df columns
#     cols = df.columns
#     price_feats = [c for c in cols if c in (
#         "open", "high", "low", "close", "volume",
#         "atr_14", "atr_sma", "obv", "obv_sma",
#         "bb_lband", "bb_hband"
#     )]

#     ratio_feats = (
#         [c for c in cols if c.startswith(("r_", "vol_"))] +
#         [c for c in cols if c in (
#             "volume_spike", "vwap_dev", "rsi", "bb_width_20",
#             "stoch_k_14", "stoch_d_3", "atr_ratio", "atr_ratio_sma"
#         )] +
#         [c for c in cols if c.startswith("eng_")]
#     )

#     indicator_feats = [
#         c for c in cols
#         if c in ("ema", "sma", "macd_line", "macd_signal", "macd_diff")
#     ]

#     # 4) Build ColumnTransformer pipelines
#     price_pipe = RobustScaler()
#     ratio_pipe = Pipeline([
#         ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
#         ("std",   StandardScaler())
#     ])
#     indicator_pipe = StandardScaler()

#     ct = ColumnTransformer(
#         [
#             ("price",     price_pipe,     price_feats),
#             ("ratio",     ratio_pipe,     ratio_feats),
#             ("indicator", indicator_pipe, indicator_feats),
#         ],
#         remainder="passthrough"  # keep cyclic, bid, ask, label intact
#     )

#     # 5) Fit on train, transform all splits with a progress bar
#     split_map = {"train": df_tr, "val": df_v, "test": df_te}
#     transformed = {}
#     for name, subset in tqdm(split_map.items(), desc="Scaling splits", unit="split"):
#         if name == "train":
#             arr = ct.fit_transform(subset)
#         else:
#             arr = ct.transform(subset)

#         # build DataFrame: scaled groups first, then the rest
#         rest = [c for c in subset.columns
#                 if c not in price_feats + ratio_feats + indicator_feats]
#         new_cols = price_feats + ratio_feats + indicator_feats + rest
#         transformed[name] = pd.DataFrame(arr, columns=new_cols, index=subset.index)

#     df_tr_s, df_v_s, df_te_s = (
#         transformed["train"], transformed["val"], transformed["test"]
#     )

#     # 6–7) Compress cyclic sin/cos pairs back to one dimension via PCA
#     cyclic_pairs = [
#         ("hour",       ["hour_sin", "hour_cos"]),
#         ("day_of_week",["dow_sin",  "dow_cos"]),
#         ("month",      ["mo_sin",   "mo_cos"]),
#     ]
#     for feat_name, cols_pair in tqdm(cyclic_pairs, desc="PCA compress", unit="feat"):
#         pca = PCA(n_components=1)
#         # fit on train
#         train_vals = df_tr_s[cols_pair].values
#         df_tr_s[feat_name] = pca.fit_transform(train_vals).ravel().round(3)
#         # apply to val/test
#         df_v_s[feat_name]  = pca.transform(df_v_s[cols_pair].values).ravel().round(3)
#         df_te_s[feat_name] = pca.transform(df_te_s[cols_pair].values).ravel().round(3)
#         # drop sin/cos
#         df_tr_s.drop(cols_pair, axis=1, inplace=True)
#         df_v_s .drop(cols_pair, axis=1, inplace=True)
#         df_te_s.drop(cols_pair, axis=1, inplace=True)

#     # 8) Reattach label + raw bid/ask, concat, and return
#     for scaled_df, orig_df in zip((df_tr_s, df_v_s, df_te_s), (df_tr, df_v, df_te)):
#         scaled_df[params.label_col] = orig_df[params.label_col].values
#         if "bid" in orig_df.columns:
#             scaled_df["bid"] = orig_df["bid"].values
#         if "ask" in orig_df.columns:
#             scaled_df["ask"] = orig_df["ask"].values

#     df_final = pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()
#     return df_final


def scale_with_splits(
    df: pd.DataFrame,
    train_prop: float = params.train_prop,
    val_prop:   float = params.val_prop
) -> pd.DataFrame:
    """
    Chronologically split, encode cyclic dates, scale features, and preserve day-level progress.

    1) Deep-copy and split df into train / val / test by chronological index.
    2) Encode hour, day_of_week, month as sine & cosine columns on each split.
    3) Dynamically group columns into:
       - price_feats     (OHLCV, ATR, OBV, Bollinger bands)
       - ratio_feats     (returns, vol-ratios, RSI, Stochastics, ATR/price, engineered)
       - indicator_feats (EMA, SMA, MACD, DI/ADX)
    4) Fit a ColumnTransformer on TRAIN only, with:
       • RobustScaler      → price_feats  
       • Yeo‐Johnson + Std  → ratio_feats  
       • StandardScaler     → indicator_feats  
       remainder kept for cyclic, bid/ask, label
    5) For each split (train/val/test):
         a) Pre-allocate one NumPy array sized (n_rows, n_output_feats).
         b) Loop per calendar day with tqdm, transform only that day’s slice,
            and write into the array at the proper row positions.
         c) Wrap the array into a DataFrame preserving index & column order.
    6) Compress each sin/cos pair back to one column via 1-component PCA (fit on TRAIN).
    7) Reattach label + raw bid/ask, concatenate splits, sort by index, and return.
    """
    # 1) split
    df = df.copy()
    N    = len(df)
    n_tr = int(N * train_prop)
    n_val= int(N * val_prop)
    if n_tr + n_val >= N:
        raise ValueError("train_prop + val_prop must sum to < 1.0")

    df_tr = df.iloc[:n_tr].copy()
    df_v  = df.iloc[n_tr:n_tr + n_val].copy()
    df_te = df.iloc[n_tr + n_val:].copy()

    # 2) encode cyclic date parts
    for part in (df_tr, df_v, df_te):
        h = part["hour"]
        part["hour_sin"], part["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)

        dow = part["day_of_week"]
        part["dow_sin"], part["dow_cos"] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)

        m = part["month"]
        part["mo_sin"], part["mo_cos"] = np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)

    # 3) dynamic feature grouping
    cols = df.columns

    price_feats = [
        c for c in cols
        if c in ("open","high","low","close","volume")
           or (c.startswith("atr_") and "ratio" not in c)
           or (c.startswith("obv") and "_sma" not in c)
           or c.startswith("bb_lband_")
           or c.startswith("bb_hband_")
    ]

    ratio_feats = [
        c for c in cols
        if c.startswith("r_")
           or c.startswith("vol_")
           or "spike" in c
           or "vwap_dev" in c
           or c.startswith("rsi_")
           or c.startswith("bb_width_")
           or c.startswith("stoch_k_")
           or c.startswith("stoch_d_")
           or "atr_ratio" in c
           or c.startswith("obv_sma_")
           or c.startswith("atr_sma_")
           or c.startswith("eng_")
    ]

    indicator_feats = [
        c for c in cols
        if c.startswith("ema_")
           or c.startswith("sma_")
           or c.startswith("macd_")
           or c.startswith("plus_di_")
           or c.startswith("minus_di_")
           or c.startswith("adx_")
    ]

    # 4) build & fit transformer on TRAIN
    ct = ColumnTransformer([
        ("price",     RobustScaler(), price_feats),
        ("ratio",     Pipeline([
            ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("std",   StandardScaler())
        ]), ratio_feats),
        ("indicator", StandardScaler(), indicator_feats),
    ], remainder="passthrough")
    ct.fit(df_tr)

    def _transform_with_progress(split_df, split_name):
        """
        Transform split_df in day-batches into one large array, showing tqdm.
        """
        # determine output columns
        rest = [c for c in split_df.columns
                if c not in price_feats + ratio_feats + indicator_feats]
        out_cols = price_feats + ratio_feats + indicator_feats + rest

        n_rows, n_feats = len(split_df), len(out_cols)
        out_arr = np.empty((n_rows, n_feats), dtype=float)

        # iterate by calendar day
        for day, day_df in tqdm(
            split_df.groupby(split_df.index.normalize()),
            desc=f"{split_name.capitalize()} days", unit="day"
        ):
            mask = split_df.index.normalize() == day
            arr  = ct.transform(day_df)
            out_arr[mask, :] = arr

        return pd.DataFrame(out_arr, columns=out_cols, index=split_df.index)

    # 5) transform each split
    df_tr_s = _transform_with_progress(df_tr, "train")
    df_v_s  = _transform_with_progress(df_v,  "val")
    df_te_s = _transform_with_progress(df_te, "test")

    # 6) PCA compress cyclic sin/cos → single feature
    cyclic_pairs = [
        ("hour",       ["hour_sin","hour_cos"]),
        ("day_of_week",["dow_sin","dow_cos"]),
        ("month",      ["mo_sin","mo_cos"]),
    ]
    for feat_name, cols_pair in tqdm(cyclic_pairs, desc="PCA compress", unit="feat"):
        pca = PCA(n_components=1)
        vals_train = df_tr_s[cols_pair].values
        df_tr_s[feat_name] = pca.fit_transform(vals_train).ravel().round(3)
        df_v_s [feat_name] = pca.transform(df_v_s [cols_pair].values).ravel().round(3)
        df_te_s[feat_name] = pca.transform(df_te_s[cols_pair].values).ravel().round(3)

        df_tr_s.drop(cols_pair, axis=1, inplace=True)
        df_v_s .drop(cols_pair, axis=1, inplace=True)
        df_te_s.drop(cols_pair, axis=1, inplace=True)

    # 7) reattach label + raw bid/ask, concat splits
    for scaled, orig in ((df_tr_s, df_tr), (df_v_s, df_v), (df_te_s, df_te)):
        scaled[params.label_col] = orig[params.label_col].values
        if "bid" in orig.columns:
            scaled["bid"] = orig["bid"].values
        if "ask" in orig.columns:
            scaled["ask"] = orig["ask"].values

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


