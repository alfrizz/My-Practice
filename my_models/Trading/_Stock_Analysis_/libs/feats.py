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

def features_creation(
    df: pd.DataFrame,
    ma_window: int = 20
) -> pd.DataFrame:
    """
    Build a rich feature set for time-series bars in three stages:

      1) Enforce a DateTimeIndex on df.
      2) Compute core technical indicators on OHLCV:
         • Trend: EMA(12), SMA(26), MACD line/signal/diff
         • Volatility: Bollinger Bands (lower, upper, relative width),
           ATR(14) + rolling ATR(14)
         • Momentum: RSI(14), Stochastic %K(14,3) and %D(14,3)
         • Directional: +DI(14), –DI(14), ADX(14)
         • Volume: OBV + rolling OBV(20), VWAP deviation, volume spike
         • Returns & vol: log-returns r_1, r_5, r_15; vol_15
         • Calendar flags: hour, day_of_week, month

      3) Copy raw open/high/low/close/volume/bid/ask/label through.
      4) Drop any rows with NaNs and return.

    Returns:
        A DataFrame of selected features ready for modeling.
    """
    # 1) Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    out = df.copy()

    # 2) Trend indicators
    out["ema"] = ta.trend.EMAIndicator(
        close=df["close"], window=12
    ).ema_indicator()

    out["sma"] = ta.trend.SMAIndicator(
        close=df["close"], window=26
    ).sma_indicator()

    macd = ta.trend.MACD(
        close=df["close"], window_slow=26, window_fast=12, window_sign=9
    )
    out["macd_line"]   = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"]   = macd.macd_diff()

    # Bollinger Bands + relative width
    bb    = ta.volatility.BollingerBands(
        close=df["close"], window=20, window_dev=2
    )
    lower = bb.bollinger_lband()
    upper = bb.bollinger_hband()
    mid   = bb.bollinger_mavg()
    out["bb_lband"]    = lower
    out["bb_hband"]    = upper
    out["bb_width_20"] = (upper - lower) / mid

    # Momentum: RSI
    out["rsi"] = ta.momentum.RSIIndicator(
        close=df["close"], window=14
    ).rsi()

    # Directional: +DI, –DI, ADX
    adx = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    out["plus_di"]  = adx.adx_pos()
    out["minus_di"] = adx.adx_neg()
    out["adx"]      = adx.adx()

    # Volatility: ATR + rolling ATR
    out["atr_14"]  = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    out["atr_sma"] = out["atr_14"].rolling(ma_window).mean()

    # ATR ratio features
    out["atr_ratio"]     = out["atr_14"] / df["close"]
    out["atr_ratio_sma"] = out["atr_ratio"].rolling(ma_window).mean()

    # Volume: OBV + rolling OBV
    out["obv"]     = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume()
    out["obv_sma"] = out["obv"].rolling(ma_window).mean()

    # VWAP deviation
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=df["high"], low=df["low"], close=df["close"],
        volume=df["volume"], window=ma_window
    ).volume_weighted_average_price()
    out["vwap_dev"] = (df["close"] - vwap) / vwap

    # Returns and short-term volatility
    for n in (1, 5, 15):
        out[f"r_{n}"] = np.log(df["close"] / df["close"].shift(n))
    out["vol_15"] = out["r_1"].rolling(ma_window).std()

    # Volume spike
    out["volume_spike"] = df["volume"] / df["volume"].rolling(ma_window).mean()

    # Stochastic oscillator %K, %D
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"],
        window=14, smooth_window=3
    )
    out["stoch_k_14"] = stoch.stoch()
    out["stoch_d_3"]  = stoch.stoch_signal()

    # Calendar features
    out["hour"]        = df.index.hour
    out["day_of_week"] = df.index.dayofweek
    out["month"]       = df.index.month

    # 3) Copy raw OHLCV + bid/ask/label
    for col in [
        "open", "high", "low", "close", "volume",
        "bid", "ask", params.label_col
    ]:
        out[col] = df[col]

    # 4) Select features, drop NaNs, return
    keep = [
        "ema", "sma",
        "macd_line", "macd_signal", "macd_diff",
        "bb_lband", "bb_hband", "bb_width_20",
        "rsi", "plus_di", "minus_di", "adx",
        "atr_14", "atr_sma", "atr_ratio", "atr_ratio_sma",
        "obv", "obv_sma", "vwap_dev",
        "r_1", "r_5", "r_15", "vol_15", "volume_spike",
        "stoch_k_14", "stoch_d_3",
        "hour", "day_of_week", "month",
        "open", "high", "low", "close", "volume",
        "bid", "ask", params.label_col
    ]
    return out.loc[:, keep].dropna()



#########################################################################################################


# def features_engineering(
#     df: pd.DataFrame,
#     low_rsi: int = 30,
#     high_rsi: int = 70,
#     adx_thresh: int = 20,
#     adx_window: int = 7
# ) -> pd.DataFrame:
#     """
#     Build higher‐level trading signals from base features:
#       1) eng_ma       : ema - sma
#       2) eng_macd     : macd_diff
#       3) eng_bb       : distance to Bollinger bands, holding state
#       4) eng_rsi      : distance to RSI thresholds, holding state
#       5) eng_adx      : ADX‐weighted DI spread with rolling scale
#       6) eng_obv      : obv - obv_sma divergence
#       7) eng_atr_div  : ATR/price ratio divergence scaled
#     """
#     price = df["close"]

#     # 1) MA spread
#     df["eng_ma"]   = (df["ema"] - df["sma"]).round(3)

#     # 2) MACD histogram
#     df["eng_macd"] = df["macd_diff"].round(3)

#     # 3) Bollinger distance (stateful)
#     lo, hi = df["bb_lband"], df["bb_hband"]
#     st_bb = (
#         pd.Series(
#             np.where(price < lo,  1,
#                      np.where(price > hi, -1, np.nan)),
#             index=df.index
#         )
#         .ffill()
#         .fillna(0)
#     )
#     df["eng_bb"] = np.where(
#         st_bb > 0,  lo - price,
#         np.where(st_bb < 0, hi - price, 0)
#     ).round(3)

#     # 4) RSI distance (stateful)
#     rsi    = df["rsi"]
#     st_rsi = (
#         pd.Series(
#             np.where(rsi < low_rsi,  1,
#                      np.where(rsi > high_rsi, -1, np.nan)),
#             index=df.index
#         )
#         .ffill()
#         .fillna(0)
#     )
#     df["eng_rsi"] = np.where(
#         st_rsi > 0,  low_rsi - rsi,
#         np.where(st_rsi < 0, high_rsi - rsi, 0)
#     ).round(3)

#     # 5) ADX‐weighted DI diff
#     plus, minus, adx = df["plus_di"], df["minus_di"], df["adx"]
#     di    = (plus - minus).abs()
#     ex    = (adx - adx_thresh).clip(lower=0)
#     raw   = di * ex
#     scale = (
#         (adx.rolling(adx_window).max() - adx_thresh) /
#         raw.rolling(adx_window).max()
#     ).replace([np.inf, -np.inf], 0).fillna(0)
#     df["eng_adx"] = np.where(
#         adx > adx_thresh,
#         np.where(plus > minus,  di * ex * scale, -di * ex * scale),
#         0
#     ).round(3)

#     # 6) OBV divergence
#     df["eng_obv"] = (df["obv"] - df["obv_sma"]).round(3)

#     # 7) ATR/price ratio divergence
#     df["eng_atr_div"] = ((df["atr_ratio"] - df["atr_ratio_sma"]) * 10_000).round(1)

#     return df


def features_engineering(
    df: pd.DataFrame,
    low_rsi: int = 30,
    high_rsi: int = 70,
    adx_thresh: int = 20,
    adx_window: int = 7,
    eps: float = 1e-8
) -> pd.DataFrame:
    """
    Build higher‐level, continuous trading signals from base OHLCV indicators:

      1) eng_ma       : difference between EMA(12) and SMA(26)
      2) eng_macd     : MACD histogram (MACD line – signal)
      3) eng_bb       : distance outside Bollinger Bands (no stateful hold)
      4) eng_rsi      : distance beyond RSI thresholds (no stateful hold)
      5) eng_adx      : ADX-weighted |+DI – –DI| spread, normalized over window
      6) eng_obv      : on-balance volume divergence from its SMA
      7) eng_atr_div  : divergence of ATR/price ratio, scaled

    Returns:
        df with seven new columns: 
        ['eng_ma','eng_macd','eng_bb','eng_rsi','eng_adx','eng_obv','eng_atr_div']
    """
    price = df["close"]

    # 1) MA spread: short-minus-long trend
    df["eng_ma"] = (df["ema"] - df["sma"]).round(3)

    # 2) MACD histogram
    df["eng_macd"] = df["macd_diff"].round(3)

    # 3) Bollinger Bands distance (zero inside bands, instant reset)
    lo = df["bb_lband"]
    hi = df["bb_hband"]
    below = price < lo
    above = price > hi
    df["eng_bb"] = np.where(
        below, (lo - price),
        np.where(above, (hi - price), 0.0)
    ).round(3)

    # 4) RSI threshold distance (zero within [low_rsi, high_rsi])
    rsi = df["rsi"]
    below = rsi < low_rsi
    above = rsi > high_rsi
    df["eng_rsi"] = np.where(
        below,  (low_rsi  - rsi),
        np.where(above, (high_rsi - rsi), 0.0)
    ).round(3)

    # 5) ADX-weighted DI spread, normalized to recent window
    plus, minus, adx = df["plus_di"], df["minus_di"], df["adx"]
    di      = (plus - minus).abs()
    excess  = (adx - adx_thresh).clip(lower=0.0)
    raw     = di * excess
    # rolling max of adx above threshold
    max_adx = (adx.rolling(adx_window).max() - adx_thresh).clip(lower=0.0)
    # rolling max of raw spread
    max_raw = raw.rolling(adx_window).max().clip(lower=eps)
    scale   = (max_adx / max_raw).fillna(0.0)
    sign    = np.sign(plus - minus)
    df["eng_adx"] = np.where(
        adx > adx_thresh,
        (sign * raw * scale).round(3),
        0.0
    )

    # 6) OBV divergence
    df["eng_obv"] = (df["obv"] - df["obv_sma"]).round(3)

    # 7) ATR/price-ratio divergence scaled
    df["eng_atr_div"] = ((df["atr_ratio"] - df["atr_ratio_sma"]) * 10_000).round(1)

    return df

#########################################################################################################


# def scale_with_splits(
#     df: pd.DataFrame,
#     train_prop: float = params.train_prop,
#     val_prop: float   = params.val_prop
# ) -> pd.DataFrame:
#     """
#     1) Chronologically split df into train/val/test by index order.
#     2) Encode calendar fields (hour, day_of_week, month) as sin/cos pairs.
#     3) Auto-identify feature groups:
#          • price_feats : open, high, low, close, volume,
#                          atr_14, atr_sma, obv, obv_sma, bb_lband, bb_hband
#          • ratio_feats : r_*, vol_15, volume_spike, vwap_dev, rsi,
#                          bb_width_20, stoch_k_14, stoch_d_3,
#                          atr_ratio, atr_ratio_sma, + all eng_* columns
#          • binary_feats: in_trading (if present)
#          • cyclic_feats: hour, day_of_week, month
#     4) Fit StandardScaler on TRAIN’s ratio_feats.
#     5) Robust-scale price_feats per calendar day using a single
#        vectorized groupby(...).expanding(): compute each bar’s
#        day-to-date median & IQR, then scale—no Python‐level loop.
#     6) Transform each split: scale price_feats + apply ratio_scaler.
#     7) Fit PCA(1) on TRAIN’s sin/cos pairs → compress back to
#        single hour, day_of_week, month columns.
#     8) Reattach label, concatenate splits, return only the
#        selected final columns in order.
#     """
#     # Make a working copy
#     df = df.copy()

#     # 1) Chronological split
#     n       = len(df)
#     n_train = int(n * train_prop)
#     n_val   = int(n * val_prop)
#     if n_train + n_val >= n:
#         raise ValueError("train_prop + val_prop must sum to < 1.0")

#     df_tr = df.iloc[:n_train].copy()
#     df_v  = df.iloc[n_train : n_train + n_val].copy()
#     df_te = df.iloc[n_train + n_val :].copy()

#     # 2) Sin/cos encode calendar features
#     for part in (df_tr, df_v, df_te):
#         h = part["hour"]
#         part["hour_sin"], part["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
#         d = part["day_of_week"]
#         part["dow_sin"], part["dow_cos"]   = np.sin(2*np.pi*d/7),  np.cos(2*np.pi*d/7)
#         m = part["month"]
#         part["mo_sin"], part["mo_cos"]     = np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)

#     # 3) Auto-discover feature groups
#     cols         = df.columns
#     price_feats  = [c for c in cols if c in (
#         "open","high","low","close","volume",
#         "atr_14","atr_sma","obv","obv_sma",
#         "bb_lband","bb_hband"
#     )]
#     ratio_feats  = (
#         [c for c in cols if c.startswith(("r_","vol_"))] +
#         [c for c in cols if c in (
#             "volume_spike","vwap_dev","rsi",
#             "bb_width_20","stoch_k_14","stoch_d_3",
#             "atr_ratio","atr_ratio_sma"
#         )] +
#         [c for c in cols if c.startswith("eng_")]
#     )
#     binary_feats = [c for c in cols if c == "in_trading"]
#     cyclic_feats = ["hour","day_of_week","month"]

#     # 4) Fit StandardScaler on TRAIN’s ratio_feats
#     ratio_scaler = StandardScaler()
#     if ratio_feats:
#         ratio_scaler.fit(df_tr[ratio_feats])

#     # 5) Vectorized robust scaling of price_feats
#     def scale_price_vec(sub: pd.DataFrame) -> pd.DataFrame:
#         out  = sub.copy()
#         days = out.index.normalize()
#         grp  = out[price_feats].groupby(days).expanding()

#         # day-to-date median & IQR for each bar
#         med = grp.median().reset_index(level=0, drop=True)
#         q75 = grp.quantile(0.75).reset_index(level=0, drop=True)
#         q25 = grp.quantile(0.25).reset_index(level=0, drop=True)
#         iqr = (q75 - q25).replace(0, 1e-6)

#         out[price_feats] = (out[price_feats] - med) / iqr
#         return out

#     # 6) Transform splits with tqdm progress
#     def transform_split(sub: pd.DataFrame) -> pd.DataFrame:
#         out = sub.copy()
#         if price_feats:
#             out = scale_price_vec(out)
#         if ratio_feats:
#             out[ratio_feats] = ratio_scaler.transform(out[ratio_feats])
#         return out

#     scaled = {}
#     for name, subset in tqdm(
#         [("train", df_tr), ("val", df_v), ("test", df_te)],
#         desc="Scaling splits", unit="split"
#     ):
#         scaled[name] = transform_split(subset)

#     df_tr_s, df_val_s, df_te_s = scaled["train"], scaled["val"], scaled["test"]

#     # 7) PCA on sin/cos → compress back to cyclic_feats
#     pca_h = PCA(n_components=1).fit(df_tr_s[["hour_sin","hour_cos"]])
#     pca_d = PCA(n_components=1).fit(df_tr_s[["dow_sin","dow_cos"]])
#     pca_m = PCA(n_components=1).fit(df_tr_s[["mo_sin","mo_cos"]])

#     def apply_pca(sub: pd.DataFrame) -> pd.DataFrame:
#         out = sub.copy()
#         out["hour"]        = pca_h.transform(out[["hour_sin","hour_cos"]])[:,0].round(3)
#         out["day_of_week"] = pca_d.transform(out[["dow_sin","dow_cos"]])[:,0].round(3)
#         out["month"]       = pca_m.transform(out[["mo_sin","mo_cos"]])[:,0].round(3)
#         out.drop([
#             "hour_sin","hour_cos",
#             "dow_sin","dow_cos",
#             "mo_sin","mo_cos"
#         ], axis=1, inplace=True)
#         return out

#     for name, subset in tqdm(
#         [("train", df_tr_s), ("val", df_val_s), ("test", df_te_s)],
#         desc="Applying PCA", unit="split"
#     ):
#         scaled[name] = apply_pca(subset)

#     df_tr_s, df_val_s, df_te_s = scaled["train"], scaled["val"], scaled["test"]

#     # 8) Reattach label and concatenate final splits
#     for part in (df_tr_s, df_val_s, df_te_s):
#         part[params.label_col] = df.loc[part.index, params.label_col]

#     df_final = pd.concat([df_tr_s, df_val_s, df_te_s]).sort_index()
#     final_cols = (
#         price_feats + ratio_feats + binary_feats +
#         cyclic_feats + ["bid","ask"] + [params.label_col]
#     )
#     return df_final[final_cols]


def scale_with_splits(
    df: pd.DataFrame,
    train_prop: float = params.train_prop,
    val_prop:   float = params.val_prop
) -> pd.DataFrame:
    """
    Chronologically split, encode cyclic date parts, and robustly scale all numeric features.

    Steps:
      1) Deep-copy and split df into train / val / test by row index.
      2) Encode hour, day_of_week, month as sine & cosine pairs.
      3) Define three feature groups:
         • price_feats     : raw price & volume indicators
         • ratio_feats     : returns, volume‐spikes, relative ratios, RSI, stochastics, eng_*
         • indicator_feats : EMA, SMA, MACD line/signal/diff
      4) Build ColumnTransformer with three pipelines:
         - RobustScaler()                 on price_feats
         - Pipeline(Yeo‐Johnson → StandardScaler) on ratio_feats
         - StandardScaler()               on indicator_feats
         remainder='passthrough' (cyclic, bid, ask, label)
      5) Fit the transformers on TRAIN only; transform train/val/test in a tqdm loop.
      6) Reassemble DataFrames preserving original index & columns.
      7) Compress each cyclic sin/cos pair back to a single value via PCA in a tqdm loop.
      8) Reattach label + raw bid/ask, concatenate splits, and return.

    Returns:
        A single DataFrame with all numeric features zero‐mean/unit‐var
        (or IQR‐scaled for price), cyclic dates collapsed, plus label, bid, ask.
    """
    # 1) Deep-copy + chronological split
    df = df.copy()
    N = len(df)
    n_tr = int(N * train_prop)
    n_val = int(N * val_prop)
    if n_tr + n_val >= N:
        raise ValueError("train_prop + val_prop must sum to < 1.0")

    df_tr = df.iloc[:n_tr].copy()
    df_v  = df.iloc[n_tr:n_tr + n_val].copy()
    df_te = df.iloc[n_tr + n_val:].copy()

    # 2) Encode cyclic date parts
    for part in (df_tr, df_v, df_te):
        h = part["hour"]
        part.loc[:, "hour_sin"] = np.sin(2 * np.pi * h / 24)
        part.loc[:, "hour_cos"] = np.cos(2 * np.pi * h / 24)

        dow = part["day_of_week"]
        part.loc[:, "dow_sin"]  = np.sin(2 * np.pi * dow / 7)
        part.loc[:, "dow_cos"]  = np.cos(2 * np.pi * dow / 7)

        m = part["month"]
        part.loc[:, "mo_sin"]   = np.sin(2 * np.pi * m / 12)
        part.loc[:, "mo_cos"]   = np.cos(2 * np.pi * m / 12)

    # 3) Define feature groups based on original df columns
    cols = df.columns
    price_feats = [c for c in cols if c in (
        "open", "high", "low", "close", "volume",
        "atr_14", "atr_sma", "obv", "obv_sma",
        "bb_lband", "bb_hband"
    )]

    ratio_feats = (
        [c for c in cols if c.startswith(("r_", "vol_"))] +
        [c for c in cols if c in (
            "volume_spike", "vwap_dev", "rsi", "bb_width_20",
            "stoch_k_14", "stoch_d_3", "atr_ratio", "atr_ratio_sma"
        )] +
        [c for c in cols if c.startswith("eng_")]
    )

    indicator_feats = [
        c for c in cols
        if c in ("ema", "sma", "macd_line", "macd_signal", "macd_diff")
    ]

    # 4) Build ColumnTransformer pipelines
    price_pipe = RobustScaler()
    ratio_pipe = Pipeline([
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("std",   StandardScaler())
    ])
    indicator_pipe = StandardScaler()

    ct = ColumnTransformer(
        [
            ("price",     price_pipe,     price_feats),
            ("ratio",     ratio_pipe,     ratio_feats),
            ("indicator", indicator_pipe, indicator_feats),
        ],
        remainder="passthrough"  # keep cyclic, bid, ask, label intact
    )

    # 5) Fit on train, transform all splits with a progress bar
    split_map = {"train": df_tr, "val": df_v, "test": df_te}
    transformed = {}
    for name, subset in tqdm(split_map.items(), desc="Scaling splits", unit="split"):
        if name == "train":
            arr = ct.fit_transform(subset)
        else:
            arr = ct.transform(subset)

        # build DataFrame: scaled groups first, then the rest
        rest = [c for c in subset.columns
                if c not in price_feats + ratio_feats + indicator_feats]
        new_cols = price_feats + ratio_feats + indicator_feats + rest
        transformed[name] = pd.DataFrame(arr, columns=new_cols, index=subset.index)

    df_tr_s, df_v_s, df_te_s = (
        transformed["train"], transformed["val"], transformed["test"]
    )

    # 6–7) Compress cyclic sin/cos pairs back to one dimension via PCA
    cyclic_pairs = [
        ("hour",       ["hour_sin", "hour_cos"]),
        ("day_of_week",["dow_sin",  "dow_cos"]),
        ("month",      ["mo_sin",   "mo_cos"]),
    ]
    for feat_name, cols_pair in tqdm(cyclic_pairs, desc="PCA compress", unit="feat"):
        pca = PCA(n_components=1)
        # fit on train
        train_vals = df_tr_s[cols_pair].values
        df_tr_s[feat_name] = pca.fit_transform(train_vals).ravel().round(3)
        # apply to val/test
        df_v_s[feat_name]  = pca.transform(df_v_s[cols_pair].values).ravel().round(3)
        df_te_s[feat_name] = pca.transform(df_te_s[cols_pair].values).ravel().round(3)
        # drop sin/cos
        df_tr_s.drop(cols_pair, axis=1, inplace=True)
        df_v_s .drop(cols_pair, axis=1, inplace=True)
        df_te_s.drop(cols_pair, axis=1, inplace=True)

    # 8) Reattach label + raw bid/ask, concat, and return
    for scaled_df, orig_df in zip((df_tr_s, df_v_s, df_te_s), (df_tr, df_v, df_te)):
        scaled_df[params.label_col] = orig_df[params.label_col].values
        if "bid" in orig_df.columns:
            scaled_df["bid"] = orig_df["bid"].values
        if "ask" in orig_df.columns:
            scaled_df["ask"] = orig_df["ask"].values

    df_final = pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()
    return df_final


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


