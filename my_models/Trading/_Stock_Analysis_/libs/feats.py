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
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA

from scipy.stats import spearmanr


##########################################################################################################


def create_features(
    df: pd.DataFrame,
    window_multiplier: float = 1.0,
    sma_short:   int         = 14,
    sma_long:    int         = 28,
    rsi_window:  int         = 14,
    macd_fast:   int         = 12,
    macd_slow:   int         = 26,
    macd_sig:    int         = 9,
    atr_window:  int         = 14,
    bb_window:   int         = 20,
    obv_sma:     int         = 14,
    vwap_window: int         = 14
) -> pd.DataFrame:
    """
    Vectorized generation of OHLCV‐derived features, including returns:

    1) Scale all window sizes by window_multiplier.
    2) Compute price‐change channels:
         • ret     = simple return
         • log_ret = log‐return
    3) Candlestick geometry: body, body_pct, upper_shad, lower_shad, range_pct.
    4) Popular indicators:
         • RSI(rsi_window)
         • MACD(line, signal, diff)
         • SMA(short/long) + pct deviations
         • ATR(atr_window) + atr_pct
         • Bollinger Bands(bb_window) + bb_width
         • +DI, –DI, ADX(atr_window)
         • OBV + obv_sma(obv_sma) + obv_pct
         • VWAP(vwap_window) + vwap_dev
         • vol_spike ratio
    5) Calendar: hour, day_of_week, month.
    6) Drop NaNs and return full DataFrame.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    def WM(x):
        return max(1, int(round(x * window_multiplier)))

    # derive window lengths
    w_sma_s = WM(sma_short)
    w_sma_l = WM(sma_long)
    w_rsi   = WM(rsi_window)
    w_atr   = WM(atr_window)
    w_bb    = WM(bb_window)
    w_obv   = WM(obv_sma)
    w_vwap  = WM(vwap_window)

    # pick base columns
    cols_in = ["open","high","low","close","volume","bid","ask", params.label_col]
    out     = df[cols_in].copy()
    c       = out["close"]

    # 2) price‐change channels
    out["ret"]     = c.pct_change()
    out["log_ret"] = np.log(c).diff()

    # 3) candlestick geometry
    o, h, l = out.open, out.high, out.low
    out["body"]       = c - o
    out["body_pct"]   = (c - o) / (o + 1e-8)
    out["upper_shad"] = h - out[["open","close"]].max(axis=1)
    out["lower_shad"] = out[["open","close"]].min(axis=1) - l
    out["range_pct"]  = (h - l) / (c + 1e-8)

    # 4) RSI
    out[f"rsi_{w_rsi}"] = ta.momentum.RSIIndicator(close=c, window=w_rsi).rsi()

    # 5) MACD
    macd = ta.trend.MACD(
        close=c,
        window_fast=macd_fast,
        window_slow=macd_slow,
        window_sign=macd_sig
    )
    out[f"macd_line_{macd_fast}_{macd_slow}_{macd_sig}"]   = macd.macd()
    out[f"macd_signal_{macd_fast}_{macd_slow}_{macd_sig}"] = macd.macd_signal()
    out[f"macd_diff_{macd_fast}_{macd_slow}_{macd_sig}"]   = macd.macd_diff()

    # 6) SMA + pct deviation
    sma_s = c.rolling(w_sma_s, min_periods=1).mean()
    sma_l = c.rolling(w_sma_l, min_periods=1).mean()
    out[f"sma_{w_sma_s}"]     = sma_s
    out[f"sma_{w_sma_l}"]     = sma_l
    out[f"sma_pct_{w_sma_s}"] = (c - sma_s) / (sma_s + 1e-8)
    out[f"sma_pct_{w_sma_l}"] = (c - sma_l) / (sma_l + 1e-8)

    # 7) ATR + pct
    atr = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=w_atr).average_true_range()
    out[f"atr_{w_atr}"]     = atr
    out[f"atr_pct_{w_atr}"] = atr / (c + 1e-8)

    # 8) Bollinger Bands + width
    bb    = ta.volatility.BollingerBands(close=c, window=w_bb, window_dev=2)
    lband = bb.bollinger_lband()
    hband = bb.bollinger_hband()
    mavg  = bb.bollinger_mavg()
    out[f"bb_lband_{w_bb}"] = lband
    out[f"bb_hband_{w_bb}"] = hband
    out[f"bb_w_{w_bb}"]     = (hband - lband) / (mavg + 1e-8)

    # 9) +DI, –DI, ADX
    adx = ta.trend.ADXIndicator(high=h, low=l, close=c, window=w_atr)
    out[f"plus_di_{w_atr}"]  = adx.adx_pos()
    out[f"minus_di_{w_atr}"] = adx.adx_neg()
    out[f"adx_{w_atr}"]      = adx.adx()

    # 10) OBV + SMA + pct
    obv = ta.volume.OnBalanceVolumeIndicator(close=c, volume=out.volume).on_balance_volume()
    out["obv"]             = obv
    out[f"obv_sma_{w_obv}"] = obv.rolling(w_obv, min_periods=1).mean()
    out[f"obv_pct_{w_obv}"] = obv / (out.volume + 1e-8)

    # 11) VWAP + deviation
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=h, low=l, close=c, volume=out.volume, window=w_vwap
    ).volume_weighted_average_price()
    out[f"vwap_{w_vwap}"]     = vwap
    out[f"vwap_dev_{w_vwap}"] = (c - vwap) / (vwap + 1e-8)

    # 12) vol_spike ratio
    vol_roll = out.volume.rolling(w_obv, min_periods=1).mean()
    out[f"vol_spike_{w_obv}"] = out.volume / (vol_roll + 1e-8)

    # 13) calendar columns
    out["hour"]        = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["month"]       = out.index.month

    # drop any NaNs from initial windows
    return out.dropna()


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
#     Build seven continuous “eng_” signals by comparing core indicators to thresholds
#     or to their own moving baselines.

#     1) eng_ma      = (SMA_short – SMA_long)    / (SMA_long + eps)
#     2) eng_macd    = MACD_diff                  / (SMA_long + eps)
#     3) eng_bb      = distance outside BollBands / (BB_width + eps)
#     4) eng_rsi     = distance beyond [rsi_low, rsi_high] / 100
#     5) eng_adx     = sign(+DI––DI–)×(|+DI––DI–|/100)×max(ADX–adx_thr,0)/100
#     6) eng_obv     = (OBV – OBV_SMA)           / (OBV_SMA + eps)
#     7) eng_atr_div = 10000×[(ATR/Close) – rolling_mean(ATR/Close, mult_w)]

#     All denominators add eps to avoid division‐by‐zero.  
#     Returns a DataFrame of these seven engineered columns, indexed as df.
#     """
#     out = pd.DataFrame(index=df.index)

#     # 1) detect the two SMA columns (exclude sma_pct_)
#     sma_cols = [c for c in df.columns 
#                 if c.startswith("sma_") and not c.startswith("sma_pct")]
#     sma_cols = sorted(sma_cols, key=lambda c: int(c.split("_")[1]))
#     sma_short_col, sma_long_col = sma_cols[:2]

#     # 2) detect MACD diff
#     macd_diff_col = next(c for c in df.columns if c.startswith("macd_diff_"))

#     # 3) detect Bollinger bands & width
#     bb_l_col   = next(c for c in df.columns if c.startswith("bb_lband_"))
#     bb_h_col   = next(c for c in df.columns if c.startswith("bb_hband_"))
#     bb_w_col   = next(c for c in df.columns if c.startswith("bb_w_"))

#     # 4) detect RSI column
#     rsi_col    = next(c for c in df.columns if c.startswith("rsi_"))

#     # 5) detect DI & ADX
#     plus_di_col  = next(c for c in df.columns if c.startswith("plus_di_"))
#     minus_di_col = next(c for c in df.columns if c.startswith("minus_di_"))
#     adx_col      = next(c for c in df.columns if c.startswith("adx_"))

#     # 6) detect OBV & its SMA
#     obv_col      = "obv"
#     obv_sma_col  = next(c for c in df.columns if c.startswith("obv_sma_"))

#     # 7) detect ATR/Close pct
#     atr_pct_col  = next(c for c in df.columns if c.startswith("atr_pct_"))

#     # 1) MA spread ratio
#     out["eng_ma"] = (
#         (df[sma_short_col] - df[sma_long_col])
#         / (df[sma_long_col] + eps)
#     ).round(3)

#     # 2) MACD diff ratio
#     out["eng_macd"] = (
#         df[macd_diff_col] / (df[sma_long_col] + eps)
#     ).round(3)

#     # 3) Bollinger deviation ratio
#     close = df["close"]
#     lo, hi, bw = df[bb_l_col], df[bb_h_col], df[bb_w_col]
#     dev = np.where(close < lo, lo - close,
#            np.where(close > hi, close - hi, 0.0))
#     out["eng_bb"] = (dev / (bw + eps)).round(3)

#     # 4) RSI threshold ratio
#     rsi = df[rsi_col]
#     low_dev  = np.clip((rsi_low  - rsi), 0, None) / 100.0
#     high_dev = np.clip((rsi       - rsi_high), 0, None) / 100.0
#     out["eng_rsi"] = np.where(
#         rsi < rsi_low, low_dev,
#         np.where(rsi > rsi_high, high_dev, 0.0)
#     ).round(3)

#     # 5) ADX‐weighted DI spread
#     plus, minus, adx = df[plus_di_col], df[minus_di_col], df[adx_col]
#     di_diff = (plus - minus)
#     diff_abs = di_diff.abs() / 100.0
#     ex = np.clip((adx - adx_thr) / 100.0, 0, None)
#     out["eng_adx"] = (np.sign(di_diff) * diff_abs * ex).round(3)

#     # 6) OBV divergence ratio
#     out["eng_obv"] = (
#         (df[obv_col] - df[obv_sma_col])
#         / (df[obv_sma_col] + eps)
#     ).round(3)

#     # 7) ATR/Close stationary deviation
#     ratio = df[atr_pct_col]  # ATR/Close
#     rm = ratio.rolling(mult_w, min_periods=1).mean()
#     out["eng_atr_div"] = ((ratio - rm) * 10_000).round(1)

#     return out


def features_engineering(
    df: pd.DataFrame,
    rsi_low:   float = 30.0,
    rsi_high:  float = 70.0,
    adx_thr:   float = 20.0,
    mult_w:    int   = 14,
    eps:       float = 1e-8
) -> pd.DataFrame:
    """
    Build continuous “eng_” signals from raw indicators + relative‐price bands.

    1) eng_ma        = (SMA_short – SMA_long) / SMA_long
    2) eng_macd      = MACD_diff / SMA_long
    3) eng_bb        = distance outside BBands / BB_width
    4) eng_bb_mid    = (BB_mid – close) / close
    5) eng_rsi       = distance beyond [rsi_low, rsi_high] / 100
    6) eng_adx       = sign(DI+–DI–) × (|DI+–DI–|/100) × max(ADX–adx_thr, 0)/100
    7) eng_obv       = (OBV – OBV_SMA) / OBV_SMA
    8) eng_atr_div   = 10 000 × [(ATR/close) – rolling_mean(ATR/close, mult_w)]
    9) eng_sma_short = (SMA_short – close) / close
   10) eng_sma_long  = (SMA_long  – close) / close
   11) eng_vwap      = (VWAP – close) / close

    Returns a DataFrame of these engineered features, indexed same as `df`.
    """
    out   = pd.DataFrame(index=df.index)
    close = df["close"]

    # 1) Find true SMA columns (exclude sma_pct_*)
    sma_cols = [
        c for c in df.columns
        if c.startswith("sma_") and c.split("_")[1].isdigit()
    ]
    sma_cols = sorted(sma_cols, key=lambda c: int(c.split("_")[1]))
    sma_s, sma_l = sma_cols[:2]

    # 2) Locate MACD diff
    macd_diff_col = next(c for c in df.columns if c.startswith("macd_diff_"))

    # 3) Locate Bollinger lband/hband/width
    bb_l_col = next(c for c in df.columns if c.startswith("bb_lband_"))
    bb_h_col = next(c for c in df.columns if c.startswith("bb_hband_"))
    bb_w_col = next(c for c in df.columns if c.startswith("bb_w_"))

    # 4) Locate RSI
    rsi_col = next(c for c in df.columns if c.startswith("rsi_"))

    # 5) Locate +DI, –DI, ADX
    plus_di_col  = next(c for c in df.columns if c.startswith("plus_di_"))
    minus_di_col = next(c for c in df.columns if c.startswith("minus_di_"))
    adx_col      = next(c for c in df.columns if c.startswith("adx_"))

    # 6) Locate OBV & its SMA
    obv_col     = "obv"
    obv_sma_col = next(c for c in df.columns if c.startswith("obv_sma_"))

    # 7) Locate ATR/close pct
    atr_pct_col = next(c for c in df.columns if c.startswith("atr_pct_"))

    # 1) MA spread ratio
    out["eng_ma"] = ((df[sma_s] - df[sma_l]) / (df[sma_l] + eps)).round(3)

    # 2) MACD diff ratio
    out["eng_macd"] = (df[macd_diff_col] / (df[sma_l] + eps)).round(3)

    # 3) Bollinger deviation ratio
    lo, hi, bw = df[bb_l_col], df[bb_h_col], df[bb_w_col]
    dev = np.where(close < lo, lo - close,
          np.where(close > hi, close - hi, 0.0))
    out["eng_bb"] = (dev / (bw + eps)).round(3)

    # 4) Bollinger mid‐band relative
    bb_mid = (lo + hi) * 0.5
    out["eng_bb_mid"] = ((bb_mid - close) / (close + eps)).round(4)

    # 5) RSI threshold ratio
    rsi_vals = df[rsi_col]
    low_dev  = np.clip((rsi_low  - rsi_vals), 0, None) / 100.0
    high_dev = np.clip((rsi_vals - rsi_high), 0, None) / 100.0
    out["eng_rsi"] = np.where(
        rsi_vals < rsi_low, low_dev,
        np.where(rsi_vals > rsi_high, high_dev, 0.0)
    ).round(3)

    # 6) ADX‐weighted DI spread
    di_diff    = df[plus_di_col] - df[minus_di_col]
    diff_abs   = di_diff.abs() / 100.0
    ex         = np.clip((df[adx_col] - adx_thr) / 100.0, 0, None)
    out["eng_adx"] = (np.sign(di_diff) * diff_abs * ex).round(3)

    # 7) OBV divergence ratio
    out["eng_obv"] = (
        (df[obv_col] - df[obv_sma_col]) / (df[obv_sma_col] + eps)
    ).round(3)

    # 8) ATR/close stationary deviation
    ratio = df[atr_pct_col]
    rm    = ratio.rolling(mult_w, min_periods=1).mean()
    out["eng_atr_div"] = ((ratio - rm) * 10_000).round(1)

    # 9) SMA short/long relative to price
    out["eng_sma_short"] = ((df[sma_s] - close) / (close + eps)).round(4)
    out["eng_sma_long"]  = ((df[sma_l] - close) / (close + eps)).round(4)

    # 10) VWAP relative to price
    vwap_col = next(c for c in df.columns if c.startswith("vwap_") and not c.endswith("_dev"))
    out["eng_vwap"] = ((df[vwap_col] - close) / (close + eps)).round(4)

    return out


##########################################################################################################


# def scale_with_splits(
#     df: pd.DataFrame,
#     train_prop: float = params.train_prop,
#     val_prop:   float = params.val_prop
# ) -> pd.DataFrame:
#     """
#     1) Build cyclical calendar features (hour, day_of_week, month) as sin/cos on the full df.
#     2) Split contiguously into train/val/test by proportions.
#     3) Fit PCA(1) on train’s sin/cos pairs → compress back to hour/day_of_week/month.
#     4) Identify indicator cols vs reserved cols *after* PCA (so sin/cos are gone).
#     5) Fit a ColumnTransformer on train indicators:
#          - bounded    (/100)
#          - ratio      (MinMax[0,1])
#          - unbounded  (Robust → Standard → MinMax[0,1])
#     6) Transform each split *per calendar day* with tqdm bars, reassemble features + reserved.
#     7) Concat train/val/test, drop raw OHLCV, return full scaled DataFrame.

#     This guarantees:
#       - No future leakage (PCA & scalers trained on train only).
#       - All indicator features reside in [0,1].
#       - Inter-day rolling continuity is preserved.
#     """
#     df = df.copy()
#     df.index = pd.to_datetime(df.index)

#     # 1) Add calendar sin/cos to full df
#     df["hour"]        = df.index.hour
#     df["day_of_week"] = df.index.dayofweek
#     df["month"]       = df.index.month

#     for name, period in [("hour",24), ("day_of_week",7), ("month",12)]:
#         vals = df[name]
#         df[f"{name}_sin"] = np.sin(2*np.pi * vals / period)
#         df[f"{name}_cos"] = np.cos(2*np.pi * vals / period)

#     # 2) Split contiguously
#     N     = len(df)
#     n_tr  = int(N * train_prop)
#     n_val = int(N * val_prop)
#     if n_tr + n_val >= N:
#         raise ValueError("train_prop + val_prop must sum to <1.0")

#     df_tr = df.iloc[:n_tr].copy()
#     df_v  = df.iloc[n_tr : n_tr+n_val].copy()
#     df_te = df.iloc[n_tr+n_val :].copy()

#     # 3) PCA compress sin/cos → keep only the scalar cal features
#     for cal in ("hour","day_of_week","month"):
#         cols = [f"{cal}_sin", f"{cal}_cos"]
#         pca  = PCA(n_components=1)
#         pca.fit(df_tr[cols])
#         for split in (df_tr, df_v, df_te):
#             split[cal] = pca.transform(split[cols])
#             split.drop(cols, axis=1, inplace=True)

#     # 4) Now identify reserved vs indicator after PCA drop
#     reserved = {
#         "open","high","low","close","volume",
#         "bid","ask", params.label_col,
#         "hour","day_of_week","month"
#     }
#     feat_cols = [c for c in df_tr.columns if c not in reserved]

#     # group features by type
#     bounded   = [c for c in feat_cols if c.startswith(("rsi_","adx_","plus_di_","minus_di_","stoch_"))]
#     ratio     = [c for c in feat_cols if c.startswith("r_")
#                                or "vol_spike" in c
#                                or "vwap_dev" in c
#                                or c.endswith("_w")]
#     unbounded = [c for c in feat_cols if c not in bounded + ratio]

#     # 5) Build a ColumnTransformer ending each branch in MinMax([0,1])
#     ct = ColumnTransformer([
#         ("bnd", FunctionTransformer(lambda X: X / 100.0),        bounded),
#         ("rat", MinMaxScaler(feature_range=(0,1)),              ratio),
#         ("unb", Pipeline([
#             ("robust", RobustScaler(quantile_range=(5,95))),
#             ("std",    StandardScaler()),
#             ("mm",     MinMaxScaler(feature_range=(0,1))),
#         ]),                                                    unbounded),
#     ], remainder="drop")

#     # fit only on train indicators
#     ct.fit(df_tr[feat_cols])

#     # 6) Transform each split per-day with tqdm
#     def transform_split(split_df, label):
#         arr = np.empty((len(split_df), len(feat_cols)), dtype=float)
#         for day, block in tqdm(
#             split_df.groupby(split_df.index.normalize()),
#             desc=f"{label} days", unit="day"
#         ):
#             mask        = split_df.index.normalize() == day
#             arr[mask,:] = ct.transform(block[feat_cols])

#         # rebuild DataFrame: features + reserved
#         df_feats    = pd.DataFrame(arr, index=split_df.index, columns=feat_cols)
#         df_reserved = split_df[list(reserved)]
#         df_scaled   = pd.concat([df_feats, df_reserved], axis=1)
#         return df_scaled[split_df.columns]

#     df_tr_s = transform_split(df_tr, "train")
#     df_v_s  = transform_split(df_v,  "val")
#     df_te_s = transform_split(df_te, "test")

#     # 7) Concat & drop raw OHLCV if desired
#     df_all = pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()
#     return df_all.drop(columns=["open","high","low","close","volume"], errors="ignore")



def assign_feature_groups(
    df: pd.DataFrame,
    cols: List[str],
    *,
    ratio_range:   float = 0.15,
    heavy_thresh:  float = 1e7,
    overrides:     Dict[str,str] = None
) -> pd.DataFrame:
    """
    Analyze each feature’s value‐distribution and assign it to one of four
    shape‐preserving scaling groups, then apply any manual overrides.

    Groups:
      • bounded   : RSI/ADX/DI/Stoch (0–100 scales)
      • heavy     : vol_spike_* or any max|min| ≥ heavy_thresh
      • ratio     : central bulk ∈ ±ratio_range
      • unbounded : everything else

    Returns a DataFrame indexed by feature with columns
    ['min','1%','5%','50%','95%','99%','max','group_raw','group_final'].
    """
    # 1) summary stats
    stats = (
        df[cols]
          .describe(percentiles=[0.01,0.05,0.5,0.95,0.99])
          .T.rename(columns={
            'min':'min','1%':'1%','5%':'5%','50%':'50%',
            '95%':'95%','99%':'99%','max':'max'
          })
    )

    # 2) name‐based sets
    bounded_names = {
        c for c in cols
        if c.startswith(("rsi_","adx_","plus_di_","minus_di_","stoch_"))
    }
    heavy_names = {c for c in cols if c.startswith("vol_spike_")}

    # 3) raw grouping by name + value
    raw_group: Dict[str,str] = {}
    for feat in cols:
        mn, mx = stats.loc[feat, ['min','max']]
        p5, p95 = stats.loc[feat, ['5%','95%']]

        if feat in bounded_names:
            raw_group[feat] = 'bounded'
        elif feat in heavy_names:
            raw_group[feat] = 'heavy'
        elif abs(mn) >= heavy_thresh or abs(mx) >= heavy_thresh:
            raw_group[feat] = 'heavy'
        elif p5 >= -ratio_range and p95 <= ratio_range:
            raw_group[feat] = 'ratio'
        else:
            raw_group[feat] = 'unbounded'

    # 4) final grouping with overrides
    overrides = overrides or {}
    final_group = {
        feat: overrides.get(feat, raw_group[feat])
        for feat in cols
    }

    # 5) assemble assignment table
    df_assign = stats.copy()
    df_assign['group_raw']   = df_assign.index.map(raw_group)
    df_assign['group_final'] = df_assign.index.map(final_group)
    return df_assign




##########################################################################################################


# def scale_with_splits(
#     df: pd.DataFrame,
#     assignment: pd.DataFrame,
#     train_prop: float = params.train_prop,
#     val_prop:   float = params.val_prop
# ) -> pd.DataFrame:
#     """
#     Split, scale and reassemble a time series DataFrame into [0,1] features
#     while preserving each indicator’s shape.  

#     1) Build cyclical calendar features (sin/cos) on the full df  
#     2) Split contiguously into train/val/test by train_prop/val_prop  
#     3) PCA‐compress each sin/cos pair → hour/day/month  
#     4) Identify feat_cols vs reserved columns  
#     5) Read final group assignment from assignment["group_final"] and
#        build lists: bounded, ratio, heavy, unbounded  
#     6) Construct ColumnTransformer:
#          • bounded   → divide by 100  
#          • ratio     → MinMaxScaler  
#          • heavy     → sym_log → MinMaxScaler  
#          • unbounded → StandardScaler → clip ±3σ → MinMaxScaler  
#     7) Fit transformer on train only  
#     8) Transform each split _per day_ to keep temporal blocks intact  
#     9) Concatenate splits, drop raw OHLCV, return scaled df  

#     Inputs:
#       df          : original OHLCV + feature DataFrame  
#       assignment  : DataFrame indexed by feature name with column
#                     "group_final" ∈ {"bounded","ratio","heavy","unbounded"}  
#       train_prop, val_prop : fractions for contiguous splits  

#     Returns:
#       full scaled DataFrame, columns = feat_scaled + reserved_cols
#     """
#     df = df.copy()
#     df.index = pd.to_datetime(df.index)

#     # 1) calendar sin/cos
#     df["hour"], df["day_of_week"], df["month"] = (
#         df.index.hour, df.index.dayofweek, df.index.month
#     )
#     for name, period in [("hour",24), ("day_of_week",7), ("month",12)]:
#         vals = df[name]
#         df[f"{name}_sin"] = np.sin(2*np.pi * vals / period)
#         df[f"{name}_cos"] = np.cos(2*np.pi * vals / period)

#     # 2) contiguous train/val/test split
#     N, n_tr = len(df), int(len(df)*train_prop)
#     n_val   = int(len(df)*val_prop)
#     if n_tr + n_val >= N:
#         raise ValueError("train_prop + val_prop must sum < 1.0")
#     df_tr = df.iloc[:n_tr].copy()
#     df_v  = df.iloc[n_tr:n_tr+n_val].copy()
#     df_te = df.iloc[n_tr+n_tr+n_val - n_tr:].copy()

#     # 3) PCA compress calendar sin/cos → hour/day/month
#     for cal in ("hour","day_of_week","month"):
#         sincos = [f"{cal}_sin", f"{cal}_cos"]
#         pca    = PCA(n_components=1)
#         pca.fit(df_tr[sincos])
#         for split in (df_tr, df_v, df_te):
#             split[cal] = pca.transform(split[sincos])
#             split.drop(sincos, axis=1, inplace=True)

#     # 4) carve features vs reserved
#     reserved = {
#         "open","high","low","close","volume","bid","ask",
#         params.label_col,"hour","day_of_week","month"
#     }
#     feat_cols = [c for c in df_tr.columns if c not in reserved]

#     # 5) build group‐to‐list mapping from assignment["group_final"]
#     mapping = assignment["group_final"].to_dict()
#     bounded   = [f for f in feat_cols if mapping.get(f) == "bounded"]
#     ratio     = [f for f in feat_cols if mapping.get(f) == "ratio"]
#     heavy     = [f for f in feat_cols if mapping.get(f) == "heavy"]
#     unbounded = [f for f in feat_cols if mapping.get(f) == "unbounded"]

#     # 6) shape-preserving pipelines
#     def sym_log(X):
#         return np.sign(X) * np.log1p(np.abs(X))

#     ct = ColumnTransformer([
#         ("bnd", FunctionTransformer(lambda X: X/100.0, validate=False), bounded),
#         ("rat", MinMaxScaler(feature_range=(0,1)),                 ratio),
#         ("hvy", Pipeline([
#             ("slog", FunctionTransformer(sym_log, validate=False)),
#             ("mm",   MinMaxScaler(feature_range=(0,1)))
#         ]),                                                       heavy),
#         ("unb", Pipeline([
#             ("std",   StandardScaler()),
#             ("clip",  FunctionTransformer(lambda X: np.clip(X, -3, 3),
#                                           validate=False)),
#             ("mm",    MinMaxScaler(feature_range=(0,1)))
#         ]),                                                     unbounded),
#     ], remainder="drop")

#     # 7) fit on train
#     ct.fit(df_tr[feat_cols])

#     # 8) per‐day transform helper
#     def transform_split(split_df, label):
#         arr = np.empty((len(split_df), len(feat_cols)), dtype=float)
#         for day, block in tqdm(
#             split_df.groupby(split_df.index.normalize()),
#             desc=f"{label} days", unit="day"
#         ):
#             mask        = split_df.index.normalize() == day
#             arr[mask,:] = ct.transform(block[feat_cols])
#         df_feats    = pd.DataFrame(arr,
#                                    index=split_df.index,
#                                    columns=feat_cols)
#         df_reserved = split_df[list(reserved)]
#         return pd.concat([df_feats, df_reserved], axis=1)[split_df.columns]

#     df_tr_s = transform_split(df_tr, "train")
#     df_v_s  = transform_split(df_v,  "val")
#     df_te_s = transform_split(df_te, "test")

#     # 9) assemble & drop raw OHLCV
#     df_all = pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()
#     return df_all.drop(
#         columns=["open","high","low","close","volume"],
#         errors="ignore"
#     )



def scale_with_splits(
    df: pd.DataFrame,
    assignment: pd.DataFrame,
    train_prop: float = params.train_prop,
    val_prop:   float = params.val_prop
) -> pd.DataFrame:
    """
    Split, scale and reassemble a time-series DataFrame into [0,1] features.

    Steps:
      1) Build cyclical calendar sin/cos.
      2) Split contiguously into train/val/test.
      3) PCA compress calendar pairs → hour/day/month.
      4) Identify feature vs. reserved columns.
      5) Fetch group_final → bounded, ratio, heavy, unbounded.
      6) Build monotonic pipelines:
           bounded   → X / 100
           ratio     → Yeo–Johnson power transform → MinMax
           heavy     → sym_log → MinMax
           unbounded → StandardScaler → clip ±3σ → MinMax
      7) Fit on train set.
      8) Per-day transform with tqdm.
      9) Concatenate splits, drop raw OHLCV.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # 1) calendar sin/cos
    df["hour"], df["day_of_week"], df["month"] = (
        df.index.hour, df.index.dayofweek, df.index.month
    )
    for name, period in [("hour",24), ("day_of_week",7), ("month",12)]:
        vals = df[name]
        df[f"{name}_sin"] = np.sin(2*np.pi * vals / period)
        df[f"{name}_cos"] = np.cos(2*np.pi * vals / period)

    # 2) train/val/test split
    N, n_tr = len(df), int(len(df)*train_prop)
    n_val   = int(len(df)*val_prop)
    if n_tr + n_val >= N:
        raise ValueError("train_prop + val_prop must sum < 1.0")
    df_tr = df.iloc[:n_tr].copy()
    df_v  = df.iloc[n_tr:n_tr+n_val].copy()
    df_te = df.iloc[n_tr+n_tr+n_val - n_tr:].copy()

    # 3) PCA compress calendar sin/cos → single column
    for cal in ("hour","day_of_week","month"):
        sincos = [f"{cal}_sin", f"{cal}_cos"]
        pca    = PCA(n_components=1)
        pca.fit(df_tr[sincos])
        for split in (df_tr, df_v, df_te):
            split[cal] = pca.transform(split[sincos])
            split.drop(sincos, axis=1, inplace=True)

    # 4) carve out features vs. reserved
    reserved = {
        "open","high","low","close","volume","bid","ask",
        params.label_col,"hour","day_of_week","month"
    }
    feat_cols = [c for c in df_tr.columns if c not in reserved]

    # 5) build lists by assignment
    mapping   = assignment["group_final"].to_dict()
    bounded   = [f for f in feat_cols if mapping.get(f) == "bounded"]
    ratio     = [f for f in feat_cols if mapping.get(f) == "ratio"]
    heavy     = [f for f in feat_cols if mapping.get(f) == "heavy"]
    unbounded = [f for f in feat_cols if mapping.get(f) == "unbounded"]

    # 6) monotonic, shape-preserving pipelines
    def sym_log(X):
        return np.sign(X) * np.log1p(np.abs(X))

    ct = ColumnTransformer([
        # 0–100 → 0–1
        ("bnd", FunctionTransformer(lambda X: X/100.0, validate=False), bounded),

        # uniformize narrow bulk while preserving order
        ("rat", Pipeline([
            ("pt", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("mm", MinMaxScaler(feature_range=(0,1)))
        ]), ratio),

        # compress heavy tails
        ("hvy", Pipeline([
            ("slog", FunctionTransformer(sym_log, validate=False)),
            ("mm",   MinMaxScaler(feature_range=(0,1)))
        ]), heavy),

        # z-score → clip → MinMax
        ("unb", Pipeline([
            ("std",   StandardScaler()),
            ("clip",  FunctionTransformer(lambda X: np.clip(X, -3, 3), validate=False)),
            ("mm",    MinMaxScaler(feature_range=(0,1)))
        ]), unbounded),
    ], remainder="drop")

    # 7) fit on train
    ct.fit(df_tr[feat_cols])

    # 8) per-day transform
    def transform_split(split_df, label):
        arr = np.empty((len(split_df), len(feat_cols)), dtype=float)
        for day, block in tqdm(
            split_df.groupby(split_df.index.normalize()),
            desc=f"{label} days", unit="day"
        ):
            mask        = split_df.index.normalize() == day
            arr[mask,:] = ct.transform(block[feat_cols])
        df_feats    = pd.DataFrame(arr, index=split_df.index, columns=feat_cols)
        df_resv     = split_df[list(reserved)]
        return pd.concat([df_feats, df_resv], axis=1)[split_df.columns]

    df_tr_s = transform_split(df_tr, "train")
    df_v_s  = transform_split(df_v,  "val")
    df_te_s = transform_split(df_te, "test")

    # 9) assemble & drop raw OHLCV
    df_all = pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()
    return df_all.drop(columns=["open","high","low","close","volume"], errors="ignore")


#########################################################################################################


def compare_raw_vs_scaled(
    df_raw:     pd.DataFrame,
    df_scaled:  pd.DataFrame,
    assignment: pd.DataFrame,
    feat_cols:  list[str] | None = None,
    tol:        float = 1e-6
) -> pd.DataFrame:
    """
    Compare raw vs scaled features, then verify:

      • min/max in [0,1] per group rules
      • Spearman ρ ≈ 1 (monotonic transform)
      • per-day monotonicity (scaled values preserve raw ordering)
    """
    if feat_cols is None:
        feat_cols = [c for c in df_raw.columns if c in df_scaled.columns]

    qs = [0.01, 0.05, 0.50, 0.95, 0.99]
    raw = (
        df_raw[feat_cols]
        .describe(percentiles=qs).T
        .loc[:, ['min','1%','5%','50%','95%','99%','max']]
    )
    raw.columns = [f"raw_{c}" for c in raw.columns]

    scaled = (
        df_scaled[feat_cols]
        .describe(percentiles=qs).T
        .loc[:, ['min','1%','5%','50%','95%','99%','max']]
    )
    scaled.columns = [f"scaled_{c}" for c in scaled.columns]

    cmp_df = pd.concat([raw, scaled], axis=1)
    cmp_df['group_final'] = assignment['group_final']

    # precompute stats for spearman & monotonicity
    spearman_rho = {}
    is_mono      = {}
    for feat in tqdm(feat_cols, desc="Checking shape"):
        x = df_raw[feat].dropna()
        y = df_scaled.loc[x.index, feat]

        rho = spearmanr(x, y).correlation
        order   = np.argsort(x.values)
        y_sorted= y.values[order]
        mono    = np.all(np.diff(y_sorted) >= -tol)

        spearman_rho[feat] = rho
        is_mono[feat]      = mono

    # final checks
    statuses, reasons = [], []
    for feat, row in cmp_df.iterrows():
        grp = row.group_final
        mn, mx = row.scaled_min, row.scaled_max
        rho    = spearman_rho[feat]
        mono   = is_mono[feat]

        errs = []
        # range check
        if grp == "bounded":
            if not (mn >= -tol and mx <= 1+tol):
                errs.append(f"range[{mn:.3f},{mx:.3f}]")
        else:
            if not (abs(mn) <= tol and abs(mx-1) <= tol):
                errs.append(f"range[{mn:.3f},{mx:.3f}]")

        # spearman check
        if abs(rho - 1) > 1e-3:
            errs.append(f"rho={rho:.3f}")

        # monotonic check
        if not mono:
            errs.append("non-monotonic")

        if errs:
            statuses.append("FAIL")
            reasons.append("; ".join(errs))
        else:
            statuses.append("OK")
            reasons.append("all checks passed")

    cmp_df['status']       = statuses
    cmp_df['reason']       = reasons
    cmp_df['spearman_rho'] = pd.Series(spearman_rho)
    cmp_df['is_monotonic'] = pd.Series(is_mono)

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


