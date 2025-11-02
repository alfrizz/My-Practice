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
import torch.nn as nn
import torch.backends.cudnn as cudnn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler, QuantileTransformer, OrdinalEncoder
from sklearn.decomposition import PCA

from scipy.stats import spearmanr, skew, kurtosis


##########################################################################################################


def create_features(
    df: pd.DataFrame,
    mult_feats_win:   float = 1.0,
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
    vol_spike_window: int = 14
) -> pd.DataFrame:
    """
    Compute raw OHLCV features and classic indicators on 1-min bars,
    scaling every lookback window by mult_feats_win.

    Steps:
      1) Scale all indicator windows via mult_feats_win (including MACD).
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
        return max(1, int(round(x * mult_feats_win)))

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
    cols_in = ["open","high","low","close","volume", params.label_col]
    out     = df[cols_in].copy()
    c, o, h, l = out["close"], out["open"], out["high"], out["low"]

    # 2) Returns
    out["ret"]     = c.pct_change()
    out["log_ret"] = np.log(c + eps).diff()
    
    # 2.1) Rate‐of‐Change over sma_short window
    roc_window = w_sma_s
    out[f"roc_{roc_window}"] = c.diff(roc_window) / (c.shift(roc_window) + eps)

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


##########################################################################################################


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


##########################################################################################################


def assign_feature_groups(
    df: pd.DataFrame,
    cols: List[str],
    *,
    ratio_range:     float = 0.15,
    heavy_thresh:    float = 1e7,
    skew_thresh:     float = 3.0,
    kurtosis_thresh: float = 5.0,
    discrete_thresh: int   = 10,
    overrides:       Dict[str, str] = None
) -> pd.DataFrame:
    """
    Inspect each feature’s raw distribution and bucket it into one of six
    scaling groups, excluding any pure price‐level columns:

      • EXCLUDE up front: calendar fields, OHLCV, raw SMA/VWAP/BBands.
      • Compute min, max, 1/5/95/99-percentiles, skew, kurtosis, unique_count, zero_ratio.
      • In order assign raw_group:
         a) discrete
         b) log_skewed
         c) ratio
         d) bounded
         e) robust_tails
         f) unbounded
      • Apply any user overrides → group_final.
      • Return DataFrame of stats + group_raw/group_final.
    """
    # 1) Build reserved set (drop from grouping)
    reserved = {
        "hour","day_of_week","month",
        "open","high","low","close"
    }
    # add raw‐level BBands/SMA/VWAP
    for c in cols:
        if (c.startswith("bb_lband_")
            or c.startswith("bb_hband_")
            or (c.startswith("sma_") and "_pct" not in c)
            or (c.startswith("vwap_") and not c.endswith("_dev"))):
            reserved.add(c)

    feats = [c for c in cols if c not in reserved]

    # 2) Replace infinities, compute descriptive stats
    data = df[feats].replace([np.inf, -np.inf], np.nan)
    descr = (
        data.describe(percentiles=[0.01,0.05,0.95,0.99])
            .T
            .rename(columns={"1%":"1%","5%":"5%","95%":"95%","99%":"99%"})
    )
    descr["skew"]         = data.skew().values
    descr["kurtosis"]     = data.kurtosis().values
    descr["unique_count"] = data.nunique().values
    descr["zero_ratio"]   = (data==0).mean().values

    # 3) Assign raw_group by priority rules
    raw_group: Dict[str,str] = {}
    for feat in feats:
        mn, mx   = descr.at[feat,"min"], descr.at[feat,"max"]
        p5, p95  = descr.at[feat,"5%"], descr.at[feat,"95%"]
        sk       = descr.at[feat,"skew"]
        kt       = descr.at[feat,"kurtosis"]
        uc       = descr.at[feat,"unique_count"]

        if uc <= discrete_thresh:
            grp = "discrete"
        elif mn >= 0 and sk > skew_thresh:
            grp = "log_skewed"
        elif p5 >= -ratio_range and p95 <= ratio_range:
            grp = "ratio"
        elif mn >= 0 and mx <= 100:
            grp = "bounded"
        elif abs(mn) >= heavy_thresh or abs(mx) >= heavy_thresh or kt >= kurtosis_thresh:
            grp = "robust_tails"
        else:
            grp = "unbounded"

        raw_group[feat] = grp

    # 4) Apply overrides
    overrides   = overrides or {}
    final_group = {f: overrides.get(f, raw_group[f]) for f in feats}

    # 5) Assemble assignment DataFrame
    df_assign = descr.copy()
    df_assign["group_raw"]   = df_assign.index.map(raw_group)
    df_assign["group_final"] = df_assign.index.map(final_group)
    return df_assign


  
##########################################################################################################


def scale_with_splits(
    df: pd.DataFrame,
    assignment: pd.DataFrame,
    train_prop: float = params.train_prop,
    val_prop:   float = params.val_prop
) -> pd.DataFrame:
    """
    1) Copy & datetime‐parse; mask ±∞ → NaN.
    2) Build cyclical calendar features, cast them to float.
    3) Split CONTIGUOUSLY into TRAIN/VAL/TEST.
    4) PCA‐compress each sin/cos pair → single calendar dimension.
    5) Define reserved = raw OHLCV, label, calendar, plus
       raw‐level BBands/SMA/VWAP.
    6) From assignment.group_final, build six pipelines: bounded, ratio,
       log_skewed, robust_tails, discrete, unbounded.
    7) Fit on TRAIN features.
    8) Transform each split day‐by‐day (tqdm) to preserve NaNs.
    9) Drop all raw OHLCV & raw‐level columns; return final scaled DataFrame.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2) calendar features + sin/cos
    df["hour"], df["day_of_week"], df["month"] = (
        df.index.hour, df.index.dayofweek, df.index.month
    )
    for name, period in [("hour", 24), ("day_of_week", 7), ("month", 12)]:
        vals = df[name]
        df[f"{name}_sin"] = np.sin(2 * np.pi * vals / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * vals / period)

    # 2.1) cast raw calendar cols to float to avoid dtype‐incompatible assignment
    for name in ("hour","day_of_week","month"):
        df[name] = df[name].astype(np.float64)

    # 3) CONTIGUOUS splits
    N    = len(df)
    n_tr = int(N * train_prop)
    n_val= int(N * val_prop)
    if n_tr + n_val >= N:
        raise ValueError("train_prop + val_prop must sum < 1.0")
    df_tr = df.iloc[:n_tr].copy()
    df_v  = df.iloc[n_tr:n_tr + n_val].copy()
    df_te = df.iloc[n_tr + n_val:].copy()

    # 4) PCA compress sin/cos
    for name in ("hour","day_of_week","month"):
        pair = [f"{name}_sin", f"{name}_cos"]
        pca  = PCA(n_components=1).fit(df_tr[pair])
        for split in (df_tr, df_v, df_te):
            # use .loc to avoid SettingWithCopyWarning
            split.loc[:, name] = pca.transform(split[pair]).ravel()
            split.drop(columns=pair, inplace=True)

    # 5) reserved vs feature columns
    reserved = {
        "open","high","low","close",
        params.label_col,
        "hour","day_of_week","month"
    }
    # also drop raw‐level BBands/SMA/VWAP
    for c in df.columns:
        if (
            c.startswith("bb_lband_")
            or c.startswith("bb_hband_")
            or (c.startswith("sma_") and "_pct" not in c)
            or (c.startswith("vwap_") and not c.endswith("_dev"))
        ):
            reserved.add(c)

    feat_cols = [c for c in df_tr.columns if c not in reserved]

    # 6) build pipelines
    mapping = assignment["group_final"].to_dict()
    groups  = {
        grp: [f for f in feat_cols if mapping.get(f) == grp]
        for grp in ["bounded","ratio","log_skewed","robust_tails","robust_tails_light","robust_tails_heavy","discrete","unbounded"]
    }

    def clip01(X: np.ndarray) -> np.ndarray:
        mask = np.isnan(X)
        out  = np.clip(X, 0.0, 1.0)
        out[mask] = np.nan
        return out

    class Winsorizer(FunctionTransformer):
        def __init__(self, lower_pct=1.0, upper_pct=99.0):
            super().__init__(func=None, inverse_func=None, validate=False)
            self.lower_pct = lower_pct
            self.upper_pct = upper_pct
    
        def fit(self, X, y=None):
            self.low_, self.high_ = (
                np.nanpercentile(X, self.lower_pct, axis=0),
                np.nanpercentile(X, self.upper_pct, axis=0)
            )
            return self
    
        def transform(self, X):
            mask = np.isnan(X)
            out  = np.clip(X, self.low_, self.high_)
            out[mask] = np.nan
            return out

    def signed_log(X: np.ndarray) -> np.ndarray:
        return np.sign(X) * np.log1p(np.abs(X))

    pipelines = [
        ("bnd", Pipeline([
            ("clip100", FunctionTransformer(lambda X: np.clip(X, 0, 100), validate=False)),
            ("mm",      MinMaxScaler()),
            ("c01",     FunctionTransformer(clip01, validate=False)),
        ]), groups["bounded"]),

        ("rat", Pipeline([
            ("pt",  PowerTransformer(method="yeo-johnson", standardize=False)),
            ("mm",  MinMaxScaler()),
            ("c01", FunctionTransformer(clip01, validate=False)),
        ]), groups["ratio"]),

        ("lgs", Pipeline([
            ("slog", FunctionTransformer(signed_log, validate=False)),
            ("mm",   MinMaxScaler()),
            ("c01",  FunctionTransformer(clip01, validate=False)),
        ]), groups["log_skewed"]),

        ("robust_tails_light", Pipeline([
            ("win", Winsorizer(lower_pct=0.005, upper_pct=99.995)),
            ("mm",  MinMaxScaler()),
            ("c01", FunctionTransformer(clip01, validate=False)),
        ]), groups["robust_tails_light"]),
    
        ("robust_tails", Pipeline([
            ("win", Winsorizer(lower_pct=1.0, upper_pct=99.0)),
            ("mm",  MinMaxScaler()),
            ("c01", FunctionTransformer(clip01, validate=False)),
        ]), groups["robust_tails"]),
    
        ("robust_tails_heavy", Pipeline([
            ("win", Winsorizer(lower_pct=20, upper_pct=80)),
            ("mm",  MinMaxScaler()),
            ("c01", FunctionTransformer(clip01, validate=False)),
        ]), groups["robust_tails_heavy"]),

        ("dis", Pipeline([
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("mm",  MinMaxScaler()),
            ("c01", FunctionTransformer(clip01, validate=False)),
        ]), groups["discrete"]),

        ("unb", Pipeline([
            ("std", StandardScaler()),
            ("c3",  FunctionTransformer(lambda X: np.clip(X, -3, 3), validate=False)),
            ("mm",  MinMaxScaler()),
            ("c01", FunctionTransformer(clip01, validate=False)),
        ]), groups["unbounded"]),
    ]
    ct = ColumnTransformer(transformers=pipelines, remainder="drop")

    # 7) fit on TRAIN
    ct.fit(df_tr[feat_cols])

    # 8) flatten names & widths
    flat_feats = [f for _,_,cols in pipelines for f in cols]
    n_feats    = len(flat_feats)

    # 9) transform day‐by‐day
    def transform_by_day(split_df: pd.DataFrame, label: str) -> pd.DataFrame:
        arr = np.empty((len(split_df), n_feats), dtype=float)
        for day, block in tqdm(
            split_df.groupby(split_df.index.normalize()),
            desc=f"{label} days", unit="day"
        ):
            mask = split_df.index.normalize() == day
            arr[mask] = ct.transform(block[feat_cols])

        scaled = pd.DataFrame(arr, index=split_df.index, columns=flat_feats)
        return pd.concat([scaled, split_df[list(reserved)]], axis=1)[split_df.columns]

    df_tr_s = transform_by_day(df_tr, "train")
    df_v_s  = transform_by_day(df_v,  "val")
    df_te_s = transform_by_day(df_te, "test")

    # 10) reassemble & drop raw columns
    df_all = pd.concat([df_tr_s, df_v_s, df_te_s]).sort_index()
    to_drop = ["open","high","low","close"] + [
        c for c in df_all.columns
        if c.startswith("bb_lband_")
        or c.startswith("bb_hband_")
        or (c.startswith("sma_") and "_pct" not in c)
        or (c.startswith("vwap_") and not c.endswith("_dev"))
    ]
    return df_all.drop(columns=to_drop, errors="ignore")


#########################################################################################################


def compare_raw_vs_scaled(
    df_raw: pd.DataFrame,
    df_scaled: pd.DataFrame,
    assignment: pd.DataFrame,
    feat_cols: Optional[List[str]] = None,
    train_prop: float = params.train_prop,
    tol_range:  float = 1e-6
) -> pd.DataFrame:
    """
    On the TRAIN slice only, verify that each scaled feature
    preserves its core invariants under our per-day [0,1] pipelines.

    Functionality:
      1) Split off the first train_prop fraction; replace ±inf with NaN.
      2) Select features common to raw, scaled, optional feat_cols,
         and present in assignment.index.
      3) Compute TRAIN‐only percentiles (min,1%,5%,50%,95%,99%,max).
      4) For each feature on TRAIN:
           a) NaN‐mask unchanged.
           b) All scaled values ∈ [0,1] ± tol_range.
           c) If 'discrete': unique_count unchanged.
           d) If 'bounded': exact linear clip(raw,0,100)/100 mapping.
           e) If constant raw: scaled is constant.
      5) Return a summary DataFrame with pass/fail flags and reasons.
    """
    # 1) TRAIN split and NaN‐clean
    N    = len(df_raw)
    n_tr = int(N * train_prop)
    raw_tr = df_raw.iloc[:n_tr].replace([np.inf, -np.inf], np.nan)
    sca_tr = df_scaled.iloc[:n_tr].replace([np.inf, -np.inf], np.nan)

    # 2) Determine features to test
    common = set(raw_tr.columns) & set(sca_tr.columns)
    if feat_cols is not None:
        common &= set(feat_cols)
    features = [f for f in common if f in assignment.index]

    # 3) Compute TRAIN‐only percentiles
    qs = [0.01, 0.05, 0.50, 0.95, 0.99]
    raw_q = (
        raw_tr[features]
        .describe(percentiles=qs).T
        .loc[:, ['min','1%','5%','50%','95%','99%','max']]
        .add_prefix('raw_')
    )
    sca_q = (
        sca_tr[features]
        .describe(percentiles=qs).T
        .loc[:, ['min','1%','5%','50%','95%','99%','max']]
        .add_prefix('scaled_')
    )
    cmp_df = pd.concat([raw_q, sca_q], axis=1)
    cmp_df['group_final'] = assignment['group_final']

    # 4) Prepare result containers
    nan_ok       = {}
    range_ok     = {}
    discrete_ok  = {}
    bounded_ok   = {}
    const_ok     = {}

    # 5) Per‐feature checks on TRAIN
    for feat in tqdm(features, desc="Validating train-split"):
        grp = assignment.at[feat, 'group_final']

        # a) NaN‐mask unchanged
        nan_ok[feat] = (raw_tr[feat].isna() == sca_tr[feat].isna()).all()

        # align non‐NaN pairs
        x = raw_tr[feat].dropna()
        y = sca_tr[feat].dropna()
        x, y = x.align(y, join='inner')

        # b) Range containment [0,1]
        if len(y):
            ymin, ymax = y.min(), y.max()
            range_ok[feat] = (ymin >= -tol_range) and (ymax <= 1 + tol_range)
        else:
            range_ok[feat] = True

        # c) Discrete cardinality
        if grp == 'discrete':
            discrete_ok[feat] = x.nunique() == y.nunique()
        else:
            discrete_ok[feat] = True

        # d) Bounded linear mapping
        if grp == 'bounded' and len(x):
            # clip raw to [0,100], then compute train-min/max
            x_clip = np.clip(x, 0, 100)
            lo, hi = x_clip.min(), x_clip.max()
            # map into [0,1]
            target = (x_clip - lo) / (hi - lo) if hi > lo else 0
            bounded_ok[feat] = np.allclose(y, target, atol=tol_range)
        else:
            bounded_ok[feat] = True

        # e) Constant‐feature behavior
        rmin, rmax = raw_q.at[feat, 'raw_min'], raw_q.at[feat, 'raw_max']
        if abs(rmax - rmin) < tol_range:
            const_ok[feat] = y.nunique() == 1
        else:
            const_ok[feat] = True

    # 6) Compile pass/fail status & reasons
    status, reason = [], []
    for feat in features:
        errs = []
        if not nan_ok[feat]:
            errs.append("nan_mask_changed")
        if not range_ok[feat]:
            mn, mx = sca_q.at[feat, 'scaled_min'], sca_q.at[feat, 'scaled_max']
            errs.append(f"range[{mn:.3f},{mx:.3f}]")
        if not discrete_ok[feat]:
            errs.append("cardinality_changed")
        if not bounded_ok[feat]:
            errs.append("non-linear_bounded")
        if not const_ok[feat]:
            errs.append("constant_not_const")

        if errs:
            status.append("FAIL")
            reason.append("; ".join(errs))
        else:
            status.append("OK")
            reason.append("all checks passed")

    cmp_df['nan_mask_ok']       = pd.Series(nan_ok)
    cmp_df['range_ok']          = pd.Series(range_ok)
    cmp_df['discrete_ok']       = pd.Series(discrete_ok)
    cmp_df['bounded_linear_ok'] = pd.Series(bounded_ok)
    cmp_df['constant_ok']       = pd.Series(const_ok)
    cmp_df['status']            = status
    cmp_df['reason']            = reason

    return cmp_df


#########################################################################################################


# def ig_feature_importance(
#     model: nn.Module,
#     loader,
#     feature_names,
#     device: torch.device,
#     n_samples: int = 100,
#     n_steps: int = 50
# ) -> pd.DataFrame:
#     """
#     Compute per‐feature Integrated Gradients attributions.

#     - Disables cuDNN (RNN backward incompatibility in eval).
#     - Puts model in eval(), clears state.
#     - Defines a pure‐float32 forward that returns the final scalar signal.
#     - Runs IG with float32 inputs and baselines.
#     - Summarizes absolute attributions over time and averages across windows.
#     - Returns DataFrame sorted by descending importance.
#     """
#     # 1) disable cuDNN for RNN backward
#     cudnn_enabled = cudnn.enabled
#     cudnn.enabled = False

#     model.eval()
#     model.h_short = model.h_long = None

#     # 2) forward that emits the final scalar (sigmoid‐activated) in float32
#     def forward_reg(x: torch.Tensor) -> torch.Tensor:
#         # x: (B, W, F), float32
#         with torch.cuda.amp.autocast(enabled=False):
#             model.float()
#             model.reset_short()
#             out = model(x.float())
#             pr  = out[0] if isinstance(out, (tuple, list)) else out
#             # pr may be (B, W, 1), (B, W, D), or (B, W)
#             if pr.dim() == 3 and pr.size(-1) == 1:
#                 pr = pr.squeeze(-1)
#             if pr.dim() == 3:
#                 pr = pr[..., 0]
#             if pr.dim() == 2:
#                 # final timestep = last window step
#                 pr = pr[:, -1]
#             return torch.sigmoid(pr)

#     ig = IntegratedGradients(forward_reg)

#     total_attr = np.zeros(len(feature_names), dtype=float)
#     count = 0

#     # 3) iterate windows (with progress bar)
#     for batch in tqdm(loader, desc="IG windows", total=n_samples):
#         xb, y_sig, *_, lengths = batch
#         W = int(lengths[0])
#         if W == 0:
#             continue

#         x = xb[0, :W].unsqueeze(0).to(device).float()   # (1, W, F)
#         baseline = torch.zeros_like(x)

#         # 4) compute IG in float32
#         atts, delta = ig.attribute(
#             inputs=x,
#             baselines=baseline,
#             n_steps=n_steps,
#             internal_batch_size=1,
#             return_convergence_delta=True
#         )

#         # 5) collapse abs attributions over time
#         attr_np = atts.detach().abs().cpu().numpy()      # (1, W, F)
#         abs_sum = attr_np.reshape(-1, attr_np.shape[-1]).sum(axis=0)  # (F,)
#         total_attr += abs_sum
#         count += 1
#         if count >= n_samples:
#             break

#         # 6) free GPU memory
#         del atts, delta, x, baseline
#         torch.cuda.empty_cache()

#     # 7) restore cuDNN
#     cudnn.enabled = cudnn_enabled

#     # 8) average and package
#     avg_attr = total_attr / max(1, count)
#     imp_df   = pd.DataFrame({
#         "feature":    feature_names,
#         "importance": avg_attr
#     }).sort_values("importance", ascending=False).reset_index(drop=True)

#     return imp_df


def ig_feature_importance(
    model: nn.Module,
    loader,
    feature_names,
    device: torch.device,
    n_samples: int = 100,
    n_steps: int = 50
) -> pd.DataFrame:
    """
    Minimal Integrated Gradients feature importances.
    Assumes loader yields tuples (xb, y_sig, ..., seq_lengths) and that
    xb has shape (B, W, F). Does not modify ModelClass source.
    """
    # disable cuDNN during attribution (RNN backward can be fragile) and remember state
    cudnn_enabled = cudnn.enabled
    cudnn.enabled = False

    model.to(device)
    model.eval()

    # simple state reset expected by your forward (no method calls, no monkey patch)
    model.h_short = model.c_short = None
    model.h_long  = model.c_long  = None

    def forward_reg(x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F) on device, float32
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            model.float()
            # clear states so repeated forward calls are independent
            model.h_short = model.c_short = None
            model.h_long  = model.c_long  = None
            out = model(x.float())
            pr = out[0] if isinstance(out, (tuple, list)) else out
            if pr.dim() == 3 and pr.size(-1) == 1:
                pr = pr.squeeze(-1)
            if pr.dim() == 3:
                pr = pr[..., 0]
            if pr.dim() == 2:
                pr = pr[:, -1]
            return torch.sigmoid(pr)

    ig = IntegratedGradients(forward_reg)

    total_attr = np.zeros(len(feature_names), dtype=float)
    count = 0

    for batch in tqdm(loader, desc="IG windows", total=n_samples):
        xb, y_sig, *rest, seq_lengths = batch
        W = int(seq_lengths[0])
        if W == 0:
            continue

        x = xb[0, :W].unsqueeze(0).to(device).float()   # (1, W, F)
        baseline = torch.zeros_like(x)

        atts, _delta = ig.attribute(
            inputs=x,
            baselines=baseline,
            n_steps=n_steps,
            internal_batch_size=1,
            return_convergence_delta=True
        )

        attr_np = atts.detach().abs().cpu().numpy()      # (1, W, F)
        abs_sum = attr_np.reshape(-1, attr_np.shape[-1]).sum(axis=0)  # (F,)
        total_attr += abs_sum
        count += 1
        if count >= n_samples:
            del atts, _delta, x, baseline
            torch.cuda.empty_cache()
            break

        del atts, _delta, x, baseline
        torch.cuda.empty_cache()

    cudnn.enabled = cudnn_enabled

    avg_attr = total_attr / max(1, count)
    imp_df = pd.DataFrame({"feature": feature_names, "importance": avg_attr}) \
             .sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df
