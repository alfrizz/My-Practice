from libs import plots, params

from typing import Sequence, List, Tuple, Optional, Union, Dict, Iterable
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


def add_session_centered_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact DataFrame of single-scalar session-centered time features mapped to [0,1].

    Produces these columns (each in [0,1]):
      - minute_feat: minute-of-day phase centered on sess_start (mapped from (-1,1] -> [0,1])
      - hour_feat: fractional-hour phase centered on sess_start (mapped to [0,1])
      - dow_feat, month_feat, day_of_year_feat, week_of_year_feat: single circular scalars

    Implementation details / guarantees:
      - Numeric numpy arrays used internally; outputs are float32 to reduce memory.
      - Logic and centering are identical to the earlier implementation: phase = 2π*(v-center)/period,
        wrapped to (-π,π] then normalized by /π to (-1,1], finally mapped to [0,1] by (x+1)/2.
      - Week extraction uses .isocalendar() safely and coerces to numpy.
    """
    def _single_centered_phase_np(v: np.ndarray, center: float, period: float) -> np.ndarray:
        # v must be numeric numpy array
        phase = 2.0 * np.pi * (v.astype(np.float64) - float(center)) / float(period)
        phase = ((phase + np.pi) % (2.0 * np.pi)) - np.pi    # wrap to (-pi, pi]
        return (phase / np.pi).astype(np.float32)            # in (-1,1] as float32

    sess_start_min = float(params.sess_start_reg.hour * 60 + params.sess_start_reg.minute)
    sess_start_hr = sess_start_min / 60.0

    # build base arrays (force numeric numpy arrays)
    minutes = (df.index.hour * 60 + df.index.minute).to_numpy(dtype=np.int32)         # 0..1439
    hour_idx = (df.index.hour).to_numpy(dtype=np.int32)                               # 0..23
    dow = df.index.dayofweek.to_numpy(dtype=np.int32)                                 # 0..6
    month0 = (df.index.month - 1).to_numpy(dtype=np.int32)                            # 0..11
    doy = (df.index.dayofyear - 1).to_numpy(dtype=np.int32)                           # 0..364

    # safe week-of-year extraction (coerce to numpy)
    iso = df.index.isocalendar()
    # iso may be DataFrame-like; ensure numeric numpy array
    week0 = (iso.week.to_numpy().astype(np.int32) - 1)                                # 0..52

    out = pd.DataFrame(index=df.index)

    # centered phases in (-1,1], then map to [0,1] and cast float32
    out["minute_time"] = (((_single_centered_phase_np(minutes, sess_start_min, 1440.0)) + 1.0) / 2.0).astype(np.float32)
    out["hour_time"]   = (((_single_centered_phase_np(hour_idx, sess_start_hr, 24.0)) + 1.0) / 2.0).astype(np.float32)
    out["dow_time"]         = (((_single_centered_phase_np(dow, 0.0, 7.0)) + 1.0) / 2.0).astype(np.float32)
    out["month_time"]       = (((_single_centered_phase_np(month0, 0.0, 12.0)) + 1.0) / 2.0).astype(np.float32)
    out["day_of_year_time"] = (((_single_centered_phase_np(doy, 0.0, 365.0)) + 1.0) / 2.0).astype(np.float32)
    out["week_of_year_time"] = (((_single_centered_phase_np(week0, 0.0, 52.0)) + 1.0) / 2.0).astype(np.float32)
    out["in_sess_time"] = ((df.index.time >= params.sess_start_reg) & (df.index.time < params.sess_end)).astype(np.float32)

    return out


###########################################




def standard_indicators(
    df: pd.DataFrame,
    extra_windows: Optional[Iterable[int]] = None,
    label_col: str = params.label_col,
    sma_short: int = 14,
    sma_long: int = 28,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_sig: int = 9,
    atr_window: int = 14,
    bb_window: int = 20,
    vwap_window: int = 14,
    vol_spike_window: int = 14,
    z_candidates: Optional[Iterable[str]] = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Compute canonical OHLCV technical indicators and consistent multi-horizon variants.

    - Produces canonical single-scale indicators from raw OHLCV (sma/ema/roc/rsi/macd/atr/bb/adx/obv/vwap/vol_spike etc.).
    - For each horizon in extra_windows emits horizon-specific variants using the correct
      rolling per indicator class (SMA/EMA for price, TA methods for ATR/BB/ADX, VWAP per window, etc.).
    - Adds one robust median/MAD z (scale-free) per window only for features listed in z_candidates.
      Robust z is computed from the canonical unsuffixed source when available to avoid
      accidental rolling-of-rolling. Output z column name: "{base}_z_{w}".
    - Preserves all original naming, eps handling and min_periods semantics; the change is minimal.
    """
    # defensive copy, ensure datetime index monotonic increasing
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    assert df.index.is_monotonic_increasing

    # prepare extra windows set (no window==1 duplicates)
    if extra_windows is None:
        W = []
    else:
        W = sorted({int(w) for w in extra_windows if (isinstance(w, (int, np.integer)) and int(w) > 1)})

    # required input columns
    cols_in = ["open", "high", "low", "close", "volume"]
    if label_col in df.columns:
        cols_in.append(label_col)
    base = df[cols_in].copy()
    o, h, l, c, v = base["open"], base["high"], base["low"], base["close"], base["volume"]

    def safe_div(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / (den + eps)
        return out

    new = {}

    # base returns
    new["ret"] = c.pct_change()
    new["log_ret"] = np.log(c + eps).diff()

    # canonical SMA/EMA
    s14 = c.rolling(sma_short, min_periods=1).mean()
    s28 = c.rolling(sma_long, min_periods=1).mean()
    new[f"sma_{sma_short}"] = s14
    new[f"sma_{sma_long}"] = s28
    new[f"sma_pct_{sma_short}"] = safe_div(c - s14, s14)
    new[f"sma_pct_{sma_long}"] = safe_div(c - s28, s28)
    new[f"ema_{sma_short}"] = c.ewm(span=sma_short, adjust=False).mean()
    new[f"ema_{sma_long}"] = c.ewm(span=sma_long, adjust=False).mean()

    # canonical ROC (skip w==1)
    new[f"roc_{sma_short}"] = c.pct_change(sma_short)
    new[f"roc_{sma_long}"] = c.pct_change(sma_long)

    # candlestick geometry
    new["body"] = c - o
    new["body_pct"] = safe_div(c - o, o)
    new["upper_shad"] = h - np.maximum(o, c)
    new["lower_shad"] = np.minimum(o, c) - l
    new["range_pct"] = safe_div(h - l, c)

    # canonical RSI, MACD, ATR, BB, DI/ADX, OBV, VWAP, vol spike
    new["rsi"] = ta.momentum.RSIIndicator(close=c, window=rsi_window).rsi()
    macd = ta.trend.MACD(close=c, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_sig)
    new["macd_line"] = macd.macd()
    new["macd_signal"] = macd.macd_signal()
    new["macd_diff"] = macd.macd_diff()
    atr = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=atr_window)
    new["atr"] = atr.average_true_range()
    new["atr_pct"] = safe_div(new["atr"], c)
    bb = ta.volatility.BollingerBands(close=c, window=bb_window, window_dev=2)
    new["bb_lband"] = bb.bollinger_lband()
    new["bb_hband"] = bb.bollinger_hband()
    new["bb_w"] = safe_div(new["bb_hband"] - new["bb_lband"], bb.bollinger_mavg())
    adx = ta.trend.ADXIndicator(high=h, low=l, close=c, window=atr_window)
    new["plus_di"] = adx.adx_pos()
    new["minus_di"] = adx.adx_neg()
    new["adx"] = adx.adx()
    new["obv"] = ta.volume.OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()
    vwap = ta.volume.VolumeWeightedAveragePrice(high=h, low=l, close=c, volume=v, window=vwap_window)
    new["vwap"] = vwap.volume_weighted_average_price()
    new["vwap_dev_pct"] = 100.0 * safe_div(c - new["vwap"], new["vwap"])

    # vol spike and z (base window uses vol_spike_window)
    vol_roll = v.rolling(vol_spike_window, min_periods=vol_spike_window).mean().fillna(eps)
    new["vol_spike"] = safe_div(v, vol_roll)
    v_mu = v.rolling(vol_spike_window, min_periods=vol_spike_window).mean()
    v_sigma = v.rolling(vol_spike_window, min_periods=vol_spike_window).std().fillna(eps).replace(0.0, eps)
    new[f"vol_z_{vol_spike_window}"] = safe_div(v - v_mu, v_sigma)

    # rolling extrema (canonical sma_long)
    new[f"rolling_max_close_{sma_long}"] = c.rolling(sma_long, min_periods=sma_long).max()
    new[f"rolling_min_close_{sma_long}"] = c.rolling(sma_long, min_periods=sma_long).min()
    new[f"dist_high_{sma_long}"] = safe_div(new[f"rolling_max_close_{sma_long}"] - c, c)
    new[f"dist_low_{sma_long}"] = safe_div(c - new[f"rolling_min_close_{sma_long}"], c)

    # OBV base and derived (use vol_spike_window as canonical smoothing length)
    new["obv_sma"] = new["obv"].rolling(vol_spike_window, min_periods=vol_spike_window).mean()
    new[f"obv_diff_{vol_spike_window}"] = new["obv"].diff(vol_spike_window)
    denom_arr = (c.rolling(vol_spike_window, min_periods=vol_spike_window).mean().abs().replace(0.0, eps)).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = new[f"obv_diff_{vol_spike_window}"].to_numpy(dtype=float) / denom_arr
    new[f"obv_pct_{vol_spike_window}"] = pd.Series(np.where(np.isnan(pct), np.nan, pct), index=base.index)
    new[f"obv_sma_{vol_spike_window}"] = new["obv"].rolling(vol_spike_window, min_periods=vol_spike_window).mean()
    obv_sigma = new[f"obv_sma_{vol_spike_window}"].rolling(vol_spike_window, min_periods=vol_spike_window).std().fillna(eps).replace(0.0, eps)
    new[f"obv_z_{vol_spike_window}"] = safe_div(new["obv"] - new[f"obv_sma_{vol_spike_window}"], obv_sigma)

    # ---------- multi-horizon generation: canonical rollings ----------
    for w in W:
        # returns
        new[f"ret_{w}"] = c.pct_change(w)

        # SMA / SMA_pct / EMA / ROC
        new[f"sma_{w}"] = c.rolling(w, min_periods=w).mean()
        new[f"sma_pct_{w}"] = safe_div(c - new[f"sma_{w}"], new[f"sma_{w}"])
        new[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()
        if w != 1:
            new[f"roc_{w}"] = c.pct_change(w)

        # RSI / ATR / ATR_pct / BB / DI/ADX
        new[f"rsi_{w}"] = ta.momentum.RSIIndicator(close=c, window=w).rsi()
        new[f"atr_{w}"] = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=w).average_true_range()
        new[f"atr_pct_{w}"] = safe_div(new[f"atr_{w}"], c)
        bb_w = ta.volatility.BollingerBands(close=c, window=w, window_dev=2)
        new[f"bb_lband_{w}"] = bb_w.bollinger_lband()
        new[f"bb_hband_{w}"] = bb_w.bollinger_hband()
        new[f"bb_w_{w}"] = safe_div(new[f"bb_hband_{w}"] - new[f"bb_lband_{w}"], bb_w.bollinger_mavg())
        adx_w = ta.trend.ADXIndicator(high=h, low=l, close=c, window=w)
        new[f"plus_di_{w}"] = adx_w.adx_pos()
        new[f"minus_di_{w}"] = adx_w.adx_neg()
        new[f"adx_{w}"] = adx_w.adx()

        # OBV-derived (prefer raw obv as the source; avoid rolling-of-rolling)
        new[f"obv_diff_{w}"] = new["obv"].diff(w)
        price_roll_mean = c.rolling(w, min_periods=w).mean().abs().fillna(eps)
        denom_arr = price_roll_mean.replace(0.0, eps).to_numpy(dtype=float)
        num = new[f"obv_diff_{w}"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = num / denom_arr
        new[f"obv_pct_{w}"] = pd.Series(np.where(np.isnan(num), np.nan, pct), index=base.index)
        # prefer z from raw obv (avoid rolling-of-rolling)
        new[f"obv_sma_{w}"] = new["obv"].rolling(w, min_periods=w).mean()
        # strict z from raw obv rolling stats
        obv_mu = new["obv"].rolling(w, min_periods=w).mean()
        obv_sigma = new["obv"].rolling(w, min_periods=w).std().fillna(eps).replace(0.0, eps)
        new[f"obv_z_{w}"] = safe_div(new["obv"] - obv_mu, obv_sigma)

        # VWAP dev + z (prefer canonical unsuffixed vwap_dev_pct when present)
        new[f"vwap_{w}"] = ta.volume.VolumeWeightedAveragePrice(high=h, low=l, close=c, volume=v, window=w).volume_weighted_average_price()
        new[f"vwap_dev_pct_{w}"] = 100.0 * safe_div(c - new[f"vwap_{w}"], new[f"vwap_{w}"])
        x_pct = new.get("vwap_dev_pct", new[f"vwap_dev_pct_{w}"])
        new[f"z_vwap_dev_{w}"] = safe_div(x_pct - x_pct.rolling(w, min_periods=w).mean(), x_pct.rolling(w, min_periods=w).std().fillna(eps).replace(0.0, eps))

        # vol spike + vol_z (use raw volume semantics for vol_z)
        vol_roll_w = v.rolling(w, min_periods=w).mean().fillna(eps)
        new[f"vol_spike_{w}"] = safe_div(v, vol_roll_w)
        v_mu = v.rolling(w, min_periods=w).mean()
        v_sigma = v.rolling(w, min_periods=w).std().fillna(eps).replace(0.0, eps)
        new[f"vol_z_{w}"] = safe_div(v - v_mu, v_sigma)

        # rolling volatility
        new[f"ret_std_{w}"] = new["ret"].rolling(w, min_periods=w).std()

        # rolling extrema and distances
        new[f"rolling_max_close_{w}"] = c.rolling(w, min_periods=w).max()
        new[f"rolling_min_close_{w}"] = c.rolling(w, min_periods=w).min()
        new[f"dist_high_{w}"] = safe_div(new[f"rolling_max_close_{w}"] - c, c)
        new[f"dist_low_{w}"] = safe_div(c - new[f"rolling_min_close_{w}"], c)

        # MACD smoothing: rolling mean of canonical outputs
        new[f"macd_line_{w}"] = new["macd_line"].rolling(w, min_periods=w).mean()
        new[f"macd_signal_{w}"] = new["macd_signal"].rolling(w, min_periods=w).mean()
        new[f"macd_diff_{w}"] = new["macd_diff"].rolling(w, min_periods=w).mean()

    # ---------- apply robust z only to selected drift-prone indicators ----------
    def _rolling_median_mad_z(s: pd.Series, w: int) -> pd.Series:
        med = s.rolling(w, min_periods=w).median()
        mad = (s - med).abs().rolling(w, min_periods=w).median() * 1.4826
        arr = s.to_numpy(dtype=float); fin = arr[np.isfinite(arr)]
        gstd = float(np.nanstd(fin)) if fin.size else eps
        floor = max(eps, gstd * 1e-6)
        mad = mad.fillna(0.0).where(mad > floor, floor)
        z = (s - med) / mad
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-20.0, 20.0)
    
    for w in W:
        for base_name in z_candidates:
            src = new.get(base_name)
            if src is None:
                src = new.get(f"{base_name}_{w}")
            if src is None:
                src = base.get(base_name)
            if src is None:
                continue
            if not isinstance(src, pd.Series):
                src = pd.Series(src, index=base.index, dtype=float)
            new[f"{base_name}_z_{w}"] = _rolling_median_mad_z(src, w)
    
    time_feats = add_session_centered_time_features(base)
    new.update(time_feats.to_dict(orient="series"))
    
    out = pd.concat([base, pd.DataFrame(new, index=base.index)], axis=1)
    return out


##########################################################################################################


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
    Build engineered signals from base indicators (uses only past/present values).
    - Works with canonical columns from standard_indicators and their windowed copies.
    - Detects relevant columns robustly (e.g. 'sma_14', 'sma_60', 'macd_diff', 'macd_diff_30').
    - Uses safe denominators and std floors to ensure finite outputs.
    Returns DataFrame of engineered columns (float, NaNs filled per fillna_zero).
    """
    out = pd.DataFrame(index=df.index)
    produced: List[str] = []

    def _find(prefix: str) -> Optional[str]:
        # find first column that startswith prefix (prefer exact canonical name before windowed)
        if prefix in df.columns:
            return prefix
        for c in df.columns:
            if c.startswith(prefix):
                return c
        return None

    def _std_floor(series: pd.Series, window: int):
        rstd = series.rolling(window, min_periods=1).std()
        arr = series.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        global_std = float(np.nanstd(finite)) if finite.size else eps
        floor = max(eps, abs(global_std) * z_std_floor_factor)
        return rstd.fillna(floor).replace(0.0, floor)

    # helpers for column detection
    close = df.get("close", pd.Series(index=df.index, dtype=float)).astype(float)

    # robust SMA detection: pick two smallest numeric windows if available
    sma_cols = sorted([c for c in df.columns if re.match(r"^sma_\d+$", c)],
                      key=lambda c: int(re.search(r"_(\d+)$", c).group(1)))
    sma_s = sma_cols[0] if sma_cols else None
    sma_l = sma_cols[1] if len(sma_cols) > 1 else None


    # find other columns (prefer canonical names, fallback to first matching windowed)
    macd_diff = _find("macd_diff")
    bb_l = _find("bb_lband")
    bb_h = _find("bb_hband")
    bb_w = _find("bb_w")
    rsi_col = _find("rsi")
    plus_di = _find("plus_di")
    minus_di = _find("minus_di")
    adx_col = _find("adx")
    obv_diff = _find("obv_diff")
    obv_sma = _find("obv_sma")
    vwap_dev = _find("vwap_dev") or _find("vwap")
    atr_pct = _find("atr_pct")

    # eng_ma
    if sma_s and sma_l and sma_s in df.columns and sma_l in df.columns:
        out["eng_ma"] = (df[sma_s].astype(float) - df[sma_l].astype(float)) / (df[sma_l].astype(float) + eps)
        produced.append("eng_ma")

    # eng_macd
    if macd_diff and sma_l and macd_diff in df.columns and sma_l in df.columns:
        out["eng_macd"] = df[macd_diff].astype(float) / (df[sma_l].astype(float) + eps)
        produced.append("eng_macd")

    # eng_bb / mid
    if bb_l and bb_h and bb_w and bb_l in df.columns and bb_h in df.columns and bb_w in df.columns:
        lo = df[bb_l].astype(float); hi = df[bb_h].astype(float); bw = df[bb_w].astype(float)
        dev = np.where(close < lo, lo - close, np.where(close > hi, close - hi, 0.0))
        out["eng_bb"] = pd.Series(dev, index=df.index) / (bw + eps)
        out["eng_bb_mid"] = ((lo + hi) / 2 - close) / (close + eps)
        produced.extend(["eng_bb", "eng_bb_mid"])

    # eng_rsi
    if rsi_col and rsi_col in df.columns:
        rv = df[rsi_col].astype(float)
        low_d = np.clip((rsi_low - rv), 0, None) / 100.0
        high_d = np.clip((rv - rsi_high), 0, None) / 100.0
        out["eng_rsi"] = np.where(rv < rsi_low, low_d, np.where(rv > rsi_high, high_d, 0.0))
        produced.append("eng_rsi")

    # eng_adx
    if plus_di and minus_di and adx_col and plus_di in df.columns and minus_di in df.columns and adx_col in df.columns:
        di_diff = df[plus_di].astype(float) - df[minus_di].astype(float)
        diff_abs = di_diff.abs() / 100.0
        ex = np.clip((df[adx_col].astype(float) - adx_thr) / 100.0, 0, None)
        out["eng_adx"] = np.sign(di_diff) * diff_abs * ex
        produced.append("eng_adx")

    # eng_obv (safe denom + winsorized numerator)
    if obv_diff and obv_sma and obv_diff in df.columns and obv_sma in df.columns:
        num = df[obv_diff].astype(float).to_numpy(dtype=float)
        try:
            finite = num[np.isfinite(num)]
            lo_cut = np.nanpercentile(finite, 0.5) if finite.size else np.nan
            hi_cut = np.nanpercentile(finite, 99.5) if finite.size else np.nan
            num_clipped = np.copy(num)
            mask_num = np.isnan(num_clipped)
            num_clipped = np.clip(num_clipped, lo_cut, hi_cut)
            num_clipped[mask_num] = np.nan
        except Exception:
            num_clipped = num

        den_s = df[obv_sma].rolling(window=mult_w, min_periods=mult_w).median().ffill().fillna(0.0)
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
    if atr_pct and atr_pct in df.columns:
        ratio = df[atr_pct].astype(float)
        rm = ratio.rolling(mult_w, min_periods=mult_w).mean()
        out["eng_atr_div"] = (ratio - rm) * 10_000
        std_r = ratio.rolling(mult_w, min_periods=mult_w).std()
        global_std = np.nanstd(ratio.to_numpy(dtype=float))
        std_floor = max(eps, abs(global_std) * z_std_floor_factor)
        std_eff = std_r.fillna(std_floor).replace(0.0, std_floor)
        out["z_eng_atr"] = (ratio - rm) / (std_eff + eps)
        produced.extend(["eng_atr_div", "z_eng_atr"])

    # eng_sma distances
    if sma_s and sma_s in df.columns:
        out["eng_sma_short"] = (df[sma_s].astype(float) - close) / (close + eps)
        produced.append("eng_sma_short")
    if sma_l and sma_l in df.columns:
        out["eng_sma_long"] = (df[sma_l].astype(float) - close) / (close + eps)
        produced.append("eng_sma_long")

    # eng_vwap and z — prefer canonical vwap_dev_pct if present
    vwap_col = _find("vwap_dev_pct") or _find("vwap")
    if vwap_col and vwap_col in df.columns:
        if vwap_col == "vwap_dev_pct":
            eng_vwap_pct = df[vwap_col].astype(float)
        else:
            vwap_base = df[vwap_col].astype(float)
            eng_vwap_pct = 100.0 * (vwap_base - close) / (close + eps)
        out["eng_vwap"] = eng_vwap_pct
        znum = eng_vwap_pct
        zm = znum.rolling(mult_w, min_periods=mult_w).mean()
        zs = znum.rolling(mult_w, min_periods=mult_w).std()
        arr = znum.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        global_std = float(np.nanstd(finite)) if finite.size else eps
        std_floor = max(eps, abs(global_std) * z_std_floor_factor)
        zs_eff = zs.fillna(std_floor).replace(0.0, std_floor)
        out["z_vwap_dev"] = (znum - zm) / (zs_eff + eps)
        produced.extend(["eng_vwap", "z_vwap_dev"])

    # z_bb_w
    if bb_w and bb_w in df.columns:
        x = df[bb_w].astype(float)
        xm = x.rolling(mult_w, min_periods=mult_w).mean()
        xs = x.rolling(mult_w, min_periods=mult_w).std()
        global_std = np.nanstd(x.to_numpy(dtype=float))
        std_floor = max(eps, abs(global_std) * z_std_floor_factor)
        xs_eff = xs.fillna(std_floor).replace(0.0, std_floor)
        out["z_bb_w"] = (x - xm) / (xs_eff + eps)
        produced.append("z_bb_w")

    # strict z of raw OBV (prefer this for scale-free ML input)
    if "obv" in df.columns:
        x = df["obv"].astype(float)
        xm = x.rolling(mult_w, min_periods=mult_w).mean()
        xs = x.rolling(mult_w, min_periods=mult_w).std()
        arr = x.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        global_std = float(np.nanstd(finite)) if finite.size else eps
        std_floor = max(eps, abs(global_std) * z_std_floor_factor)
        xs_eff = xs.fillna(std_floor).replace(0.0, std_floor)
        out["z_obv"] = (x - xm) / (xs_eff + eps)
        produced.append("z_obv")

    # momentum aggregates: detect available ret_{H} columns (use any found in df)
    ret_cols = [c for c in df.columns if re.match(r"^ret_(\d+)$", c)]
    horizons = sorted({int(re.search(r"^ret_(\d+)$", c).group(1)) for c in ret_cols}) if ret_cols else [1, 5, 15, 60]
    for H in horizons:
        ret_col = f"ret_{H}"
        if ret_col in df.columns:
            out[f"mom_sum_{H}"] = df[ret_col].rolling(H, min_periods=H).sum()
            out[f"mom_std_{H}"] = df[ret_col].rolling(H, min_periods=H).std()
            produced.extend([f"mom_sum_{H}", f"mom_std_{H}"])

    # ema cross flags: find sma numeric windows used for ema names
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

    # finalize: ensure order, types and optional fillna
    produced = [p for p in produced if p in out.columns]
    for col in produced:
        out[col] = out[col].astype(float)
        if fillna_zero:
            out[col] = out[col].fillna(0.0)

    return out[produced].copy()


##########################################################################################################


def prune_and_diag(
    df_unsc: pd.DataFrame,
    train_prop: float = 0.7,
    pct_shift_thresh: float = 0.16,
    frac_outside_thresh: float = 0.06,
    ks_pval_thresh: float = 1e-6,
    min_train_samples: int = 50,
    conc_bins: int = 80,
    min_failures = 1,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Prune numeric features and return diagnostics.

    Summary
    - Split rows contiguously into TRAIN / VAL / TEST by `train_prop`.
    - For each numeric column (except the supervised label column if `params.label_col`
      exists in the calling scope) compute TRAIN percentiles (q01,q25,q75,q99 with
      tiny-tail fallback), medians, relative median shifts, fraction of VAL/TEST
      outside TRAIN tails, optional KS p-value (TRAIN vs TEST), and a simple spread
      score derived from a histogram over [q01,q99].
    - Drop rule: mark feature DROP when any of:
        * relative median shift in VAL or TEST > pct_shift_thresh
        * fraction of VAL or TEST outside TRAIN tails > frac_outside_thresh
        * TRAIN is effectively constant
        * KS p-value < ks_pval_thresh AND (pct_shift_fail OR frac_out_fail)
    - Returns (df_pruned, to_drop, diag). `diag` contains per-feature diagnostics.
    - Minimal safety: skip pruning for features with too few TRAIN non-null samples.
    """
    # small constants
    zero_tol = 1e-8
    denom_eps = 1e-6
    const_eps = 1e-12
    tail_min_count = 3

    if len(df_unsc) == 0:
        return df_unsc.copy(), [], pd.DataFrame()

    # split
    n_tr = int(len(df_unsc) * train_prop)
    n_val = (len(df_unsc) - n_tr) // 2
    tr = df_unsc.iloc[:n_tr].replace([np.inf, -np.inf], np.nan)
    vl = df_unsc.iloc[n_tr:n_tr + n_val].replace([np.inf, -np.inf], np.nan)
    te = df_unsc.iloc[n_tr + n_val:].replace([np.inf, -np.inf], np.nan)

    numeric_cols = [c for c in df_unsc.columns
                    if pd.api.types.is_numeric_dtype(df_unsc[c]) and c not in params.label_col]

    rows = []

    def safe_denom(arr: np.ndarray) -> float:
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

    # skip-prune threshold for tiny TRAIN samples
    min_non_na_train = max(5, int(0.01 * max(1, len(tr))))

    for feat in tqdm(numeric_cols, desc="prune_and_diag", unit="feat"):
        tr_s = tr[feat].dropna()
        vl_s = vl[feat].dropna()
        te_s = te[feat].dropna()

        if tr_s.shape[0] < min_non_na_train:
            rows.append({
                "feature": feat, "q01": np.nan, "q25": np.nan, "q75": np.nan, "q99": np.nan,
                "full_span": np.nan, "iqr": np.nan, "med_tr": np.nan, "med_val": np.nan, "med_te": np.nan,
                "pct_shift_val": np.nan, "pct_shift_te": np.nan, "abs_shift_val": np.nan, "abs_shift_te": np.nan,
                "frac_val_out": np.nan, "frac_te_out": np.nan, "ks_p": np.nan,
                "nan_mask_train": tr[feat].isna().any(), "const_on_train": False,
                "status": "OK", "reason": "too_few_train", "score": 0.0,
                "mode_frac": 0.0, "zero_frac": 0.0, "top_bin_share": 0.0,
            })
            continue

        # medians and denom
        med_tr = float(np.nanmedian(tr_s)) if len(tr_s) else np.nan
        med_val = float(np.nanmedian(vl_s)) if len(vl_s) else np.nan
        med_te = float(np.nanmedian(te_s)) if len(te_s) else np.nan
        denom = safe_denom(tr_s.to_numpy()) if len(tr_s) else denom_eps

        pct_shift_val = abs(med_val - med_tr) / denom if not np.isnan(med_tr) else 0.0
        pct_shift_te = abs(med_te - med_tr) / denom if not np.isnan(med_tr) else 0.0
        abs_shift_val = abs(med_val - med_tr)
        abs_shift_te = abs(med_te - med_tr)

        # percentiles with tiny-tail fallback
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

        # simple diagnostics
        mode_frac = 0.0
        zero_frac = 0.0
        top_bin_share = 0.0
        spread_score = 0.0

        if len(tr_s) > 0:
            mode_frac = float(tr_s.value_counts(normalize=True).iloc[0])
            zero_frac = float(((tr_s.abs() <= zero_tol)).sum()) / len(tr_s)

            if not (np.isnan(q01) or np.isnan(q99) or q99 <= q01):
                edges = np.linspace(q01, q99, conc_bins + 1)
                h, _ = np.histogram(tr_s, bins=edges)
                total = h.sum()
                p = h / total if total > 0 else np.zeros_like(h)
                top_bin_share = float(p.max()) if p.size else 0.0
                p_nz = p[p > 0]
                if p_nz.size:
                    entropy = -float((p_nz * np.log(p_nz)).sum())
                    max_entropy = np.log(len(p)) if len(p) > 0 else 1.0
                    spread_score = float(np.clip(entropy / max_entropy, 0.0, 1.0))
                else:
                    spread_score = 0.0
            else:
                top_bin_share = 1.0 if tr_s.nunique() == 1 else 0.0
                spread_score = 0.0 if top_bin_share == 1.0 else 1.0

        # failure tests
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
            ks_p = np.nan
            ks_fail = False

        const_on_train = False
        if len(tr_s) and np.isfinite(tr_s.min()) and np.isfinite(tr_s.max()):
            const_tol = max(const_eps, denom * 1e-12)
            const_on_train = (abs(tr_s.max() - tr_s.min()) < const_tol)
        const_fail = const_on_train

        ks_support = ks_fail and (pct_shift_fail or frac_out_fail)
        
        # voting: require at least two failing signals to drop
        vote = int(pct_shift_fail) + int(frac_out_fail) + int(const_fail) + int(ks_support)
        drop_flag = (vote >= min_failures)
        
        drop_reasons: List[str] = []
        if pct_shift_fail:
            drop_reasons.append(f"pct_shift_val={pct_shift_val:.3f};te={pct_shift_te:.3f}")
        if frac_out_fail:
            drop_reasons.append(f"frac_out_val={frac_val_out:.3f};te={frac_te_out:.3f}")
        if ks_support:
            drop_reasons.append(f"ks_p={ks_p:.4g}")
        if const_fail:
            drop_reasons.append("constant_on_train")
        
        status = "DROP" if drop_flag else "OK"
        reason = "; ".join(drop_reasons)

        rows.append({
            "feature": feat,
            "q01": q01, "q25": q25, "q75": q75, "q99": q99,
            "full_span": full_span, "iqr": iqr,
            "med_tr": med_tr, "med_val": med_val, "med_te": med_te,
            "pct_shift_val": pct_shift_val, "pct_shift_te": pct_shift_te,
            "abs_shift_val": abs_shift_val, "abs_shift_te": abs_shift_te,
            "frac_val_out": frac_val_out, "frac_te_out": frac_te_out,
            "ks_p": ks_p,
            "nan_mask_train": tr[feat].isna().any(),
            "const_on_train": const_on_train,
            "status": status,
            "reason": reason,
            "score": spread_score,
            "mode_frac": float(mode_frac),
            "zero_frac": float(zero_frac),
            "top_bin_share": float(top_bin_share),
        })

    diag = pd.DataFrame(rows).set_index("feature").sort_values(["status"], ascending=[True])
    to_drop = diag[diag["status"] == "DROP"].index.tolist()
    df_pruned = df_unsc.drop(columns=to_drop, errors="ignore")
    return df_pruned, to_drop, diag

    
############################################################################ 


def assign_percentiles_from_diag(
    diag: pd.DataFrame,
    custom_range: Tuple[float, float],
    standard_range: Tuple[float, float],
    base_range: Tuple[float, float],
    narrow_thresh: float
) -> pd.DataFrame:
    """
    Assign final percentile pairs from diagnostics.

    Inputs
      - diag: DataFrame indexed by feature containing at least these columns:
          score, zero_frac, mode_frac, frac_in, top_bin_share
      - custom_range: percentile pair assigned to "kept_narrow" features (e.g. (30,70))
      - standard_range: percentile pair assigned to all other features by default
      - base_range: percentile pair assigned to features with names ending in '_time' and to the target signal

    Behavior
      - zero_mass := (zero_frac >= 0.05) and (mode_frac >= 0.05)
      - center_dom := (frac_in >= 0.60) and (top_bin_share >= 0.35)
      - excluded_local := zero_mass OR center_dom  -> assigned "excluded"
      - kept_narrow := not excluded_local AND score > 0 -> assigned custom_range
      - otherwise -> assigned standard_range
      - features ending with '_time' and target signal get base_range regardless of other rules
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
    base_lo, base_hi = _norm_range(base_range)

    low_list, high_list, reason_list = [], [], []

    for feat, r in out.iterrows():
        # feature-level diagnostics
        mode_frac = r.get("mode_frac", 0.0)
        zero_frac = r.get("zero_frac", 0.0)
        frac_in = r.get("frac_in", 0.0)
        top_bin = r.get("top_bin_share", 0.0)
        score = r.get("score", 0.0)

        # time features override everything
        if str(feat).endswith("_time"):
            lo, hi = base_lo, base_hi
            assigned = "time_feature"
        elif str(feat) == str(params.label_col):
            lo, hi = base_lo, base_hi
            assigned = "target_signal"
        else:
            zero_mass_flag = (zero_frac >= 0.05) and (mode_frac >= 0.05)
            center_dom_flag = (frac_in >= 0.60) and (top_bin >= 0.35)
            excluded_local = bool(zero_mass_flag or center_dom_flag)

            if excluded_local:
                lo, hi = standard_lo, standard_hi
                assigned = "excluded"
            elif top_bin > narrow_thresh:
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
        return out[feat_cols].astype(np.float32)

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

    df_out = df.copy()  # single output copy

    # assign scaled blocks (they are DataFrames with the same index and feat_cols)
    # use get_indexer once for efficiency
    col_idx = df_out.columns.get_indexer(feat_cols)
    df_out.iloc[tr_slice, col_idx] = tr_scaled.to_numpy()
    df_out.iloc[val_slice, col_idx] = v_scaled.to_numpy()
    df_out.iloc[te_slice, col_idx] = te_scaled.to_numpy()

    # free intermediates and force garbage collection
    del tr_scaled, v_scaled, te_scaled, df_tr
    gc.collect()

    return df_out.sort_index()


#########################################################################################################


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


