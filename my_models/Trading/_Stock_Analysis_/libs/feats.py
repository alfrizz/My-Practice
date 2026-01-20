from libs import plots, params

from typing import Sequence, List, Tuple, Optional, Union, Dict, Iterable
Number = Union[int, float]

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
from pathlib import Path

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
    out["time_minute"] = (((_single_centered_phase_np(minutes, sess_start_min, 1440.0)) + 1.0) / 2.0).astype(np.float32)
    out["time_hour"]   = (((_single_centered_phase_np(hour_idx, sess_start_hr, 24.0)) + 1.0) / 2.0).astype(np.float32)
    out["time_dow"]         = (((_single_centered_phase_np(dow, 0.0, 7.0)) + 1.0) / 2.0).astype(np.float32)
    out["time_month"]       = (((_single_centered_phase_np(month0, 0.0, 12.0)) + 1.0) / 2.0).astype(np.float32)
    out["time_day_of_year"] = (((_single_centered_phase_np(doy, 0.0, 365.0)) + 1.0) / 2.0).astype(np.float32)
    out["time_week_of_year"] = (((_single_centered_phase_np(week0, 0.0, 52.0)) + 1.0) / 2.0).astype(np.float32)
    out["time_in_sess"] = ((df.index.time >= params.sess_start_reg) & (df.index.time < params.sess_end)).astype(np.float32)
    out["time_premark"] = (df.index.time < params.sess_start_reg).astype(np.float32)
    out["time_afthour"] = (df.index.time >= params.sess_end).astype(np.float32)

    return out


###########################################


def standard_indicators(
    df: pd.DataFrame,
    *,
    sma: Optional[List[int]] = None,
    ema: Optional[List[int]] = None,
    atr: Optional[List[int]] = None,
    bb: Optional[Union[List[int], Dict[int, List[Number]]]] = None,  # window -> list of devs
    rsi: Optional[List[int]] = None,
    roc: Optional[List[int]] = None,
    vol_spike: Optional[List[int]] = None,
    obv_roll: Optional[List[int]] = None,
    ret_std: Optional[List[int]] = None,
    macd: Optional[List[Dict[str, int]]] = None,  # each dict: fast, slow, signal
    stoch: Optional[List[Dict[str, int]]] = None,  # each dict: k, d (signal), smooth
    cci: Optional[List[int]] = None,
    mfi: Optional[List[int]] = None,
    cmf: Optional[List[int]] = None,
    donch: Optional[List[int]] = None,  # Donchian channel window
    roll_vwap: Optional[List[int]] = None,  # rolling VWAP window
    linreg_slope: Optional[List[int]] = None,  # slope of close over window
    kc: Optional[List[Dict[str, Number]]] = None,  # Keltner: ema_window, atr_window, atr_mult
    psar: Optional[Dict[str, Number]] = None,  # step, max_step
) -> pd.DataFrame:
    """
    Compute OHLCV indicators suitable for intraday/scalping & ML features.

    Conventions:
      - Windowed indicators use list[int].
      - BB can be list[int] (dev=2.0) or dict {window: [devs...]}.
      - MACD is list of dicts with fast/slow/signal.
      - Stoch is list of dicts with k, d, smooth.
      - Keltner uses a custom implementation with atr_mult; pass dicts.

    Outputs use descriptive suffixes, e.g.:
      sma_9, ema_21, atr_14, rsi_6, roc_5,
      bb_lband_20_2, bb_hband_20_2, bb_w_20_2,
      macd_line_6_13_5, macd_signal_6_13_5, macd_diff_6_13_5,
      stoch_k_14_3_3, stoch_d_14_3_3,
      cci_20, mfi_14, cmf_20,
      donch_h_20, donch_l_20, donch_w_20,
      roll_vwap_20, slope_close_20,
      kc_mid_20_20, kc_l_20_20_1.5, kc_h_20_20_1.5, kc_w_20_20_1.5,
      psar, psar_dir.
    """

    macd = macd or []
    stoch = stoch or []
    kc = kc or []
    psar = psar or {}

    def norm_windows(x):
        return sorted({int(i) for i in x}) if x else []

    sma_ws = norm_windows(sma)
    ema_ws = norm_windows(ema)
    atr_ws = norm_windows(atr)
    rsi_ws = norm_windows(rsi)
    roc_ws = norm_windows(roc)
    vol_ws = norm_windows(vol_spike)
    obv_roll_ws = norm_windows(obv_roll)
    ret_std_ws = norm_windows(ret_std)
    cci_ws = norm_windows(cci)
    mfi_ws = norm_windows(mfi)
    cmf_ws = norm_windows(cmf)
    donch_ws = norm_windows(donch)
    roll_vwap_ws = norm_windows(roll_vwap)
    slope_ws = norm_windows(linreg_slope)

    # BB normalization
    bb_items = []
    if bb:
        if isinstance(bb, dict):
            for w, devs in bb.items():
                bb_items.append((int(w), [float(d) for d in devs]))
        else:  # list of ints
            for w in norm_windows(bb):
                bb_items.append((w, [2.0]))
        bb_items.sort(key=lambda t: t[0])

    # Prepare df
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    EPS = 1e-9

    def safe_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return a / (b.replace(0, np.nan) if isinstance(b, pd.Series) else b + EPS)

    def linreg_series(series: pd.Series, window: int) -> pd.Series:
        idx = np.arange(window)
        denom = (idx - idx.mean()) @ (idx - idx.mean())
        def slope(x):
            y = x.values
            return ((idx - idx.mean()) @ (y - y.mean())) / denom if denom != 0 else np.nan
        return series.rolling(window, min_periods=window).apply(slope, raw=False)

    # Progress bar
    total_windows = (
        len(sma_ws) + len(ema_ws) + len(atr_ws) + len(rsi_ws) +
        len(roc_ws) + len(vol_ws) + len(obv_roll_ws) + len(ret_std_ws) +
        len(cci_ws) + len(mfi_ws) + len(cmf_ws) + len(donch_ws) +
        len(roll_vwap_ws) + len(slope_ws)
    )
    single_shot = ("returns", "geometry", "obv", "rolling_extrema", "vwap_session")
    pbar = tqdm(total=total_windows + len(single_shot) + len(bb_items) +
                len(macd) + len(stoch) + len(kc) + (1 if psar else 0),
                desc="Indicators", unit="task")

    new = {}

    # returns
    pbar.set_description("ret/log_ret")
    new["ret"] = c.pct_change()
    new["log_ret"] = np.log(c + EPS).diff()
    pbar.update(1)

    # SMA
    for w in sma_ws:
        pbar.set_description(f"SMA {w}")
        s = c.rolling(w, min_periods=1).mean()
        new[f"sma_{w}"] = s
        new[f"sma_pct_{w}"] = safe_div(c - s, s)
        pbar.update(1)

    # EMA
    for w in ema_ws:
        pbar.set_description(f"EMA {w}")
        new[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()
        pbar.update(1)

    # ROC
    for w in roc_ws:
        pbar.set_description(f"ROC {w}")
        new[f"roc_{w}"] = c.pct_change(w)
        pbar.update(1)

    # geometry
    pbar.set_description("geometry")
    new["body"] = c - o
    new["body_pct"] = safe_div(c - o, o)
    new["upper_shad"] = h - np.maximum(o, c)
    new["lower_shad"] = np.minimum(o, c) - l
    new["range_pct"] = safe_div(h - l, c)
    pbar.update(1)

    # RSI
    for w in rsi_ws:
        pbar.set_description(f"RSI {w}")
        new[f"rsi_{w}"] = ta.momentum.RSIIndicator(close=c, window=w).rsi()
        pbar.update(1)

    # MACD(s)
    for cfg in macd:
        fast = int(cfg["fast"]); slow = int(cfg["slow"]); sig = int(cfg["signal"])
        pbar.set_description(f"MACD {fast}/{slow}/{sig}")
        macd_obj = ta.trend.MACD(close=c, window_fast=fast, window_slow=slow, window_sign=sig)
        triple = f"{fast}_{slow}_{sig}"
        new[f"macd_line_{triple}"] = macd_obj.macd()
        new[f"macd_signal_{triple}"] = macd_obj.macd_signal()
        new[f"macd_diff_{triple}"] = macd_obj.macd_diff()
        pbar.update(1)

    # ATR and ADX
    for w in atr_ws:
        pbar.set_description(f"ATR {w}")
        atr_s = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=w).average_true_range()
        new[f"atr_{w}"] = atr_s
        new[f"atr_pct_{w}"] = safe_div(atr_s, c)
        pbar.update(1)

    for w in atr_ws:
        pbar.set_description(f"ADX {w}")
        adx = ta.trend.ADXIndicator(high=h, low=l, close=c, window=w)
        new[f"plus_di_{w}"] = adx.adx_pos()
        new[f"minus_di_{w}"] = adx.adx_neg()
        new[f"adx_{w}"] = adx.adx()
        pbar.update(1)

    # Bollinger Bands
    for w, devs in bb_items:
        for dev in devs:
            pbar.set_description(f"BB {w} dev={dev}")
            bb_obj = ta.volatility.BollingerBands(close=c, window=int(w), window_dev=float(dev))
            dev_suffix = f"{dev}".replace(".", "p")
            lb = bb_obj.bollinger_lband(); hb = bb_obj.bollinger_hband(); ma = bb_obj.bollinger_mavg()
            new[f"bb_lband_{w}_{dev_suffix}"] = lb
            new[f"bb_hband_{w}_{dev_suffix}"] = hb
            new[f"bb_w_{w}_{dev_suffix}"] = safe_div(hb - lb, ma)
        pbar.update(1)

    # OBV
    pbar.set_description("OBV")
    new["obv"] = ta.volume.OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()
    pbar.update(1)

    # vol_spike
    for w in vol_ws:
        pbar.set_description(f"Vol {w}")
        vol_roll = v.rolling(w, min_periods=w).mean().fillna(EPS)
        new[f"vol_spike_{w}"] = safe_div(v, vol_roll)
        pbar.update(1)

    # rolling extrema & distances (largest SMA if present)
    pbar.set_description("rolling_extrema")
    long_w = max(sma_ws) if sma_ws else None
    if long_w:
        new[f"rolling_max_close_{long_w}"] = c.rolling(long_w, min_periods=1).max()
        new[f"rolling_min_close_{long_w}"] = c.rolling(long_w, min_periods=1).min()
        new[f"dist_high_{long_w}"] = safe_div(new[f"rolling_max_close_{long_w}"] - c, c)
        new[f"dist_low_{long_w}"] = safe_div(c - new[f"rolling_min_close_{long_w}"], c)
    pbar.update(1)

    # OBV-derived percent changes
    for w in obv_roll_ws:
        pbar.set_description(f"OBV pct {w}")
        new[f"obv_diff_{w}"] = new["obv"].diff(w)
        denom = c.rolling(w, min_periods=w).mean().abs().fillna(EPS).replace(0.0, EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = new[f"obv_diff_{w}"].to_numpy(dtype=float) / denom.to_numpy(dtype=float)
        new[f"obv_pct_{w}"] = pd.Series(np.where(np.isnan(pct), np.nan, pct), index=df.index)
        pbar.update(1)

    # ret_std
    for w in ret_std_ws:
        pbar.set_description(f"ret_std {w}")
        new[f"ret_std_{w}"] = new["ret"].rolling(w, min_periods=w).std()
        pbar.update(1)

    # Stochastic %K/%D (root fix: EPS guard + min_periods=1)
    for cfg in stoch:
        k = int(cfg["k"]); d = int(cfg.get("d", 3)); smooth = int(cfg.get("smooth", 3))
        pbar.set_description(f"Stoch {k}/{d}/{smooth}")
        low_k = l.rolling(window=k, min_periods=1).min()
        high_k = h.rolling(window=k, min_periods=1).max()
        # guard denominator with tiny EPS to avoid divide-by-zero
        denom = (high_k - low_k).replace(0.0, EPS)
        sk_raw = (c - low_k) / denom
        sk = sk_raw.rolling(window=smooth, min_periods=1).mean()
        sd = sk.rolling(window=d, min_periods=1).mean()
        # keep values in expected range
        sk = sk.clip(lower=0.0, upper=1.0)
        sd = sd.clip(lower=0.0, upper=1.0)
        new[f"stoch_k_{k}_{d}_{smooth}"] = sk
        new[f"stoch_d_{k}_{d}_{smooth}"] = sd
        pbar.update(1)

    # CCI
    for w in cci_ws:
        pbar.set_description(f"CCI {w}")
        tp = (h + l + c) / 3.0
        sma = tp.rolling(window=w, min_periods=1).mean()
        mad = (tp - sma).abs().rolling(window=w, min_periods=1).mean().replace(0.0, EPS)
        new[f"cci_{w}"] = ((tp - sma) / (0.015 * mad)).clip(-500.0, 500.0)
        pbar.update(1)

    # MFI
    for w in mfi_ws:
        pbar.set_description(f"MFI {w}")
        tp = (h + l + c) / 3.0
        mf = tp * v
        pos = mf.where(tp.diff() > 0, 0.0).rolling(w, min_periods=1).sum()
        neg = (-mf).where(tp.diff() < 0, 0.0).rolling(w, min_periods=1).sum().replace(0.0, EPS)
        new[f"mfi_{w}"] = (100.0 - 100.0 / (1.0 + pos / neg)).clip(0.0, 100.0)
        pbar.update(1)

    # CMF
    for w in cmf_ws:
        pbar.set_description(f"CMF {w}")
        new[f"cmf_{w}"] = ta.volume.ChaikinMoneyFlowIndicator(high=h, low=l, close=c, volume=v, window=w).chaikin_money_flow()
        pbar.update(1)

    # Donchian channels
    for w in donch_ws:
        pbar.set_description(f"Donch {w}")
        hband = h.rolling(w, min_periods=w).max()
        lband = l.rolling(w, min_periods=w).min()
        new[f"donch_h_{w}"] = hband
        new[f"donch_l_{w}"] = lband
        new[f"donch_w_{w}"] = safe_div(hband - lband, c)
        pbar.update(1)

    # Rolling VWAP (windowed)
    for w in roll_vwap_ws:
        pbar.set_description(f"rVWAP {w}")
        pv = (c * v).rolling(w, min_periods=w).sum()
        volw = v.rolling(w, min_periods=w).sum().replace(0, np.nan)
        new[f"roll_vwap_{w}"] = pv / volw
        pbar.update(1)

    # Linear regression slope of close
    for w in slope_ws:
        pbar.set_description(f"slope {w}")
        new[f"slope_close_{w}"] = linreg_series(c, w)
        pbar.update(1)

    # Keltner Channel (custom: ema_window, atr_window, atr_mult)
    for cfg in kc:
        ema_w = int(cfg.get("ema_window", 20))
        atr_w = int(cfg.get("atr_window", ema_w))
        mult = float(cfg.get("atr_mult", 1.5))
        pbar.set_description(f"KC ema{ema_w}/atr{atr_w}*{mult}")
        ema_mid = c.ewm(span=ema_w, adjust=False).mean()
        atr_s = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=atr_w).average_true_range()
        lband = ema_mid - mult * atr_s
        hband = ema_mid + mult * atr_s
        suffix = f"{ema_w}_{atr_w}_{mult}"
        new[f"kc_mid_{suffix}"] = ema_mid
        new[f"kc_l_{suffix}"] = lband
        new[f"kc_h_{suffix}"] = hband
        new[f"kc_w_{suffix}"] = safe_div(hband - lband, ema_mid)
        pbar.update(1)

    # PSAR
    if psar:
        pbar.set_description("PSAR")
        step = float(psar.get("step", 0.02))
        max_step = float(psar.get("max_step", 0.2))
        psar_ind = ta.trend.PSARIndicator(high=h, low=l, close=c, step=step, max_step=max_step)
        psar_series = psar_ind.psar()
        new["psar"] = psar_series

        pbar.update(1)

    # VWAP session (daily)
    pbar.set_description("vwap_session")
    pv = (c * v).groupby(df.index.date).cumsum()
    vol_cum = v.groupby(df.index.date).cumsum().replace(0, np.nan)
    new["vwap_ohlc_close_session"] = (pv / vol_cum).reindex(df.index)
    pbar.update(1)

    pbar.set_description("finalizing")
    pbar.close()

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


##########################################################################################################


# def engineered_indicators(
#     df: pd.DataFrame,
#     rsi_low: float = 30.0,
#     rsi_high: float = 70.0,
#     adx_thr: float = 20.0,
#     mult_w: int =7,
#     eps: float = 1e-9,
#     fillna_zero: bool = True,
#     small_factor: float = 1e-3,
#     ratio_clip_abs: float = 1e6,
#     z_std_floor_factor: float = 1e-3
# ) -> pd.DataFrame:
#     """
#     Build engineered signals from base indicators (uses only past/present values).
#     - Works with canonical columns from standard_indicators and their windowed copies.
#     - Detects relevant columns robustly (e.g. 'sma_14', 'sma_60', 'macd_diff', 'macd_diff_30').
#     - Uses safe denominators and std floors to ensure finite outputs.
#     Returns DataFrame of engineered columns (float, NaNs filled per fillna_zero).
#     """
#     out = pd.DataFrame(index=df.index)
#     produced: List[str] = []

#     def _find(prefix: str) -> Optional[str]:
#         # find first column that startswith prefix (prefer exact canonical name before windowed)
#         if prefix in df.columns:
#             return prefix
#         for c in df.columns:
#             if c.startswith(prefix):
#                 return c
#         return None

#     def _std_floor(series: pd.Series, window: int):
#         rstd = series.rolling(window, min_periods=1).std()
#         arr = series.to_numpy(dtype=float)
#         finite = arr[np.isfinite(arr)]
#         global_std = float(np.nanstd(finite)) if finite.size else eps
#         floor = max(eps, abs(global_std) * z_std_floor_factor)
#         return rstd.fillna(floor).replace(0.0, floor)

#     # helpers for column detection
#     close = df.get("close", pd.Series(index=df.index, dtype=float)).astype(float)

#     # robust SMA detection: pick two smallest numeric windows if available
#     sma_cols = sorted([c for c in df.columns if re.match(r"^sma_\d+$", c)],
#                       key=lambda c: int(re.search(r"_(\d+)$", c).group(1)))
#     sma_s = sma_cols[0] if sma_cols else None
#     sma_l = sma_cols[1] if len(sma_cols) > 1 else None


#     # find other columns (prefer canonical names, fallback to first matching windowed)
#     macd_diff = _find("macd_diff")
#     bb_l = _find("bb_lband")
#     bb_h = _find("bb_hband")
#     bb_w = _find("bb_w")
#     rsi_col = _find("rsi")
#     plus_di = _find("plus_di")
#     minus_di = _find("minus_di")
#     adx_col = _find("adx")
#     obv_diff = _find("obv_diff")
#     obv_sma = _find("obv_sma")
#     vwap_dev = _find("vwap_dev") or _find("vwap")
#     atr_pct = _find("atr_pct")

#     # eng_ma
#     if sma_s and sma_l and sma_s in df.columns and sma_l in df.columns:
#         out["eng_ma"] = (df[sma_s].astype(float) - df[sma_l].astype(float)) / (df[sma_l].astype(float) + eps)
#         produced.append("eng_ma")

#     # eng_macd
#     if macd_diff and sma_l and macd_diff in df.columns and sma_l in df.columns:
#         out["eng_macd"] = df[macd_diff].astype(float) / (df[sma_l].astype(float) + eps)
#         produced.append("eng_macd")

#     # eng_bb / mid
#     if bb_l and bb_h and bb_w and bb_l in df.columns and bb_h in df.columns and bb_w in df.columns:
#         lo = df[bb_l].astype(float); hi = df[bb_h].astype(float); bw = df[bb_w].astype(float)
#         dev = np.where(close < lo, lo - close, np.where(close > hi, close - hi, 0.0))
#         out["eng_bb"] = pd.Series(dev, index=df.index) / (bw + eps)
#         out["eng_bb_mid"] = ((lo + hi) / 2 - close) / (close + eps)
#         produced.extend(["eng_bb", "eng_bb_mid"])

#     # eng_rsi
#     if rsi_col and rsi_col in df.columns:
#         rv = df[rsi_col].astype(float)
#         low_d = np.clip((rsi_low - rv), 0, None) / 100.0
#         high_d = np.clip((rv - rsi_high), 0, None) / 100.0
#         out["eng_rsi"] = np.where(rv < rsi_low, low_d, np.where(rv > rsi_high, high_d, 0.0))
#         produced.append("eng_rsi")

#     # eng_adx
#     if plus_di and minus_di and adx_col and plus_di in df.columns and minus_di in df.columns and adx_col in df.columns:
#         di_diff = df[plus_di].astype(float) - df[minus_di].astype(float)
#         diff_abs = di_diff.abs() / 100.0
#         ex = np.clip((df[adx_col].astype(float) - adx_thr) / 100.0, 0, None)
#         out["eng_adx"] = np.sign(di_diff) * diff_abs * ex
#         produced.append("eng_adx")

#     # eng_obv (safe denom + winsorized numerator)
#     if obv_diff and obv_sma and obv_diff in df.columns and obv_sma in df.columns:
#         num = df[obv_diff].astype(float).to_numpy(dtype=float)
#         try:
#             finite = num[np.isfinite(num)]
#             lo_cut = np.nanpercentile(finite, 0.5) if finite.size else np.nan
#             hi_cut = np.nanpercentile(finite, 99.5) if finite.size else np.nan
#             num_clipped = np.copy(num)
#             mask_num = np.isnan(num_clipped)
#             num_clipped = np.clip(num_clipped, lo_cut, hi_cut)
#             num_clipped[mask_num] = np.nan
#         except Exception:
#             num_clipped = num

#         den_s = df[obv_sma].rolling(window=mult_w, min_periods=mult_w).median().ffill().fillna(0.0)
#         den_arr = den_s.to_numpy(dtype=float)
#         den_floor = np.maximum(np.abs(den_arr) * small_factor, eps)
#         finite_nonzero = np.isfinite(den_arr) & (np.abs(den_arr) > 0.0)
#         den_safe = np.where(finite_nonzero, np.sign(den_arr) * np.maximum(np.abs(den_arr), den_floor), den_floor)

#         with np.errstate(divide='ignore', invalid='ignore'):
#             ratio = num_clipped / den_safe
#         ratio = np.where(np.isnan(num_clipped), np.nan, np.clip(ratio, -ratio_clip_abs, ratio_clip_abs))
#         out["eng_obv"] = pd.Series(ratio, index=df.index).astype(float)
#         produced.append("eng_obv")

#     # eng_atr_div, z_eng_atr
#     if atr_pct and atr_pct in df.columns:
#         ratio = df[atr_pct].astype(float)
#         rm = ratio.rolling(mult_w, min_periods=mult_w).mean()
#         out["eng_atr_div"] = (ratio - rm) * 10_000
#         std_r = ratio.rolling(mult_w, min_periods=mult_w).std()
#         global_std = np.nanstd(ratio.to_numpy(dtype=float))
#         std_floor = max(eps, abs(global_std) * z_std_floor_factor)
#         std_eff = std_r.fillna(std_floor).replace(0.0, std_floor)
#         out["z_eng_atr"] = (ratio - rm) / (std_eff + eps)
#         produced.extend(["eng_atr_div", "z_eng_atr"])

#     # eng_sma distances
#     if sma_s and sma_s in df.columns:
#         out["eng_sma_short"] = (df[sma_s].astype(float) - close) / (close + eps)
#         produced.append("eng_sma_short")
#     if sma_l and sma_l in df.columns:
#         out["eng_sma_long"] = (df[sma_l].astype(float) - close) / (close + eps)
#         produced.append("eng_sma_long")

#     # eng_vwap and z — prefer canonical vwap_dev_pct if present
#     vwap_col = _find("vwap_dev_pct") or _find("vwap")
#     if vwap_col and vwap_col in df.columns:
#         if vwap_col == "vwap_dev_pct":
#             eng_vwap_pct = df[vwap_col].astype(float)
#         else:
#             vwap_base = df[vwap_col].astype(float)
#             eng_vwap_pct = 100.0 * (vwap_base - close) / (close + eps)
#         out["eng_vwap"] = eng_vwap_pct
#         znum = eng_vwap_pct
#         zm = znum.rolling(mult_w, min_periods=mult_w).mean()
#         zs = znum.rolling(mult_w, min_periods=mult_w).std()
#         arr = znum.to_numpy(dtype=float)
#         finite = arr[np.isfinite(arr)]
#         global_std = float(np.nanstd(finite)) if finite.size else eps
#         std_floor = max(eps, abs(global_std) * z_std_floor_factor)
#         zs_eff = zs.fillna(std_floor).replace(0.0, std_floor)
#         out["z_vwap_dev"] = (znum - zm) / (zs_eff + eps)
#         produced.extend(["eng_vwap", "z_vwap_dev"])

#     # z_bb_w
#     if bb_w and bb_w in df.columns:
#         x = df[bb_w].astype(float)
#         xm = x.rolling(mult_w, min_periods=mult_w).mean()
#         xs = x.rolling(mult_w, min_periods=mult_w).std()
#         global_std = np.nanstd(x.to_numpy(dtype=float))
#         std_floor = max(eps, abs(global_std) * z_std_floor_factor)
#         xs_eff = xs.fillna(std_floor).replace(0.0, std_floor)
#         out["z_bb_w"] = (x - xm) / (xs_eff + eps)
#         produced.append("z_bb_w")

#     # strict z of raw OBV (prefer this for scale-free ML input)
#     if "obv" in df.columns:
#         x = df["obv"].astype(float)
#         xm = x.rolling(mult_w, min_periods=mult_w).mean()
#         xs = x.rolling(mult_w, min_periods=mult_w).std()
#         arr = x.to_numpy(dtype=float)
#         finite = arr[np.isfinite(arr)]
#         global_std = float(np.nanstd(finite)) if finite.size else eps
#         std_floor = max(eps, abs(global_std) * z_std_floor_factor)
#         xs_eff = xs.fillna(std_floor).replace(0.0, std_floor)
#         out["z_obv"] = (x - xm) / (xs_eff + eps)
#         produced.append("z_obv")

#     # momentum aggregates: detect available ret_{H} columns (use any found in df)
#     ret_cols = [c for c in df.columns if re.match(r"^ret_(\d+)$", c)]
#     horizons = sorted({int(re.search(r"^ret_(\d+)$", c).group(1)) for c in ret_cols}) if ret_cols else [1, 5, 15, 60]
#     for H in horizons:
#         ret_col = f"ret_{H}"
#         if ret_col in df.columns:
#             out[f"mom_sum_{H}"] = df[ret_col].rolling(H, min_periods=H).sum()
#             out[f"mom_std_{H}"] = df[ret_col].rolling(H, min_periods=H).std()
#             produced.extend([f"mom_sum_{H}", f"mom_std_{H}"])

#     # ema cross flags: find sma numeric windows used for ema names
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

#     # finalize: ensure order, types and optional fillna
#     produced = [p for p in produced if p in out.columns]
#     for col in produced:
#         out[col] = out[col].astype(float)
#         if fillna_zero:
#             out[col] = out[col].fillna(0.0)

#     return out[produced].copy()


########################################################################################################## 


def flag_indicators(
    df: pd.DataFrame,
    train_prop: float = 0.7,
    pct_shift_thresh: float = 0.16,     # relative median shift threshold
    frac_outside_thresh: float = 0.06,  # fraction outside train 1–99% range
    min_train_samples: int = 50,
    na_rate_thresh: float = 0.4,        # flag if >40% NaN in train or overall
    const_tol: float = 1e-12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inspect numeric features and flag simple issues.

    Returns (df_out, diag):
      - diag: one row per numeric feature with status in {FEW_TRAIN, CONST, HIGH_NA, DRIFT, OK}
      - df_out: copy of input where every feature flagged DRIFT is replaced by "<feature>_DRIFT"
              (original column dropped and duplicate named "<feature>_DRIFT" kept).
    Notes:
      - Splits df into contiguous train/val/test slices (no shuffling).
      - This function only flags and reorganizes columns; it does not compute RZ.
    """
    if len(df) == 0:
        return df.copy(), pd.DataFrame()

    # contiguous train/val/test split
    n_tr = int(len(df) * train_prop)
    n_val = (len(df) - n_tr) // 2
    tr = df.iloc[:n_tr].replace([np.inf, -np.inf], np.nan)
    vl = df.iloc[n_tr:n_tr + n_val].replace([np.inf, -np.inf], np.nan)
    te = df.iloc[n_tr + n_val:].replace([np.inf, -np.inf], np.nan)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def robust_scale(arr: np.ndarray) -> float:
        """Robust denominator for relative shifts: max(|med|, MAD, std, tiny)."""
        if arr.size == 0:
            return 1e-6
        med = float(np.nanmedian(arr))
        mad = float(np.nanmedian(np.abs(arr - med)))
        std = float(np.nanstd(arr))
        return max(abs(med), mad, std, 1e-6)

    def frac_outside(s: pd.Series, q01, q99) -> float:
        s2 = s.dropna()
        if len(s2) == 0 or np.isnan(q01) or np.isnan(q99):
            return 0.0
        return float(((s2 < q01) | (s2 > q99)).mean())

    rows = []
    # iterate with progress bar for visibility
    for feat in tqdm(numeric_cols, desc="Flagging indicators", unit="feat"):
        tr_s = tr[feat].dropna()
        vl_s = vl[feat].dropna()
        te_s = te[feat].dropna()

        na_rate_train = float(tr[feat].isna().mean())
        na_rate_all = float(df[feat].isna().mean())

        # Priority 1: too few train samples
        if len(tr_s) < min_train_samples:
            rows.append({
                "feature": feat, "status": "FEW_TRAIN", "reason": "too_few_train",
                "pct_shift_val": np.nan, "pct_shift_te": np.nan,
                "frac_val_out": np.nan, "frac_te_out": np.nan,
                "na_rate_train": na_rate_train, "na_rate_all": na_rate_all
            })
            continue

        # Priority 2: constant on train
        const_fail = (tr_s.max() - tr_s.min()) < const_tol
        if const_fail:
            rows.append({
                "feature": feat, "status": "CONST", "reason": "constant_train",
                "pct_shift_val": np.nan, "pct_shift_te": np.nan,
                "frac_val_out": np.nan, "frac_te_out": np.nan,
                "na_rate_train": na_rate_train, "na_rate_all": na_rate_all
            })
            continue

        # Compute drift stats (median shifts relative to robust scale)
        med_tr = float(np.nanmedian(tr_s))
        med_val = float(np.nanmedian(vl_s)) if len(vl_s) else np.nan
        med_te = float(np.nanmedian(te_s)) if len(te_s) else np.nan
        denom = robust_scale(tr_s.to_numpy())

        pct_shift_val = abs(med_val - med_tr) / denom if not np.isnan(med_val) else 0.0
        pct_shift_te = abs(med_te - med_tr) / denom if not np.isnan(med_te) else 0.0

        # train tails (fallback to 5/95 if too few tail samples)
        q01 = np.nanpercentile(tr_s, 1)
        q99 = np.nanpercentile(tr_s, 99)
        if (tr_s <= q01).sum() < 3:
            q01 = np.nanpercentile(tr_s, 5)
        if (tr_s >= q99).sum() < 3:
            q99 = np.nanpercentile(tr_s, 95)

        frac_val_out = frac_outside(vl_s, q01, q99)
        frac_te_out = frac_outside(te_s, q01, q99)

        drift_fail = (
            (pct_shift_val > pct_shift_thresh) or (pct_shift_te > pct_shift_thresh)
            or (frac_val_out > frac_outside_thresh) or (frac_te_out > frac_outside_thresh)
        )
        high_na = (na_rate_train > na_rate_thresh) or (na_rate_all > na_rate_thresh)

        if high_na:
            status = "HIGH_NA"
            reason = f"na_train={na_rate_train:.2f}; na_all={na_rate_all:.2f}"
        elif drift_fail:
            status = "DRIFT"
            reason = (
                f"pct_shift_val={pct_shift_val:.3f},te={pct_shift_te:.3f}; "
                f"frac_out_val={frac_val_out:.3f},te={frac_te_out:.3f}"
            )
        else:
            status = "OK"
            reason = ""

        rows.append({
            "feature": feat,
            "status": status,
            "reason": reason,
            "pct_shift_val": pct_shift_val,
            "pct_shift_te": pct_shift_te,
            "frac_val_out": frac_val_out,
            "frac_te_out": frac_te_out,
            "na_rate_train": na_rate_train,
            "na_rate_all": na_rate_all,
        })

    diag = pd.DataFrame(rows)

    # Build output df: replace DRIFT originals with <feature>_DRIFT and drop originals
    df_out = df.copy()
    drift_feats = diag.loc[diag["status"] == "DRIFT", "feature"].astype(str).tolist()
    for feat in tqdm(drift_feats, desc="Marking DRIFT columns", unit="feat"):
        if feat in df_out.columns:
            # create duplicate named "<feat>_DRIFT"
            df_out[f"{feat}_DRIFT"] = df_out[feat].copy()
            # drop original column
            df_out.drop(columns=[feat], inplace=True)

    return df_out, diag


########################################################################################################## 

 
def apply_rz_to_drifts(
    df: pd.DataFrame,
    diag: pd.DataFrame,
    rz_window: int = 60,
    min_periods: Optional[int] = None,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Convert selected DRIFT features to rolling robust z-scores and drop all _DRIFT columns.

    Steps:
      - For each feature flagged DRIFT in diag, look for <feature>_DRIFT in df.
      - If the feature matches the patterns (EMA/SMA/MACD/ATR/PSAR/VWAP/slopes/extrema), create <feature>_RZ
        using rolling median/MAD, then drop <feature>_DRIFT.
      - If it does not match, simply drop <feature>_DRIFT.
    Returns a new DataFrame.
    """
    df_out = df.copy()
    if min_periods is None:
        min_periods = max(1, rz_window // 3)

    def is_rz_feature(name: str) -> bool:
        # pattern matches for selected level-like indicators
        for pat in ("macd_line", "macd_signal", "macd_diff", "psar", "vwap_ohlc_close_session"):
            if name.startswith(pat) or name == pat:
                return True
        for prefix in ("ema_", "sma_", "atr_", "rolling_max_close", "rolling_min_close",
                       "roll_vwap", "vwap", "slope_close"):
            if name.startswith(prefix):
                return True
        return False

    drift_feats = diag.loc[diag["status"] == "DRIFT", "feature"].astype(str).tolist()

    for feat in tqdm(drift_feats, desc="Applying RZ to DRIFTs", unit="feat"):
        drift_col = f"{feat}_DRIFT"
        if drift_col not in df_out.columns:
            continue

        if is_rz_feature(feat):
            s = df_out[drift_col].astype('float32')
            if feat == "vwap_ohlc_close_session": # only indicator calculated per session, to be normalized by the close price
                s = ((s.astype(float) / df_out["close_raw"].astype(float)) - 1.0).astype('float32')
            med = s.rolling(window=rz_window, min_periods=min_periods).median()
            mad = (s - med).abs().rolling(window=rz_window, min_periods=min_periods).median()
            mad_safe = mad.replace(0, np.nan).fillna(eps)
            df_out[f"{feat}_RZ"] = (s - med) / mad_safe

        # always drop the _DRIFT column
        df_out.drop(columns=[drift_col], inplace=True)

    return df_out
    
########################################################################################################## 


def scale_features(
    df: pd.DataFrame,
    train_prop: float = 0.7,
    p_lo=1.0, 
    p_hi=99.0,
    include_rz: bool = False,  # False: skip *_RZ columns
    eps: float = 1e-12,        # tiny to avoid divide-by-zero if max==min
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train-based MinMax scaling to [0,1] on numeric columns.
    - Fits per-feature min/max on the train slice (first train_prop of rows).
    - Scales the full series with those train min/max.
    - No clipping; purely linear rescale.
    - Optionally skip *_RZ columns (they’re already standardized).
    Returns (df_scaled, stats_df with min/max).
    """
    df_out = df.copy()
    
    n_tr = int(len(df_out) * train_prop)
    train = df_out.iloc[:n_tr]

    num_cols = [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c]) and c != "close_raw"]
    if not include_rz:
        num_cols = [c for c in num_cols if not c.endswith("_RZ")]

    stats_rows = []
    for c in tqdm(num_cols, desc="MinMax p1/p99", unit="feat"):
        tr = train[c].replace([np.inf, -np.inf], np.nan).astype(float)
        lo = np.nanpercentile(tr, p_lo)
        hi = np.nanpercentile(tr, p_hi)
        span = hi - lo if np.isfinite(hi - lo) and (hi - lo) != 0 else eps

        stats_rows.append({"feature": c, "min": lo, "max": hi})

        s = df_out[c].astype(float).clip(lo, hi)
        df_out[c] = (s - lo) / span

    stats = pd.DataFrame(stats_rows).set_index("feature")
    return df_out, stats
 

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
    for col in tqdm(cols, desc="Pruning features", total=len(cols)):
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


def extract_windows_from_loader(loader, h5_path = str(Path(params.models_folder) / "Xy.h5")):
    """
    Stream (batch[0], batch[1]) from a DataLoader into an extendable HDF5 file.
    Preserves exact per-batch logic: batch[0].detach().cpu().numpy() and
    batch[1].detach().cpu().numpy().reshape(-1).
    Returns (X_ds, y_ds, h5_file) where X_ds/y_ds are h5py.Dataset objects.
    """
    it = iter(loader)
    b0 = next(it)                      
    xb0, yb0 = b0[0], b0[1]
    L, F = int(xb0.shape[1]), int(xb0.shape[2])
    dx = xb0.detach().cpu().numpy().dtype
    dy = yb0.detach().cpu().numpy().reshape(-1).dtype

    f = h5py.File(h5_path, "w")
    X = f.create_dataset("X", shape=(0, L, F), maxshape=(None, L, F), dtype=dx, chunks=True)
    y = f.create_dataset("y", shape=(0,), maxshape=(None,), dtype=dy, chunks=True)

    def _append(ds, arr):
        n0 = ds.shape[0]
        ds.resize(n0 + arr.shape[0], axis=0)
        ds[n0:n0 + arr.shape[0]] = arr

    with torch.no_grad():
        _append(X, xb0.detach().cpu().numpy())
        _append(y, yb0.detach().cpu().numpy().reshape(-1))
        for batch in tqdm(it, desc="Extracting windows"):
            xb = batch[0].detach().cpu().numpy()
            yb = batch[1].detach().cpu().numpy().reshape(-1)
            _append(X, xb)
            _append(y, yb)

    return X, y, f


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


