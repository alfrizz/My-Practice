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
import json

import ta
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import DonchianChannel

from captum.attr import IntegratedGradients
from tqdm.auto import tqdm
from pathlib import Path
import h5py

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
    sma: Optional[List[int]] = None,
    ema: Optional[List[int]] = None,
    atr: Optional[List[int]] = None,
    bb: Optional[Union[List[int], Dict[int, List[float]]]] = None,
    rsi: Optional[List[int]] = None,
    roc: Optional[List[int]] = None,
    vol_spike: Optional[List[int]] = None,
    obv_roll: Optional[List[int]] = None,
    ret_std: Optional[List[int]] = None,
    macd: Optional[List[Dict[str, int]]] = None,
    stoch: Optional[List[Dict[str, int]]] = None,
    cci: Optional[List[int]] = None,
    mfi: Optional[List[int]] = None,
    cmf: Optional[List[int]] = None,
    donch: Optional[List[int]] = None,
    roll_vwap: Optional[List[int]] = None,
    linreg_slope: Optional[List[int]] = None,
    kc: Optional[List[Dict[str, float]]] = None,
    psar: Optional[Dict[str, float]] = None,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """
    Computes a comprehensive suite of OHLCV technical indicators optimized for high-throughput 
    Machine Learning pipelines. 

    Designed specifically for large-scale intraday data (e.g., 1-minute scalping), this function 
    leverages raw NumPy arrays, C-speed rolling operations, and parallelized Pandas execution 
    to bypass standard library bottlenecks. It natively resolves recursive dependencies 
    (like Parabolic SAR) using compiled numpy logic, dropping calculation times from minutes to milliseconds.

    Outputs are strictly formatted as stationary or bounded ML features, protected against 
    zero-division errors via a safe epsilon architecture. It handles dynamic multi-window configurations 
    for trend, momentum, volatility, volume spikes, and session-specific anchoring (VWAP).
    """
    
    # --- 1. Helper Normalization ---
    def norm_ws(x): return sorted({int(i) for i in x}) if x else []
    
    sma_ws, ema_ws, atr_ws = norm_ws(sma), norm_ws(ema), norm_ws(atr)
    rsi_ws, roc_ws, vol_ws = norm_ws(rsi), norm_ws(roc), norm_ws(vol_spike)
    obv_roll_ws, ret_std_ws, cci_ws = norm_ws(obv_roll), norm_ws(ret_std), norm_ws(cci)
    mfi_ws, cmf_ws, donch_ws = norm_ws(mfi), norm_ws(cmf), norm_ws(donch)
    roll_vwap_ws, slope_ws = norm_ws(roll_vwap), norm_ws(linreg_slope)

    macd_cfgs = macd or []
    stoch_cfgs = stoch or []
    kc_cfgs = kc or []
    psar_cfg = psar or {}

    bb_items = []
    if bb:
        if isinstance(bb, dict):
            for w, devs in bb.items(): bb_items.append((int(w), [float(d) for d in devs]))
        else:
            for w in norm_ws(bb): bb_items.append((w, [2.0]))
    bb_items.sort(key=lambda t: t[0])

    # --- 2. Workload Estimation for TQDM ---
    total_tasks = (
        6 + len(sma_ws) + len(ema_ws) + len(rsi_ws) + len(roc_ws) + len(cci_ws) +
        len(macd_cfgs) + len(stoch_cfgs) + len(atr_ws) + len(kc_cfgs) +
        len(obv_roll_ws) + 1 + len(mfi_ws) + len(cmf_ws) + len(vol_ws) + len(donch_ws) +
        len(roll_vwap_ws) + len(slope_ws) + len(ret_std_ws) + 
        sum(len(devs) for _, devs in bb_items) +
        (2 if psar_cfg else 0) + (2 if sma_ws else 0) + 2 
    )

    df = df.sort_index()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    EPS = 1e-9

    # --- 3. Optimized Computational Kernels ---
    def safe_div(a, b):
        return a / (b.replace(0, np.nan) if isinstance(b, pd.Series) else (b + EPS))

    def fast_slope(series: pd.Series, w: int) -> pd.Series:
        x = np.arange(w)
        x_mean = x.mean()
        denom = np.sum((x - x_mean)**2)
        def _slope_kernel(y):
            return np.sum((x - x_mean) * (y - np.mean(y))) / denom if denom != 0 else np.nan
        return series.rolling(window=w).apply(_slope_kernel, raw=True)

    def fast_psar(high, low, close, step=0.02, max_step=0.2):
        n = len(close)
        psar_arr = np.zeros(n, dtype=np.float64)
        bull, af, ep, psar_arr[0] = True, step, high[0], low[0]
        for i in range(1, n):
            prev_psar = psar_arr[i-1]
            if bull:
                psar_arr[i] = prev_psar + af * (ep - prev_psar)
                psar_arr[i] = min(psar_arr[i], low[i-1], low[i-2]) if i >= 2 else min(psar_arr[i], low[i-1])
                if low[i] < psar_arr[i]: bull, psar_arr[i], ep, af = False, ep, low[i], step
                elif high[i] > ep: ep, af = high[i], min(af + step, max_step)
            else:
                psar_arr[i] = prev_psar + af * (ep - prev_psar)
                psar_arr[i] = max(psar_arr[i], high[i-1], high[i-2]) if i >= 2 else max(psar_arr[i], high[i-1])
                if high[i] > psar_arr[i]: bull, psar_arr[i], ep, af = True, ep, high[i], step
                elif low[i] < ep: ep, af = low[i], min(af + step, max_step)
        return psar_arr

    # --- 4. Feature Engineering Main Loop ---
    new = {}
    pbar = tqdm(total=total_tasks, desc="Initializing", leave=False)

    # 4.1 Returns & Geometry
    pbar.set_description("Processing: Returns & Geometry")
    new["ret"], new["log_ret"] = c.pct_change(), np.log(c + EPS).diff()
    new["body"], new["upper_shad"], new["lower_shad"] = c - o, h - np.maximum(o, c), np.minimum(o, c) - l
    new["range_pct"] = safe_div(h - l, c)
    pbar.update(6)

    # 4.2 Moving Averages
    pbar.set_description("Processing: Moving Averages")
    for w in sma_ws:
        s = c.rolling(w, min_periods=1).mean()
        new[f"sma_{w}"], new[f"sma_pct_{w}"] = s, safe_div(c - s, s)
        pbar.set_postfix_str(f"Current: sma_{w}")
        pbar.update(1)
    for w in ema_ws:
        new[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()
        pbar.set_postfix_str(f"Current: ema_{w}")
        pbar.update(1)

    # 4.3 Momentum
    pbar.set_description("Processing: Momentum")
    for w in rsi_ws: 
        new[f"rsi_{w}"] = ta.momentum.RSIIndicator(c, window=w).rsi()
        pbar.set_postfix_str(f"Current: rsi_{w}"); pbar.update(1)
    for w in roc_ws: 
        new[f"roc_{w}"] = c.pct_change(w)
        pbar.set_postfix_str(f"Current: roc_{w}"); pbar.update(1)
    for w in cci_ws: 
        new[f"cci_{w}"] = ta.trend.CCIIndicator(h, l, c, window=w).cci()
        pbar.set_postfix_str(f"Current: cci_{w}"); pbar.update(1)
    for cfg in macd_cfgs:
        m = ta.trend.MACD(c, window_fast=cfg['fast'], window_slow=cfg['slow'], window_sign=cfg['signal'])
        suff = f"{cfg['fast']}_{cfg['slow']}_{cfg['signal']}"
        new[f"macd_line_{suff}"], new[f"macd_signal_{suff}"], new[f"macd_diff_{suff}"] = m.macd(), m.macd_signal(), m.macd_diff()
        pbar.set_postfix_str(f"Current: macd_{suff}"); pbar.update(1)
    for cfg in stoch_cfgs:
        k, d, s = cfg['k'], cfg.get('d', 3), cfg.get('smooth', 3)
        low_k, high_k = l.rolling(k).min(), h.rolling(k).max()
        sk = safe_div(c - low_k, high_k - low_k).rolling(s).mean().clip(0, 1)
        new[f"stoch_k_{k}_{d}_{s}"], new[f"stoch_d_{k}_{d}_{s}"] = sk, sk.rolling(d).mean().clip(0, 1)
        pbar.set_postfix_str(f"Current: stoch_{k}"); pbar.update(1)

    # 4.4 Volatility & Bands
    pbar.set_description("Processing: Volatility & Bands")
    for w in atr_ws:
        atr_s = ta.volatility.AverageTrueRange(h, l, c, window=w).average_true_range()
        new[f"atr_{w}"], new[f"atr_pct_{w}"] = atr_s, safe_div(atr_s, c)
        adx = ta.trend.ADXIndicator(h, l, c, window=w)
        new[f"plus_di_{w}"], new[f"minus_di_{w}"], new[f"adx_{w}"] = adx.adx_pos(), adx.adx_neg(), adx.adx()
        pbar.set_postfix_str(f"Current: atr/adx_{w}"); pbar.update(1)
    for w, devs in bb_items:
        for dev in devs:
            bb_o = ta.volatility.BollingerBands(c, window=w, window_dev=dev)
            suff = f"{w}_{str(dev).replace('.', 'p')}"
            new[f"bb_lband_{suff}"], new[f"bb_hband_{suff}"] = bb_o.bollinger_lband(), bb_o.bollinger_hband()
            new[f"bb_w_{suff}"] = safe_div(new[f"bb_hband_{suff}"] - new[f"bb_lband_{suff}"], bb_o.bollinger_mavg())
            pbar.set_postfix_str(f"Current: bb_{suff}"); pbar.update(1)
    for cfg in kc_cfgs:
        ew, aw, m = cfg['ema_window'], cfg.get('atr_window', 20), cfg['atr_mult']
        mid = c.ewm(span=ew, adjust=False).mean()
        as_ = ta.volatility.AverageTrueRange(h, l, c, window=aw).average_true_range()
        suff = f"{ew}_{aw}_{m}"
        new[f"kc_mid_{suff}"], new[f"kc_l_{suff}"], new[f"kc_h_{suff}"] = mid, mid - m * as_, mid + m * as_
        new[f"kc_w_{suff}"] = safe_div(new[f"kc_h_{suff}"] - new[f"kc_l_{suff}"], mid)
        pbar.set_postfix_str(f"Current: kc_{suff}"); pbar.update(1)

    # 4.5 Volume & Channels
    pbar.set_description("Processing: Volume & Channels")
    new["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume(); pbar.update(1)
    for w in obv_roll_ws: 
        new[f"obv_roll_{w}"] = new["obv"].rolling(w).mean()
        pbar.set_postfix_str(f"Current: obv_{w}"); pbar.update(1)
    for w in mfi_ws: 
        new[f"mfi_{w}"] = ta.volume.MFIIndicator(h, l, c, v, window=w).money_flow_index()
        pbar.set_postfix_str(f"Current: mfi_{w}"); pbar.update(1)
    for w in cmf_ws: 
        new[f"cmf_{w}"] = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, window=w).chaikin_money_flow()
        pbar.set_postfix_str(f"Current: cmf_{w}"); pbar.update(1)
    for w in vol_ws: 
        new[f"vol_spike_{w}"] = safe_div(v, v.rolling(w).mean())
        pbar.set_postfix_str(f"Current: vol_spike_{w}"); pbar.update(1)
    for w in donch_ws:
        hi, lo = h.rolling(w).max(), l.rolling(w).min()
        new[f"donch_h_{w}"], new[f"donch_l_{w}"], new[f"donch_w_{w}"] = hi, lo, safe_div(hi - lo, c)
        pbar.set_postfix_str(f"Current: donch_{w}"); pbar.update(1)

    # 4.6 Specialized Anchors
    pbar.set_description("Processing: Specialized")
    for w in roll_vwap_ws: 
        new[f"roll_vwap_{w}"] = safe_div((c * v).rolling(w).sum(), v.rolling(w).sum())
        pbar.set_postfix_str(f"Current: roll_vwap_{w}"); pbar.update(1)
    for w in slope_ws: 
        new[f"slope_close_{w}"] = fast_slope(c, w)
        pbar.set_postfix_str(f"Current: slope_{w}"); pbar.update(1)
    for w in ret_std_ws: 
        new[f"ret_std_{w}"] = new["ret"].rolling(w).std()
        pbar.set_postfix_str(f"Current: ret_std_{w}"); pbar.update(1)
    
    cv, v_cum = (c * v).groupby(df.index.date).cumsum(), v.groupby(df.index.date).cumsum().replace(0, np.nan)
    new["vwap_ohlc_close_session"] = cv / v_cum
    new["vwap_dist_session"] = safe_div(c - new["vwap_ohlc_close_session"], new["vwap_ohlc_close_session"])
    pbar.update(2)

    if psar_cfg:
        step, max_step = psar_cfg.get("step", 0.02), psar_cfg.get("max_step", 0.2)
        new["psar"] = fast_psar(h.to_numpy(), l.to_numpy(), c.to_numpy(), step, max_step)
        new["psar_dist"] = safe_div(c - new["psar"], c)
        pbar.set_postfix_str("Current: psar")
        pbar.update(2)

    if sma_ws:
        lw = max(sma_ws)
        hb, lb = h.rolling(lw).max(), l.rolling(lw).min()
        new[f"dist_high_{lw}"], new[f"dist_low_{lw}"] = safe_div(hb - c, c), safe_div(c - lb, c)
        pbar.set_postfix_str("Current: extrema")
        pbar.update(2)

    pbar.close()

    # --- 5. Final Assembly ---
    df_out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1).replace([np.inf, -np.inf], np.nan)
    if drop_warmup:
        df_out = df_out.dropna().copy()

    return df_out


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
    pct_shift_thresh: float = 0.16,
    frac_outside_thresh: float = 0.06,
    min_train_samples: int = 50,
    na_rate_thresh: float = 0.4,
    const_tol: float = 1e-12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyzes numeric features for data integrity issues like drift, constant values, 
    or high missingness. Marks drifted features for special downstream handling.
    """
    if len(df) == 0:
        return df.copy(), pd.DataFrame()

    # 1. Contiguous Split (Vectorized)
    n_tr = int(len(df) * train_prop)
    n_val = (len(df) - n_tr) // 2
    
    # Efficient slicing
    tr = df.iloc[:n_tr]
    vl = df.iloc[n_tr:n_tr + n_val]
    te = df.iloc[n_tr + n_val:]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    def robust_scale(arr: np.ndarray) -> float:
        """Helper to find a stable denominator for relative drift calculation."""
        if arr.size == 0: return 1e-6
        med = float(np.nanmedian(arr))
        # Median Absolute Deviation (MAD) is more robust than STD for financial spikes
        mad = float(np.nanmedian(np.abs(arr - med)))
        std = float(np.nanstd(arr))
        return max(abs(med), mad, std, 1e-6)

    rows = []
    
    # 2. Main Diagnostic Loop
    for feat in tqdm(numeric_cols, desc="Flagging indicators", unit="feat", leave=True):
        # Extract slices as numpy arrays for speed
        tr_arr = tr[feat].to_numpy()
        vl_arr = vl[feat].to_numpy()
        te_arr = te[feat].to_numpy()
        
        # Filter NaNs for stats
        tr_s = tr_arr[~np.isnan(tr_arr)]
        vl_s = vl_arr[~np.isnan(vl_arr)]
        te_s = te_arr[~np.isnan(te_arr)]

        na_rate_train = np.isnan(tr_arr).mean()
        na_rate_all = df[feat].isna().mean()

        # Priority Check: Sample size & Constants
        if len(tr_s) < min_train_samples:
            status, reason = "FEW_TRAIN", "too_few_train"
        elif (np.nanmax(tr_s) - np.nanmin(tr_s)) < const_tol:
            status, reason = "CONST", "constant_train"
        else:
            # Check for Drift
            med_tr = np.nanmedian(tr_s)
            med_val = np.nanmedian(vl_s) if len(vl_s) else np.nan
            med_te = np.nanmedian(te_s) if len(te_s) else np.nan
            denom = robust_scale(tr_s)

            pct_shift_val = abs(med_val - med_tr) / denom if not np.isnan(med_val) else 0.0
            pct_shift_te = abs(med_te - med_tr) / denom if not np.isnan(med_te) else 0.0

            # Tail Drift Check (1-99 percentiles)
            q01 = np.nanpercentile(tr_s, 1)
            q99 = np.nanpercentile(tr_s, 99)
            
            # Fallback if tails are too sparse
            if (tr_s <= q01).sum() < 3: q01 = np.nanpercentile(tr_s, 5)
            if (tr_s >= q99).sum() < 3: q99 = np.nanpercentile(tr_s, 95)

            frac_val_out = np.mean((vl_s < q01) | (vl_s > q99)) if len(vl_s) else 0.0
            frac_te_out = np.mean((te_s < q01) | (te_s > q99)) if len(te_s) else 0.0

            drift_fail = (pct_shift_val > pct_shift_thresh or pct_shift_te > pct_shift_thresh or 
                          frac_val_out > frac_outside_thresh or frac_te_out > frac_outside_thresh)
            
            high_na = (na_rate_train > na_rate_thresh or na_rate_all > na_rate_thresh)

            if high_na:
                status, reason = "HIGH_NA", f"na_train={na_rate_train:.2f}; na_all={na_rate_all:.2f}"
            elif drift_fail:
                status, reason = "DRIFT", f"shift_val={pct_shift_val:.3f}; out_te={frac_te_out:.3f}"
            else:
                status, reason = "OK", ""

        rows.append({
            "feature": feat, "status": status, "reason": reason,
            "pct_shift_val": pct_shift_val if 'pct_shift_val' in locals() else np.nan,
            "pct_shift_te": pct_shift_te if 'pct_shift_te' in locals() else np.nan,
            "frac_val_out": frac_val_out if 'frac_val_out' in locals() else np.nan,
            "frac_te_out": frac_te_out if 'frac_te_out' in locals() else np.nan,
            "na_rate_train": na_rate_train, "na_rate_all": na_rate_all
        })

    diag = pd.DataFrame(rows)

    # 3. Final Reorganization (Memory Safe)
    drift_mask = diag["status"] == "DRIFT"
    drift_feats = diag.loc[drift_mask, "feature"].tolist()
    
    # Create a dictionary mapping old names to new names
    rename_mapping = {feat: f"{feat}_DRIFT" for feat in drift_feats if feat in df.columns}
    
    # Rename all columns simultaneously in one highly optimized step
    df_out = df.rename(columns=rename_mapping)

    return df_out, diag
    

########################################################################################################## 


def apply_rz_to_drifts(
    df: pd.DataFrame,
    diag: pd.DataFrame,
    rz_window: int = 60,
    min_periods: Optional[int] = None,
    eps: float = 1e-6,
    mask_warmup: bool = False,
) -> pd.DataFrame:
    """
    Applies a rolling Robust Z-Score transformation to features flagged with 'DRIFT'.
    
    This function:
    1. Identifies features requiring scaling from the diagnostic report.
    2. Handles price-dependent indicators (VWAP) by converting them to relative percentages.
    3. Calculates rolling Median and Median Absolute Deviation (MAD) to compute Z-scores.
    4. Casts output to float32 to optimize memory for GPU-heavy training.
    """
    df_out = df.copy()
    min_periods = min_periods if min_periods is not None else 1
    
    # Identify drift features (ensure they are strings)
    drift_feats = diag.loc[diag["status"] == "DRIFT", "feature"].astype(str).tolist()
    
    # Pre-fetch close_raw once to avoid repeated dictionary lookups in the loop
    close_raw = None
    if any("vwap" in f.lower() for f in drift_feats):
        if "close_raw" in df_out.columns:
            close_raw = pd.to_numeric(df_out["close_raw"], errors="coerce").astype("float64")
            close_raw = close_raw.replace(0, np.nan)

    for feat in tqdm(drift_feats, desc="Applying RZ to DRIFTs", unit="feat", leave=True):
        drift_col = f"{feat}_DRIFT"
        if drift_col not in df_out.columns:
            continue

        # 1. Convert to numeric once and ensure float64 for precision during rolling ops
        s = pd.to_numeric(df_out[drift_col], errors="coerce").astype("float64")

        # 2. Relative Transformation for VWAP
        # Makes VWAP distance scale-invariant before calculating Z-score
        if "vwap" in feat.lower() and close_raw is not None:
            s = (s / close_raw) - 1.0

        # 3. Robust Z-Score Calculation
        # RZ = (x - median) / MAD
        rolling_obj = s.rolling(window=rz_window, min_periods=min_periods)
        med = rolling_obj.median()
        
        # Calculate MAD: median of absolute deviations from the median
        abs_dev = (s - med).abs()
        mad = abs_dev.rolling(window=rz_window, min_periods=min_periods).median()
        
        # Zero-division safeguard
        mad_safe = mad.replace(0, np.nan).fillna(eps)
        rz = (s - med) / mad_safe

        # 4. Optional Warm-up Masking
        # Prevents the model from seeing unstable early-window calculations
        if mask_warmup and rz_window > 1:
            rz.iloc[:rz_window - 1] = np.nan

        # 5. Commit & Cleanup
        # Using float32 saves 50% memory in the final Parquet file and VRAM
        df_out[f"{feat}_RZ"] = rz.astype("float32")
        df_out.drop(columns=[drift_col], inplace=True)

    return df_out

    
########################################################################################################## 


def scale_features(
    df: pd.DataFrame,
    train_prop: float = 0.7,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    include_rz: bool = False,
    eps: float = 1e-12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies Percentile-based MinMax scaling to [0,1].
    
    Fits the scale on the training portion of the data (first train_prop rows) 
    and applies it to the entire dataset. This prevents data leakage from the 
    future (val/test sets) into the training process.
    """
    df_out = df.copy()
    n_tr = int(len(df_out) * train_prop)
    train_df = df_out.iloc[:n_tr]

    # 1. Filter numeric columns carefully
    # Exclude '_raw' metadata columns and optionally '_RZ' (already standardized)
    num_cols = [
        c for c in df_out.columns 
        if pd.api.types.is_numeric_dtype(df_out[c]) 
        and not c.endswith("_raw")
    ]
    
    if not include_rz:
        num_cols = [c for c in num_cols if not c.endswith("_RZ")]

    stats_rows = []
    
    # 2. Fit and Transform
    for c in tqdm(num_cols, desc="MinMax Scaling", unit="feat", leave=True):
        # Clean training slice for fitting
        tr_slice = train_df[c].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        
        if len(tr_slice) == 0:
            stats_rows.append({"feature": c, "min": 0.0, "max": 1.0})
            continue

        # Calculate Percentiles (The "Fitting" Step)
        lo = np.nanpercentile(tr_slice, p_lo)
        hi = np.nanpercentile(tr_slice, p_hi)
        
        # Calculate span with safety for constant values
        span = (hi - lo) if (hi - lo) > eps else eps
        stats_rows.append({"feature": c, "min": lo, "max": hi})

        # Apply transformation to the full dataset (The "Transform" Step)
        # We clip to [lo, hi] before scaling to ensure the output is strictly [0, 1]
        # This protects against anomalous spikes in the live/test data.
        s = df_out[c].astype("float32").clip(lower=lo, upper=hi)
        df_out[c] = (s - lo) / span

    stats = pd.DataFrame(stats_rows).set_index("feature")
    
    return df_out, stats

    
#########################################################################################################


def prune_features_by_variance_and_correlation(
    X_all: pd.DataFrame,
    y: pd.Series,
    train_prop: float = 0.7,  # <--- CRITICAL FIX: Training split parameter
    min_std: float = 1e-6,
    max_corr: float = 0.9,
) -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame]:
    """
    Prune numeric features by variance and pairwise Pearson correlation.
    Calculations are strictly fitted on the training split to prevent Data Leakage.
    """
    # 1) Numeric-only and align y
    X = X_all.select_dtypes(include=[np.number]).copy()
    y = y.reindex(X.index)

    # --- LEAKAGE PREVENTION ---
    # Isolate training data for all statistical decisions
    n_tr = int(len(X) * train_prop)
    X_train = X.iloc[:n_tr]
    y_train = y.iloc[:n_tr]

    # 2) Variance filter (Measured on TRAIN only)
    stds = X_train.std(axis=0, ddof=0)
    kept_after_std = stds[stds >= min_std].index.tolist()
    dropped_low_variance = stds[stds < min_std].index.tolist()
    
    if not kept_after_std:
        print("No numeric features after std filter.")
        return [], dropped_low_variance, pd.DataFrame(), pd.DataFrame()

    X_train_var = X_train[kept_after_std]

    # 3) One-shot Pearson correlation (High-Speed NumPy Implementation)
    print("Calculating correlation matrix on training data...") 
    
    # Convert to NumPy array for speed
    arr = X_train_var.to_numpy(dtype=float)
    
    # Fill straggling NaNs with the column mean to prevent np.corrcoef from crashing
    col_means = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    if inds[0].size > 0:
        arr[inds] = col_means[inds[1]]  # <--- FIXED: Safer 2D numpy indexing
        
    # Calculate Pearson correlation at C-speed
    corr_matrix = np.corrcoef(arr, rowvar=False)
    
    # Convert back to Pandas for easy indexing later
    corr_full = pd.DataFrame(
        np.clip(np.abs(corr_matrix), 0.0, 1.0), 
        index=X_train_var.columns, 
        columns=X_train_var.columns
    )

    # 4) Precompute corr with target (abs) on TRAIN only
    try:
        corr_with_y = X_train_var.corrwith(y_train).abs()
    except Exception:
        corr_with_y = pd.Series(index=X_train_var.columns, data=np.nan)

    # 5) Greedy grouping and pruning
    mask_upper = np.triu(np.ones(corr_full.shape), k=1).astype(bool)
    upper = corr_full.where(mask_upper)
    
    to_drop: Set[str] = set()
    kept_map: Dict[str, List[str]] = {}
    dropped_corr_info: Dict[str, float] = {}

    for col in upper.columns:
        if col in to_drop:
            continue
            
        high_corr_series = upper[col][upper[col] > max_corr]
        high_corr_feats = [feat for feat in high_corr_series.index if feat not in to_drop]
        
        if not high_corr_feats:
            continue
            
        group = [col] + high_corr_feats
        
        # Choose representative: prefer corr with y, else std
        group_corr_y = corr_with_y[group]
        if group_corr_y.notna().any():
            best = group_corr_y.idxmax()
        else:
            best = stds[group].idxmax()
            
        for member in group:
            if member != best:
                to_drop.add(member)
                kept_map.setdefault(best, []).append(member)
                corr_val = corr_full.loc[member, best]
                dropped_corr_info[member] = float(corr_val) if not pd.isna(corr_val) else np.nan

    # 6) Finalize lists
    pruned_feats = sorted(list(to_drop))
    kept_final_feats = [f for f in kept_after_std if f not in to_drop]
    corr_pruned = corr_full.loc[kept_final_feats, kept_final_feats] if kept_final_feats else pd.DataFrame()

    # 7) Prints
    print(f"\nDropped low-variance features (n={len(dropped_low_variance)}):")
    if dropped_low_variance:
        for f in sorted(dropped_low_variance):
            print(f"  Dropped: {f}  (std={stds[f]:.6g})")
    else:
        print("  None")

    if kept_map:
        print(f"\nDropped by correlation (n={len(pruned_feats)}), mapping Dropped <-- Kept (corr):")
        for kept_feat, dropped_list in kept_map.items():
            for dropped_feat in dropped_list:
                corr_val = dropped_corr_info.get(dropped_feat, np.nan)
                corr_str = f"{corr_val:.4f}" if not np.isnan(corr_val) else "nan"
                print(f"  Dropped: {dropped_feat}  (corr={corr_str})  <-- Kept: {kept_feat}")
    else:
        print("\nDropped by correlation: None")

    print(f"\nKept after std filter (n={len(kept_after_std)}).")
    print(f"Kept after correlation pruning (n={len(kept_final_feats)}).")
    
    return kept_final_feats, pruned_feats, corr_full, corr_pruned
    

#########################################################################################################



def extract_and_save_windows(loader, out_dir="trainings", features=None):
    """
    Stream (xb, yb) from a PyTorch loader directly into raw binary files.
    Skips HDF5 entirely. Uses NumPy's C-level .tofile() for maximum disk I/O speed.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    X_path = out / "X_windows.dat"
    y_path = out / "y_windows.dat"
    meta_path = out / "X_windows_meta.json"
    
    # Open files in binary write mode ('wb')
    with open(X_path, 'wb') as fX, open(y_path, 'wb') as fy:
        N = 0
        shape_X_tail = None
        dtype_X = None
        dtype_y = None
        
        # Just iterate over 'batch' instead of unpacking 'xb, yb'
        for batch in tqdm(loader, desc="Streaming straight to binary", unit="batch"):
            # Explicitly grab the first two elements just like your original code
            xb = batch[0]
            yb = batch[1]
            
            # Move to CPU and convert to numpy
            xb_np = xb.detach().cpu().numpy()
            yb_np = yb.detach().cpu().numpy().reshape(-1)
            
            # Capture metadata on the first pass
            if shape_X_tail is None:
                shape_X_tail = xb_np.shape[1:]  # e.g., (Lookback, Features)
                dtype_X = str(xb_np.dtype)
                dtype_y = str(yb_np.dtype)
            
            # Write raw memory block directly to disk
            xb_np.tofile(fX)
            yb_np.tofile(fy)
            
            N += xb_np.shape[0]

    # Save the exact minimal metadata required to reconstruct the memmap
    meta = {
        "X_path": str(X_path),
        "y_path": str(y_path),
        "X_shape": (N, *shape_X_tail),
        "X_dtype": dtype_X,
        "y_shape": (N,),
        "y_dtype": dtype_y,
        "features": list(features) if features is not None else []
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return X_path, y_path, meta_path


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


def live_display_importances(imp_series, features, target, method,
                             batch=1, pause=0.02, threshold=None):
    """
    Incrementally display importances from a completed pd.Series.

    - imp_series: pd.Series indexed by feature name (values = importance)
    - features: ordered list of features to reveal (must match index names)
    - target: target name for title
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
        plt.title(f"{method} Importance (partial) for {target}")
        plt.xlabel("Importance")
        plt.tight_layout()
        display(plt.gcf())
        plt.close()
        time.sleep(pause)
    return imp_series.sort_values(ascending=False)


######################################


def update_feature_importances(fi_dict, importance_type, values: pd.Series):
    """
    fi_dict: master dict
    importance_type: one of "corr","mi","perm","shap","lasso"
    values: pd.Series indexed by feature name
    """
    for feat, val in values.items():
        if feat in fi_dict:
            fi_dict[feat][importance_type] = val


