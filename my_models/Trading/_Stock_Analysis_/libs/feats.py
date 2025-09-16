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


##########################################################################################################


def create_features(
    df: pd.DataFrame,
    sma_short:   int = 20,
    sma_long:    int = 100,
    rsi_window:  int = 14,
    macd_fast:   int = 12,
    macd_slow:   int = 26,
    macd_sig:    int = 9,
    atr_window:  int = 14,
    bb_window:   int = 20,
    obv_sma:     int = 14,
    vwap_window: int = 20,
    base_w:      int | None = None
) -> pd.DataFrame:
    """
    1) Compute raw OHLCV + bid/ask + label.
    2) Compute standard textbook indicators at 1m:
       • rsi_{rsi_window}
       • macd_line_{macd_fast}_{macd_slow}_{macd_sig}, macd_signal_…, macd_diff_…
       • sma_{sma_short}, sma_{sma_long}
       • atr_{atr_window}
       • bb_lband_{bb_window}, bb_hband_{bb_window}, bb_width_{bb_window}
       • plus_di_{atr_window}, minus_di_{atr_window}, adx_{atr_window}
       • obv, obv_sma_{obv_sma}
       • vwap_{vwap_window}, vol_spike_{obv_sma}
       • hour, day_of_week, month
    3) If base_w > 1, compute custom-window versions at half/base/double—
       skipping any that would collide with standard-window names.
    4) Prevent division-by-zero by adding eps to every denominator.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    eps = 1e-8
    out = df[["open","high","low","close","volume","bid","ask", params.label_col]].copy()

    # 1) STANDARD WINDOWS

    # RSI
    out[f"rsi_{rsi_window}"] = (
        ta.momentum.RSIIndicator(out["close"], window=rsi_window)
          .rsi().round(3)
    )

    # MACD
    macd = ta.trend.MACD(
        close=out["close"],
        window_fast=macd_fast,
        window_slow=macd_slow,
        window_sign=macd_sig
    )
    out[f"macd_line_{macd_fast}_{macd_slow}_{macd_sig}"]   = macd.macd().round(3)
    out[f"macd_signal_{macd_fast}_{macd_slow}_{macd_sig}"] = macd.macd_signal().round(3)
    out[f"macd_diff_{macd_fast}_{macd_slow}_{macd_sig}"]   = macd.macd_diff().round(3)

    # SMAs
    out[f"sma_{sma_short}"] = out["close"].rolling(sma_short, min_periods=1).mean().round(3)
    out[f"sma_{sma_long}"]  = out["close"].rolling(sma_long,  min_periods=1).mean().round(3)

    # ATR
    out[f"atr_{atr_window}"] = (
        ta.volatility.AverageTrueRange(
            high=out["high"], low=out["low"], close=out["close"],
            window=atr_window
        ).average_true_range().round(3)
    )

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(out["close"], window=bb_window, window_dev=2)
    out[f"bb_lband_{bb_window}"] = bb.bollinger_lband().round(3)
    out[f"bb_hband_{bb_window}"] = bb.bollinger_hband().round(3)
    m = bb.bollinger_mavg()
    out[f"bb_width_{bb_window}"] = ((out[f"bb_hband_{bb_window}"]
                                     - out[f"bb_lband_{bb_window}"]) / (m + eps)
                                   ).round(3)

    # Directional Movement
    adx = ta.trend.ADXIndicator(
        high=out["high"], low=out["low"],
        close=out["close"], window=atr_window
    )
    out[f"plus_di_{atr_window}"]  = adx.adx_pos().round(3)
    out[f"minus_di_{atr_window}"] = adx.adx_neg().round(3)
    out[f"adx_{atr_window}"]      = adx.adx().round(3)

    # OBV
    out["obv"] = (
        ta.volume.OnBalanceVolumeIndicator(
            close=out["close"], volume=out["volume"]
        ).on_balance_volume().round(3)
    )
    out[f"obv_sma_{obv_sma}"] = (
        out["obv"].rolling(obv_sma, min_periods=1).mean().round(3)
    )

    # VWAP & vol_spike
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=out["high"], low=out["low"],
        close=out["close"], volume=out["volume"],
        window=vwap_window
    ).volume_weighted_average_price().round(6)
    out[f"vwap_{vwap_window}"] = vwap
    vol_roll = out["volume"].rolling(obv_sma, min_periods=1).mean()
    out[f"vol_spike_{obv_sma}"] = (out["volume"] / (vol_roll + eps)).round(3)
    out[f"vwap_dev_{vwap_window}"] = ((out["close"] - vwap) / (vwap + eps)).round(6)

    # Calendar
    out["hour"]        = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["month"]       = out.index.month

    # 2) CUSTOM-WINDOW EXTRAS
    if base_w and base_w > 1:
        half, double = max(2, base_w // 2), base_w * 2
        sig = max(1, base_w // 4)
        skip_sma     = {sma_short, sma_long}
        skip_rsi     = {rsi_window}
        skip_atr     = {atr_window}
        skip_bb      = {bb_window}
        skip_obv_sma = {obv_sma}

        cust = pd.DataFrame(index=out.index)

        # EMA at half
        cust[f"ema_{half}"] = (
            ta.trend.EMAIndicator(out["close"], window=half)
              .ema_indicator().round(3)
        )

        # SMAs at half, base, double
        for w in (half, base_w, double):
            if w not in skip_sma:
                cust[f"sma_{w}"] = out["close"].rolling(w, min_periods=1).mean().round(3)

        # MACD diff at half/base/sig
        macd2 = ta.trend.MACD(out["close"], window_fast=half,
                              window_slow=base_w, window_sign=sig)
        cust[f"macd_diff_{half}_{base_w}_{sig}"] = macd2.macd_diff().round(3)

        # ATR at base & double
        for w in (base_w, double):
            if w not in skip_atr:
                cust[f"atr_{w}"] = (
                    ta.volatility.AverageTrueRange(
                        high=out["high"], low=out["low"],
                        close=out["close"], window=w
                    ).average_true_range().round(3)
                )

        # BBands at base
        if base_w not in skip_bb:
            bb2 = ta.volatility.BollingerBands(out["close"],
                                               window=base_w, window_dev=2)
            cust[f"bb_lband_{base_w}"] = bb2.bollinger_lband().round(3)
            cust[f"bb_hband_{base_w}"] = bb2.bollinger_hband().round(3)
            m2 = bb2.bollinger_mavg()
            cust[f"bb_width_{base_w}"] = ((cust[f"bb_hband_{base_w}"]
                                           - cust[f"bb_lband_{base_w}"]) / (m2 + eps)
                                         ).round(3)

        # RSI at base
        if base_w not in skip_rsi:
            cust[f"rsi_{base_w}"] = (
                ta.momentum.RSIIndicator(out["close"], window=base_w)
                  .rsi().round(3)
            )

        # Stochastic
        st = ta.momentum.StochasticOscillator(
            high=out["high"], low=out["low"], close=out["close"],
            window=base_w, smooth_window=sig
        )
        cust[f"stoch_k_{base_w}"] = st.stoch().round(3)
        cust[f"stoch_d_{sig}"]     = st.stoch_signal().round(3)

        # DI/ADX at base
        if base_w not in skip_atr:
            adx2 = ta.trend.ADXIndicator(out["high"], out["low"],
                                         out["close"], window=base_w)
            cust[f"plus_di_{base_w}"]  = adx2.adx_pos().round(3)
            cust[f"minus_di_{base_w}"] = adx2.adx_neg().round(3)
            cust[f"adx_{base_w}"]      = adx2.adx().round(3)

        # OBV_SMA at base
        if base_w not in skip_obv_sma:
            cust[f"obv_sma_{base_w}"] = (
                out["obv"].rolling(base_w, min_periods=1).mean().round(3)
            )

        # VWAP_dev & vol_spike at base
        v2 = ta.volume.VolumeWeightedAveragePrice(
                high=out["high"], low=out["low"],
                close=out["close"], volume=out["volume"],
                window=base_w
            ).volume_weighted_average_price().round(6)
        cust[f"vwap_dev_{base_w}"]  = ((out["close"] - v2) / (v2 + eps)).round(6)
        roll2 = out["volume"].rolling(base_w, min_periods=1).mean()
        cust[f"vol_spike_{base_w}"] = (out["volume"] / (roll2 + eps)).round(3)

        # Returns & rolling vol at 1, base, double
        for p in (1, base_w, double):
            cust[f"r_{p}"] = np.log(out["close"] / out["close"].shift(p)).round(6)
        cust[f"vol_{base_w}"] = cust["r_1"].rolling(base_w, min_periods=1).std().round(6)

        out = pd.concat([out, cust], axis=1)

    return out.dropna()


##########################################################################################################


def features_engineering(
    df: pd.DataFrame,
    rsi_low:  float = 30.0,
    rsi_high: float = 70.0,
    adx_thr:  float = 20.0,
    mult_w:   int   = 14,
    eps:      float = 1e-8
) -> pd.DataFrame:
    """
    Build seven eng_* signals as stationary ratios or deviations:
      1) eng_ma      = (sma_short – sma_long)  / sma_long
      2) eng_macd    = macd_diff               / sma_long
      3) eng_bb      = distance outside BBands / bb_width
      4) eng_rsi     = threshold distance      / 100
      5) eng_adx     = sign(DI+–DI–)/100 × (ADX–thr)/100
      6) eng_obv     = (obv – obv_sma)        / obv_sma
      7) eng_atr_div = 10 000×[(atr/close) – roll_mean(atr/close)]
    All divisions protect against zero by adding eps to denominators.
    """
    out = pd.DataFrame(index=df.index)

    # detect short/long SMA columns
    sma_cols = sorted([c for c in df if c.startswith("sma_")],
                      key=lambda x: int(x.split("_")[-1]))
    short, long = sma_cols[0], sma_cols[1]

    # 1) MA spread ratio
    out["eng_ma"] = ((df[short] - df[long]) / (df[long] + eps)).round(3)

    # 2) MACD diff ratio
    out["eng_macd"] = (df["macd_diff_12_26_9"] / (df[long] + eps)).round(3)

    # 3) Bollinger deviation ratio
    lo, hi, bw = df["bb_lband_20"], df["bb_hband_20"], df["bb_width_20"]
    dev = np.where(df["close"] < lo, lo - df["close"],
          np.where(df["close"] > hi, df["close"] - hi, 0.0))
    out["eng_bb"] = (dev / (bw + eps)).round(3)

    # 4) RSI threshold ratio
    rsi = df["rsi_14"]
    low_dev  = np.clip((rsi_low - rsi),   0, None) / 100.0
    high_dev = np.clip((rsi - rsi_high), 0, None) / 100.0
    out["eng_rsi"] = np.where(rsi < rsi_low, low_dev,
                       np.where(rsi > rsi_high, high_dev, 0.0)).round(3)

    # 5) ADX-weighted DI spread
    plus, minus, adx = df["plus_di_14"], df["minus_di_14"], df["adx_14"]
    diff = (plus - minus).abs() / 100.0
    ex   = np.clip((adx - adx_thr) / 100.0, 0, None)
    out["eng_adx"] = (np.sign(plus - minus) * diff * ex).round(3)

    # 6) OBV divergence ratio
    out["eng_obv"] = (
        (df["obv"] - df["obv_sma_14"]) / (df["obv_sma_14"] + eps)
    ).round(3)

    # 7) ATR/price stationary deviation
    ratio = df["atr_14"] / (df["close"] + eps)
    rm    = ratio.rolling(mult_w, min_periods=1).mean()
    out["eng_atr_div"] = ((ratio - rm) * 10_000).round(1)

    return out


##########################################################################################################


def scale_with_splits(
    df: pd.DataFrame,
    train_prop: float = params.train_prop,
    val_prop:   float = params.val_prop
) -> pd.DataFrame:
    """
    Split, encode time, PCA‐compress, and scale indicator features without leaking future info.

    1) Chronologically split df → train / val / test.
    2) In each split:
       a) encode hour, day_of_week, month as sin/cos;
       b) immediately PCA‐compress each sin/cos pair back to a single 'hour', 'day_of_week', 'month'.
    3) Identify `indicator_cols` = all columns except:
       - raw price (open, high, low, close, volume),
       - bid, ask, label,
       - calendar dims (hour, day_of_week, month).
       Those raw / label / calendar fields pass through unchanged.
    4) On TRAIN[indicator_cols], fit a ColumnTransformer:
       - bounded_inds (RSI/Stoch/+DI/-DI/ADX)       → divide by 100  
       - ratio_inds (returns, vol_spike, vwap_dev, bb_width, eng_*) → MinMaxScaler(0,1)  
       - unbounded  (EMA/SMA/MACD/ATR/BB midbands/OBV/VWAP)       → RobustScaler(5–95%) → StandardScaler → MinMaxScaler(0,1)
    5) For each split & each calendar day (tqdm):
       transform only `indicator_cols`, reattach raw/label/calendar.
    6) Concatenate train/val/test, sort by index, return.
    """
    # 1) Chronological split
    df = df.copy()
    N     = len(df)
    n_tr  = int(N * train_prop)
    n_val = int(N * val_prop)
    if n_tr + n_val >= N:
        raise ValueError("train_prop + val_prop must sum to <1.0")
    df_tr = df.iloc[:n_tr].copy()
    df_v  = df.iloc[n_tr : n_tr + n_val].copy()
    df_te = df.iloc[n_tr + n_val :].copy()

    # 2) Encode sin/cos and PCA‐compress in each split
    for split_name, split_df in zip(
        ["train","val","test"], [df_tr, df_v, df_te]
    ):
        # 2a) encode cyclic time
        h   = split_df["hour"]
        dow = split_df["day_of_week"]
        m   = split_df["month"]
        split_df["hour_sin"], split_df["hour_cos"] = (
            np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
        )
        split_df["dow_sin"],  split_df["dow_cos"]  = (
            np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
        )
        split_df["mo_sin"],   split_df["mo_cos"]   = (
            np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)
        )

        # 2b) PCA‐compress each pair back to one calendar dim
        for feat, (c1, c2) in zip(
                ["hour","day_of_week","month"],
                [("hour_sin","hour_cos"),
                 ("dow_sin","dow_cos"),
                 ("mo_sin","mo_cos")]):
            pca_vals = split_df[[c1, c2]].values
            comp     = PCA(n_components=1).fit_transform(pca_vals).ravel().round(3)
            split_df[feat] = comp
            split_df.drop([c1, c2], axis=1, inplace=True)

    # 3) Identify indicator columns to scale, excluding the 'reserved'
    reserved = {
        "open","high","low","close","volume",  # we don´t need the raw ohlcv
        "bid","ask", params.label_col,         # we need them raw
        "hour","day_of_week","month"           # already PCA‐compressed calendar, no need to rescale them
    }
    indicator_cols = [c for c in df.columns if c not in reserved]

    # 4) Domain‐aware scaling on TRAIN[indicators]
    bounded   = [c for c in indicator_cols
                 if c.startswith(("rsi_","stoch_","plus_di_","minus_di_","adx_"))]
    ratio     = [c for c in indicator_cols
                 if c.startswith("r_")
                 or "vol_spike" in c
                 or "vwap_dev"  in c
                 or c.endswith("_width")
                 or c.startswith("eng_")]
    unbounded = [c for c in indicator_cols if c not in bounded + ratio]

    ct = ColumnTransformer([
        ("bnd", FunctionTransformer(lambda X: X / 100.0), bounded),
        ("rat", MinMaxScaler(feature_range=(0,1)),       ratio),
        ("unb", Pipeline([
            ("robust", RobustScaler(quantile_range=(5,95))),
            ("std",    StandardScaler()),
            ("mm",     MinMaxScaler(feature_range=(0,1))),
        ]),                                           unbounded),
    ], remainder="drop")

    # fit on TRAIN indicators only
    ct.fit(df_tr[indicator_cols])

    # 5) Per‐day transform & reattach
    def transform_per_day(split_df, split_name):
        arr = np.empty((len(split_df), len(indicator_cols)), dtype=float)
        for day, block in tqdm(
            split_df.groupby(split_df.index.normalize()),
            desc=f"{split_name} days", unit="day"
        ):
            arr[split_df.index.normalize() == day] = ct.transform(block[indicator_cols])

        # build scaled DataFrame
        scal_cols = [c for c in indicator_cols]
        df_scaled = pd.DataFrame(arr, index=split_df.index, columns=scal_cols)

        # reattach raw, bid/ask, label, calendar dims
        for c in split_df.columns:
            if c not in indicator_cols:
                df_scaled[c] = split_df[c]
        return df_scaled

    df_tr_s = transform_per_day(df_tr, "train")
    df_v_s  = transform_per_day(df_v,  "val")
    df_te_s = transform_per_day(df_te, "test")

    # 6) Concatenate splits and return
    df_all = pd.concat([df_tr_s, df_v_s, df_te_s])
    df_all.drop(["open", "high", "low", "close", "volume"], axis=1, inplace=False)
    return df_all.sort_index()


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


