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

from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import pandas_ta as ta

from sklearn.preprocessing import StandardScaler, RobustScaler

from captum.attr import IntegratedGradients

#########################################################################################################


def features_creation(
    df: pd.DataFrame,
    ma_window: int = 20
) -> pd.DataFrame:
    """
    1) Ensure DateTimeIndex.
    2) Compute base technical indicators on close, high, low, volume:
       - ema(12), sma(26)
       - macd line/signal/diff
       - bollinger bands (lband, hband, width)
       - rsi(14)
       - +DI, -DI, adx(14)
       - atr(14) and rolling atr_sma
       - atr_ratio and rolling atr_ratio_sma
       - obv and rolling obv_sma
       - vwap deviation
       - log‐returns r_1, r_5, r_15 and vol_15
       - volume_spike
       - stoch_k_14, stoch_d_3
       - calendar flags: hour, day_of_week, month
    3) Copy raw open/high/low/close/volume.
    4) Drop initial NaNs and return all features + bid, ask, label.
    """
    # 1) enforce datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    out = df.copy()

    # 2) base indicators
    out["ema"]        = ta.ema(df["close"], length=12)
    out["sma"]        = ta.sma(df["close"], length=26)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    out["macd_line"]   = macd["MACD_12_26_9"]
    out["macd_signal"] = macd["MACDs_12_26_9"]
    out["macd_diff"]   = macd["MACDh_12_26_9"]

    bb = ta.bbands(df["close"], length=20, std=2)
    out["bb_lband"]    = bb["BBL_20_2.0"]
    out["bb_hband"]    = bb["BBU_20_2.0"]
    out["bb_width_20"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / bb["BBM_20_2.0"]

    out["rsi"] = ta.rsi(df["close"], length=14)

    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    out["plus_di"]  = adx["DMP_14"]
    out["minus_di"] = adx["DMN_14"]
    out["adx"]      = adx["ADX_14"]

    out["atr_14"]     = ta.atr(df["high"], df["low"], df["close"], length=14)
    out["atr_sma"]    = out["atr_14"].rolling(ma_window).mean()

    out["atr_ratio"]        = out["atr_14"] / df["close"]
    out["atr_ratio_sma"]    = out["atr_ratio"].rolling(ma_window).mean()

    out["obv"]        = ta.obv(df["close"], df["volume"])
    out["obv_sma"]    = out["obv"].rolling(ma_window).mean()

    vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    out["vwap_dev"]   = (df["close"] - vwap) / vwap

    for n in (1, 5, 15):
        out[f"r_{n}"] = np.log(df["close"] / df["close"].shift(n))
    out["vol_15"] = out["r_1"].rolling(ma_window).std()

    out["volume_spike"] = df["volume"] / df["volume"].rolling(ma_window).mean()

    st = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    out["stoch_k_14"] = st["STOCHk_14_3_3"]
    out["stoch_d_3"]  = st["STOCHd_14_3_3"]

    # calendar flags
    out["hour"]        = df.index.hour
    out["day_of_week"] = df.index.dayofweek
    out["month"]       = df.index.month

    # 3) raw OHLCV
    out["open"]   = df["open"]
    out["high"]   = df["high"]
    out["low"]    = df["low"]
    out["close"]  = df["close"]
    out["volume"] = df["volume"]

    # 4) select & drop NaNs
    keep = [
        "ema","sma",
        "macd_line","macd_signal","macd_diff",
        "bb_lband","bb_hband","bb_width_20",
        "rsi","plus_di","minus_di","adx",
        "atr_14","atr_sma","atr_ratio","atr_ratio_sma",
        "obv","obv_sma","vwap_dev",
        "r_1","r_5","r_15","vol_15","volume_spike",
        "stoch_k_14","stoch_d_3",
        "hour","day_of_week","month",
        "open","high","low","close","volume",
        "bid","ask", params.label_col
    ]
    return out.loc[:, keep].dropna()


#########################################################################################################


def features_engineering(
    df: pd.DataFrame,
    low_rsi: int = 30,
    high_rsi: int = 70,
    adx_thresh: int = 20,
    adx_window: int = 7
) -> pd.DataFrame:
    """
    Build higher‐level trading signals from base features:
      1) eng_ma       : ema - sma
      2) eng_macd     : macd_diff
      3) eng_bb       : distance to Bollinger bands, holding state
      4) eng_rsi      : distance to RSI thresholds, holding state
      5) eng_adx      : ADX‐weighted DI spread with rolling scale
      6) eng_obv      : obv - obv_sma divergence
      7) eng_atr_div  : ATR/price ratio divergence scaled
    """
    price = df["close"]

    # 1) MA spread
    df["eng_ma"]   = (df["ema"] - df["sma"]).round(3)

    # 2) MACD histogram
    df["eng_macd"] = df["macd_diff"].round(3)

    # 3) Bollinger distance (stateful)
    lo, hi = df["bb_lband"], df["bb_hband"]
    st_bb = (
        pd.Series(
            np.where(price < lo,  1,
                     np.where(price > hi, -1, np.nan)),
            index=df.index
        )
        .ffill()
        .fillna(0)
    )
    df["eng_bb"] = np.where(
        st_bb > 0,  lo - price,
        np.where(st_bb < 0, hi - price, 0)
    ).round(3)

    # 4) RSI distance (stateful)
    rsi    = df["rsi"]
    st_rsi = (
        pd.Series(
            np.where(rsi < low_rsi,  1,
                     np.where(rsi > high_rsi, -1, np.nan)),
            index=df.index
        )
        .ffill()
        .fillna(0)
    )
    df["eng_rsi"] = np.where(
        st_rsi > 0,  low_rsi - rsi,
        np.where(st_rsi < 0, high_rsi - rsi, 0)
    ).round(3)

    # 5) ADX‐weighted DI diff
    plus, minus, adx = df["plus_di"], df["minus_di"], df["adx"]
    di    = (plus - minus).abs()
    ex    = (adx - adx_thresh).clip(lower=0)
    raw   = di * ex
    scale = (
        (adx.rolling(adx_window).max() - adx_thresh) /
        raw.rolling(adx_window).max()
    ).replace([np.inf, -np.inf], 0).fillna(0)
    df["eng_adx"] = np.where(
        adx > adx_thresh,
        np.where(plus > minus,  di * ex * scale, -di * ex * scale),
        0
    ).round(3)

    # 6) OBV divergence
    df["eng_obv"] = (df["obv"] - df["obv_sma"]).round(3)

    # 7) ATR/price ratio divergence
    df["eng_atr_div"] = ((df["atr_ratio"] - df["atr_ratio_sma"]) * 10_000).round(1)

    return df


#########################################################################################################


def scale_with_splits(
    df: pd.DataFrame,
    train_prop: float = params.train_prop,
    val_prop: float   = params.val_prop
) -> pd.DataFrame:
    """
    1) Chronologically split df into train/val/test by index order.
    2) Encode calendar fields (hour, day_of_week, month) as sin/cos pairs.
    3) Auto-identify feature groups:
         • price_feats : open, high, low, close, volume,
                         atr_14, atr_sma, obv, obv_sma, bb_lband, bb_hband
         • ratio_feats : r_*, vol_15, volume_spike, vwap_dev, rsi,
                         bb_width_20, stoch_k_14, stoch_d_3,
                         atr_ratio, atr_ratio_sma, + all eng_* columns
         • binary_feats: in_trading (if present)
         • cyclic_feats: hour, day_of_week, month
    4) Fit StandardScaler on TRAIN’s ratio_feats.
    5) Robust-scale price_feats per calendar day using a single
       vectorized groupby(...).expanding(): compute each bar’s
       day-to-date median & IQR, then scale—no Python‐level loop.
    6) Transform each split: scale price_feats + apply ratio_scaler.
    7) Fit PCA(1) on TRAIN’s sin/cos pairs → compress back to
       single hour, day_of_week, month columns.
    8) Reattach label, concatenate splits, return only the
       selected final columns in order.
    """
    # Make a working copy
    df = df.copy()

    # 1) Chronological split
    n       = len(df)
    n_train = int(n * train_prop)
    n_val   = int(n * val_prop)
    if n_train + n_val >= n:
        raise ValueError("train_prop + val_prop must sum to < 1.0")

    df_tr = df.iloc[:n_train].copy()
    df_v  = df.iloc[n_train : n_train + n_val].copy()
    df_te = df.iloc[n_train + n_val :].copy()

    # 2) Sin/cos encode calendar features
    for part in (df_tr, df_v, df_te):
        h = part["hour"]
        part["hour_sin"], part["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
        d = part["day_of_week"]
        part["dow_sin"], part["dow_cos"]   = np.sin(2*np.pi*d/7),  np.cos(2*np.pi*d/7)
        m = part["month"]
        part["mo_sin"], part["mo_cos"]     = np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)

    # 3) Auto-discover feature groups
    cols         = df.columns
    price_feats  = [c for c in cols if c in (
        "open","high","low","close","volume",
        "atr_14","atr_sma","obv","obv_sma",
        "bb_lband","bb_hband"
    )]
    ratio_feats  = (
        [c for c in cols if c.startswith(("r_","vol_"))] +
        [c for c in cols if c in (
            "volume_spike","vwap_dev","rsi",
            "bb_width_20","stoch_k_14","stoch_d_3",
            "atr_ratio","atr_ratio_sma"
        )] +
        [c for c in cols if c.startswith("eng_")]
    )
    binary_feats = [c for c in cols if c == "in_trading"]
    cyclic_feats = ["hour","day_of_week","month"]

    # 4) Fit StandardScaler on TRAIN’s ratio_feats
    ratio_scaler = StandardScaler()
    if ratio_feats:
        ratio_scaler.fit(df_tr[ratio_feats])

    # 5) Vectorized robust scaling of price_feats
    def scale_price_vec(sub: pd.DataFrame) -> pd.DataFrame:
        out  = sub.copy()
        days = out.index.normalize()
        grp  = out[price_feats].groupby(days).expanding()

        # day-to-date median & IQR for each bar
        med = grp.median().reset_index(level=0, drop=True)
        q75 = grp.quantile(0.75).reset_index(level=0, drop=True)
        q25 = grp.quantile(0.25).reset_index(level=0, drop=True)
        iqr = (q75 - q25).replace(0, 1e-6)

        out[price_feats] = (out[price_feats] - med) / iqr
        return out

    # 6) Transform splits with tqdm progress
    def transform_split(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.copy()
        if price_feats:
            out = scale_price_vec(out)
        if ratio_feats:
            out[ratio_feats] = ratio_scaler.transform(out[ratio_feats])
        return out

    scaled = {}
    for name, subset in tqdm(
        [("train", df_tr), ("val", df_v), ("test", df_te)],
        desc="Scaling splits", unit="split"
    ):
        scaled[name] = transform_split(subset)

    df_tr_s, df_val_s, df_te_s = scaled["train"], scaled["val"], scaled["test"]

    # 7) PCA on sin/cos → compress back to cyclic_feats
    pca_h = PCA(n_components=1).fit(df_tr_s[["hour_sin","hour_cos"]])
    pca_d = PCA(n_components=1).fit(df_tr_s[["dow_sin","dow_cos"]])
    pca_m = PCA(n_components=1).fit(df_tr_s[["mo_sin","mo_cos"]])

    def apply_pca(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.copy()
        out["hour"]        = pca_h.transform(out[["hour_sin","hour_cos"]])[:,0].round(3)
        out["day_of_week"] = pca_d.transform(out[["dow_sin","dow_cos"]])[:,0].round(3)
        out["month"]       = pca_m.transform(out[["mo_sin","mo_cos"]])[:,0].round(3)
        out.drop([
            "hour_sin","hour_cos",
            "dow_sin","dow_cos",
            "mo_sin","mo_cos"
        ], axis=1, inplace=True)
        return out

    for name, subset in tqdm(
        [("train", df_tr_s), ("val", df_val_s), ("test", df_te_s)],
        desc="Applying PCA", unit="split"
    ):
        scaled[name] = apply_pca(subset)

    df_tr_s, df_val_s, df_te_s = scaled["train"], scaled["val"], scaled["test"]

    # 8) Reattach label and concatenate final splits
    for part in (df_tr_s, df_val_s, df_te_s):
        part[params.label_col] = df.loc[part.index, params.label_col]

    df_final = pd.concat([df_tr_s, df_val_s, df_te_s]).sort_index()
    final_cols = (
        price_feats + ratio_feats + binary_feats +
        cyclic_feats + ["bid","ask"] + [params.label_col]
    )
    return df_final[final_cols]


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


