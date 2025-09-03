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

#########################################################################################################


# def features_creation(df: pd.DataFrame,
#                       features_cols: list,
#                       label_col: str) -> pd.DataFrame:
#     """
#     Compute core “feat_main_…” indicator series needed downstream.

#     This function:
#       1) Ensures the index is a DateTimeIndex.
#       2) Augments features_cols so all indicators required by the
#          engineering step are always computed: ema, sma, macd_diff,
#          bb_hband, bb_lband, rsi, plus_di, minus_di, adx, obv, obv_sma.
#       3) Precomputes True Range if ATR or ADX are needed.
#       4) For each feature in the (augmented) features_cols, computes:
#          • EMA & SMA
#          • MACD line, signal, histogram
#          • Bollinger bands (upper, lower, width)
#          • RSI (14)
#          • +DI, –DI, ADX (14)
#          • ATR (14)
#          • OBV & 20-period OBV_SMA
#          • VWAP deviation
#          • Log-returns (r_1, r_5, r_15) and rolling vol_15
#          • Volume spike
#          • Stochastic %K (14) & %D (3)
#          • Calendar flags (hour, day_of_week, month)
#       5) Always copies raw OHLC into feat_main_open/high/low/close.
#       6) Returns all columns starting with “feat_main_” plus 'bid', 'ask', and label_col.
#     """
#     # 1) ensure a DateTimeIndex
#     if not isinstance(df.index, pd.DatetimeIndex):
#         df.index = pd.to_datetime(df.index)

#     # 2) guarantee main features for engineering exist
#     required = {"ema","sma","macd_diff","bb_hband","bb_lband",
#                 "rsi","plus_di","minus_di","adx","obv","obv_sma"}
#     features_cols = list(set(features_cols) | required)

#     # 3) precompute True Range if any ADX/ATR feature requested
#     if any(f in features_cols for f in ("atr_14","plus_di","minus_di","adx")):
#         hl = df["high"] - df["low"]
#         hp = (df["high"] - df["close"].shift()).abs()
#         lp = (df["low"]  - df["close"].shift()).abs()
#         tr = pd.concat([hl, hp, lp], axis=1).max(axis=1)

#     # 4) compute each requested main feature
#     for feat in features_cols:
#         if feat == "ema":
#             df["feat_main_ema"] = df["close"].ewm(span=12, adjust=False).mean()

#         elif feat == "sma":
#             df["feat_main_sma"] = df["close"].rolling(26).mean()

#         elif feat in ("macd_diff","macd_line","macd_signal"):
#             # ensure EMA & SMA exist
#             macd_line   = df["feat_main_ema"] - df["feat_main_sma"]
#             macd_signal = macd_line.ewm(span=9, adjust=False).mean()
#             df["feat_main_macd_line"]   = macd_line
#             df["feat_main_macd_signal"] = macd_signal
#             df["feat_main_macd_diff"]   = macd_line - macd_signal

#         elif feat in ("bb_hband","bb_lband"):
#             m20, s20 = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
#             df["feat_main_bb_hband"] = m20 + 2*s20
#             df["feat_main_bb_lband"] = m20 - 2*s20

#         elif feat == "bb_width_20":
#             m20, s20 = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
#             df["feat_main_bb_width_20"] = ( (m20+2*s20) - (m20-2*s20) ) / m20

#         elif feat == "rsi":
#             d    = df["close"].diff()
#             gain = d.clip(lower=0).rolling(14).mean()
#             loss = -d.clip(upper=0).rolling(14).mean()
#             rs   = gain.div(loss).replace([np.inf, -np.inf], 0).fillna(0)
#             df["feat_main_rsi"] = 100 - (100/(1+rs))

#         elif feat in ("plus_di","minus_di","adx"):
#             atr14 = tr.rolling(14).mean()
#             up    = df["high"].diff().clip(lower=0)
#             dn    = df["low"].diff().clip(upper=0).abs()
#             plus_dm  = up.where(up>dn, 0).rolling(14).sum()
#             minus_dm = dn.where(dn>up, 0).rolling(14).sum()
#             df["feat_main_plus_di"]  = 100*(plus_dm/atr14)
#             df["feat_main_minus_di"] = 100*(minus_dm/atr14)
#             df["feat_main_adx"]      = 100*((plus_dm-minus_dm).abs()/(plus_dm+minus_dm)).rolling(14).mean()

#         elif feat == "atr_14":
#             df["feat_main_atr_14"] = tr.rolling(14).mean()

#         elif feat == "obv":
#             dir_ = np.sign(df["close"].diff()).fillna(0)
#             df["feat_main_obv"] = (dir_*df["volume"]).cumsum()

#         elif feat == "obv_sma":
#             df["feat_main_obv_sma"] = df["feat_main_obv"].rolling(20).mean()

#         elif feat == "vwap_dev":
#             tp     = (df["high"] + df["low"] + df["close"]) / 3
#             cum_vp = (tp*df["volume"]).cumsum()
#             cum_v  = df["volume"].cumsum()
#             vwap   = cum_vp.div(cum_v)
#             df["feat_main_vwap_dev"] = (df["close"] - vwap)/vwap

#         elif feat in ("r_1","r_5","r_15"):
#             n = int(feat.split("_")[1])
#             df[f"feat_main_{feat}"] = np.log(df["close"]/df["close"].shift(n))

#         elif feat == "vol_15":
#             df["feat_main_vol_15"] = df.get("feat_main_r_1",
#                                             np.log(df["close"]/df["close"].shift(1))
#                                            ).rolling(15).std()

#         elif feat == "volume_spike":
#             df["feat_main_volume_spike"] = df["volume"]/df["volume"].rolling(15).mean()

#         elif feat in ("stoch_k_14","stoch_d_3"):
#             lo14 = df["low"].rolling(14).min()
#             hi14 = df["high"].rolling(14).max()
#             k    = 100*(df["close"]-lo14)/(hi14-lo14)
#             df["feat_main_stoch_k_14"] = k
#             df["feat_main_stoch_d_3"]  = k.rolling(3).mean()

#         elif feat == "hour":
#             df["feat_main_hour"] = df.index.hour

#         elif feat == "day_of_week":
#             df["feat_main_day_of_week"] = df.index.dayofweek

#         elif feat == "month":
#             df["feat_main_month"] = df.index.month

#     # 5) always copy raw OHLC
#     df["feat_main_open"]  = df["open"]
#     df["feat_main_high"]  = df["high"]
#     df["feat_main_low"]   = df["low"]
#     df["feat_main_close"] = df["close"]
#     df["feat_main_volume"] = df["volume"]

#     # 6) select and return
#     cols = [c for c in df.columns if c.startswith("feat_main_")]
#     cols += ["bid", "ask", label_col]
#     return df.loc[:, cols].dropna()




def features_creation(
    df: pd.DataFrame,
    ma_window: int = 20
) -> pd.DataFrame:
    """
    Compute core “feat_main_…” indicator series using pandas_ta.

    Functionality:
      1) Ensures df.index is DateTimeIndex.
      2) Computes exactly the requested feat_main_<…> columns via pandas_ta
         or simple vectorized ops:
         – EMA(12), SMA(26)
         – MACD line, signal, histogram
         – Bollinger upper, lower, width (20,2)
         – RSI(14)
         – +DI(14), –DI(14), ADX(14)
         – ATR(14)
         – ATR_SMA(ma_window)
         – ATR/price ratio & its SMA(ma_window)
         – OBV & OBV_SMA(ma_window)
         – VWAP deviation
         – Log-returns r_1,r_5,r_15 & rolling vol_15
         – Volume spike over ma_window bars
         – Stochastic %K(14), %D(3)
         – Calendar flags: hour, day_of_week, month
      3) Always copies raw OHLCV to feat_main_open/high/low/close/volume.
      4) Returns all feat_main_… plus bid, ask, and label_col, dropping NaNs.
    """
    # 1) ensure DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    out = df.copy()

    # 2) compute indicators in bulk where possible

    # EMA & SMA
    out["feat_main_ema"] = ta.ema(df["close"], length=12)
    out["feat_main_sma"] = ta.sma(df["close"], length=26)

    # MACD trio
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    out["feat_main_macd_line"]   = macd["MACD_12_26_9"]
    out["feat_main_macd_signal"] = macd["MACDs_12_26_9"]
    out["feat_main_macd_diff"]   = macd["MACDh_12_26_9"]

    # Bollinger bands
    bb = ta.bbands(df["close"], length=20, std=2)
    out["feat_main_bb_lband"]    = bb["BBL_20_2.0"]
    out["feat_main_bb_hband"]    = bb["BBU_20_2.0"]
    out["feat_main_bb_width_20"] = (
        (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) /
         bb["BBM_20_2.0"]
    )

    # RSI
    out["feat_main_rsi"] = ta.rsi(df["close"], length=14)

    # +DI, -DI, ADX
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    out["feat_main_plus_di"]  = adx["DMP_14"]
    out["feat_main_minus_di"] = adx["DMN_14"]
    out["feat_main_adx"]      = adx["ADX_14"]

    # ATR and its rolling SMA
    out["feat_main_atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    out["feat_main_atr_sma"] = out["feat_main_atr_14"].rolling(ma_window).mean()

    # ATR/price ratio and its SMA
    out["feat_main_atr_ratio"] = out["feat_main_atr_14"] / df["close"]
    out["feat_main_atr_ratio_sma"] = out["feat_main_atr_ratio"].rolling(ma_window).mean()

    # OBV and its SMA
    out["feat_main_obv"] = ta.obv(df["close"], df["volume"])
    out["feat_main_obv_sma"] = out["feat_main_obv"].rolling(ma_window).mean()

    # VWAP deviation
    vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    out["feat_main_vwap_dev"] = (df["close"] - vwap) / vwap

    # log-returns and vol_15
    for n in (1,5,15):
        key = f"r_{n}"
        out[f"feat_main_{key}"] = np.log(df["close"] / df["close"].shift(n))
    out["feat_main_vol_15"] = out["feat_main_r_1"].rolling(ma_window).std()

    # volume spike
    out["feat_main_volume_spike"] = df["volume"] / df["volume"].rolling(ma_window).mean()

    # Stochastic %K and %D
    st = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    out["feat_main_stoch_k_14"] = st["STOCHk_14_3_3"]
    out["feat_main_stoch_d_3"]  = st["STOCHd_14_3_3"]

    # calendar flags
    out["feat_main_hour"] = df.index.hour
    out["feat_main_day_of_week"] = df.index.dayofweek
    out["feat_main_month"] = df.index.month

    # 3) copy raw OHLCV
    out["feat_main_open"]   = df["open"]
    out["feat_main_high"]   = df["high"]
    out["feat_main_low"]    = df["low"]
    out["feat_main_close"]  = df["close"]
    out["feat_main_volume"] = df["volume"]

    # 4) select and return
    keep = [c for c in out.columns if c.startswith("feat_main_")]
    keep += ["bid", "ask", params.label_col]
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
    From feat_main_… series, compute continuous buy/sell hint features:

      1) feat_eng_ma      = EMA − SMA
      2) feat_eng_macd    = MACD histogram
      3) feat_eng_bb      = distance to active Bollinger band
      4) feat_eng_rsi     = RSI distance to thresholds
      5) feat_eng_adx     = ADX‐weighted DI diff (rolling‐window)
      6) feat_eng_obv     = OBV − OBV_SMA
      7) feat_eng_atr_div = feat_main_atr_ratio − feat_main_atr_ratio_sma

    Inputs:
      • low_rsi, high_rsi : RSI thresholds
      • adx_thresh,ad x_window : ADX entry & scaling window
    """
    price = df["feat_main_close"]

    # 1) MA spread
    df["feat_eng_ma"] = (df["feat_main_ema"] - df["feat_main_sma"]).round(3)

    # 2) MACD histogram
    df["feat_eng_macd"] = df["feat_main_macd_diff"].round(3)

    # 3) Bollinger distance (hold state no gaps)
    lo, hi = df["feat_main_bb_lband"], df["feat_main_bb_hband"]
    st_bb = pd.Series(
        np.where(price<lo,1,np.where(price>hi,-1,np.nan)),
        index=df.index
    ).ffill().fillna(0)
    df["feat_eng_bb"] = np.where(
        st_bb>0, lo-price,
        np.where(st_bb<0, hi-price, 0)
    ).round(3)

    # 4) RSI distance (hold state)
    rsi = df["feat_main_rsi"]
    st_rsi = pd.Series(
        np.where(rsi<low_rsi,1,np.where(rsi>high_rsi,-1,np.nan)),
        index=df.index
    ).ffill().fillna(0)
    df["feat_eng_rsi"] = np.where(
        st_rsi>0, low_rsi-rsi,
        np.where(st_rsi<0, high_rsi-rsi, 0)
    ).round(3)

    # 5) ADX‐weighted DI diff (rolling scaling)
    plus, minus, adx = df["feat_main_plus_di"], df["feat_main_minus_di"], df["feat_main_adx"]
    di = (plus-minus).abs()
    ex = (adx-adx_thresh).clip(lower=0)
    raw = di*ex
    scale = (
        (adx.rolling(adx_window).max()-adx_thresh) /
        raw.rolling(adx_window).max()
    ).replace([np.inf,-np.inf],0).fillna(0)
    df["feat_eng_adx"] = np.where(
        adx>adx_thresh,
        np.where(plus>minus, di*ex*scale, -di*ex*scale),
        0
    ).round(3)

    # 6) OBV divergence
    df["feat_eng_obv"] = (df["feat_main_obv"] - df["feat_main_obv_sma"]).round(3)

    # 7) ATR ratio divergence
    df["feat_eng_atr_div"] = ((df["feat_main_atr_ratio"] - df["feat_main_atr_ratio_sma"])* 10_000).round(1)

    return df



#########################################################################################################


# def scale_with_splits(
#     df: pd.DataFrame,
#     features_cols: list[str],
#     label_col: str,
#     train_prop: float = 0.70,
#     val_prop: float   = 0.15
# ) -> pd.DataFrame:
#     """
#     1) Copy and chronologically split df into train/val/test by row index.
#     2) Encode 'hour', 'day_of_week', 'month' as cyclic sin/cos features.
#     3) Define feature groups:
#        - price_feats   : open, high, low, close, volume, ATR, moving averages, etc.
#        - ratio_feats   : returns, vol_15, vwap_dev, indicators, etc.
#        - binary_feats  : in_trading flag
#        - cyclic_feats  : hour, day_of_week, month (post‐PCA)
#     4) Fit StandardScaler on TRAIN’s ratio_feats.
#     5) **Per-day expanding robust scaling** on price_feats:
#        for each calendar day, use only data ≤ current bar to compute 
#        median & IQR, then scale that bar.
#     6) Apply ratio and price scalers to train/val/test splits.
#     7) Fit PCA(1) on train’s sin/cos pairs and transform all splits back to single
#        'hour', 'day_of_week', 'month' columns.
#     8) Reattach label_col, concat splits, select and return final columns.
#     """
#     df = df.copy()

#     # 1) Split into train/val/test
#     n       = len(df)
#     n_train = int(n * train_prop)
#     n_val   = int(n * val_prop)
#     if n_train + n_val >= n:
#         raise ValueError("train_prop + val_prop must sum to < 1.0")

#     df_tr = df.iloc[:n_train].copy()
#     df_v  = df.iloc[n_train : n_train + n_val].copy()
#     df_te = df.iloc[n_train + n_val :].copy()

#     # 2) Generate cyclic sin/cos for time features
#     for sub in (df_tr, df_v, df_te):
#         h = sub["hour"]
#         sub["hour_sin"],     sub["hour_cos"]     = np.sin(2*np.pi*h/24),     np.cos(2*np.pi*h/24)
#         d = sub["day_of_week"]
#         sub["day_of_week_sin"], sub["day_of_week_cos"] = np.sin(2*np.pi*d/7),  np.cos(2*np.pi*d/7)
#         m = sub["month"]
#         sub["month_sin"],    sub["month_cos"]    = np.sin(2*np.pi*m/12),    np.cos(2*np.pi*m/12)

#     # 3) Define feature groups
#     all_price = ["open","high","low","close","volume","atr_14",
#                  "ma_5","ma_20","ma_diff","macd_12_26","macd_signal_9","obv"]
#     all_ratio = ["r_1","r_5","r_15","vol_15","volume_spike",
#                  "vwap_dev","rsi_14","bb_width_20","stoch_k_14","stoch_d_3"]

#     price_feats  = [f for f in all_price  if f in features_cols]
#     ratio_feats  = [f for f in all_ratio  if f in features_cols]
#     binary_feats = [f for f in ["in_trading"] if f in features_cols]
#     cyclic_feats = ["hour","day_of_week","month"]

#     # 4) Fit StandardScaler on TRAIN ratio_feats
#     ratio_scaler = StandardScaler()
#     if ratio_feats:
#         ratio_scaler.fit(df_tr[ratio_feats])

#     # 5) Per-day expanding robust scaling on price_feats (no leakage)
#     def scale_price_per_day(sub: pd.DataFrame, desc: str) -> pd.DataFrame:
#         out = sub.copy()
#         days = out.index.normalize().unique()
#         for day in tqdm(days, desc=f"Scaling price per day ({desc})", unit="day"):
#             mask  = out.index.normalize() == day
#             block = out.loc[mask, price_feats]

#             # expanding median & IQR up to each bar
#             med = block.expanding().median()
#             q75 = block.expanding().quantile(0.75)
#             q25 = block.expanding().quantile(0.25)
#             iqr = (q75 - q25).replace(0, 1e-6)

#             # apply element-wise: bar i uses only bars ≤ i
#             out.loc[mask, price_feats] = (block - med) / iqr
#         return out

#     # 6) Transform splits
#     def transform(sub: pd.DataFrame, split_name: str) -> pd.DataFrame:
#         out = sub.copy()
#         if price_feats:
#             out = scale_price_per_day(out, split_name)
#         if ratio_feats:
#             out[ratio_feats] = ratio_scaler.transform(sub[ratio_feats])
#         return out

#     df_tr_s  = transform(df_tr,  "train")
#     df_val_s = transform(df_v,   "val")
#     df_te_s  = transform(df_te,  "test")

#     # 7) PCA(1) on sin/cos pairs, fit on TRAIN → transform all
#     pca_hour = PCA(n_components=1).fit(df_tr_s[["hour_sin","hour_cos"]])
#     pca_dow  = PCA(n_components=1).fit(df_tr_s[["day_of_week_sin","day_of_week_cos"]])
#     pca_mo   = PCA(n_components=1).fit(df_tr_s[["month_sin","month_cos"]])

#     def apply_cyclic_pca(sub: pd.DataFrame) -> pd.DataFrame:
#         out = sub.copy()
#         out["hour"]        = np.round(pca_hour.transform(out[["hour_sin","hour_cos"]])[:,0], 3)
#         out["day_of_week"] = np.round(pca_dow.transform(out[["day_of_week_sin","day_of_week_cos"]])[:,0], 3)
#         out["month"]       = np.round(pca_mo.transform(out[["month_sin","month_cos"]])[:,0], 3)
#         out.drop([
#             "hour_sin","hour_cos",
#             "day_of_week_sin","day_of_week_cos",
#             "month_sin","month_cos"
#         ], axis=1, inplace=True)
#         return out

#     df_tr_s  = apply_cyclic_pca(df_tr_s)
#     df_val_s = apply_cyclic_pca(df_val_s)
#     df_te_s  = apply_cyclic_pca(df_te_s)

#     # 8) Reattach label and recombine
#     for part in (df_tr_s, df_val_s, df_te_s):
#         part[label_col] = df.loc[part.index, label_col]

#     df_final = pd.concat([df_tr_s, df_val_s, df_te_s]).sort_index()

#     # 9) Return only final ordered columns
#     final_cols = (
#         price_feats +
#         ratio_feats +
#         binary_feats +
#         cyclic_feats +
#         ["bid", "ask"] +
#         [label_col]
#     )
#     return df_final[final_cols]




def scale_with_splits(
    df: pd.DataFrame,
    train_prop: float = params.train_prop,
    val_prop: float   = params.val_prop
) -> pd.DataFrame:
    """
    Scale price, ratio, and cyclic features reliably without look-ahead:

      1) Chronologically split df into train/val/test by row index.
      2) Create sin/cos for feat_main_hour, feat_main_day_of_week, feat_main_month.
      3) Discover feature‐groups:
         • price_feats : raw OHLCV, ATR, OBV, Bollinger bands (robust‐scaled per bar)
         • ratio_feats : returns, vol_15, vwap_dev, RSI, stoch, feat_eng_… (standard‐scaled)
         • cyclic_feats: hour, day_of_week, month (PCA of sin/cos)
         • binary_feats: any “in_trading” flag
      4) Fit StandardScaler on TRAIN’s ratio_feats.
      5) **Vectorized** per‐bar robust scaling on price_feats:
         use groupby(...).expanding() to compute each bar’s day‐to‐date median & IQR,
         and scale in one go—no Python loop over days.
      6) Apply ratio‐scaler to ratio_feats.
      7) Fit PCA(1) on train’s sin/cos, transform all splits back to single cyclic cols.
      8) Reattach label_col, concat splits, and return columns in order.
    """
    df = df.copy()
    # -------------------------------------------------------------------------
    # 1) split
    # -------------------------------------------------------------------------
    n, n_tr = len(df), int(len(df)*train_prop)
    n_val   = int(len(df)*val_prop)
    if n_tr + n_val >= n:
        raise ValueError("train_prop + val_prop must sum to < 1.0")
    df_tr = df.iloc[:n_tr].copy()
    df_v  = df.iloc[n_tr: n_tr+n_val].copy()
    df_te = df.iloc[n_tr+n_val:].copy()

    # -------------------------------------------------------------------------
    # 2) sin/cos
    # -------------------------------------------------------------------------
    for part in (df_tr, df_v, df_te):
        h = part["feat_main_hour"]
        part["hour_sin"], part["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
        d = part["feat_main_day_of_week"]
        part["dow_sin"], part["dow_cos"]   = np.sin(2*np.pi*d/7),  np.cos(2*np.pi*d/7)
        m = part["feat_main_month"]
        part["mo_sin"], part["mo_cos"]     = np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)

    # -------------------------------------------------------------------------
    # 3) feature groups
    # -------------------------------------------------------------------------
    cols = df.columns

    price_feats = [
        c for c in cols if c in (
            "feat_main_open","feat_main_high","feat_main_low","feat_main_close",
            "feat_main_volume","feat_main_atr_14","feat_main_atr_sma",
            "feat_main_obv","feat_main_obv_sma",
            "feat_main_bb_lband","feat_main_bb_hband"
        )
    ]

    ratio_feats = (
        [c for c in cols if c.startswith(("feat_main_r_","feat_main_vol_"))]
        + [c for c in cols if c in (
            "feat_main_volume_spike","feat_main_vwap_dev","feat_main_rsi",
            "feat_main_bb_width_20","feat_main_stoch_k_14","feat_main_stoch_d_3",
            "feat_main_atr_ratio","feat_main_atr_ratio_sma"
        )]
        + [c for c in cols if c.startswith("feat_eng_")]
    )

    binary_feats = [c for c in cols if c == "in_trading"]
    cyclic_feats = ["hour","day_of_week","month"]

    # -------------------------------------------------------------------------
    # 4) fit ratio scaler
    # -------------------------------------------------------------------------
    ratio_scaler = StandardScaler()
    if ratio_feats:
        ratio_scaler.fit(df_tr[ratio_feats])

    # -------------------------------------------------------------------------
    # 5) vectorized robust scaling of price_feats
    # -------------------------------------------------------------------------
    def scale_price_vec(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.copy()
        # group-key: normalized date for each row
        days = out.index.normalize()

        # expanding() will give a MultiIndex (date, timestamp).
        grp = out[price_feats].groupby(days).expanding()

        # compute per-bar day‐to‐date medians & quantiles
        med = grp.median().reset_index(level=0, drop=True)
        q75 = grp.quantile(0.75).reset_index(level=0, drop=True)
        q25 = grp.quantile(0.25).reset_index(level=0, drop=True)
        iqr = (q75 - q25).replace(0, 1e-6)

        # assign robust‐scaled values
        out[price_feats] = (out[price_feats] - med) / iqr
        return out

    def transform_split(sub: pd.DataFrame, tag: str) -> pd.DataFrame:
        out = sub.copy()
        if price_feats:
            out = scale_price_vec(out)
        if ratio_feats:
            out[ratio_feats] = ratio_scaler.transform(out[ratio_feats])
        return out

    df_tr_s  = transform_split(df_tr,  "train")
    df_val_s = transform_split(df_v,   "val")
    df_te_s  = transform_split(df_te,  "test")

    # -------------------------------------------------------------------------
    # 6) PCA on sin/cos → single cyclic columns
    # -------------------------------------------------------------------------
    pca_h = PCA(1).fit(df_tr_s[["hour_sin","hour_cos"]])
    pca_d = PCA(1).fit(df_tr_s[["dow_sin","dow_cos"]])
    pca_m = PCA(1).fit(df_tr_s[["mo_sin","mo_cos"]])

    def apply_pca(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.copy()
        out["hour"]        = pca_h.transform(out[["hour_sin","hour_cos"]])[:,0].round(3)
        out["day_of_week"] = pca_d.transform(out[["dow_sin","dow_cos"]])[:,0].round(3)
        out["month"]       = pca_m.transform(out[["mo_sin","mo_cos"]])[:,0].round(3)

        out.drop([
            "feat_main_hour","feat_main_day_of_week","feat_main_month",
            "hour_sin","hour_cos","dow_sin","dow_cos","mo_sin","mo_cos"
        ], axis=1, inplace=True)
        return out

    df_tr_s  = apply_pca(df_tr_s)
    df_val_s = apply_pca(df_val_s)
    df_te_s  = apply_pca(df_te_s)

    # -------------------------------------------------------------------------
    # 7) reattach label, concat, select order
    # -------------------------------------------------------------------------
    for part in (df_tr_s, df_val_s, df_te_s):
        part[params.label_col] = df.loc[part.index, params.label_col]

    df_final = pd.concat([df_tr_s, df_val_s, df_te_s]).sort_index()
    final_cols = price_feats + ratio_feats + binary_feats + cyclic_feats + ["bid","ask"] + [params.label_col]
    return df_final[final_cols]
