from pathlib import Path 
from datetime import datetime
import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as Funct

import os
import glob
import json

#########################################################################################################
ticker = 'AAPL'
save_path  = Path("dfs_training")

createCSVsign = False
date_to_check = '2024-06' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stocks_folder  = "intraday_stocks" 
optuna_folder = "optuna_results" 

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidasktoclose_pct = 0.075 # percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes): 0.1% is a conservative one

base_csv = save_path / f"{ticker}_1_base.csv"
sign_csv = save_path / f"{ticker}_2_sign.csv"
feat_csv = save_path / f"{ticker}_3_feat.csv"
pred_csv = save_path / f"{ticker}_4_pred.csv"

label_col = "signal" 

feats_cols_all = [
    "open",             # Opening price
    "high",             # Highest price
    "low",              # Lowest price
    "close",            # Closing price
    "volume",           # Traded volume
    "r_1",              # 1-period log return
    "r_5",              # 5-period log return
    "r_15",             # 15-period log return
    "vol_15",           # Rolling 15-period volatility of r_1
    "volume_spike",     # Current volume / avg volume over 15
    "atr_14",           # 14-period Average True Range
    "vwap_dev",         # (close – VWAP) / VWAP
    "rsi_14",           # 14-period Relative Strength Index
    "bb_width_20",      # (BB upper – BB lower) / MA20
    "stoch_k_14",       # %K of 14-period Stochastic
    "stoch_d_3",        # 3-period SMA of %K
    "ma_5",             # 5-period simple moving average
    "ma_20",            # 20-period simple moving average
    "ma_diff",          # ma_5 – ma_20
    "macd_12_26",       # EMA12 – EMA26
    "macd_signal_9",    # 9-period EMA of MACD
    "obv",              # On-Balance Volume
    "in_trading",       # Within regular trading time
    "hour",             # Hour of the day (0–23)
    "day_of_week",      # Day of week (0=Mon…6=Sun)
    "month",            # Month (1–12)
]

# Market Session	        US Market Time (ET)	             Corresponding Time in Datasheet (UTC)
# Premarket             	~4:00 AM – 9:30 AM	             9:00 – 14:30
# Regular Trading	        9:30 AM – 4:00 PM	             14:30 – 21:00
# After-Hours	           ~4:00 PM – 7:00 PM	             21:00 – 00:00

sess_start         = datetime.strptime('14:30', '%H:%M').time()  
sess_premark       = datetime.strptime('09:00' , '%H:%M').time()  
sess_end           = datetime.strptime('21:00' , '%H:%M').time() 


#########################################################################################################

def load_best_optuna_record(optuna_folder=optuna_folder, ticker=ticker):
    """
    Scan an Optuna‐output folder for JSON files matching '{ticker}_<value>.json',
    ignore any '*_all.json' summary files, pick the one with the highest <value>,
    and return its numeric value plus the 'params' dict.

    Args:
        optuna_folder: path to the directory containing ticker-specific JSONs
        ticker:        stock ticker prefix for each JSON filename

    Returns:
        best_value: float  # the largest numeric prefix extracted from the filename
        best_params: dict  # the 'params' field from that JSON record

    Raises:
        FileNotFoundError: if no matching JSON files are found
        ValueError:       if the numeric prefix cannot be parsed
    """
    # 1) find all files named like "TICKER_*.json", excluding the "*_all.json" summary
    pattern = os.path.join(optuna_folder, f"{ticker}_*.json")
    files = [f for f in glob.glob(pattern) if not f.endswith("_all.json")]
    if not files:
        raise FileNotFoundError(f"No JSON checkpoints found for ticker '{ticker}' in {optuna_folder!r}")

    # 2) pick the file whose suffix (after the underscore) is the largest float
    def extract_value(path: str) -> float:
        name = os.path.splitext(os.path.basename(path))[0]
        # name is "TICKER_<value>", split at first underscore
        _, val_str = name.split("_", 1)
        return float(val_str)

    best_file = max(files, key=extract_value)

    # 3) load and return the record
    with open(best_file, "r") as fp:
        record = json.load(fp)

    best_value  = record["value"]
    best_params = record["params"]
    return best_value, best_params

# automatically executed function to get the optuna values and parameters
best_optuna_value, best_optuna_params = load_best_optuna_record()

#########################################################################################################

def signal_parameters(ticker):
    '''
    # to define the trades
    min_prof_thr ==> # percent of minimum profit to define a potential trade
    max_down_prop ==> # float (percent/100) of maximum allowed drop of a potential trade
    gain_tightening_factor ==> # as gain grows, tighten the stop 'max_down_prop' by this factor.
    merging_retracement_thr ==> # intermediate retracement, relative to the first trade's full range
    merging_time_gap_thr ==> # time gap between trades, relative to the first and second trade durations
    
    # to define the  signal
    pre_entry_decay ==> # per-minute decay rate for the profit-gap:  range: 0.01–0.5  
                                                               #     ↓ → slower fade (signal lingers longer)  
                                                               #     ↑ → faster fade (signal drops off sooner)
    short_penal_decay ==> # strength of the duration penalty:        range: 0.5–2.5  
                                                               #     ↓ → milder penalty (short trades less suppressed)  
                                                               #     ↑ → stronger penalty (short trades heavily suppressed)
    
    # to define the final buy and sell triggers
    buy_threshold ==> # (percent/100) threshold of the true signal to trigger the final trade
    pred_threshold ==> # (percent/100) threshold of the predicted signal to trigger the final trade
    trailing_stop_thresh ==> # (percent/100) of the trailing stop loss of the final trade
    trailing_stop_pred ==> # (percent/100) of the trailing stop loss of the predicted signal

    # to train the model
    look_back ==> length of historical window (how many minutes of history each training example contains): number of past time‐steps fed into the LSTM to predict next value
    '''
    if ticker == 'AAPL':
        look_back = 30
        sess_start_pred = dt.time(*divmod((sess_start.hour * 60 + sess_start.minute) - look_back, 60))
        sess_start_shift = dt.time(*divmod((sess_start.hour * 60 + sess_start.minute) - 2*look_back, 60))
        features_cols = ['vol_15', 'ma_5', 'ma_20', 'close', 'hour', 'bb_width_20', 'high', 'low', 'open', 'vwap_dev', 'atr_14', 'r_5', 'r_1', 'r_15', 'obv']
        trailing_stop_pred = 0.02
        pred_threshold = 0.3
        
    return look_back, sess_start_pred, sess_start_shift, features_cols, trailing_stop_pred, pred_threshold

# automatically executed function to get the parameters for the selected ticker
look_back_tick, sess_start_pred_tick, sess_start_shift_tick, features_cols_tick, trailing_stop_pred_tick, pred_threshold_tick = signal_parameters(ticker)

#########################################################################################################

hparams = {
    # ── Architecture Parameters ────────────────────────────────────────
    "SHORT_UNITS":           128,   # hidden size of daily LSTM; ↑ adds capacity (risk overfitting + slower), ↓ reduces capacity (risk underfitting)
    "LONG_UNITS":            128,   # hidden size of weekly LSTM; ↑ more temporal context (slower/increased memory), ↓ less context (may underfit)
    "DROPOUT_SHORT":         0.1,   # dropout after residual+attention; ↑ stronger regularization (may underlearn), ↓ lighter regularization (risk overfit)
    "DROPOUT_LONG":          0.1,   # dropout after weekly LSTM; ↑ reduces co-adaptation (can underfit), ↓ retains more signal (risk overfit)
    "ATT_HEADS":             16,     # number of attention heads; ↑ finer multi-head subspaces (compute↑), ↓ coarser attention (expressivity↓)
    "ATT_DROPOUT":           0.1,   # dropout inside attention; ↑ more regularization in attention maps, ↓ less regularization (risk overfit)
    "WEIGHT_DECAY":          1e-5,  # L2 penalty on weights; ↑ stronger shrinkage (better generalization/risk underfit), ↓ lighter shrinkage (risk overfit)
    "HUBER_BETA":            0.5,   # β Huber loss: ↑ larger β → more MSE-like (sensitive to big errors), ↓ smaller β → more MAE-like (robust to outliers)
    "CLS_LOSS_WEIGHT":       0.5,   # α classification loss: ↑ emphasize spike/event detection (risk underfitting amplitude), ↓ emphasize regression accuracy (risk smoothing spikes)

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":           16,    # training batch size; ↑ more stable gradients (memory↑, slower per step), ↓ more noisy grads (memory↓, faster per step)
    "VAL_BATCH":             1,     # validation batch size; ↑ faster eval but uses more memory, ↓ slower eval but uses less memory
    "NUM_WORKERS":           2,     # DataLoader workers; ↑ parallel loading (bus error risk + overhead), ↓ safer but less parallelism
    "TRAIN_PREFETCH_FACTOR": 1,     # batches to prefetch per worker; ↑ more overlap (shm↑), ↓ less overlap (GPU may stall)
    "MAX_EPOCHS":            90,    # maximum training epochs; ↑ more training (risk wasted compute), ↓ shorter runs (risk undertraining)
    "EARLY_STOP_PATIENCE":   15,    # epochs without val-improve before stop; ↑ more patience (risk overtrain), ↓ less patience (may stop too early)

    # ── Optimizer Settings ─────────────────────────────────────────────
    "LR_EPOCHS_WARMUP":      5,     # epochs to keep LR constant before decay; ↑ longer warmup (stable start/slower), ↓ shorter warmup (faster ramp/risk overshoot)
    "INITIAL_LR":            1e-3,  # starting learning rate; ↑ speeds convergence (risk divergence), ↓ safer steps (slower training)
    "CLIPNORM":              1.0,  # max-gradient norm; ↑ higher clip threshold (less clipping, risk explosion), ↓ lower threshold (more clipping, risk under-update)
    
    # ── CosineAnnealingWarmRestarts Scheduler ──────────────────────────
    "ETA_MIN":               1e-5,  # floor LR in each cosine cycle
    "T_0":                   90,    # epochs before first cosine restart
    "T_MULT":                1,     # cycle length multiplier after each restart

    # ── ReduceLROnPlateau Scheduler ───────────────────────────────────
    "PLATEAU_FACTOR":        0.9,   # multiply LR by this factor on plateau
    "PLATEAU_PATIENCE":      0,     # epochs with no val-improve before LR cut
    "MIN_LR":                1e-6,  # lower bound on LR after reductions
    "PLAT_EPOCHS_WARMUP":    999    # epochs to wait before triggering plateau logic
}
