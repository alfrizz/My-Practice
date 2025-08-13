from pathlib import Path 
from datetime import datetime
import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as Funct

#########################################################################################################
ticker = 'AAPL'
save_path  = Path("dfs_training")
model_path = save_path / f"{ticker}_0.1638.pth" # model RMSE

createCSVsign = True
date_to_check = '2015-05' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stocks_folder  = "intraday_stocks" 
optuna_folder = "optuna_results" 

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidasktoclose_spread = 0.03

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
    pre_entry_decay ==> # pre-trade decay of the trades' raw signal (higher: quicker decay [0.01 - 1])
    short_penalty ==> # duration penalty factor (lower: higher penalization [0.01 - 1])
    
    # to define the final buy and sell triggers
    buy_threshold ==> # (percent/100) threshold of the true signal to trigger the final trade
    pred_threshold ==> # (percent/100) threshold of the predicted signal to trigger the final trade
    trailing_stop_thresh ==> # (percent/100) of the trailing stop loss of the final trade
    trailing_stop_pred ==> # (percent/100) of the trailing stop loss of the predicted signal
    '''
    if ticker == 'AAPL':
        features_cols = ['vol_15', 'bb_width_20', 'hour', 'ma_20', 'macd_signal_9', 'low', 'atr_14', 'obv', 'vwap_dev', 'volume_spike', 'r_15', 'close', 'ma_5', 'open', 'high']
        trailing_stop_pred=0.045
        pred_threshold=0.17
        

    return features_cols, trailing_stop_pred, pred_threshold

# automatically executed function to get the parameters for the selected ticker
features_cols_tick, trailing_stop_pred_tick, pred_threshold_tick = signal_parameters(ticker)

#########################################################################################################

# Market Session	        US Market Time (ET)	             Corresponding Time in Datasheet (UTC)
# Premarket             	~4:00 AM – 9:30 AM	             9:00 – 14:30
# Regular Trading	        9:30 AM – 4:00 PM	             14:30 – 21:00
# After-Hours	           ~4:00 PM – 7:00 PM	             21:00 – 00:00

regular_start  = datetime.strptime('14:30', '%H:%M').time()  
regular_start_premarket = datetime.strptime('09:00' , '%H:%M').time()  
regular_end = datetime.strptime('21:00' , '%H:%M').time()   


#########################################################################################################

hparams = {
    # ── Architecture Parameters ────────────────────────────────────────
    "SHORT_UNITS":           64,     # hidden size of daily LSTM; ↑ adds capacity (risk overfitting + slower), ↓ reduces capacity (risk underfitting)
    "LONG_UNITS":            64,    # hidden size of weekly LSTM; ↑ more temporal context (slower/increased memory), ↓ less context (may underfit)
    "DROPOUT_SHORT":         0.4,   # dropout after residual+attention; ↑ stronger regularization (may underlearn), ↓ lighter regularization (risk overfit)
    "DROPOUT_LONG":          0.4,   # dropout after weekly LSTM; ↑ reduces co-adaptation (can underfit), ↓ retains more signal (risk overfit)
    "ATT_HEADS":             4,      # number of attention heads; ↑ finer multi-head subspaces (compute↑), ↓ coarser attention (expressivity↓)
    "ATT_DROPOUT":           0.2,    # dropout inside attention; ↑ more regularization in attention maps, ↓ less regularization (risk overfit)
    "WEIGHT_DECAY":          5e-4,   # L2 penalty on weights; ↑ stronger shrinkage (better generalization/risk underfit), ↓ lighter shrinkage (risk overfit)

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":           32,     # training batch size; ↑ more stable gradients (memory↑, slower per step), ↓ more noisy grads (memory↓, faster per step)
    "VAL_BATCH":             1,      # validation batch size; ↑ faster eval but uses more memory, ↓ slower eval but uses less memory
    "NUM_WORKERS":           2,      # DataLoader workers; ↑ parallel loading (bus error risk + overhead), ↓ safer but less parallelism
    "TRAIN_PREFETCH_FACTOR": 1,      # batches Sto prefetch per worker; ↑ more overlap (shm↑), ↓ less overlap (GPU may stall)
    "MAX_EPOCHS":            60,     # maximum training epochs; ↑ more training (risk wasted compute), ↓ shorter runs (risk undertraining)
    "EARLY_STOP_PATIENCE":   12,     # epochs without val-improve before stop; ↑ more patience (risk overtrain), ↓ less patience (may stop too early)

    # ── Optimizer Settings ─────────────────────────────────────────────
    "LR_EPOCHS_WARMUP":      5,      # epochs to keep LR constant before decay; ↑ longer warmup (stable start/slower), ↓ shorter warmup (faster ramp/risk overshoot)
    "INITIAL_LR":            3e-5,   # starting learning rate; ↑ speeds convergence (risk divergence), ↓ safer steps (slower training)
    "CLIPNORM":              0.5,      # max-gradient norm; ↑ higher clip threshold (less clipping, risk explosion), ↓ lower threshold (more clipping, risk under-update)
    
    # ── CosineAnnealingWarmRestarts Scheduler ──────────────────────────
    "ETA_MIN":               1e-6,   # floor LR in each cosine cycle
    "T_0":                   60,     # epochs before first cosine restart
    "T_MULT":                1,      # cycle length multiplier after each restart

    # ── ReduceLROnPlateau Scheduler ───────────────────────────────────
    "PLATEAU_FACTOR":        0.9,    # multiply LR by this factor on plateau
    "PLATEAU_PATIENCE":      0,      # epochs with no val-improve before LR cut
    "MIN_LR":                1e-6,   # lower bound on LR after reductions
    "PLAT_EPOCHS_WARMUP":    999     # epochs to wait before triggering plateau logic
}
