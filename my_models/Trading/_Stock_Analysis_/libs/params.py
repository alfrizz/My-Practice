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
model_path = save_path / f"{ticker}_0.2419.pth" # model RMSE

date_to_check = None # to analyze all dates save the final CSV
# date_to_check = '2004-03' # set to None to analyze all dates and save the "ready" CSV

date_to_test = '2004-03' # in the ML_Results notebook and ML_DF_prep (if date_to_check==None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stocks_folder  = "intraday_stocks" 
optuna_folder = "optuna_results" 

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
is_centered = True # smoothing and centering using past and future data (True) or only with past data without centering (False)
bidasktoclose_spread = 0.03

base_csv = save_path / f"{ticker}_1_base.csv"
ready_csv = save_path / f"{ticker}_2_ready.csv"
final_csv = save_path / f"{ticker}_3_final.csv"
pred_csv = save_path / f"{ticker}_4_preds.csv"

label_col = "signal" 

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
    smooth_win_sig ==> # smoothing window of the signal used for the identification of the final trades 
    pre_entry_decay ==> # pre-trade decay of the trades' raw signal (higher: quicker decay [0.01 - 1])
    short_penalty ==> # duration penalty factor (lower: higher penalization [0.01 - 1])
    percentile_ref ==> # percentile (eg 50 if median) of the percentage profit to use as reference to scale the signal
    
    # to define the final buy and sell triggers
    buy_threshold ==> # (percent/100) threshold of the true signal to trigger the final trade
    pred_threshold ==> # (percent/100) threshold of the predicted signal to trigger the final trade
    trailing_stop_thresh ==> # (percent/100) of the trailing stop loss of the final trade
    trailing_stop_pred ==> # (percent/100) of the trailing stop loss of the predicted signal
    '''
    if ticker == 'AAPL':
        features_cols = ['obv', 'hour', 'high', 'low', 'vwap_dev', 'open', 'ma_20', 'ma_5', 'close', 'atr_14', 'macd_12_26', 'bb_width_20', 'in_trading']
        look_back=90
        # to define the initial trades:
        min_prof_thr=0.05
        max_down_prop=0.86
        gain_tightening_factor=0.2
        merging_retracement_thr=0.13
        merging_time_gap_thr=0.12
        # to define the true signal:
        smooth_win_sig=1
        pre_entry_decay=0.5
        short_penalty=0.39
        percentile_ref=75
        # to define the final buy and sell triggers:
        trailing_stop_thresh=0.01
        trailing_stop_pred=0.01
        buy_threshold=0.01
        pred_threshold=0.01
        
    if ticker == 'GOOGL':
        features_cols = ['obv', 'hour', 'high', 'low', 'vwap_dev', 'open', 'ma_20', 'ma_5', 'close', 'atr_14', 'macd_12_26', 'bb_width_20', 'in_trading']
        look_back=120
        # to define the initial trades:
        min_prof_thr= 0.1376
        max_down_prop=0.4278
        gain_tightening_factor=0.5446
        merging_retracement_thr=0.1240
        merging_time_gap_thr=0.2310
        # to define the true signal:
        smooth_win_sig=1
        pre_entry_decay=0.4659
        short_penalty=0.0508
        percentile_ref=75
        # to define the final buy and sell triggers:
        trailing_stop_thresh=0.0654
        trailing_stop_pred=0.03
        buy_threshold=0.1806
        pred_threshold=0.33
 
    if ticker == 'TSLA':
        # features_cols = ['obv', 'hour', 'high', 'low', 'vwap_dev', 'open', 'ma_20', 'ma_5', 'close', 'atr_14', 'macd_12_26', 'bb_width_20', 'in_trading']
        look_back=90
        # to define the initial trades:
        min_prof_thr=0.45 
        max_down_prop=0.3
        gain_tightening_factor=0.02
        merging_retracement_thr=0.9
        merging_time_gap_thr=0.7
        # to define the true signal:
        smooth_win_sig=3  
        pre_entry_decay=0.6
        short_penalty=0.1
        percentile_ref=75
        # to define the final buy and sell triggers:
        trailing_stop_thresh=0.1 
        trailing_stop_pred=0.6 #0.16
        buy_threshold=0.1 
        pred_threshold=0.4 #0.3

    return features_cols, look_back, min_prof_thr, max_down_prop, gain_tightening_factor, merging_retracement_thr, merging_time_gap_thr,  \
        smooth_win_sig, pre_entry_decay, short_penalty, percentile_ref, trailing_stop_thresh, trailing_stop_pred, buy_threshold, pred_threshold

# automatically executed function to get the parameters for the selected ticker
features_cols_tick, look_back_tick, min_prof_thr_tick, max_down_prop_tick, gain_tightening_factor_tick, merging_retracement_thr_tick, merging_time_gap_thr_tick, smooth_win_sig_tick, \
pre_entry_decay_tick, short_penalty_tick, percentile_ref_tick, trailing_stop_thresh_tick, trailing_stop_pred_tick, buy_threshold_tick, pred_threshold_tick = signal_parameters(ticker)

#########################################################################################################

# Market Session	        US Market Time (ET)	             Corresponding Time in Datasheet (UTC)
# Premarket             	~4:00 AM – 9:30 AM	             9:00 – 14:30
# Regular Trading	        9:30 AM – 4:00 PM	             14:30 – 21:00
# After-Hours	           ~4:00 PM – 7:00 PM	             21:00 – 00:00

regular_start  = datetime.strptime('14:30', '%H:%M').time()  
regular_start_pred = dt.time(*divmod(regular_start.hour * 60 + regular_start.minute - look_back_tick, 60))
regular_start_shifted = dt.time(*divmod(regular_start.hour * 60 + regular_start.minute - look_back_tick*2, 60))
regular_end = datetime.strptime('21:00' , '%H:%M').time()   


#########################################################################################################

hparams = {
    # ── Architecture Parameters ────────────────────────────────────────
    "SHORT_UNITS":           96,     # hidden size of daily LSTM; ↑ adds capacity (risk overfitting + slower), ↓ reduces capacity (risk underfitting)
    "LONG_UNITS":            128,    # hidden size of weekly LSTM; ↑ more temporal context (slower/increased memory), ↓ less context (may underfit)
    "DROPOUT_SHORT":         0.2,   # dropout after residual+attention; ↑ stronger regularization (may underlearn), ↓ lighter regularization (risk overfit)
    "DROPOUT_LONG":          0.25,   # dropout after weekly LSTM; ↑ reduces co-adaptation (can underfit), ↓ retains more signal (risk overfit)
    "ATT_HEADS":             8,      # number of attention heads; ↑ finer multi-head subspaces (compute↑), ↓ coarser attention (expressivity↓)
    "ATT_DROPOUT":           0.15,    # dropout inside attention; ↑ more regularization in attention maps, ↓ less regularization (risk overfit)
    "WEIGHT_DECAY":          1e-5,   # L2 penalty on weights; ↑ stronger shrinkage (better generalization/risk underfit), ↓ lighter shrinkage (risk overfit)

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":           32,     # training batch size; ↑ more stable gradients (memory↑, slower per step), ↓ more noisy grads (memory↓, faster per step)
    "VAL_BATCH":             1,      # validation batch size; ↑ faster eval but uses more memory, ↓ slower eval but uses less memory
    "NUM_WORKERS":           2,      # DataLoader workers; ↑ parallel loading (bus error risk + overhead), ↓ safer but less parallelism
    "TRAIN_PREFETCH_FACTOR": 1,      # batches to prefetch per worker; ↑ more overlap (shm↑), ↓ less overlap (GPU may stall)
    "MAX_EPOCHS":            60,     # maximum training epochs; ↑ more training (risk wasted compute), ↓ shorter runs (risk undertraining)
    "EARLY_STOP_PATIENCE":   12,     # epochs without val-improve before stop; ↑ more patience (risk overtrain), ↓ less patience (may stop too early)

    # ── Optimizer Settings ─────────────────────────────────────────────
    "LR_EPOCHS_WARMUP":      1,      # epochs to keep LR constant before decay; ↑ longer warmup (stable start/slower), ↓ shorter warmup (faster ramp/risk overshoot)
    "INITIAL_LR":            5e-4,   # starting learning rate; ↑ speeds convergence (risk divergence), ↓ safer steps (slower training)
    "CLIPNORM":              1,      # max-gradient norm; ↑ higher clip threshold (less clipping, risk explosion), ↓ lower threshold (more clipping, risk under-update)
    
    # ── CosineAnnealingWarmRestarts Scheduler ──────────────────────────
    "ETA_MIN":               2e-5,   # floor LR in each cosine cycle
    "T_0":                   60,     # epochs before first cosine restart
    "T_MULT":                1,      # cycle length multiplier after each restart

    # ── ReduceLROnPlateau Scheduler ───────────────────────────────────
    "PLATEAU_FACTOR":        0.9,    # multiply LR by this factor on plateau
    "PLATEAU_PATIENCE":      0,      # epochs with no val-improve before LR cut
    "MIN_LR":                1e-6,   # lower bound on LR after reductions
    "PLAT_EPOCHS_WARMUP":    999     # epochs to wait before triggering plateau logic
}
