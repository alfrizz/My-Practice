from pathlib import Path 
from datetime import datetime
import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as Funct

#########################################################################################################

date_to_check = None
date_to_check = '2021-04' # set to None to analyze all dates save the final CSV

date_to_test = '2025-04'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ticker         = 'GOOGL'
save_path      = Path("dfs training")

# dataset split proportions
train_prop, val_prop = 0.70, 0.15

look_back = 120
is_centered = True # smoothing and centering using past and future data (True) or only with past data without centering (False)

bidasktoclose_spread = 0.03

# Market Session	        US Market Time (ET)	             Corresponding Time in Datasheet (UTC)
# Premarket             	~4:00 AM – 9:30 AM	             9:00 – 14:30
# Regular Trading	        9:30 AM – 4:00 PM	             14:30 – 21:00
# After-Hours	           ~4:00 PM – 7:00 PM	             21:00 – 00:00

premarket_start  = datetime.strptime('09:00', '%H:%M').time()   

regular_start  = datetime.strptime('14:30', '%H:%M').time()  

regular_start_pred = dt.time(*divmod(regular_start.hour * 60 + regular_start.minute - look_back, 60))
regular_start_shifted = dt.time(*divmod(regular_start.hour * 60 + regular_start.minute - look_back*2, 60))

regular_end = datetime.strptime('21:00' , '%H:%M').time()   

afterhours_end = datetime.strptime('00:00' , '%H:%M').time()  

features_cols = [
    "open", "high", "low", "close", "volume",   # raw OHLCV
    "r_1", "r_5", "r_15",                       # momentum
    "vol_15", "volume_spike",                   # volatility & volume
    "vwap_dev",                                 # intraday bias
    "rsi_14"                                    # overbought/oversold
]

label_col = "signal_smooth" 


#########################################################################################################

def signal_parameters(ticker):
    '''
    # to define the trades
    min_prof_thr ==> # percent of minimum profit to define a potential trade
    max_down_prop ==> # float (percent/100) of maximum allowed drop of a potential trade
    gain_tightening_factor ==> # as gain grows, tighten the stop 'max_down_prop' by this factor.
    merging_retracement_thr ==> # intermediate retracement, relative to the first trade's full range
    merging_time_gap_thr ==> # time gap between trades, relative to the first and second trade durations
    
    # to define the smoothed signal
    smooth_win_sig ==> # smoothing window of the signal used for the identification of the final trades 
    pre_entry_decay ==> # pre-trade decay of the final trades' raw signal (higher: quicker decay [0.01 - 1])
    short_penalty ==> # duration penalty factor (lower: higher penalization [0.01 - 1])
    
    # to define the final buy and sell triggers
    buy_threshold ==> # float (percent/100) threshold of the smoothed signal to trigger the final trade
    pred_threshold ==> # float (percent/100) threshold of the predicted signal to trigger the final trade
    trailing_stop_thresh ==> # percent of the trailing stop loss of the final trade
    '''
    if ticker == 'AAPL':
        # to define the initial trades:
        min_prof_thr=0.2 
        max_down_prop=0.4
        gain_tightening_factor=0.1
        merging_retracement_thr=0.9
        merging_time_gap_thr=0.7
        # to define the smoothed signal:
        smooth_win_sig=5
        pre_entry_decay=0.77
        short_penalty=0.1
        # to define the final buy and sell triggers:
        buy_threshold=0.1
        pred_threshold=0.3
        trailing_stop_thresh=0.16
        
    if ticker == 'GOOGL':
        # to define the initial trades:
        min_prof_thr=0.689 
        max_down_prop=0.896
        gain_tightening_factor=0.202
        merging_retracement_thr=0.193
        merging_time_gap_thr=0.509
        # to define the smoothed signal:
        smooth_win_sig=2
        pre_entry_decay=0.487
        short_penalty= 0.149
        # to define the final buy and sell triggers:
        buy_threshold=0.120
        pred_threshold=0.120
        trailing_stop_thresh=0.130
        
    if ticker == 'TSLA':
        # to define the initial trades:
        min_prof_thr=0.45 
        max_down_prop=0.3
        gain_tightening_factor=0.02
        merging_retracement_thr=0.9
        merging_time_gap_thr=0.7
        # to define the smoothed signal:
        smooth_win_sig=3  
        pre_entry_decay=0.6
        short_penalty=0.1
        # to define the final buy and sell triggers:
        buy_threshold=0.1 
        pred_threshold=0.3
        trailing_stop_thresh=0.1 

    return min_prof_thr, max_down_prop, gain_tightening_factor, smooth_win_sig, pre_entry_decay, short_penalty, \
        buy_threshold, pred_threshold, trailing_stop_thresh, merging_retracement_thr, merging_time_gap_thr

# run the function to get the parameters ("_man": manually assigned)
min_prof_thr_man, max_down_prop_man, gain_tightening_factor_man, smooth_win_sig_man, pre_entry_decay_man, short_penalty_man, \
buy_threshold_man, pred_threshold_man, trailing_stop_thresh_man, merging_retracement_thr_man, merging_time_gap_thr_man = signal_parameters(ticker)


#########################################################################################################


hparams = {
    # ── Architecture Parameters ────────────────────────────────────────
    "SHORT_UNITS":         32,      # hidden size of each daily LSTM layer
    "LONG_UNITS":          64,      # hidden size of the weekly LSTM
    "DROPOUT_SHORT":       0.3,     # dropout after residual+attention block
    "DROPOUT_LONG":        0.4,     # dropout after weekly LSTM outputs
    "ATT_HEADS":           4,       # number of self-attention heads
    "ATT_DROPOUT":         0.2,     # dropout rate inside attention
    "WEIGHT_DECAY":        2e-4,    # L2 weight decay on all model weights

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":         16,      # training batch size
    "VAL_BATCH":           1,       # validation batch size
    "NUM_WORKERS":         1,       # DataLoader workers
    "MAX_EPOCHS":          80,      # upper limit on training epochs
    "EARLY_STOP_PATIENCE": 12,      # stop if no val-improve for this many epochs

    # ── Optimizer Settings ─────────────────────────────────────────────
    "LR_EPOCHS_WARMUP":    0,       # epochs to wait before decreasing the LR
    "INITIAL_LR":          1e-3,    # AdamW initial learning rate
    "CLIPNORM":            0.5,     # max-norm gradient clipping

    # ── CosineAnnealingWarmRestarts Scheduler ──────────────────────────
    "T_0":                 100,     # epochs before first cosine restart
    "T_MULT":              1,       # cycle length multiplier after each restart
    "ETA_MIN":             5e-5,    # floor LR in each cosine cycle

    # ── ReduceLROnPlateau Scheduler ───────────────────────────────────
    "PLATEAU_FACTOR":      0.9,     # multiply LR by this factor on plateau
    "PLATEAU_PATIENCE":    0,       # epochs with no val-improve before LR cut
    "MIN_LR":              1e-6,    # lower bound on LR after reductions
    "PLAT_EPOCHS_WARMUP":  999      # epochs to wait before triggering plateau logic
}
