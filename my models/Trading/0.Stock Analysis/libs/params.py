from pathlib import Path 
from datetime import datetime
import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as Funct

#########################################################################################################
ticker = 'GOOGL'

date_to_check = None # to analyze all dates save the final CSV
# date_to_check = '2020-04' # set to None to analyze all dates save the final CSV

date_to_test = '2020-04'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path      = Path("dfs training")

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
is_centered = True # smoothing and centering using past and future data (True) or only with past data without centering (False)
bidasktoclose_spread = 0.03

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
        look_back=90
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
        trailing_stop_thresh=0.16
        buy_threshold=0.1
        pred_threshold=0.3
        
    if ticker == 'GOOGL':
        look_back=90
        # to define the initial trades:
        min_prof_thr= 0.8847
        max_down_prop=0.7716
        gain_tightening_factor=0.8953
        merging_retracement_thr=0.9373
        merging_time_gap_thr=0.1317
        # to define the smoothed signal:
        smooth_win_sig=15
        pre_entry_decay=0.4431
        short_penalty=0.1835
        # to define the final buy and sell triggers:
        trailing_stop_thresh=0.0962
        buy_threshold=0.0846
        pred_threshold=0.0846
# {'look_back': 30, 'min_prof_thr': 0.8873023692019155, 'max_down_prop': 0.5542992413400941, 'gain_tightening_factor': 0.3800101769222284, 'merging_retracement_thr': 0.48504854872127645, 'merging_time_gap_thr': 0.6496630991359298, 'smooth_win_sig': 3, 'pre_entry_decay': 0.46541344789493516, 'short_penalty': 0.3830993632494481, 'trailing_stop_thresh': 0.39542503158784303, 'buy_threshold': 0.21133224027063702}
    if ticker == 'TSLA':
        look_back=90
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
        trailing_stop_thresh=0.1 
        buy_threshold=0.1 
        pred_threshold=0.3

    return look_back, min_prof_thr, max_down_prop, gain_tightening_factor, merging_retracement_thr, merging_time_gap_thr,  \
        smooth_win_sig, pre_entry_decay, short_penalty, trailing_stop_thresh, buy_threshold, pred_threshold

# automatically executed function to get the parameters for the selected ticker
look_back_tick, min_prof_thr_tick, max_down_prop_tick, gain_tightening_factor_tick, merging_retracement_thr_tick, merging_time_gap_thr_tick, \
smooth_win_sig_tick, pre_entry_decay_tick, short_penalty_tick, trailing_stop_thresh_tick, buy_threshold_tick, pred_threshold_tick = signal_parameters(ticker)

#########################################################################################################

# Market Session	        US Market Time (ET)	             Corresponding Time in Datasheet (UTC)
# Premarket             	~4:00 AM – 9:30 AM	             9:00 – 14:30
# Regular Trading	        9:30 AM – 4:00 PM	             14:30 – 21:00
# After-Hours	           ~4:00 PM – 7:00 PM	             21:00 – 00:00

regular_start  = datetime.strptime('14:30', '%H:%M').time()  
regular_start_pred = dt.time(*divmod(regular_start.hour * 60 + regular_start.minute - look_back_tick, 60))
regular_start_shifted = dt.time(*divmod(regular_start.hour * 60 + regular_start.minute - look_back_tick*2, 60))
regular_end = datetime.strptime('21:00' , '%H:%M').time()   

features_cols = [
    "open", "high", "low", "close", "volume",   # raw OHLCV
    "r_1", "r_5", "r_15",                       # momentum
    "vol_15", "volume_spike",                   # volatility & volume
    "vwap_dev",                                 # intraday bias
    "rsi_14"                                    # overbought/oversold
]

label_col = "signal_smooth" 

#########################################################################################################

hparams = {
    # ── Architecture Parameters ────────────────────────────────────────
    "SHORT_UNITS":         32,      # hidden size of each daily LSTM layer
    "LONG_UNITS":          32,      # hidden size of the weekly LSTM
    "DROPOUT_SHORT":       0.3,     # dropout after residual+attention block
    "DROPOUT_LONG":        0.4,     # dropout after weekly LSTM outputs
    "ATT_HEADS":           4,       # number of self-attention heads
    "ATT_DROPOUT":         0.2,     # dropout rate inside attention
    "WEIGHT_DECAY":        2e-4,    # L2 weight decay on all model weights

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":         32,      # training batch size
    "VAL_BATCH":           1,       # validation batch size
    "NUM_WORKERS":         8,       # DataLoader workers
    "MAX_EPOCHS":          60,      # upper limit on training epochs
    "EARLY_STOP_PATIENCE": 12,      # stop if no val-improve for this many epochs

    # ── Optimizer Settings ─────────────────────────────────────────────
    "LR_EPOCHS_WARMUP":    0,       # epochs to wait before decreasing the LR
    "INITIAL_LR":          8e-4,    # AdamW initial learning rate
    "CLIPNORM":            0.5,     # max-norm gradient clipping

    # ── CosineAnnealingWarmRestarts Scheduler ──────────────────────────
    "T_0":                 60,     # epochs before first cosine restart
    "T_MULT":              1,       # cycle length multiplier after each restart
    "ETA_MIN":             8e-5,    # floor LR in each cosine cycle

    # ── ReduceLROnPlateau Scheduler ───────────────────────────────────
    "PLATEAU_FACTOR":      0.9,     # multiply LR by this factor on plateau
    "PLATEAU_PATIENCE":    0,       # epochs with no val-improve before LR cut
    "MIN_LR":              1e-6,    # lower bound on LR after reductions
    "PLAT_EPOCHS_WARMUP":  999      # epochs to wait before triggering plateau logic
}
