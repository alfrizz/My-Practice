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

from libs.models import simple_lstm, dual_lstm_smooth

#########################################################################################################

ticker = 'AAPL'
label_col  = "signal" 
month_to_check = '2024-06'

smooth_sign_win = 15 # smoothing of the continuous target signal
mult_feats_win = 1 # use 1 to generate features indicators with their standard windows size

createCSVbase = False # set to True to regenerate the 'base' csv
createCSVsign = False # set to True to regenerate the 'sign' csv
train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidask_spread_pct = 0.05 # conservative 5 percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes)

model_selected = simple_lstm # the correspondent .py model file must also be imported from libs.models
sel_val_rmse = 0.26212

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stocks_folder  = "intraday_stocks" 
optuna_folder = "optuna_results" 
models_folder = "trainings" 
log_file = Path(models_folder) / "training_diagnostics.txt"

save_path  = Path("dfs")
base_csv = save_path / f"{ticker}_1_base.csv"
sign_csv = save_path / f"{ticker}_2_sign.csv"
feat_all_csv = save_path / f"{ticker}_3_feat_all.csv"
test_csv = save_path / f"{ticker}_4_test.csv"
trainval_csv = save_path / f"{ticker}_4_trainval.csv"

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
        return None, {}          # or provide some hard-coded defaults

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
    look_back ==> length of historical window (how many minutes of history each training example contains): number of past time‐steps fed into the LSTM to predict next value
    '''
    if ticker == 'AAPL':
        look_back = 90
        sess_start_pred = dt.time(*divmod((sess_start.hour * 60 + sess_start.minute) - look_back, 60))
        sess_start_shift = dt.time(*divmod((sess_start.hour * 60 + sess_start.minute) - 2*look_back, 60))
        features_cols = ['sma_pct_14',
                         'atr_pct_14',
                         'rsi_14',
                         'bb_w_20',
                         'plus_di_14',
                         'range_pct',
                         'eng_ma',
                         'minus_di_14',
                         'eng_macd',
                         'macd_diff_12_26_9',
                         'body_pct',
                         'macd_line_12_26_9',
                         'volume',
                         'obv_diff_14',
                         'eng_rsi',
                         'eng_atr_div',
                         'eng_adx',
                         'adx_14',
                         'hour',
                         'body']
        trailing_stop_pred = 0.5
        pred_threshold = 0.1
        return_threshold = 0.01
        
    return look_back, sess_start_pred, sess_start_shift, features_cols, trailing_stop_pred, pred_threshold, return_threshold

# automatically executed function to get the parameters for the selected ticker
look_back_tick, sess_start_pred_tick, sess_start_shift_tick, features_cols_tick, trailing_stop_pred_tick, pred_threshold_tick, return_threshold_tick \
= signal_parameters(ticker)


#########################################################################################################


hparams = {
    #  ────────────────────────────────────────
    "CONV_K":                5,      # input conv1d kernel; ↑local smoothing, ↓fine-detail capture
    "CONV_DILATION":         2,      # input conv dilation; ↑receptive field, ↓signal granularity
    # "SMOOTH_K":             3,      # regression-head conv kernel (unused in forward); ↑smoothing window, ↓reactivity
    # "SMOOTH_DILATION":      1,      # regression conv dilation (unused); ↑lag smoothing, ↓immediate response
    "ALPHA_SMOOTH":          10,     # increase to smooth the predicted signal
    
    # ── lstm ─────────────────────────────
    "SHORT_UNITS":           192,      # short LSTM total hidden dim (bidirectional); ↑capacity for spike detail, ↓overfit & latency
    "LONG_UNITS":            256,      # projection / pred input dim (formerly long LSTM size); ↑feature width, ↓bottleneck risk
    "DROPOUT_SHORT":         0.2,      # after short LSTM; ↑regularization, ↓retains sharp spikes
    "DROPOUT_LONG":          0.2,      # after projection; ↑overfitting guard, ↓reactivity at head
    
    # ── Projection (short -> final feature space) ──────────────────────
    # "PROJ_HIDDEN":          192,     # hidden dim for optional short2long MLP (Option B) -- commented until used
    # "PROJ_USE_MLP":         False,   # toggle single-linear vs small-MLP (unused) -- commented until used
    "PRED_HIDDEN":           128,      # optional head hidden dim 
    
    # ── Optimizer & Scheduler Settings ──────────────────────────────────
    "MAX_EPOCHS":            50,     # max epochs
    "EARLY_STOP_PATIENCE":   5,      # no-improve epochs; ↑robustness to noise, ↓max training time 
    "WEIGHT_DECAY":          2e-4,   # L2 penalty; ↑weight shrinkage (smoother), ↓model expressivity
    "CLIPNORM":              1,      # max grad norm; ↑training stability, ↓gradient expressivity
    "ONECYCLE_MAX_LR":       2e-4,   # peak LR in the cycle
    "ONECYCLE_DIV_FACTOR":   10,   # start_lr = max_lr / div_factor
    "ONECYCLE_FINAL_DIV":    1000,    # end_lr   = max_lr / final_div_factor
    "ONECYCLE_PCT_START":    0.1,   # fraction of total steps spent rising
    "ONECYCLE_STRATEGY":     'cos', # or 'linear'

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":           16,     # sequences per train batch; ↑GPU efficiency, ↓stochasticity
    "VAL_BATCH":             1,      # sequences per val batch
    "NUM_WORKERS":           12,     # DataLoader workers; ↑throughput, ↓CPU contention
    "TRAIN_PREFETCH_FACTOR": 4,      # prefetch factor; ↑loader speed, ↓memory overhead

}
