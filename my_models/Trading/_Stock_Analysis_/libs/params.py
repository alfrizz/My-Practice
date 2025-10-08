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
month_to_check = '2023-10'

createCSVbase = False # set to True to regenerate the 'base' csv
createCSVsign = False # set to True to regenerate the 'sign' csv
train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidask_spread_pct = 0.05 # conservative 5 percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes)

model_selected = simple_lstm # the correspondent .py model file must also be imported from libs.models
sel_val_rmse = 0.25078

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
        look_back = 60
        sess_start_pred = dt.time(*divmod((sess_start.hour * 60 + sess_start.minute) - look_back, 60))
        sess_start_shift = dt.time(*divmod((sess_start.hour * 60 + sess_start.minute) - 2*look_back, 60))
        smooth_sign_win = 15
        features_cols = ['sma_pct_14',
                         'atr_pct_14',
                         'rsi_14',
                         'bb_w_20',
                         'plus_di_14',
                         'range_pct',
                         'eng_ma',
                         'minus_di_14',
                         'macd_diff_12_26_9',
                         'ret',
                         'eng_macd',
                         'macd_line_12_26_9',
                         'obv_diff_14',
                         'eng_atr_div',
                         'eng_adx',
                         'hour',
                         'adx_14']
        trailing_stop_pred = 0.2
        pred_threshold = 0.2
        return_threshold = 0.01
        
    return look_back, sess_start_pred, sess_start_shift, features_cols, smooth_sign_win, trailing_stop_pred, pred_threshold, return_threshold

# automatically executed function to get the parameters for the selected ticker
look_back_tick, sess_start_pred_tick, sess_start_shift_tick, features_cols_tick, smooth_sign_win_tick, trailing_stop_pred_tick, pred_threshold_tick, return_threshold_tick \
= signal_parameters(ticker)


#########################################################################################################


hparams = {
    # ── Input conv (first layer) ────────────────────────────────────────
    "CONV_K":                3,      # input conv1d kernel; ↑local smoothing, ↓fine-detail capture
    "CONV_DILATION":         1,      # input conv dilation; ↑receptive field, ↓signal granularity
    # "SMOOTH_K":             3,      # regression-head conv kernel (unused in forward); ↑smoothing window, ↓reactivity
    # "SMOOTH_DILATION":      1,      # regression conv dilation (unused); ↑lag smoothing, ↓immediate response
    
    # ── Short-term encoder (short Bi-LSTM) ─────────────────────────────
    "SHORT_UNITS":           192,    # short LSTM total hidden dim (bidirectional); ↑capacity for spike detail, ↓overfit & latency
    "DROPOUT_SHORT":         0.2,   # after short LSTM; ↑regularization, ↓retains sharp spikes
    
    # ── Projection (short -> final feature space) ──────────────────────
    "LONG_UNITS":            256,    # projection / pred input dim (formerly long LSTM size); ↑feature width, ↓bottleneck risk
    # "PROJ_HIDDEN":          192,    # hidden dim for optional short2long MLP (Option B) -- commented until used
    # "PROJ_USE_MLP":         False,  # toggle single-linear vs small-MLP (unused) -- commented until used
    "PRED_HIDDEN":           128,      # optional head hidden dim 
    
    # ── Final normalization / dropout (applied to projection before head) ─
    "DROPOUT_LONG":          0.2,   # after projection; ↑overfitting guard, ↓reactivity at head
    
    # ── Regression head (last layer) ───────────────────────────────────
    # "PRED_HIDDEN":          None,   # optional hidden dim for extra head layer (unused)
    
    # ── Attention / classification (kept for backward compatibility; unused) ─
    # "ATT_HEADS":            6,      # kept for compatibility (attention not used) -- commented
    # "ATT_DROPOUT":          0.15,   # kept for compatibility (unused) -- commented
    
    # —— Active Loss & Smoothing Hyperparameters —— 
    # "DIFF1_WEIGHT":          1.0,    # L2 on negative Δ; ↑drop resistance, ↓upward bias
    # "DIFF2_WEIGHT":          2.0,    # L2 on curvature; ↑smooth curves, ↓spike sharpness
    # "SMOOTH_ALPHA":          0.05,   # EWMA decay; ↑weight on latest (more reactive), ↓history smoothing
    # "SMOOTH_BETA":           100.0,  # Huber weight on slips; ↑drop resistance, ↓sensitivity to dips
    # "SMOOTH_DELTA":          0.02,   # Huber δ for slip; ↑linear tolerance, ↓quadratic penalization
    # "CLS_LOSS_WEIGHT":       0.10,   # BCE head weight (kept but unused); ↑spike emphasis, ↓regression focus
    
    # ── Optimizer & Scheduler Settings ──────────────────────────────────
    "LR_EPOCHS_WARMUP":      3,      # constant LR before scheduler; ↑stable start, ↓early adaptation
    "INITIAL_LR":            1e-4,   # start LR; ↑fast convergence, ↓risk of instability (test 2e-4 briefly)
    "WEIGHT_DECAY":          2e-4,   # L2 penalty; ↑weight shrinkage (smoother), ↓model expressivity
    "CLIPNORM":              2,    # max grad norm; ↑training stability, ↓gradient expressivity
    "ETA_MIN":               1e-6,   # min LR in cosine cycle; ↑fine-tuning tail, ↓floor on updates
    "T_0":                   10,    # cosine cycle length (unchanged)
    "T_MULT":                2,      # cycle multiplier (unchanged)
    
    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":           64,     # sequences per train batch; ↑GPU efficiency, ↓stochasticity
    "VAL_BATCH":             1,      # sequences per val batch
    "NUM_WORKERS":           12,     # DataLoader workers; ↑throughput, ↓CPU contention
    "TRAIN_PREFETCH_FACTOR": 4,      # prefetch factor; ↑loader speed, ↓memory overhead
    "MAX_EPOCHS":            100,    # max epochs
    "EARLY_STOP_PATIENCE":   7,      # no-improve epochs; ↑robustness to noise, ↓max training time

    # ── ReduceLROnPlateau Scheduler ───
    # "PLATEAU_FACTOR":        0.9,    # multiply LR by this factor on plateau
    # "PLATEAU_PATIENCE":      0,      # epochs with no val-improve before LR cut
    # "MIN_LR":                1e-6,   # lower bound on LR after reductions
    # "PLAT_EPOCHS_WARMUP":    999     # epochs to wait before triggering plateau logic
}
