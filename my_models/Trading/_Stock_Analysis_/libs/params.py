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
label_col  = "signal" 
feat_sel = 'man' # 'auto' or 'man'

month_to_check = '2023-10'
createCSVsign = True

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidasktoclose_pct = 0.075 # percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes): 0.1% is a conservative one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stocks_folder  = "intraday_stocks" 
optuna_folder = "optuna_results" 
models_folder = "trainings" 

save_path  = Path("dfs")
base_csv = save_path / f"{ticker}_1_base.csv"
sign_csv = save_path / f"{ticker}_2_sign.csv"
feat_all_csv = save_path / f"{ticker}_3_feat_all.csv"
feat_sel_auto_csv = save_path / f"{ticker}_3_feat_sel_auto.csv"
feat_sel_man_csv = save_path / f"{ticker}_3_feat_sel_man.csv"
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
        features_cols = ['atr_ratio', 'atr_ratio_sma', 'vol_15',  'bb_width_20', 'r_5', 'r_15', 'rsi', 'stoch_d_3', 'stoch_k_14', 'hour', 'eng_adx', 'eng_obv', 'eng_ma']
        trailing_stop_pred = 0.05
        pred_threshold = 0.35
        
    return look_back, sess_start_pred, sess_start_shift, features_cols, trailing_stop_pred, pred_threshold

# automatically executed function to get the parameters for the selected ticker
look_back_tick, sess_start_pred_tick, sess_start_shift_tick, features_cols_tick, trailing_stop_pred_tick, pred_threshold_tick = signal_parameters(ticker)

#########################################################################################################

hparams = {
    # ── Architecture Parameters ────────────────────────────────────────
    "SHORT_UNITS":           96,    # hidden size of daily LSTM; high capacity to model fine-grained daily patterns
    "LONG_UNITS":            128,    # hidden size of weekly LSTM; large context window for long-term trends
    "DROPOUT_SHORT":         0.2,  # light dropout after daily LSTM+attention; preserves spike information
    "DROPOUT_LONG":          0.25,  # moderate dropout after weekly LSTM; balances overfitting and information retention
    "ATT_HEADS":             8,     # number of multi-head attention heads; more heads capture diverse interactions
    "ATT_DROPOUT":           0.15,   # dropout inside attention layers; regularizes attention maps
    "WEIGHT_DECAY":          1e-4,  # L2 penalty on all weights; prevents extreme magnitudes

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":           32,    # number of sequences per training batch
    "VAL_BATCH":             1,     # number of sequences per validation batch
    "NUM_WORKERS":           2,     # DataLoader CPU workers
    "TRAIN_PREFETCH_FACTOR": 1,     # prefetch factor for DataLoader

    "MAX_EPOCHS":            90,    # maximum number of epochs
    "EARLY_STOP_PATIENCE":   6,     # epochs with no val–RMSE improvement before stopping

    # ── Optimizer & Scheduler Settings ────────────────────────────────
    "LR_EPOCHS_WARMUP":      3,     # epochs to keep LR constant before cosine decay
    "INITIAL_LR":            1e-4,  # starting learning rateS
    "CLIPNORM":              1,   # max gradient norm for clipping
    "ETA_MIN":               1e-5,  # floor LR in CosineAnnealingWarmRestarts
    "T_0":                   90,    # period (in epochs) of first cosine decay cycle
    "T_MULT":                1,     # multiplier for cycle length after each restart

    ############################───────────────NOT USED────────────────###################################
    # —— Active Loss Hyperparameters —— 
    "HUBER_BETA":            0.1,   # δ threshold for SmoothL1 (Huber) loss; range: 0.01–1.0: lower→more like MAE (heavier spike penalty), higher→more like MSE (tolerate spikes)
    "CLS_LOSS_WEIGHT":       0.05,  # α weight for the binary BCE loss head; range: 0.1–10.0: lower→less emphasis on threshold-crossing signal, higher→stronger spike detection
    
    # ── ReduceLROnPlateau Scheduler ───
    "PLATEAU_FACTOR":        0.9,   # multiply LR by this factor on plateau
    "PLATEAU_PATIENCE":      0,     # epochs with no val-improve before LR cut
    "MIN_LR":                1e-6,  # lower bound on LR after reductions
    "PLAT_EPOCHS_WARMUP":    999    # epochs to wait before triggering plateau logic
}
