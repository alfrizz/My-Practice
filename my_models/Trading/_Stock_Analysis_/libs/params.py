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
month_to_check = '2024-06'

smooth_sign_win = 15 # smoothing of the continuous target signal
mult_inds_win = [0.5, 1, 2, 4]

createCSVbase = False # set to True to regenerate the 'base' csv
createCSVsign = False # set to True to regenerate the 'sign' csv
train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidask_spread_pct = 0.05 # conservative 5 percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes)

feats_min_std = 0.01
feats_max_corr = 0.99

sel_val_rmse = 0.08990

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


hparams = {
    # ── Input convolution toggle ──────────────────────────
    "USE_CONV":              False,  # enable Conv1d + BatchNorm1d
    "CONV_K":                3,      # Conv1d kernel size; ↑local smoothing, ↓fine-detail
    "CONV_DILATION":         1,      # Conv1d dilation;   ↑receptive field, ↓granularity
    "CONV_CHANNELS":         64,     # Conv1d output channels; ↑early-stage capacity, ↓compute

    # ── Temporal ConvNet (TCN) toggle ────────────────────
    "USE_TCN":               False,  # enable dilated Conv1d stack
    "TCN_LAYERS":            3,      # number of dilated Conv1d layers
    "TCN_KERNEL":            3,      # kernel size for each TCN layer
    "TCN_CHANNELS":          64,     # TCN output channels; independent from CONV_CHANNELS for flexibility

    # ── Short Bi-LSTM toggle ──────────────────────────────
    "USE_SHORT_LSTM":       False,   # enable bidirectional “short” LSTM
    "SHORT_UNITS":          128,     # short-LSTM total output width (bidirectional); per-dir hidden = SHORT_UNITS // 2
    "DROPOUT_SHORT":        0.1,     # dropout after short-LSTM; ↑regularization

    # ── Transformer toggle ────────────────────────────────
    "USE_TRANSFORMER":      True,    # enable TransformerEncoder
    "TRANSFORMER_D_MODEL":  64,     # transformer embedding width (d_model); adapter maps upstream features into this
    "TRANSFORMER_LAYERS":   2,       # number of encoder layers
    "TRANSFORMER_HEADS":    4,       # attention heads in each layer
    "TRANSFORMER_FF_MULT":  4,       # FFN expansion factor (d_model * MULT)
    "DROPOUT_TRANS":        0.15,     # transformer dropout; ↑regularization

    # ── Long Bi-LSTM ──────────────
    "USE_LONG_LSTM":        False,   # enable bidirectional “long” LSTM
    "DROPOUT_LONG":         0.1,     # dropout after projection (or long-LSTM)
    "LONG_UNITS":           128,     # long-LSTM total output width (bidirectional); per-dir hidden = LONG_UNITS // 2

    # ── Regression head, smooting, huber and delta  ───────────────────────────────────────
    "FLATTEN_MODE":          "pool", # format to be provided to regression head: "flatten" | "last" | "pool" | "attn"
    "PRED_HIDDEN":           64,    # head MLP hidden dim; ↑capacity, ↓over-parameterization
    
    "ALPHA_SMOOTH":          0.0,    # derivative slope-penalty weight; ↑smoothness, ↓spike fidelity
    "WARMUP_STEPS":          5,      # linear warmup for slope penalty (0 = no warmup)
    
    "USE_HUBER":             False,  # if True use Huber for level term instead of MSE
    "HUBER_DELTA":           0.1,    # Huber delta (transition threshold); scale to your typical error
    
    "USE_DELTA":             False,  # enable Delta baseline vs features predictions head
    "LAMBDA_DELTA":          0.1,    # Delta residual loss weight  ↑: stronger residual fit  ↓: safer base learning

    # ── Optimizer & Scheduler Settings ──────────────────────────────────
    "MAX_EPOCHS":            80,     # max epochs
    "EARLY_STOP_PATIENCE":   8,      # no-improve epochs; ↑robustness to noise, ↓max training time 
    "WEIGHT_DECAY":          8e-5,   # L2 penalty; ↑weight shrinkage (smoother), ↓model expressivity
    "CLIPNORM":              1,      # max grad norm; ↑training stability, ↓gradient expressivity
    "ONECYCLE_MAX_LR":       8e-4,   # peak LR in the cycle
    "ONECYCLE_DIV_FACTOR":   10,     # start_lr = max_lr / div_factor
    "ONECYCLE_FINAL_DIV":    100,    # end_lr   = max_lr / final_div_factor
    "ONECYCLE_PCT_START":    0.1,    # fraction of total steps spent rising
    "ONECYCLE_STRATEGY":     'cos',  # 'cos' or 'linear'

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":           16,     # sequences per train batch; ↑GPU efficiency, ↓stochasticity
    "VAL_BATCH":             1,      # sequences per val batch
    "TRAIN_WORKERS":         8,      # DataLoader workers; ↑throughput, ↓CPU contention
    "TRAIN_PREFETCH_FACTOR": 4,      # prefetch factor; ↑loader speed, ↓memory overhead

    "LOOK_BACK":             60,     # length of each input window (how many minutes of history each training example contains)
    
    "MICRO_SAMPLE_K":        16,     # sample K per-segment forwards to compute p50/p90 latencies (cost: extra forward calls; recommend 16 for diagnostics)
}


#########################################################################################################


def signal_parameters(ticker):

    if ticker == 'AAPL':
        sess_start_pred = dt.time(*divmod((sess_start.hour * 60 + sess_start.minute) - hparams["LOOK_BACK"], 60))
        sess_start_shift = dt.time(*divmod((sess_start.hour * 60 + sess_start.minute) - 2*hparams["LOOK_BACK"], 60))
        
        features_cols = ['bb_w_20', 'sma_pct_14', 'atr_14', 'hour', 'eng_bb', 'rsi_14',
       'obv_diff_14', 'range_pct', 'ret', 'sma_pct_28', 'plus_di_14', 'eng_rsi',
       'lower_shad', 'adx_14', 'atr_pct_14', 'body_pct', 'minus_di_14', 'eng_atr_div', 'eng_macd',
       'upper_shad']
        
        trailing_stop_pred = 0.2
        pred_threshold = 0.1
        return_threshold = 0.01
        
    return sess_start_pred, sess_start_shift, features_cols, trailing_stop_pred, pred_threshold, return_threshold

# automatically executed function to get the parameters for the selected ticker
sess_start_pred_tick, sess_start_shift_tick, features_cols_tick, trailing_stop_pred_tick, pred_threshold_tick, return_threshold_tick = signal_parameters(ticker)



