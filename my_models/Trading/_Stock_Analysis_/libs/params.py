from pathlib import Path 
from datetime import datetime
import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as Funct

import pandas as pd
import os
import glob
import json
import re

#########################################################################################################

ticker = 'AAPL'
label_col  = "signal"
shares_per_trade = 1

month_to_check = '2024-03'
sel_val_rmse = 0.09436

smooth_sign_win = 15 # smoothing of the continuous target signal
extra_windows = [30, 60, 90] #  to produce additional smoothed/rolling copies of selected indicators for each window 

createCSVbase = False # set to True to regenerate the 'base' csv
createCSVsign = False # set to True to regenerate the 'sign' csv
since_year = 2009

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidask_spread_pct = 0.02 # conservative 2 percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes)

feats_min_std = 0.03
feats_max_corr = 0.997
thresh_gb = 56 # use ram instead of memmap, if X_buf below this value

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
    "USE_SHORT_LSTM":       True,    # enable bidirectional “short” LSTM
    "SHORT_UNITS":          128,     # short-LSTM total output width (bidirectional); per-dir hidden = SHORT_UNITS // 2
    "DROPOUT_SHORT":        0.1,     # dropout after short-LSTM; ↑regularization

    # ── Transformer toggle ────────────────────────────────
    "USE_TRANSFORMER":      True,    # enable TransformerEncoder
    "TRANSFORMER_D_MODEL":  64,     # transformer embedding width (d_model); adapter maps upstream features into this
    "TRANSFORMER_LAYERS":   4,       # number of encoder layers
    "TRANSFORMER_HEADS":    4,       # attention heads in each layer
    "TRANSFORMER_FF_MULT":  4,       # FFN expansion factor (d_model * MULT)
    "DROPOUT_TRANS":        0.01,     # transformer dropout; ↑regularization

    # ── Long Bi-LSTM ──────────────
    "USE_LONG_LSTM":        False,    # enable bidirectional “long” LSTM
    "DROPOUT_LONG":         0.2,     # dropout after projection (or long-LSTM)
    "LONG_UNITS":           128,     # long-LSTM total output width (bidirectional); per-dir hidden = LONG_UNITS // 2

    # ── Regression head, smooting, huber and delta  ───────────────────────────────────────
    "FLATTEN_MODE":          "attn", # format to be provided to regression head: "flatten" | "last" | "pool" | "attn"
    "PRED_HIDDEN":           128,    # head MLP hidden dim; ↑capacity, ↓over-parameterization
    
    "ALPHA_SMOOTH":          0,      # derivative slope-penalty weight; ↑smoothness, ↓spike fidelity
    "WARMUP_STEPS":          3,      # linear warmup for slope penalty (0 = no warmup)
    
    "USE_HUBER":             False,  # if True use Huber for level term instead of MSE
    "HUBER_DELTA":           0.1,    # Huber delta (transition threshold); scale to your typical error
    
    "USE_DELTA":             False,  # enable Delta baseline vs features predictions head
    "LAMBDA_DELTA":          0.1,    # Delta residual loss weight  ↑: stronger residual fit  ↓: safer base learning

    # ── Optimizer & Scheduler Settings ──────────────────────────────────
    "MAX_EPOCHS":            90,     # max epochs
    "EARLY_STOP_PATIENCE":   9,      # no-improve epochs; ↑robustness to noise, ↓max training time 
    "WEIGHT_DECAY":          1e-6,   # L2 penalty; ↑weight shrinkage (smoother), ↓model expressivity
    "CLIPNORM":              10,     # max grad norm; ↑training stability, ↓gradient expressivity
    
    "ONECYCLE_MAX_LR":       1e-4,   # peak LR in the cycle
    "HEAD_LR_PCT":           1,      # percentage of learning rate to apply to the head (1 default)
    "ONECYCLE_DIV_FACTOR":   10,     # start_lr = max_lr / div_factor
    "ONECYCLE_FINAL_DIV":    100,    # end_lr   = max_lr / final_div_factor
    "ONECYCLE_PCT_START":    0.1,    # fraction of total steps spent rising
    "ONECYCLE_STRATEGY":     'cos',  # 'cos' or 'linear'

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":           16,     # sequences per train batch; ↑GPU efficiency, ↓stochasticity
    "VAL_BATCH":             1,      # sequences per val batch
    "TRAIN_WORKERS":         8,      # DataLoader workers; ↑throughput, ↓CPU contention
    "TRAIN_PREFETCH_FACTOR": 4,      # prefetch factor; ↑loader speed, ↓memory overhead

    "LOOK_BACK":             45,     # length of each input window (how many minutes of history each training example contains)
    
    "MICRO_SAMPLE_K":        16,     # sample K per-segment forwards to compute p50/p90 latencies (cost: extra forward calls; recommend 16 for diagnostics)
}

# Market Session	        US Market Time (ET)	             Corresponding Time in Datasheet (UTC)
# Premarket             	~4:00 AM – 9:30 AM	             9:00 – 14:30
# Regular Trading	        9:30 AM – 4:00 PM	             14:30 – 21:00
# After-Hours	           ~4:00 PM – 7:00 PM	             21:00 – 00:00

sess_premark     = datetime.strptime('09:00' , '%H:%M').time()  
sess_end         = datetime.strptime('21:00' , '%H:%M').time() 
sess_start_reg   = datetime.strptime('14:30', '%H:%M').time()  
sess_start_pred  = dt.time(*divmod((sess_start_reg.hour * 60 + sess_start_reg.minute) - hparams["LOOK_BACK"], 60))
sess_start_shift = dt.time(*divmod((sess_start_reg.hour * 60 + sess_start_reg.minute) - 2*hparams["LOOK_BACK"], 60))


#########################################################################################################


def load_target_sign_optuna_record(sig_type, optuna_folder=optuna_folder, ticker=ticker):
    """
    Find the Optuna JSON file named like '{ticker}_{value}_target.json' with the
    largest numeric <value> and return (value, params) from that file.

    Assumes at least one matching file exists; will raise if none are found.
    """
    # build glob pattern for files like "AAPL_1.8870_target.json"
    pattern = os.path.join(optuna_folder, f"{ticker}_*_{sig_type}.json")

    # compile regex to extract the numeric suffix between ticker_ and _target.json
    rx = re.compile(
        rf'^{re.escape(ticker)}_(?P<val>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)_{sig_type}\.json$'
    )

    # list all candidate files matching the glob pattern
    files = glob.glob(pattern)

    # keep only files that match the regex and parse their numeric suffix
    matches = [
        (p, float(rx.match(os.path.basename(p)).group("val"))) 
        for p in files
    ]

    # pick the file with the largest numeric suffix (will raise if matches is empty)
    best_file = max(matches, key=lambda t: t[1])[0]

    # load the JSON record and return the stored value and params
    with open(best_file, "r") as fp:
        record = json.load(fp)

    if sig_type == 'target':
        record["params"]["sess_start"] = pd.to_datetime(record["params"]["sess_start"]).time()
        return record["value"], record["params"]

    if sig_type == 'predicted':
        return record["best_value"], record["best_params"]


#########################################################################################################


def ticker_parameters(ticker, sellmin_idx_tick, sess_start_tick, trailstop_pct_tick, buy_thresh_tick, sign_smoothwin_tick, return_thresh_tick):

    if ticker == 'AAPL':
        
        features_cols_tick =['dist_low_30', 'in_sess_time', 'dist_low_60', 'dist_low_28', 'eng_ema_cross_up', 'minute_time', 'rsi', 'hour_time', 'z_vwap_dev', 'dist_high_30', 'dist_high_60', 'eng_bb_mid', 'sma_pct_28', 'rsi_30', 'eng_vwap', 'plus_di', 'sma_pct_14', 'z_obv', 'adx_30', 'rsi_60', 'dist_high_28', 'roc_14', 'vwap_dev_pct_30', 'obv_diff_60', 'adx_60', 'volume_z_90', 'adx', 'z_vwap_dev_90', 'z_vwap_dev_60', 'obv_pct_14', 'minus_di', 'roc_28', 'obv_diff_14', 'obv_diff_30', 'vwap_dev_pct_60', 'plus_di_30', 'obv_pct_60', 'mom_sum_60', 'eng_macd', 'vwap_dev_pct_90', 'sma_pct_60', 'sma_pct_90', 'eng_ma', 'roc_30', 'obv_diff_90', 'obv_pct_30', 'vwap_dev_pct_z_90', 'volume_z_60', 'obv_z_90', 'obv_sma_90', 'adx_90', 'rsi_90', 'obv_pct_90', 'obv_sma_60', 'vwap_dev_pct_z_60', 'eng_obv', 'z_vwap_dev_30', 'ret_std_z_90', 'roc_60', 'atr_z_90', 'bb_w_z_90', 'macd_diff_z_60', 'obv_z_60', 'body_pct', 'vol_z_30', 'bb_w_z_60', 'minus_di_30', 'volume_z_30', 'macd_diff_z_90', 'plus_di_60']

# ['adx', 'adx_30', 'adx_60', 'adx_90', 'eng_ma', 'dow_time', 'hour_time', 'minute_time', 'in_sess_time', 'mom_sum_30', 'mom_sum_60', 'mom_sum_90', 'macd_diff_z_30', 'macd_diff_z_60', 'macd_diff_z_90', 'eng_macd', 'bb_w_z_30', 'bb_w_z_60', 'bb_w_z_90']

        # trailstop_pct_tick = max(1.5 * bidask_spread_pct, trailstop_pct_tick) # safe minimum trail stop set to 'factor' times the bid spread (so bid starts enough higher than the trail)
        
    return features_cols_tick, sellmin_idx_tick, sess_start_tick, trailstop_pct_tick, buy_thresh_tick, sign_smoothwin_tick, return_thresh_tick

# automatically executed function to get the parameters for the selected ticker
features_cols_tick, sellmin_idx_tick, sess_start_tick, trailstop_pct_tick, buy_thresh_tick, sign_smoothwin_tick, return_thresh_tick \
    = ticker_parameters(ticker              = ticker,
                        sign_smoothwin_tick = 1,
                        sellmin_idx_tick    = None,
                        sess_start_tick     = sess_start_pred,
                        buy_thresh_tick     = 0.15,
                        trailstop_pct_tick  = 3.5,
                        return_thresh_tick  = 0) # TBD
                                                                                                           




