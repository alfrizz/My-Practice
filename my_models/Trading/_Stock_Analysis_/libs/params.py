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

from tqdm import tqdm

#########################################################################################################

ticker = 'AAPL'
label_col  = "signal"
shares_per_trade = 1

month_to_check = '2024-03'
sel_val_rmse = 0.10347

smooth_sign_win = 15 # smoothing of the continuous target signal
extra_windows = [30, 45, 60] #  to produce additional smoothed/rolling copies of selected indicators for each window 

createCSVbase = False # set to True to regenerate the 'base' csv
createCSVsign = False # set to True to regenerate the 'sign' csv
since_year = 2009

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidask_spread_pct = 0.02 # conservative 2 percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes)

feats_min_std = 0.03
feats_max_corr = 0.997
thresh_gb = 56 # use ram instead of memmap, if X_buf below this value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################################################################################################


stocks_folder  = "intraday_stocks" 
optuna_folder = "optuna_results" 
models_folder = "trainings" 
log_file = Path(models_folder) / "training_diagnostics.txt"

save_path  = Path("dfs")
base_csv = save_path / f"{ticker}_1_base.csv"
sign_csv = save_path / f"{ticker}_2_sign.csv"
feat_all_csv = save_path / f"{ticker}_3_feat_all.csv"
indunsc_test_csv = save_path / f"{ticker}_4_indunsc_test.csv"
indunsc_trainval_csv = save_path / f"{ticker}_4_indunsc_trainval.csv"
test_csv = save_path / f"{ticker}_5_test.csv"
trainval_csv = save_path / f"{ticker}_5_trainval.csv"


def _human(n):
    for u in ("B","KB","MB","GB"):
        if abs(n) < 1024: return f"{n:3.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

def to_csv_with_progress(df, path, chunksize=100_000, index=True):
    with open(path, "w", newline="") as f:
        df.iloc[:0].to_csv(f, index=index, date_format="%Y-%m-%d %H:%M:%S")  # header only
        total = len(df)
        pbar = tqdm(total=total, desc="Saving CSV", unit="rows")
        for start in range(0, total, chunksize):
            end = start + chunksize
            df.iloc[start:end].to_csv(f, index=index, header=False, date_format="%Y-%m-%d %H:%M:%S")
            f.flush()
            pbar.update(min(end, total) - start)
            pbar.set_postfix_str(f"size={_human(f.tell())}")
        pbar.close()


#########################################################################################################


hparams = {
    # ── Input convolution toggle ──────────────────────────
    "USE_CONV":              False,  # enable Conv1d + BatchNorm1d
    "CONV_K":                3,      # Conv1d kernel size; ↑local smoothing, ↓fine-detail
    "CONV_DILATION":         1,      # Conv1d dilation;   ↑receptive field, ↓granularity
    "CONV_CHANNELS":         64,     # Conv1d output channels; ↑early-stage capacity, ↓compute

    # ── Temporal ConvNet (TCN) toggle ────────────────────
    "USE_TCN":               False,  # enable dilated Conv1d stack
    "TCN_LAYERS":            1,      # number of dilated Conv1d layers
    "TCN_KERNEL":            3,      # kernel size for each TCN layer
    "TCN_CHANNELS":          64,     # TCN output channels; independent from CONV_CHANNELS for flexibility

    # ── Short Bi-LSTM toggle ──────────────────────────────
    "USE_SHORT_LSTM":        False,  # enable bidirectional “short” LSTM
    "SHORT_UNITS":           64,     # short-LSTM total output width (bidirectional); per-dir hidden = SHORT_UNITS // 2
    "DROPOUT_SHORT":         0.1,    # dropout after short-LSTM; ↑regularization

    # ── Transformer toggle ────────────────────────────────
    "USE_TRANSFORMER":       True,   # enable TransformerEncoder
    "TRANSFORMER_D_MODEL":   64,     # transformer embedding width (d_model); adapter maps upstream features into this
    "TRANSFORMER_LAYERS":    3,      # number of encoder layers
    "TRANSFORMER_HEADS":     4,      # attention heads in each layer
    "TRANSFORMER_FF_MULT":   4,      # FFN expansion factor (d_model * MULT)
    "DROPOUT_TRANS":         0.03,   # transformer dropout; ↑regularization

    # ── Long Bi-LSTM ──────────────
    "USE_LONG_LSTM":         False,  # enable bidirectional “long” LSTM
    "LONG_UNITS":            64,     # long-LSTM total output width (bidirectional); per-dir hidden = LONG_UNITS // 2
    "DROPOUT_LONG":          0.1,    # dropout after projection (or long-LSTM)

    # ── Regression head, smooting, huber and delta  ───────────────────────────────────────
    "FLATTEN_MODE":          "attn", # format to be provided to regression head: "flatten" | "last" | "pool" | "attn"
    "PRED_HIDDEN":           96,     # head MLP hidden dim; ↑capacity, ↓over-parameterization
    
    "ALPHA_SMOOTH":          0,      # derivative slope-penalty weight; ↑smoothness, ↓spike fidelity
    "WARMUP_STEPS":          3,      # linear warmup for slope penalty (0 = no warmup)
    
    "USE_HUBER":             False,  # if True use Huber for level term instead of MSE
    "HUBER_DELTA":           0.1,    # Huber delta (transition threshold); scale to your typical error
    
    "USE_DELTA":             False,  # enable Delta baseline vs features predictions head
    "LAMBDA_DELTA":          0.1,    # Delta residual loss weight  ↑: stronger residual fit  ↓: safer base learning

    # ── Optimizer & Scheduler Settings ──────────────────────────────────
    "MAX_EPOCHS":            90,     # max epochs
    "EARLY_STOP_PATIENCE":   9,      # no-improve epochs; ↑robustness to noise, ↓max training time 
    "WEIGHT_DECAY":          3e-6,   # L2 penalty; ↑weight shrinkage (smoother), ↓model expressivity
    "CLIPNORM":              3,      # max grad norm; ↑training stability, ↓gradient expressivity
    
    "ONECYCLE_MAX_LR":       3e-4,   # peak LR in the cycle
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

    "LOOK_BACK":             60,     # length of each input window (how many minutes of history each training example contains)
    
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


def load_sign_optuna_record(sig_type, optuna_folder=optuna_folder, ticker=ticker):
    """
    Find the Optuna JSON file named like '{ticker}_{value}_{sig_type "target" or "predicted"}.json' 
    with the largest numeric <value> and return (value, params) from that file.

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


def ticker_parameters(ticker, sellmin_idx_tick, sess_start_tick, trailstop_pct_tick, atr_mult_tick, vwap_atr_mult_tick, \
                      rsi_thresh_tick, buy_thresh_tick, sign_smoothwin_tick):

    if ticker == 'AAPL':
        
        features_cols_tick = ['dist_low_28', 'dist_low_60', 'dist_low_30', 'in_sess_time', 'dist_high_60', 'dist_high_30', 'dist_high_28', 'minute_time', 'hour_time', 'ret_std_z_90', 'adx_60', 'rsi', 'volume_z_60', 'volume_z_90', 'sma_pct_14', 'atr_z_90', 'adx_90', 'adx', 'eng_bb_mid', 'obv_diff_14', 'eng_rsi', 'volume_z_30', 'eng_vwap', 'z_obv', 'obv_diff_30', 'z_vwap_dev_60', 'plus_di', 'z_vwap_dev',  'vol_z_90', 'z_vwap_dev_90', 'sma_pct_60', 'obv_pct_30', 'bb_w_z_60', 'obv_diff_60', 'vol_z_60', 'roc_14', 'vol_spike_90', 'obv_pct_14', 'rsi_30', 'sma_pct_28', 'vwap_dev_pct_30', 'plus_di_30', 'vol_spike_60', 'vwap_dev_pct_90', 'vwap_dev_pct_60', 'plus_di_90', 'eng_macd', 'z_vwap_dev_30',  'minus_di', 'ret_std_z_30', 'sma_pct_90', 'bb_w_z_30', 'vwap_dev_pct_z_30', 'z_bb_w', 'vwap_dev_pct_z_60', 'obv_sma_60', 'body_pct', 'roc_28', 'ret', 'eng_ma', 'vwap_dev_pct_z_90']

        # 'roc_30', 'plus_di_60', 'obv_pct_60', 'bb_w_z_90', 'obv_sma_90', 'obv_diff_90', 'adx_30', 'obv_pct_90', 'roc_60',

        # trailstop_pct_tick = max(1.5 * bidask_spread_pct, trailstop_pct_tick) # safe minimum trail stop set to 'factor' times the bid spread (so bid starts enough higher than the trail)
        
    return features_cols_tick, sellmin_idx_tick, sess_start_tick, trailstop_pct_tick, atr_mult_tick, vwap_atr_mult_tick, \
    rsi_thresh_tick, buy_thresh_tick, sign_smoothwin_tick

# automatically executed function to get the parameters for the selected ticker
features_cols_tick, sellmin_idx_tick, sess_start_tick, trailstop_pct_tick, atr_mult_tick, vwap_atr_mult_tick, \
rsi_thresh_tick, buy_thresh_tick, sign_smoothwin_tick\
    = ticker_parameters(ticker              = ticker,
                        sign_smoothwin_tick = 180,
                        sellmin_idx_tick    = None,
                        sess_start_tick     = sess_start_pred,
                        buy_thresh_tick     = 0.1,
                        trailstop_pct_tick  = 0.3,
                        atr_mult_tick       = 10,
                        vwap_atr_mult_tick  = 10,
                        rsi_thresh_tick     = 35)
                        # return_thresh_tick  = 0 # TBD
                                                                                                           




