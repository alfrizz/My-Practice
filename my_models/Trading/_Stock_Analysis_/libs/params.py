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
init_cash = 100000

month_to_check = '2021-09'
sel_val_rmse = 0.30414

# shares_per_trade = 1
# createCSVbase = False # set to True to regenerate the 'base' csv
# createCSVsign = False # set to True to regenerate the 'sign' csv

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidask_spread_pct = 0.02 # conservative 2 percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes)

feats_min_std = 0.03
feats_max_corr = 0.999
thresh_gb = 56 # use ram instead of memmap, if X_buf below this value

device = torch.device("cuda") 


#########################################################################################################


optuna_folder = "optuna_results" 
models_folder = "trainings" 
log_file = Path(models_folder) / "training_diagnostics.txt"

save_path  = Path("dfs")
alpaca_csv = save_path / f"{ticker}_0_alpaca.csv"
base_csv = save_path / f"{ticker}_1_base.csv"
indunsc_csv = save_path / f"{ticker}_2_indunsc.csv"
feat_all_csv = save_path / f"{ticker}_3_feat_all.csv"
sign_featall_csv = save_path / f"{ticker}_4_sign_featall.csv"
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
    "HEAD_LR_PCT":           1,      # percentage of learning rate to apply to the head ([0-1])
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
# After-Hours	           ~4:00 PM – 8:00 PM	             21:00 – 01:00

sess_start_reg   = datetime.strptime('13:30', '%H:%M').time()  
sess_start_pred  = dt.time(*divmod((sess_start_reg.hour * 60 + sess_start_reg.minute) - hparams["LOOK_BACK"], 60))
sess_start_shift = dt.time(*divmod((sess_start_reg.hour * 60 + sess_start_reg.minute) - 2*hparams["LOOK_BACK"], 60))

sess_premark     = datetime.strptime('08:00' , '%H:%M').time()  
sess_end         = datetime.strptime('20:00' , '%H:%M').time() 
sess_afthour     = datetime.strptime('00:00' , '%H:%M').time() 


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

if ticker == 'AAPL':

    min_prof_thr_tick   = 0.00245986446131613   # minimum % gain to accept a swing
    max_down_prop_tick  = 0.0041145609656908525  # base retracement threshold (fraction of move)
    gain_tightfact_tick = 0.0594101600369578     # tighter retracement for larger gains
    tau_time_tick       = 8.000579617527153      # minutes half-life for temporal decay
    tau_dur_tick        = 5.634526369801502      # minutes half-life for duration boost
    thresh_mode_tick    = "median_nonzero"
    thresh_window_tick  = None                   # rolling window (bars) for rolling modes
    
    col_atr_tick        = "atr_14"
    col_adx_tick        = "adx_14"
    col_rsi_tick        = "rsi_6"
    col_vwap_tick       = "vwap_ohlc_close_session"
    
    col_signal_tick     = 'signal_raw'                # 'signal_raw' for target
    sign_thresh_tick    = 'signal_thresh'                # 'signal_thresh' for target
    
    reset_peak_tick     = False
    rsi_min_thresh_tick = 0
    rsi_max_thresh_tick = 95
    adx_thresh_tick     = 7.577417782611593
    atr_mult_tick       = 0.10407349500377423
    vwap_atr_mult_tick  = 0.11168380289115962
    buy_factor_tick     = 0.007323139245182256
    sell_factor_tick    = 0.002947796369068403
    trailstop_pct_tick  = 6.117895367737556

    features_cols_tick  = ['range_pct', 'atr_pct_7', 'atr_pct_28', 'time_afthour', 'time_premark', 'kc_w_20_20_2.0', 'bb_w_20_2p0', 'donch_w_20', 'ret_std_63', 'atr_pct_14', 'donch_w_55', 'ret_std_21', 'upper_shad', 'time_in_sess', 'dist_high_200', 'bb_w_50_2p0', 'lower_shad', 'time_hour', 'dist_low_200', 'time_week_of_year', 'trade_count', 'volume', 'atr_7_RZ', 'atr_14_RZ', 'atr_28_RZ', 'time_day_of_year', 'time_month', 'vol_spike_28', 'plus_di_28', 'stoch_k_14_3_3', 'rolling_max_close_200_RZ', 'adx_14', 'minus_di_28', 'minus_di_14', 'cci_20', 'plus_di_7', 'plus_di_14', 'adx_28', 'rsi_6', 'rolling_min_close_200_RZ', 'vol_spike_14', 'stoch_d_9_3_3', 'sma_5_RZ', 'minus_di_7', 'sma_21_RZ', 'sma_9_RZ', 'cci_14', 'sma_pct_200', 'stoch_k_9_3_3', 'cmf_14']
    signals_cols_tick   = ['close_raw', 'signal_raw', 'signal_thresh']
