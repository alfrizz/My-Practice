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
init_cash = 100000

month_to_check = '2022-01'
sel_val_rmse = 0.10347

smooth_sign_win = 15 # smoothing of the continuous target signal
extra_windows = [30, 45, 60] #  to produce additional smoothed/rolling copies of selected indicators for each window 

createCSVbase = False # set to True to regenerate the 'base' csv
createCSVsign = False # set to True to regenerate the 'sign' csv

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidask_spread_pct = 0.02 # conservative 2 percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes)

feats_min_std = 0.03
feats_max_corr = 0.997
thresh_gb = 56 # use ram instead of memmap, if X_buf below this value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################################################################################################


optuna_folder = "optuna_results" 
models_folder = "trainings" 
log_file = Path(models_folder) / "training_diagnostics.txt"

save_path  = Path("dfs")
alpaca_csv = save_path / f"{ticker}_0_alpaca.csv"
base_csv = save_path / f"{ticker}_1_base.csv"
indunsc_csv = save_path / f"{ticker}_2_indunsc.csv"
sign_csv = save_path / f"{ticker}_3_sign.csv"
feat_all_csv = save_path / f"{ticker}_4_feat_all.csv"
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
# Params => tc_id=tc_ema_21_50; reset_peak=False; rsi_min_thresh=36; rsi_max_thresh=63; vol_thresh=4.792337693948495; adx_thresh=35.4613512679662; atr_mult=2.1082213436587214; vwap_atr_mult=-4.991044535157611; buy_factor=0.9862755372912152; sell_factor=0.0006975427281712354; trailstop_pct=8.893346370633799; sess_start=08:00:00
# Trial 1878: 100%|██████████| 364/364 [00:36<00:00,  9.84it/s]
# [Results] mean_pnl:86.3430 mean_excess:91.2474 improv:-1960.52%
# Action counts: {'Buy': 208, 'Sell': 118514, 'Hold': 216092}
# Best trial is: 1878 with best_val: 91.2474

if ticker == 'AAPL':
    # min_prof_thr_tick   = 0.5  # minimum % gain to accept a swing
    # max_down_prop_tick  = 0.25 # base retracement threshold (fraction of move)
    # gain_tightfact_tick = 0.02 # tighter retracement for larger gains
    # tau_time_tick       = 60.0 # minutes half-life for temporal decay
    # tau_dur_tick        = 60.0 # minutes half-life for duration boost
    # thresh_mode_tick    = "median_nonzero"
    # thresh_window_tick  = 7    # rolling window (bars) for rolling modes
    # col_signal_tick     = 'ema_9'
    # sign_thresh_tick    = 'ema_21'
    # col_atr_tick        = "atr_14"
    # col_adx_tick        = "adx_14"
    # col_vol_spike_tick  = 'vol_spike_14'
    # col_rsi_tick        = "rsi_6"
    # col_vwap_tick       = "vwap_ohlc_close_session"
    # reset_peak_tick     = False
    # rsi_min_thresh_tick = 36
    # rsi_max_thresh_tick = 63
    # vol_thresh_tick     = 4.792337693948495
    # adx_thresh_tick     = 35.4613512679662
    # atr_mult_tick       = 2.1082213436587214
    # vwap_atr_mult_tick  = -4.991044535157611
    # buy_factor_tick     = 0.9862755372912152
    # sell_factor_tick    = 0.0006975427281712354
    # trailstop_pct_tick  = 8.893346370633799 
    min_prof_thr_tick   = 0.10793969867338143  # minimum % gain to accept a swing
    max_down_prop_tick  = 0.3111564451111338   # base retracement threshold (fraction of move)
    gain_tightfact_tick = 0.02304060785274687  # tighter retracement for larger gains
    tau_time_tick       = 531.7958126071431    # minutes half-life for temporal decay
    tau_dur_tick        = 510.1403461829883    # minutes half-life for duration boost
    thresh_mode_tick    = "median_nonzero"
    thresh_window_tick  = 7                    # rolling window (bars) for rolling modes
    col_signal_tick     = 'ema_9'
    sign_thresh_tick    = 'ema_21'
    col_atr_tick        = "atr_14"
    col_adx_tick        = "adx_14"
    col_vol_spike_tick  = 'vol_spike_14'
    col_rsi_tick        = "rsi_6"
    col_vwap_tick       = "vwap_ohlc_close_session"
    reset_peak_tick     = False
    rsi_min_thresh_tick = 26
    rsi_max_thresh_tick = 43
    vol_thresh_tick     = 1.6068182237266366
    adx_thresh_tick     = 17.754197354951703
    atr_mult_tick       = 5.394179205319159
    vwap_atr_mult_tick  = -4.459095574686781
    buy_factor_tick     = 0.27182961472178646
    sell_factor_tick    = 0.3770809163036939
    trailstop_pct_tick  = 7.301574488995097
    sess_start_tick     = sess_premark
    features_cols_tick  = ['dist_low_28', 'dist_low_60', 'dist_low_30', 'in_sess_time', 'dist_high_60', 'dist_high_30', 'dist_high_28', 'minute_time', 'hour_time', 'ret_std_z_90', 'adx_60', 'rsi', 'volume_z_60', 'volume_z_90', 'sma_pct_14', 'atr_z_90', 'adx_90', 'adx', 'eng_bb_mid', 'obv_diff_14', 'eng_rsi', 'volume_z_30', 'eng_vwap', 'z_obv', 'obv_diff_30', 'z_vwap_dev_60', 'plus_di', 'z_vwap_dev',  'vol_z_90', 'z_vwap_dev_90', 'sma_pct_60', 'obv_pct_30', 'bb_w_z_60', 'obv_diff_60', 'vol_z_60', 'roc_14', 'vol_spike_90', 'obv_pct_14', 'rsi_30', 'sma_pct_28', 'vwap_dev_pct_30', 'plus_di_30', 'vol_spike_60', 'vwap_dev_pct_90', 'vwap_dev_pct_60', 'plus_di_90', 'eng_macd', 'z_vwap_dev_30',  'minus_di', 'ret_std_z_30', 'sma_pct_90', 'bb_w_z_30', 'vwap_dev_pct_z_30', 'z_bb_w', 'vwap_dev_pct_z_60', 'obv_sma_60', 'body_pct', 'roc_28', 'ret', 'eng_ma', 'vwap_dev_pct_z_90']
