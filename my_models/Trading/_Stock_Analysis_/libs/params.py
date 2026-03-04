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
import sys

from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

#########################################################################################################

ticker = 'AAPL'
init_cash = 100000
init_df_year = 2016
month_to_check = '2021-01'
sel_val_rmse = '0.16086'

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidask_spread_pct = 0.02 # conservative 2 percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes)

feats_min_std = 0.03
feats_max_corr = 0.999
mmap_thresh_gb = 16 # use ram instead of memmap, if X_buf below this value

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
pred_test_pqt = save_path / f"{ticker}_5_pred_test.parquet"
pred_trainval_pqt = save_path / f"{ticker}_5_pred_trainval.parquet"


# def _human(n):
#     for u in ("B","KB","MB","GB","TB"):
#         if abs(n) < 1024:
#             return f"{n:3.1f}{u}"
#         n /= 1024
#     return f"{n:.1f}PB"

# def to_csv_with_progress(df, path, chunksize=10_000, index=True, show_size_every=5, leave=True):
#     with open(path, "w", newline="") as f:
#         df.iloc[:0].to_csv(f, index=index, date_format="%Y-%m-%d %H:%M:%S")  # header only
#         total = len(df)
#         pbar = tqdm(total=total, desc="Saving CSV", unit="rows", leave=leave, dynamic_ncols=True)
#         chunk_count = 0
#         for start in range(0, total, chunksize):
#             end = start + chunksize
#             df.iloc[start:end].to_csv(f, index=index, header=False, date_format="%Y-%m-%d %H:%M:%S")
#             # no per-chunk flush; update bar
#             written = min(end, total) - start
#             pbar.update(written)
#             chunk_count += 1
#             if show_size_every and (chunk_count % show_size_every == 0):
#                 # use f.tell() for bytes; call infrequently to avoid overhead
#                 pbar.set_postfix(size=_human(f.tell()))
#                 pbar.refresh()
#         pbar.close()



def to_parquet_with_progress(df: pd.DataFrame, filepath, chunksize: int = 25000):
    """
    Saves a DataFrame to Parquet format while displaying a progress bar.
    
    Functionality:
    - Converts the DataFrame to an Apache Arrow table.
    - Slices the table into chunks (default 25k rows) and streams them to disk.
    - Preserves all datatypes and dramatically reduces file size compared to CSV.
    """
    # Convert pandas DataFrame to an Apache Arrow Table
    table = pa.Table.from_pandas(df)
    
    # Calculate total chunks for the progress bar
    total_chunks = (len(table) + chunksize - 1) // chunksize
    
    # Write the table in chunks to update the progress bar
    with pq.ParquetWriter(filepath, table.schema) as writer:
        for i in tqdm(range(0, len(table), chunksize), total=total_chunks, desc=f"Saving Parquet"):
            writer.write_table(table.slice(i, chunksize))


#########################################################################################################


hparams = {
    # ── Input convolution toggle ──────────────────────────
    "USE_CONV":              False,  # enable Conv1d + BatchNorm1d
    "CONV_K":                3,      # Conv1d kernel size; ↑local smoothing, ↓fine-detail
    "CONV_DILATION":         1,      # Conv1d dilation; ↑receptive field, ↓granularity
    "CONV_CHANNELS":         64,     # Conv1d output channels; ↑early-stage capacity, ↓compute

    # ── Temporal ConvNet (TCN) toggle ────────────────────
    "USE_TCN":                False, # enable dilated Conv1d stack
    "TCN_LAYERS":             1,      # number of dilated Conv1d layers
    "TCN_KERNEL":             3,      # kernel size for each TCN layer
    "TCN_CHANNELS":           64,     # TCN output channels; independent from CONV_CHANNELS

    # ── Short Bi-LSTM toggle ──────────────────────────────
    "USE_SHORT_LSTM":        False, # enable bidirectional “short” LSTM
    "SHORT_UNITS":           64,     # short-LSTM total output width (bidirectional); per-dir hidden = UNITS // 2
    "DROPOUT_SHORT":         0.1,    # dropout after short-LSTM; ↑regularization, ↓overfitting

    # ── Transformer toggle ────────────────────────────────
    "USE_TRANSFORMER":       True,   # enable TransformerEncoder
    "TRANSFORMER_D_MODEL":   64,     # transformer embedding width (d_model); adapter maps features into this
    "TRANSFORMER_LAYERS":    1,      # number of encoder layers; ↑depth/complexity, ↓speed/stability
    "TRANSFORMER_HEADS":     4,      # attention heads; must divide d_model; ↑multi-aspect focus
    "TRANSFORMER_FF_MULT":   2,      # FFN expansion factor (d_model * MULT); ↑internal capacity
    "DROPOUT_TRANS":         0.1,    # transformer dropout; ↑regularization

    # ── Long Bi-LSTM ──────────────
    "USE_LONG_LSTM":         False, # enable bidirectional “long” LSTM
    "LONG_UNITS":            64,     # long-LSTM total output width (bidirectional); per-dir hidden = UNITS // 2
    "DROPOUT_LONG":          0.12,   # dropout after projection or long-LSTM layer

    # ── Regression head, smoothing, huber and delta ───────────────────────────────────────
    "FLATTEN_MODE":          "attn", # head input: "flatten" | "last" | "pool" | "attn" (attention-based pooling)
    "PRED_HIDDEN":           96,     # head MLP hidden dim; ↑capacity, ↓generalization
    
    "ALPHA_SMOOTH":          0,      # derivative slope-penalty weight; ↑smoothness, ↓temporal precision
    "WARMUP_STEPS":          3,      # steps to ramp slope penalty from 0 to ALPHA_SMOOTH
    
    "USE_HUBER":             False, # if True use Huber loss; ↑robustness to outliers vs MSE
    "HUBER_DELTA":           0.1,    # Huber transition threshold; scale to typical target error
    
    "USE_DELTA":             False, # enable Delta residual prediction vs base signal
    "LAMBDA_DELTA":          0.1,    # Delta loss weight; ↑residual fit focus, ↓base learning stability

    # ── Optimizer & Scheduler Settings ──────────────────────────────────
    "MAX_EPOCHS":            90,     # max training epochs
    "EARLY_STOP_PATIENCE":   9,      # epochs to wait without improvement; ↑robustness to noise
    "WEIGHT_DECAY":          2e-6,   # L2 penalty; ↑weight shrinkage, ↓overfitting
    "CLIPNORM":              2,      # max grad norm; ↑stability (prevents exploding gradients)
    
    "ONECYCLE_MAX_LR":       7e-4,   # peak LR in 1cycle policy
    "HEAD_LR_PCT":           1,      # LR multiplier for head vs backbone (0.0 to 1.0)
    "ONECYCLE_DIV_FACTOR":   10,     # initial_lr = max_lr / div_factor
    "ONECYCLE_FINAL_DIV":    100,    # end_lr = max_lr / final_div_factor
    "ONECYCLE_PCT_START":    0.1,    # fraction of cycle spent increasing LR
    "ONECYCLE_STRATEGY":     'cos',  # 'cos' or 'linear' LR annealing

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":           16,     # sequences per train batch; ↑throughput, ↓stochasticity/GPU heat
    "VAL_TEST_BATCH":        16,     # sequences per val batch; must be 1 for stateful LSTMs
    "TRAIN_WORKERS":         0,      # DataLoader sub-processes; 0 = main thread (safest for laptop heat)
    "TRAIN_PREFETCH_FACTOR": None,   # batches to pre-load; ignored if TRAIN_WORKERS = 0
    
    "PIN_MEMORY":            False,  # Locks RAM for faster GPU transfer; ↓latency, ↑kernel pressure (Set False if experiencing reboots/Watchdog errors)
    "LOOK_BACK":             60,     # sequence length; minutes of history per training example
    "MICRO_SAMPLE_K":        1,      # sample count for latency metrics; ↑diagnostics, ↓speed
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

# rsi_min_thresh31
# rsi_max_thresh100
# adx_thresh7.002976435560177
# atr_mult0.9096852009472672
# vwap_atr_mult1.5900548665418703
# buy_factor0.9719112773876222
# sell_factor0.6074313558823258
# trailstop_pct0.3260636449258955
# thresh_mode"roll_median"
# thresh_window77
# best_value612.4890526315789

if ticker == 'AAPL':

    min_prof_thr_tick    = 0.0018291260396256649  # minimum % gain to accept a swing
    max_down_prop_tick   = 0.004030139474703384   # base retracement threshold (fraction of move)
    gain_tightfact_tick  = 0.025006471074875608   # tighter retracement for larger gains
    tau_time_tick        = 5.400049943858033      # minutes half-life for temporal decay
    tau_dur_tick         = 6.357695320055652      # minutes half-life for duration boost
    
    col_atr_tick         = "atr_21"
    col_adx_tick         = "adx_21"
    col_rsi_tick         = "rsi_21"
    col_vwap_tick        = "vwap_ohlc_close_session"
    
    col_signal_tick      = "pred_signal"          # 'targ_signal' for target, 'pred_signal' for ML, eg "ema_*" for IND
    sign_thresh_tick     = "signal_thresh"        # 'signal_thresh' for target or ML, constant or eg "ema_*" for IND
    
    rsi_min_thresh_tick  = 31
    rsi_max_thresh_tick  = 100
    adx_thresh_tick      = 7.002976435560177
    atr_mult_tick        = 0.9096852009472672
    vwap_atr_mult_tick   = 1.5900548665418703
    buy_factor_tick      = 0.9719112773876222
    sell_factor_tick     = 0.6074313558823258
    trailstop_pct_tick   = 0.3260636449258955

    thresh_mode_tick     = "roll_median"          # "median_nonzero","mean_nonzero","p90","p95","p99","median","mean","roll_mean","roll_median","roll_p90","roll_p95","numeric"
    thresh_window_tick   = 77                    # rolling window (bars) for rolling thresh_modes
    thresh_mode_num_tick = 0.01562252543390733    # numeric threshold for "numeric" thresh_mode

    strategy_cols_tick   = [col_atr_tick, col_adx_tick, col_rsi_tick, col_vwap_tick]
    signals_cols_tick    = ['close_raw', 'targ_signal', 'signal_thresh']
    features_cols_tick   = ['range_pct', 'atr_pct_7', 'bb_w_50_2p0', 'ret_std_21', 'bb_w_20_3p0', 'time_in_sess', 'atr_pct_14', 'donch_w_20', 'time_premark', 'kc_w_20_20_2.0', 'atr_pct_28', 'ret_std_63', 'time_afthour', 'upper_shad', 'donch_w_55', 'dist_low_100', 'lower_shad', 'trade_count', 'dist_high_100', 'time_hour', 'volume', 'time_day_of_year', 'vol_spike_28', 'time_month', 'atr_7_RZ', 'atr_14_RZ', 'minus_di_28', 'kc_h_20_20_1.5_RZ', 'ret', 'adx_28', 'time_minute', 'minus_di_14', 'plus_di_7', 'atr_21_RZ', 'atr_28_RZ', 'sma_pct_21', 'time_week_of_year', 'minus_di_21', 'donch_h_55_RZ', 'bb_lband_50_2p0_RZ', 'kc_h_20_20_2.0_RZ', 'minus_di_7', 'plus_di_28', 'bb_lband_20_3p0_RZ', 'bb_hband_50_2p0_RZ', 'plus_di_21', 'rsi_28', 'plus_di_14', 'stoch_d_9_3_3', 'roc_21']


