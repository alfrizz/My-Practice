from pathlib import Path 
from datetime import datetime
import datetime as dt

import torch # Kept only for torch.device
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
start_date_sim = '2021-09-01'
end_date_sim = '2023-03-01'
month_to_check = '2022-01'
sel_val_rmse = '0.06561'

train_prop, val_prop = 0.70, 0.15 # dataset split proportions
bidask_spread_pct = 0.02 # conservative 2 percent (per leg) to compensate for conservative all-in scenario (spreads, latency, queuing, partial fills, spikes)

feats_min_std = 0.03
feats_max_corr = 0.999
mmap_thresh_gb = 16 # use ram instead of memmap, if X_buf below this value

device = torch.device("cuda") 


#########################################################################################################

save_path  = Path("dfs")
alpaca_parquet = save_path / f"{ticker}_0_alpaca.parquet"
base_parquet = save_path / f"{ticker}_1_base.parquet"
indunsc_parquet = save_path / f"{ticker}_2_indunsc.parquet"
feat_all_parquet = save_path / f"{ticker}_3_feat_all.parquet"
sign_featall_parquet = save_path / f"{ticker}_4_sign_featall.parquet"

pred_test_pqt = save_path / f"{ticker}_5_pred_test.parquet"
pred_trainval_pqt = save_path / f"{ticker}_5_pred_trainval.parquet"


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
    "TRANSFORMER_D_MODEL":   128,     # transformer embedding width (d_model); adapter maps features into this
    "TRANSFORMER_LAYERS":    2,      # number of encoder layers; ↑depth/complexity, ↓speed/stability
    "TRANSFORMER_HEADS":     8,      # attention heads; must divide d_model; ↑multi-aspect focus
    "TRANSFORMER_FF_MULT":   1,      # FFN expansion factor (d_model * MULT); ↑internal capacity
    "DROPOUT_TRANS":         0.1,    # transformer dropout; ↑regularization

    # ── Long Bi-LSTM ──────────────
    "USE_LONG_LSTM":         False, # enable bidirectional “long” LSTM
    "LONG_UNITS":            64,     # long-LSTM total output width (bidirectional); per-dir hidden = UNITS // 2
    "DROPOUT_LONG":          0.12,   # dropout after projection or long-LSTM layer

    # ── Regression head, smoothing, huber and delta ───────────────────────────────────────
    "FLATTEN_MODE":          "attn", # head input: "flatten" | "last" | "pool" | "attn" (attention-based pooling)
    "PRED_HIDDEN":           128,     # head MLP hidden dim; ↑capacity, ↓generalization
    
    "ALPHA_SMOOTH":          0,      # derivative slope-penalty weight; ↑smoothness, ↓temporal precision
    "WARMUP_STEPS":          3,      # steps to ramp slope penalty from 0 to ALPHA_SMOOTH
    
    "USE_HUBER":             False, # if True use Huber loss; ↑robustness to outliers vs MSE
    "HUBER_DELTA":           0.1,    # Huber transition threshold; scale to typical target error
    
    "USE_DELTA":             False, # enable Delta residual prediction vs base signal
    "LAMBDA_DELTA":          0.1,    # Delta loss weight; ↑residual fit focus, ↓base learning stability

    # ── Optimizer & Scheduler Settings ──────────────────────────────────
    "MAX_EPOCHS":            90,     # max training epochs
    "EARLY_STOP_PATIENCE":   15,      # epochs to wSait without improvement; ↑robustness to noise
    "WEIGHT_DECAY":          2e-6,   # L2 penalty; ↑weight shrinkage, ↓overfitting
    "CLIPNORM":              2,      # max grad norm; ↑stability (prevents exploding gradients)
    
    "ONECYCLE_MAX_LR":       5e-4,   # peak LR in 1cycle policy
    "HEAD_LR_PCT":           1,      # LR multiplier for head vs backbone (0.0 to 1.0)
    "ONECYCLE_DIV_FACTOR":   10,     # initial_lr = max_lr / div_factor
    "ONECYCLE_FINAL_DIV":    100,    # end_lr = max_lr / final_div_factor
    "ONECYCLE_PCT_START":    0.1,    # fraction of cycle spent increasing LR
    "ONECYCLE_STRATEGY":     'cos',  # 'cos' or 'linear' LR annealing

    # ── Training Control Parameters ────────────────────────────────────
    "TRAIN_BATCH":           8,     # sequences per train batch; ↑throughput, ↓stochasticity/GPU heat
    "VAL_TEST_BATCH":        8,     # sequences per val batch; must be 1 for stateful LSTMs
    "TRAIN_WORKERS":         2,      # DataLoader sub-processes; 0 = main thread (safest for laptop heat)
    "TRAIN_PREFETCH_FACTOR": 8,   # batches to pre-load; ignored if TRAIN_WORKERS = 0
    "PIN_MEMORY":            True,  # Locks RAM for faster GPU transfer; ↓latency, ↑kernel pressure (Set False if experiencing reboots/Watchdog errors)
    
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


if ticker == 'AAPL':
    # # --- Swing Detection Parameters ---
    # min_prof_thr_tick    = 0.05097433784612173   # Minimum price move required to identify a valid swing
    # max_down_prop_tick   = 0.694720761798232    # Fraction of the move allowed for retracement before exit
    # gain_tightfact_tick  = 3.133710750083402    # Factor that tightens exit tolerance as unrealized profit grows
    # tau_time_tick        = 46.37000086515304     # Half-life constant for decaying signal strength over time
    # tau_dur_tick         = 69.2091040513691    # Constant used to boost the score based on swing duration

    # # --- Strategy Filter Thresholds ---
    # rsi_min_thresh_tick  = 100     # Ceiling value for the RSI filter during entry conditions
    # rsi_max_thresh_tick  = 0     # Floor value for the RSI filter during exit conditions
    # adx_thresh_tick      = 15      # Threshold for minimum required trend strength
    # trailstop_pct_tick   = 1.044410522412767      # Distance used for the trailing stop-loss mechanism
    # atr_mult_tick        = 1   # Coefficient for setting volatility-based price stops
    # vwap_atr_mult_tick   = -28.173758909259544   # Coefficient for the safety offset relative to the VWAP
    # buy_factor_tick      = 4.106794878301534   # Multiplier applied to calculate the buy position weight
    # sell_factor_tick     = 5.897963476949319    # Multiplier applied to calculate the sell position weight
    
    # --- Indicator Columns ---
    col_atr_tick         = "atr_21"               # Column label for the Average True Range indicator
    col_adx_tick         = "adx_21"               # Column label for the Average Directional Index
    col_rsi_tick         = "rsi_21"               # Column label for the Relative Strength Index
    col_vwap_tick        = "vwap_ohlc_close_session" # Column label for the Volume Weighted Average Price
    
    # --- Logic Configuration ---
    col_signal_tick      = "cci_20"          # The primary signal column used for decision making ("targ_signal" or "pred_signal" or any indicator signal)
    sign_thresh_tick     = 0.0        # The reference column or value for signal activation ("signal_thresh" or any indicator threshold)

    # --- Thresholding Logic ---
    thresh_mode_tick     = "median_nonzero"       # Statistical method for defining the signal threshold
    thresh_window_tick   = 0                      # Lookback period used for dynamic thresholding calculations

    strategy_cols_tick   = [col_atr_tick, col_adx_tick, col_rsi_tick, col_vwap_tick]
    signals_cols_tick    = ['close_raw', 'targ_signal', 'signal_thresh']
    features_cols_tick   = [
            'donch_w_20', 'atr_28_RZ', 'range_pct', 'time_premark', 'dist_low_100', 
            'stoch_k_14_3_3', 'kc_h_20_20_2.0_RZ', 'kc_w_20_20_2.0', 'bb_w_50_2p0', 
            'donch_h_20_RZ', 'ret_std_21', 'sma_pct_5', 'trade_count', 'atr_pct_7', 
            'minus_di_14', 'sma_pct_21', 'time_day_of_year', 'upper_shad', 'ret', 
            'time_in_sess', 'high_RZ', 'bb_hband_20_3p0_RZ', 'minus_di_28', 'roc_5', 
            'time_afthour', 'stoch_d_9_3_3', 'macd_diff_12_26_9_RZ', 'atr_pct_14', 
            'rsi_7', 'adx_21', 'time_minute', 'sma_pct_9', 'sma_9_RZ', 'donch_w_55', 
            'rsi_21', 'minus_di_7', 'rsi_28', 'time_hour', 'rsi_14', 'cci_20', 
            'volume', 'atr_pct_28'
        ]


