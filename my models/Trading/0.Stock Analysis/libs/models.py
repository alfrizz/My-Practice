from typing import Sequence, List, Tuple, Optional, Union

import pandas as pd
import numpy  as np
import math

import datetime as dt
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as Funct

from sklearn.preprocessing import StandardScaler

from libs import params


#########################################################################################################

def build_lstm_tensors(
    df: pd.DataFrame,
    *,
    look_back: int,                  # number of past minutes to feed into each LSTM sample
    features_cols: Sequence[str],    # list of column names to use as inputs
    label_col: str,                  # name of the column we’ll predict (next‐step)
    regular_start: dt.time,          # only keep windows whose end‐time ≥ this market‐open time
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert 1-minute bar data into PyTorch tensors for a stateful LSTM.

    Each day is processed independently (so we can reset hidden state daily),
    features are standardized per day, then sliding windows of length `look_back`
    are built, next-step targets aligned, and finally RTH filtering applied.
    Everything is concatenated across days, then moved once to the target device.

    Returns:
      X         – float32 Tensor, shape (N, look_back, F): standardized input windows
      y         – float32 Tensor, shape (N,       ): next-step scalar targets
      raw_close – float32 Tensor, shape (N,       ): actual close price at predict time
      raw_bid   – float32 Tensor, shape (N,       ): actual bid   price at predict time
      raw_ask   – float32 Tensor, shape (N,       ): actual ask   price at predict time

    Where:
      N = total windows across all days passing the RTH filter,
      F = number of feature columns,
      look_back = minutes per input window.
    """

    # 0) Ensure we have a valid device
    device = device or torch.device("cpu")

    # Prepare lists to collect per-day tensors (all on CPU for now)
    X_days, y_days = [], []
    c_days, b_days, a_days = [], [], []  # for raw close/bid/ask prices

    # 1) Process each calendar day separately
    for _, day_df in df.groupby(df.index.normalize(), sort=False):
        # a) sort by timestamp to ensure chronological order
        day_df = day_df.sort_index()

        # b) extract raw price series as CPU tensors (length T = minutes in this day)
        close_t = torch.from_numpy(day_df["close"].to_numpy(dtype=np.float32))
        bid_t   = torch.from_numpy(day_df["bid"].to_numpy(dtype=np.float32))
        ask_t   = torch.from_numpy(day_df["ask"].to_numpy(dtype=np.float32))

        # c) standardize feature columns *per day*
        feats_np = StandardScaler().fit_transform(day_df[features_cols].to_numpy())
        feats_t  = torch.from_numpy(feats_np.astype(np.float32))  # shape (T, F)

        # d) extract next-step labels as a CPU tensor
        labels_t = torch.from_numpy(day_df[label_col].to_numpy(dtype=np.float32))  # shape (T,)

        # 2) build sliding windows of length `look_back`
        #    unfold creates (T - look_back + 1, look_back, F)
        windows = feats_t.unfold(0, look_back, 1)

        # 3) align windows with next-step targets:
        #    drop the last window so we can pair each window with the label at t+1
        windows = windows[:-1]               # now (T - look_back, look_back, F)
        targets  = labels_t[look_back:]      # shape (T - look_back,)
        c_pts    = close_t[look_back:]       # raw close at prediction time
        b_pts    = bid_t[look_back:]         # raw bid   at prediction time
        a_pts    = ask_t[look_back:]         # raw ask   at prediction time

        # 4) RTH filter: keep only windows whose end-time ≥ `regular_start`
        end_times = day_df.index.time[look_back:]        # length = T - look_back
        mask      = np.array(end_times) >= regular_start # boolean array
        if not mask.any():
            # no valid windows this day → skip
            continue

        # convert to torch mask and apply to all per-day tensors
        mask_t = torch.from_numpy(mask)
        windows = windows[mask_t]
        targets = targets[mask_t]
        c_pts   = c_pts[mask_t]
        b_pts   = b_pts[mask_t]
        a_pts   = a_pts[mask_t]

        # 5) collect filtered windows, labels, and raw prices
        X_days.append(windows)
        y_days.append(targets)
        c_days.append(c_pts)
        b_days.append(b_pts)
        a_days.append(a_pts)

    # 6) if nothing survived RTH filter, alert the user
    if not X_days:
        raise ValueError(
            "No windows passed the RTH filter; check your regular_start or input data."
        )

    # 7) concatenate all days along the sample dimension (dim=0), still on CPU
    X_cpu         = torch.cat(X_days, dim=0)  # shape: (N, look_back, F)
    y_cpu         = torch.cat(y_days, dim=0)  # shape: (N,)
    raw_close_cpu = torch.cat(c_days, dim=0)  # shape: (N,)
    raw_bid_cpu   = torch.cat(b_days, dim=0)  # shape: (N,)
    raw_ask_cpu   = torch.cat(a_days, dim=0)  # shape: (N,)

    # 8) one‐shot move to the target device
    #    non_blocking=True may overlap host→device copies with compute
    X         = X_cpu.to(device, non_blocking=True)
    y         = y_cpu.to(device, non_blocking=True)
    raw_close = raw_close_cpu.to(device, non_blocking=True)
    raw_bid   = raw_bid_cpu.to(device, non_blocking=True)
    raw_ask   = raw_ask_cpu.to(device, non_blocking=True)

    # 9) return fully‐prepared tensors for training or inference
    return X, y, raw_close, raw_bid, raw_ask


#########################################################################################################


def chronological_split(
    X: torch.Tensor,
    y: torch.Tensor,
    raw_close: torch.Tensor,
    raw_bid: torch.Tensor,
    raw_ask: torch.Tensor,
    df: pd.DataFrame,
    *,
    look_back: int,
    regular_start: dt.time,
    train_prop: float,
    val_prop: float,
    train_batch: int,
    device = torch.device("cpu")
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    List[int],
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Split the big (N, look_back, F) dataset into train/val/test by calendar day
    using index‐range slicing (no giant boolean masks). All splits happen on `device`.

    Returns exactly:
      (X_tr, y_tr),
      (X_val, y_val),
      (X_te, y_te, close_te, bid_te, ask_te),
      samples_per_day,
      day_id_tr, day_id_val, day_id_te
    """

    # 0) pick a real device & move the full dataset there
    device = device or (X.device if X.device is not None else torch.device("cpu"))
    X         = X.to(device)
    y         = y.to(device)
    raw_close = raw_close.to(device)
    raw_bid   = raw_bid.to(device)
    raw_ask   = raw_ask.to(device)

    # 1) Count how many windows come from each calendar day
    samples_per_day: List[int] = []
    all_days: List[pd.Timestamp] = []

    for day, day_df in df.groupby(df.index.normalize(), sort=False):
        all_days.append(day)
        end_times = day_df.index.time[look_back:]
        mask_rth  = np.array([t >= regular_start for t in end_times])
        samples_per_day.append(int(mask_rth.sum()))

    # 2) Sanity‐check total windows
    total = sum(samples_per_day)
    if total != X.size(0):
        raise ValueError(f"Window count mismatch: {total} vs {X.size(0)}")

    # 3) Decide how many days go to train/val/test
    D               = len(samples_per_day)
    train_days_orig = int(D * train_prop)
    batches_needed  = (train_days_orig + train_batch - 1) // train_batch
    train_days      = min(D, batches_needed * train_batch)
    cut_train       = train_days - 1
    cut_val         = int(D * (train_prop + val_prop))  # inclusive last val‐day index

    # 4) Build a cumulative‐sum array of sample counts
    #    cum[i] = total windows in days[0..i-1], so slices are views
    cum = np.concatenate([[0], np.cumsum(samples_per_day)])

    # 5) Compute slice indices
    end_train = int(cum[train_days])          # start of day train_days
    end_val   = int(cum[cut_val + 1])        # start of day cut_val+1

    # 6) Range‐slice the big tensors (these are views, not copies)
    X_tr       = X[:end_train]
    y_tr       = y[:end_train]

    X_val      = X[end_train:end_val]
    y_val      = y[end_train:end_val]

    X_te       = X[end_val:]
    y_te       = y[end_val:]
    close_te   = raw_close[end_val:]
    bid_te     = raw_bid[end_val:]
    ask_te     = raw_ask[end_val:]

    # 7) Build day‐ID tags via repeat_interleave per split
    #    (you can drop these if you don’t strictly need them)
    def make_day_ids(start_day: int, end_day: int) -> torch.Tensor:
        # day indices start_day .. end_day inclusive
        counts = samples_per_day[start_day : end_day + 1]
        days   = torch.arange(start_day, end_day + 1, device=device, dtype=torch.long)
        return torch.repeat_interleave(days, torch.tensor(counts, device=device))

    day_id_tr  = make_day_ids(0,          cut_train)
    day_id_val = make_day_ids(cut_train+1, cut_val)
    day_id_te  = make_day_ids(cut_val+1,  D-1)

    # ────────────────────────────────────────────────────────────────────────────
    # SIDE‐EFFECT: dump the raw‐bar test‐period DF exactly as before
    test_days = [all_days[i] for i in range(D) if i > cut_val]
    df_test   = df.loc[df.index.normalize().isin(test_days)]
    df_test.to_csv(f"dfs training/{params.ticker}_test_DF.csv", index=True)

    # 8) Return splits + metadata
    return (
        (X_tr, y_tr),
        (X_val, y_val),
        (X_te, y_te, close_te, bid_te, ask_te),
        samples_per_day,
        day_id_tr, day_id_val, day_id_te
    )


#########################################################################################################


class DayWindowDataset(Dataset):
    """
    Dataset of calendar‐day windows. Each __getitem__(i) returns:
      - x_day: (1, W_i, look_back, F)
      - y_day: (1, W_i)
      - weekday: int
    Or, for test days:
      - x_day, y_day, raw_close, raw_bid, raw_ask, weekday
    """
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        day_id: torch.Tensor,
        weekday: torch.Tensor,
        raw_close: Optional[torch.Tensor] = None,
        raw_bid:   Optional[torch.Tensor] = None,
        raw_ask:   Optional[torch.Tensor] = None
    ):
        self.X    = X
        self.y    = y
        self.day_id = day_id
        self.weekday = weekday
        self.raw_close = raw_close
        self.raw_bid   = raw_bid
        self.raw_ask   = raw_ask
        # compute counts per day and build boundaries
        counts = torch.bincount(day_id)
        boundaries = torch.cat([
            torch.tensor([0], dtype=torch.long),
            torch.cumsum(counts, dim=0)
        ])
        self.start = boundaries[:-1]
        self.end   = boundaries[1:]
        self.has_raw = raw_close is not None

    def __len__(self):
        return len(self.start)

    def __getitem__(self, idx: int):
        s = self.start[idx].item()
        e = self.end[idx].item()
        x_day = self.X[s:e].unsqueeze(0)   # (1, W_i, look_back, F)
        y_day = self.y[s:e].unsqueeze(0)   # (1, W_i)
        wd    = int(self.weekday[s].item())
        if self.has_raw:
            rc = self.raw_close[s:e]
            rb = self.raw_bid[s:e]
            ra = self.raw_ask[s:e]
            return x_day, y_day, rc, rb, ra, wd
        return x_day, y_day, wd

        
#########################################################################################################


def pad_collate(
    batch: List[Union[
        Tuple[torch.Tensor, torch.Tensor, int],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ]]
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    Pads each-day examples to the maximum window-length in `batch`.

    Supports elements:
      (x_day, y_day, weekday)
    or
      (x_day, y_day, raw_close, raw_bid, raw_ask, weekday)

    Returns either:
      (batch_x, batch_y, batch_wd)
    or:
      (batch_x, batch_y, batch_rc, batch_rb, batch_ra, batch_wd)
    All on CPU.
    """
    # find maximum number of windows W_max
    W_max = max(elem[0].shape[1] for elem in batch)

    xs, ys, rcs, rbs, ras, wds = [], [], [], [], [], []
    for elem in batch:
        x_day, y_day, *rest = elem
        weekday = rest[-1]
        has_raw = len(rest) == 4

        W_i = x_day.shape[1]
        pad_amt = W_max - W_i

        # pad x_day: (1, W_i, look_back, F) → (1, W_max, look_back, F)
        x_p = Funct.pad(
            x_day,
            pad=(0, 0, 0, 0, 0, pad_amt, 0, 0),
            mode='constant', value=0.0
        )
        xs.append(x_p)

        # pad y_day: (1, W_i) → (1, W_max)
        y_p = Funct.pad(y_day, pad=(0, pad_amt), mode='constant', value=0.0)
        ys.append(y_p)

        if has_raw:
            rc, rb, ra = rest[:-1]
            # raw vectors: (W_i,) → (1, W_i) → pad → (1, W_max)
            rc_p = Funct.pad(rc.unsqueeze(0), pad=(0, pad_amt), mode='constant', value=0.0)
            rb_p = Funct.pad(rb.unsqueeze(0), pad=(0, pad_amt), mode='constant', value=0.0)
            ra_p = Funct.pad(ra.unsqueeze(0), pad=(0, pad_amt), mode='constant', value=0.0)
            rcs.append(rc_p)
            rbs.append(rb_p)
            ras.append(ra_p)

        wds.append(weekday)

    batch_x  = torch.cat(xs, dim=0)          # (B, W_max, look_back, F)
    batch_y  = torch.cat(ys, dim=0)          # (B, W_max)
    batch_wd = torch.tensor(wds, dtype=torch.int64)

    if rcs:
        batch_rc = torch.cat(rcs, dim=0)     # (B, W_max)
        batch_rb = torch.cat(rbs, dim=0)
        batch_ra = torch.cat(ras, dim=0)
        return batch_x, batch_y, batch_rc, batch_rb, batch_ra, batch_wd

    return batch_x, batch_y, batch_wd

###################

def split_to_day_datasets(
    X_tr:         torch.Tensor,
    y_tr:         torch.Tensor,
    day_id_tr:    torch.Tensor,
    X_val:        torch.Tensor,
    y_val:        torch.Tensor,
    day_id_val:   torch.Tensor,
    X_te:         torch.Tensor,
    y_te:         torch.Tensor,
    day_id_te:    torch.Tensor,
    raw_close_te: torch.Tensor,
    raw_bid_te:   torch.Tensor,
    raw_ask_te:   torch.Tensor,
    *,
    df:           pd.DataFrame,  # full-minute DataFrame for weekday lookup
    train_batch:  int = 8,       # days per training batch
    train_workers:int = 0,       # no workers default
    device = torch.device("cpu")
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build three DataLoaders over DayWindowDataset:
      - train_loader (batch_size=train_batch days, padded)
      - val_loader   (1 day per batch)
      - test_loader  (1 day per batch, includes raw prices)

    """

    print("▶️ Entered split_to_day_datasets")

    # 1) Build weekday‐code tensors for each split
    print("1) building weekday arrays")
    all_wd = df.index.dayofweek.to_numpy(np.int64)
    n_tr   = X_tr.size(0)
    n_val  = X_val.size(0)
    wd_tr  = torch.from_numpy(all_wd[:n_tr])
    wd_val = torch.from_numpy(all_wd[n_tr : n_tr + n_val])
    wd_te  = torch.from_numpy(
        all_wd[n_tr + n_val : n_tr + n_val + X_te.size(0)]
    )
    print(f"   Weekdays counts → tr={len(wd_tr)}, val={len(wd_val)}, te={len(wd_te)}")

    # 2) Move all splits to CPU for indexing
    print("2) moving all splits to CPU")
    X_tr, y_tr, day_id_tr = X_tr.cpu(), y_tr.cpu(), day_id_tr.cpu()
    X_val, y_val, day_id_val = X_val.cpu(), y_val.cpu(), day_id_val.cpu()
    X_te, y_te, day_id_te = X_te.cpu(), y_te.cpu(), day_id_te.cpu()
    rc_te, rb_te, ra_te   = raw_close_te.cpu(), raw_bid_te.cpu(), raw_ask_te.cpu()
    print("   CPU casts done")

    # 3) Zero-base the val/test day IDs so each split’s day indices start at 0
    #    This makes len(ds_val)==410, len(ds_te)==422 instead of thousands.
    print("3) zero-bas­ing day_id for val & test")
    val_min = int(day_id_val.min().item())
    test_min= int(day_id_te.min().item())
    day_id_val = (day_id_val - val_min)
    day_id_te  = (day_id_te  - test_min)
    print(f"   val_day_id ∈ [0..{int(day_id_val.max().item())}], total days={day_id_val.max().item()+1}")
    print(f"   te_day_id  ∈ [0..{int(day_id_te .max().item())}], total days={day_id_te .max().item()+1}")

    # 4) Instantiate your DayWindowDataset exactly as before
    print("4) instantiating DayWindowDatasets")
    ds_tr = DayWindowDataset(X_tr, y_tr, day_id_tr, wd_tr)
    print("   ds_tr days:", len(ds_tr))
    ds_val = DayWindowDataset(X_val, y_val, day_id_val, wd_val)
    print("   ds_val days:", len(ds_val))
    ds_te = DayWindowDataset(
        X_te, y_te, day_id_te, wd_te,
        raw_close=rc_te,
        raw_bid=rb_te,
        raw_ask=ra_te
    )
    print("   ds_te days:", len(ds_te))

    # # 5) save datasets directly
    # torch.save(
    #     ds_val,
    #     params.save_path / f"{params.ticker}_val_ds.pt",
    #     _use_new_zipfile_serialization=False
    # )
    # torch.save(
    #     ds_te,
    #     params.save_path / f"{params.ticker}_test_ds.pt",
    #     _use_new_zipfile_serialization=False
    # )
    # print("datasets saved!")

    # 6) Build DataLoaders with our pad_collate:
    print("5) building DataLoaders")
    train_loader = DataLoader(
        ds_tr,
        batch_size=train_batch,
        shuffle=False,
        drop_last=True,
        collate_fn=pad_collate,
        num_workers=train_workers, # how many background processes are preparing data
        pin_memory=True, # ets the GPU DMA engine pull data off page-locked buffers without blocking the CPU (faster)
        persistent_workers=False, # so they stay alive across epochs (faster)
        prefetch_factor=None # how many batches per worker to pre-load into the DataLoader queue
    )
    print("   train_loader ready")

    val_loader = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print("   val_loader ready")

    test_loader = DataLoader(
        ds_te,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print("   test_loader ready")

    return train_loader, val_loader, test_loader



#########################################################################################################

def naive_rmse(data_loader):
    """
    Zero‐forecast baseline RMSE for any DayDataset loader:
      – always predicts 0
      – works for val_loader (xb, yb, wd)
      – and test_loader (xb, yb, raw_close, raw_bid, raw_ask, wd)
    """
    total_se = 0.0
    total_n  = 0

    for batch in data_loader:
        # batch[1] is always y_day regardless of extra fields
        y_day = batch[1]  
        # y_day: shape (1, W) → squeeze→ (W,)
        y = y_day.squeeze(0).view(-1)

        # accumulate (0 - y)^2 = y^2
        total_se += float((y ** 2).sum().item())
        total_n  += y.numel()

    return math.sqrt(total_se / total_n)
    

#########################################################################################################


class DualMemoryLSTM(nn.Module):
    """
    Two-stage, stateful sequence model with:
      • One short-term (daily) LSTM
      • One long-term (weekly) LSTM
      • Window-level self-attention over each day's LSTM output
      • Variational Dropout + LayerNorm after every major block
      • Automatic resets of hidden states at day/week boundaries
      • Time-distributed linear head producing one scalar per time-step

    Note on attention state:
      – The MultiheadAttention module is stateless: it recomputes
        attention weights on each forward pass over that day's window.
      – Only the LSTM hidden states carry memory across windows.
    """

    def __init__(
        self,
        n_feats: int,
        short_units: int,
        long_units: int,
        dropout_short: float = 0.4,
        dropout_long: float  = 0.5,
        att_heads: int    = 4,
        att_drop: float   = 0.1,
    ):
        """
        Args:
          n_feats       – number of input features per time-step
          short_units   – hidden size of the daily LSTM
          long_units    – hidden size of the weekly LSTM
          dropout_short – dropout rate after attention on daily features
          dropout_long  – dropout rate after weekly LSTM outputs
          att_heads     – number of heads in self-attention
          att_drop      – dropout inside the attention module
        """
        super().__init__()
        self.n_feats     = n_feats
        self.short_units = short_units
        self.long_units  = long_units

        # 1) Daily LSTM (single layer, carries hidden state across batches)
        self.short_lstm = nn.LSTM(
            input_size  = n_feats,
            hidden_size = short_units,
            batch_first = True,
            num_layers  = 1,       # one layer only
            dropout     = 0.0      # no built-in inter-layer dropout
        )

        # 2) Self-attention over the day's short-LSTM outputs
        self.attn = nn.MultiheadAttention(
            embed_dim   = short_units,
            num_heads   = att_heads,
            dropout     = att_drop,
            batch_first = True     # expects (B, S, C) format
        )

        # 3) Variational dropout + layer norm on the attended short features
        self.do_short = nn.Dropout(dropout_short)
        self.ln_short = nn.LayerNorm(short_units)

        # 4) Weekly LSTM (single layer, carries hidden state across days)
        self.long_lstm = nn.LSTM(
            input_size  = short_units,
            hidden_size = long_units,
            batch_first = True,
            num_layers  = 1,
            dropout     = 0.0
        )
        self.do_long = nn.Dropout(dropout_long)
        self.ln_long = nn.LayerNorm(long_units)

        # 5) Time-distributed linear head: project each time-step to a scalar
        self.pred = nn.Linear(long_units, 1)

        # 6) Buffers for hidden & cell states; inited lazily on first forward
        self.h_short = None
        self.c_short = None
        self.h_long  = None
        self.c_long  = None

    def _init_states(self, B: int, device: torch.device):
        """
        Create zero initial hidden & cell states for both LSTMs.
        Shapes:
          h_short, c_short – (1, B, short_units)
          h_long,  c_long  – (1, B, long_units)
        """
        self.h_short = torch.zeros(1, B, self.short_units, device=device)
        self.c_short = torch.zeros(1, B, self.short_units, device=device)
        self.h_long  = torch.zeros(1, B, self.long_units,  device=device)
        self.c_long  = torch.zeros(1, B, self.long_units,  device=device)

    def reset_short(self):
        """
        Zero out daily LSTM states at each new day.
        """
        if self.h_short is not None:
            B, dev = self.h_short.size(1), self.h_short.device
            # Re-init both short and long to keep shapes in sync
            # (long state preserved by reset_long if you wish)
            self._init_states(B, dev)

    def reset_long(self):
        """
        Zero out weekly LSTM states at each new week,
        preserving the daily LSTM state across the reset.
        """
        if self.h_long is not None:
            B, dev = self.h_long.size(1), self.h_long.device
            # stash daily states
            hs, cs = self.h_short, self.c_short
            # re-init both
            self._init_states(B, dev)
            # restore daily only
            self.h_short, self.c_short = hs.to(dev), cs.to(dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying:
          1) daily LSTM → out_short_raw
          2) self-attention over the day window
          3) dropout + layernorm → out_short
          4) weekly LSTM → out_long
          5) dropout + layernorm → out_long
          6) linear head → (B, S, 1) predictions

        Input:
          x: (B, S, F) or extra dims → will be reshaped to (B, S, F)
        Returns:
          (B, S, 1) per-timestep scalar outputs
        """
        # Collapse extra dims so x is (B, S, F)
        if x.dim() > 3:
            *lead, S, F = x.shape
            x = x.view(-1, S, F)

        # Ensure feature-last ordering
        if x.dim() == 3 and x.size(-1) != self.n_feats:
            x = x.transpose(1, 2).contiguous()

        B, S, _ = x.size()
        dev      = x.device

        # Lazy init states on first forward or batch-size change
        if self.h_short is None or self.h_short.size(1) != B:
            self._init_states(B, dev)

        # 1) Daily LSTM pass
        out_short_raw, (h_s, c_s) = self.short_lstm(x, (self.h_short, self.c_short))
        # detach so no backprop through time across days
        self.h_short, self.c_short = h_s.detach(), c_s.detach()

        # 2) Self-attention on the day's outputs
        #    Query = Key = Value = out_short_raw
        attn_out, _ = self.attn(out_short_raw, out_short_raw, out_short_raw)
        # Residual connection
        out_short = out_short_raw + attn_out

        # 3) Dropout + LayerNorm on attended features
        out_short = self.do_short(out_short)
        out_short = self.ln_short(out_short)

        # 4) Weekly LSTM pass on short representations
        out_long, (h_l, c_l) = self.long_lstm(out_short, (self.h_long, self.c_long))
        self.h_long, self.c_long = h_l.detach(), c_l.detach()

        # 5) Dropout + LayerNorm on weekly outputs
        out_long = self.do_long(out_long)
        out_long = self.ln_long(out_long)

        # 6) Time-distributed linear prediction
        return self.pred(out_long)


