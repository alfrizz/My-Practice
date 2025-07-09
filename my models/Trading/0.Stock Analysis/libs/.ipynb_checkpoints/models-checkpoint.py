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

from libs import params
###############################################################################


class DayDataset(Dataset):
    """
    A Dataset that returns one *calendar day*’s worth of sliding windows at a time.

    This is designed to avoid CUDA initialization issues in DataLoader workers by
    keeping all data and index tensors on CPU.  You can optionally include raw
    price series (close, bid, ask) alongside your standardized features.

    Usage:
      - Build with arrays or tensors for X, y, day_id, weekday, [raw_close, raw_bid, raw_ask].
      - DataLoader over DayDataset yields, per index:
          * x_day: Tensor shape (1, W, look_back, n_feats)
          * y_day: Tensor shape (1, W)
          * weekday: Python int (0=Monday…6=Sunday)
        or, if raw prices provided, also:
          * raw_close_day, raw_bid_day, raw_ask_day: each Tensor (1, W)

    Arguments:
      X         : CPU array or Tensor of shape (N, look_back, n_feats)
      y         : CPU array or Tensor of shape (N,)
      day_id    : CPU array or Tensor of shape (N,) with integer day indices [0..D-1]
      weekday   : CPU array or Tensor of shape (N,) with weekday codes 0..6
      raw_close : (optional) CPU array or Tensor of shape (N,)
      raw_bid   : (optional) CPU array or Tensor of shape (N,)
      raw_ask   : (optional) CPU array or Tensor of shape (N,)

    Returns:
      len(dataset) = D = number of unique days
      __getitem__(i) returns data for the i-th calendar day only.
    """

    def __init__(
        self,
        X:         Union[np.ndarray, torch.Tensor],  # (N, look_back, n_feats)
        y:         Union[np.ndarray, torch.Tensor],  # (N,)
        day_id:    Union[np.ndarray, torch.Tensor],  # (N,) integer day indices
        weekday:   Union[np.ndarray, torch.Tensor],  # (N,) 0=Mon…6=Sun
        raw_close: Optional[Union[np.ndarray, torch.Tensor]] = None,
        raw_bid:   Optional[Union[np.ndarray, torch.Tensor]] = None,
        raw_ask:   Optional[Union[np.ndarray, torch.Tensor]] = None
    ):
        # 1) Convert all inputs to CPU‐resident torch.Tensors with appropriate dtypes
        #    This prevents DataLoader workers from touching the GPU.
        self.X       = torch.as_tensor(X,       dtype=torch.float32, device='cpu')
        self.y       = torch.as_tensor(y,       dtype=torch.float32, device='cpu')
        self.day_id  = torch.as_tensor(day_id,  dtype=torch.int64,   device='cpu')
        self.weekday = torch.as_tensor(weekday, dtype=torch.int64,   device='cpu')

        if raw_close is not None:
            # All-or-none: if one raw price is provided, expect all three
            self.raw_close = torch.as_tensor(raw_close, dtype=torch.float32, device='cpu')
            self.raw_bid   = torch.as_tensor(raw_bid,   dtype=torch.float32, device='cpu')
            self.raw_ask   = torch.as_tensor(raw_ask,   dtype=torch.float32, device='cpu')
        else:
            self.raw_close = self.raw_bid = self.raw_ask = None

        # 2) Stable-sort by day_id so that each day’s windows are contiguous in memory.
        #    `stable=True` preserves the order of samples within each day.
        order = torch.argsort(self.day_id, stable=True)  # CPU indices
        self.X       = self.X      [order]
        self.y       = self.y      [order]
        self.day_id  = self.day_id [order]
        self.weekday = self.weekday[order]
        if self.raw_close is not None:
            self.raw_close = self.raw_close[order]
            self.raw_bid   = self.raw_bid  [order]
            self.raw_ask   = self.raw_ask  [order]

        # 3) Identify slice‐ranges for each calendar day
        #    A “break” occurs wherever day_id changes between consecutive samples.
        #    We add +1 to shift break indices to the *start* of the next day.
        changes = (self.day_id[1:] != self.day_id[:-1]).nonzero(as_tuple=False).view(-1)
        day_starts = [0] + (changes + 1).tolist()
        day_ends   = (changes + 1).tolist() + [len(self.day_id)]

        #    Build a list of 1D index‐tensors, one per day,
        #    so __getitem__ can slice out windows by index.
        self.day_slices: List[torch.Tensor] = []
        for start, end in zip(day_starts, day_ends):
            # torch.arange on CPU returns a 1D tensor of indices [start, start+1, ..., end-1].
            idx_tensor = torch.arange(start, end, dtype=torch.int64, device='cpu')
            self.day_slices.append(idx_tensor)

        # 4) Sanity check: ensure no NaNs slipped into features or targets
        assert not torch.isnan(self.X).any(), "NaNs found in X tensor!"
        assert not torch.isnan(self.y).any(), "NaNs found in y tensor!"

    def __len__(self) -> int:
        """
        Return the number of unique calendar days.
        Each __getitem__ call returns data for exactly one day.
        """
        return len(self.day_slices)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, int],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ]:
        """
        Fetch the sliding‐window batch for the `idx`-th calendar day.

        Returns either:
          (x_day, y_day, weekday)
        or:
          (x_day, y_day, raw_close_day, raw_bid_day, raw_ask_day, weekday)

        Where:
          x_day           - Tensor, shape (1, W, look_back, n_feats)
          y_day           - Tensor, shape (1, W)
          raw_close_day   - Tensor, shape (1, W) [if provided]
          raw_bid_day     - Tensor, shape (1, W)
          raw_ask_day     - Tensor, shape (1, W)
          weekday         - Python int, same for all windows this day
        """
        # a) Get the list of sample‐indices for this day (on CPU)
        idx_tensor = self.day_slices[idx]  # 1D Tensor of length W

        # b) Slice out feature windows and targets for the entire day
        #    x_block: (W, look_back, n_feats)
        x_block = self.X[idx_tensor]
        #    y_block: (W,)
        y_block = self.y[idx_tensor]

        # c) Retrieve the weekday code (all samples same day → pick first)
        weekday_code = int(self.weekday[idx_tensor[0]].item())

        # d) Add a leading “batch” dimension of size 1 for compatibility
        #    so that downstream code sees shape (batch, W, look_back, n_feats)
        x_day = x_block.unsqueeze(0)  # (1, W, look_back, n_feats)
        y_day = y_block.unsqueeze(0)  # (1, W)

        if self.raw_close is None:
            # Minimal return: no raw prices included
            return x_day, y_day, weekday_code
        else:
            # e) Also slice & unsqueeze raw price arrays
            c_day = self.raw_close[idx_tensor].unsqueeze(0)  # (1, W)
            b_day = self.raw_bid  [idx_tensor].unsqueeze(0)
            a_day = self.raw_ask  [idx_tensor].unsqueeze(0)
            return x_day, y_day, c_day, b_day, a_day, weekday_code




# -----------------------------------------------------------------------------
# collate_fn for variable-length training batches
# -----------------------------------------------------------------------------

def pad_collate(
    batch: List[Union[
        Tuple[torch.Tensor, torch.Tensor, int],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A custom collate_fn that pads each calendar-day batch to the maximum window-length.

    Inputs (per batch element):
      x_day : Tensor of shape (1, W_i, look_back, F)
      y_day : Tensor of shape (1, W_i)
      weekday: Python int

    Optionally, batch elements may include raw prices:
      (x_day, y_day, raw_close, raw_bid, raw_ask, weekday)

    Steps:
      1. Determine W_max = max_i W_i across the batch.
      2. For each element:
         a. Compute pad_amt = W_max - W_i.
         b. Pad the window‐axis (dim=1) of x_day with zeros on the right.
         c. Pad the window‐axis (last dim) of y_day with zeros on the right.
         d. Collect padded x_day, y_day, and weekday.
      3. Concatenate all x_days along batch‐dim → batch_x: (B, W_max, look_back, F).
      4. Concatenate all y_days along batch‐dim → batch_y: (B, W_max).
      5. Stack weekday ints into a 1D tensor → batch_wd: (B,).

    Returns:
      batch_x : Tensor (B, W_max, look_back, F), dtype float32, on CPU
      batch_y : Tensor (B, W_max),               dtype float32, on CPU
      batch_wd: Tensor (B,),                     dtype int64,   on CPU

    Note:
      - We keep everything on CPU here. In your training loop you call
          xb, yb, wdb = batch
          xb = xb.to(device, non_blocking=True)
          yb = yb.to(device, non_blocking=True)
          wdb= wdb.to(device, non_blocking=True)
    """
    # 1) find the maximum window-length across the batch
    W_max = max(elem[0].shape[1] for elem in batch)

    padded_xs = []
    padded_ys = []
    weekdays  = []

    for elem in batch:
        # Unpack a batch element: x_day, y_day, optional raw prices, then weekday at end
        x_day, y_day, *rest = elem
        weekday = rest[-1]  # last item is the weekday int

        # Current window-length
        W_i = x_day.shape[1]
        pad_amt = W_max - W_i


        x_padded = Funct.pad(
            x_day,
            pad=(   # pad on [batch, window, look_back, feature] dims:
                0, 0,         # no padding on feature dim (last dim)
                0, 0,         # no padding on look_back dim
                0, pad_amt,   # pad right side of window dim
                0, 0          # no padding on batch dim
            ),
            mode='constant',
            value=0.0
        )

        # 2b) Pad y_day on its last dimension (window axis).
        #    y_day shape is (1, W_i), so pad=(left, right) on last dim
        y_padded = Funct.pad(
            y_day,
            pad=(0, pad_amt),
            mode='constant',
            value=0.0
        )

        padded_xs.append(x_padded)
        padded_ys.append(y_padded)
        weekdays.append(weekday)

    # 3) Concatenate padded x_days along batch‐dim → shape (B, W_max, look_back, F)
    batch_x = torch.cat(padded_xs, dim=0)
    # 4) Concatenate padded y_days along batch‐dim → shape (B, W_max)
    batch_y = torch.cat(padded_ys, dim=0)
    # 5) Build weekday tensor → shape (B,)
    batch_wd = torch.tensor(weekdays, dtype=torch.int64)

    return batch_x, batch_y, batch_wd





#-----------------------------------------------------------------------------
# Final split_to_day_datasets
# -----------------------------------------------------------------------------
def split_to_day_datasets(
    X_tr:         np.ndarray,
    y_tr:         np.ndarray,
    day_id_tr:    np.ndarray,
    X_val:        np.ndarray,
    y_val:        np.ndarray,
    day_id_val:   np.ndarray,
    X_te:         np.ndarray,
    y_te:         np.ndarray,
    day_id_te:    np.ndarray,
    raw_close_te: np.ndarray,
    raw_bid_te:   np.ndarray,
    raw_ask_te:   np.ndarray,
    *,
    df:           pd.DataFrame,   # original minute‐bar DataFrame for weekday lookup
    train_batch:  int = 8,        # number of days per training batch
    train_workers:int = 4,        # DataLoader workers for train set
    train_pin:    bool = True     # whether to pin memory on train loader
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build three DataLoaders that iterate over *whole days* of sliding-window data:

      1) train_loader
         - Dataset: DayDataset(X_tr, y_tr, day_id_tr, weekday_tr)
         - batch_size = train_batch days
         - shuffle=False     (chronological ordering)
         - drop_last=True    (only full train batches)
         - collate_fn=pad_collate
         - num_workers=train_workers, pin_memory=train_pin
         - persistent_workers=True

      2) val_loader
         - Dataset: DayDataset(X_val, y_val, day_id_val, weekday_val)
         - batch_size=1
         - num_workers=0, pin_memory=False

      3) test_loader
         - Dataset: DayDataset(X_te, y_te, day_id_te, weekday_te,
                               raw_close=raw_close_te, raw_bid=raw_bid_te,
                               raw_ask=raw_ask_te)
         - batch_size=1, num_workers=0, pin_memory=False

    After calling this you get:
       train_loader, val_loader, test_loader

    In your training loop do:
       xb, yb, wdb = batch
       xb = xb.to(device, non_blocking=True)
       yb = yb.to(device, non_blocking=True)
       wdb = wdb.to(device, non_blocking=True)
    """
    # A) Build a full-length weekday array from the original DataFrame index
    #    shape = (total_minutes,)
    all_weekdays = df.index.dayofweek.to_numpy(dtype=np.int64)

    # B) Partition weekdays into train/val/test segments by sample count
    n_tr  = len(X_tr)
    n_val = len(X_val)
    # weekday_tr  covers the first n_tr windows
    weekday_tr  = all_weekdays[:n_tr]
    # weekday_val covers the next n_val windows
    weekday_val = all_weekdays[n_tr : n_tr + n_val]
    # weekday_te covers the remaining windows (should match len(X_te))
    weekday_te  = all_weekdays[n_tr + n_val : n_tr + n_val + len(X_te)]

    # C) Instantiate DayDatasets (all internal tensors are on CPU)
    ds_tr  = DayDataset(X_tr,  y_tr,  day_id_tr,  weekday_tr)
    ds_val = DayDataset(X_val, y_val, day_id_val, weekday_val)
    ds_te  = DayDataset(
        X_te, y_te, day_id_te, weekday_te,
        raw_close=raw_close_te,
        raw_bid=raw_bid_te,
        raw_ask=raw_ask_te
    )

    # save test dataset
    torch.save(ds_te, params.save_path / f"{params.ticker}_test_ds.pt")
    # save validation dataset
    torch.save(ds_val, params.save_path / f"{params.ticker}_val_ds.pt")

    # D) Training DataLoader: multi-day batches, padding, multiple workers
    train_loader = DataLoader(
        ds_tr,
        batch_size=train_batch,
        shuffle=False,
        drop_last=True,
        collate_fn=pad_collate,
        num_workers=train_workers,
        pin_memory=train_pin,
        persistent_workers=True
    )

    # E) Validation DataLoader: one day per batch, CPU-only
    val_loader = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # F) Test DataLoader: same as val_loader but includes raw prices
    test_loader = DataLoader(
        ds_te,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

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


