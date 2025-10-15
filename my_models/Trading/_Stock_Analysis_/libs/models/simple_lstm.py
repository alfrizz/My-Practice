from libs import plots, params, models_core

from typing import Optional, Dict, Tuple, List, Sequence, Union, Any

import gc 
import os
import io
import tempfile
import copy
import re
import warnings

import pandas as pd
import numpy  as np
import math

import datetime as dt
from datetime import datetime, time
from pathlib import Path

import torchmetrics
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32   = True
torch.backends.cudnn.allow_tf32         = True
torch.backends.cudnn.benchmark          = True


######################################################################################################


# class ModelClass(nn.Module):
#     """
#     Stateful CNN → BiLSTM (short) → projection → (optional) BiLSTM (long) → regression head

#     Architecture (when fully enabled):
#       0) Conv1d → BatchNorm1d → ReLU
#       1) Stateful bidirectional LSTM (short) → LayerNorm → Dropout
#       2) Linear projection short_units→long_units → LayerNorm → Dropout
#       3) Stateful bidirectional LSTM (long) → LayerNorm → Dropout
#       4) Time-distributed MLP head: Linear(long_units→pred_hidden) → ReLU → Linear(pred_hidden→1)

#     Gating flags:
#       use_conv        : if False, skip Conv1d+BN entirely
#       use_short_lstm  : if False, bypass the short Bi-LSTM block
#       use_long_lstm   : if False, bypass the long  Bi-LSTM block

#     State handling helpers:
#       _init_states  : allocate both short & long LSTM buffers if enabled
#       reset_short   : re-initialize only the short-LSTM buffers
#       reset_long    : re-initialize only the long-LSTM buffers
#     """
#     def __init__(
#         self,
#         n_feats:         int,
#         short_units:     int,
#         long_units:      int,
#         dropout_short:   float,
#         dropout_long:    float,
#         conv_k:          int,
#         conv_dilation:   int,
#         pred_hidden:     int,
#         use_conv:        bool = True,
#         use_short_lstm:  bool = True,
#         use_long_lstm:   bool = False,
#     ):
#         super().__init__()
#         self.n_feats        = n_feats
#         self.short_units    = short_units
#         self.long_units     = long_units
#         self.pred_hidden    = pred_hidden
#         self.use_conv       = use_conv
#         self.use_short_lstm = use_short_lstm
#         self.use_long_lstm  = use_long_lstm

#         # 0) Conv1d + BatchNorm1d or identity
#         if use_conv:
#             pad = (conv_k // 2) * conv_dilation
#             self.conv = nn.Conv1d(n_feats, n_feats, conv_k,
#                                   dilation=conv_dilation, padding=pad)
#             self.bn   = nn.BatchNorm1d(n_feats)
#         else:
#             self.conv = nn.Identity()
#             self.bn   = nn.Identity()

#         # 1) Short Bi-LSTM or bypass
#         if use_short_lstm:
#             assert short_units % 2 == 0, "short_units must be even"
#             self.short_lstm = nn.LSTM(
#                 input_size    = n_feats,
#                 hidden_size   = short_units // 2,
#                 batch_first   = True,
#                 bidirectional = True,
#             )
#             self.ln_short = nn.LayerNorm(short_units)
#             self.do_short = nn.Dropout(dropout_short)
#         else:
#             self.short_lstm = None
#             self.ln_short   = nn.Identity()
#             self.do_short   = nn.Identity()

#         # 2) Projection: maps (short_units or n_feats) → long_units
#         proj_in = short_units if use_short_lstm else n_feats
#         self.short2long = nn.Linear(proj_in, long_units)
#         self.ln_proj    = nn.LayerNorm(long_units)
#         self.do_proj    = nn.Dropout(dropout_long)

#         # 3) Long Bi-LSTM or bypass
#         if use_long_lstm:
#             assert long_units % 2 == 0, "long_units must be even"
#             self.long_lstm = nn.LSTM(
#                 input_size    = long_units,
#                 hidden_size   = long_units // 2,
#                 batch_first   = True,
#                 bidirectional = True,
#             )
#             self.ln_long = nn.LayerNorm(long_units)
#             self.do_long = nn.Dropout(dropout_long)
#         else:
#             self.long_lstm = None
#             self.ln_long   = nn.Identity()
#             self.do_long   = nn.Identity()

#         # 4) Time-distributed regression head
#         self.pred = nn.Sequential(
#             nn.Linear(long_units, pred_hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(pred_hidden, 1)
#         )

#         # 5) Stateful buffers (start empty)
#         self.h_short = None
#         self.c_short = None
#         self.h_long  = None
#         self.c_long  = None

#     def _init_states(self, B: int):
#         """
#         Allocate hidden/cell for short & long LSTMs if their flags are True.
#         """
#         dev = next(self.parameters()).device

#         if self.use_short_lstm:
#             self.h_short = torch.zeros(2, B, self.short_units // 2, device=dev)
#             self.c_short = torch.zeros(2, B, self.short_units // 2, device=dev)
#         else:
#             self.h_short = self.c_short = None

#         if self.use_long_lstm:
#             self.h_long = torch.zeros(2, B, self.long_units // 2, device=dev)
#             self.c_long = torch.zeros(2, B, self.long_units // 2, device=dev)
#         else:
#             self.h_long = self.c_long = None

#     def reset_short(self):
#         """
#         Re-initialize only the short-LSTM buffers.
#         Safe no-op if use_short_lstm=False.
#         """
#         if self.h_short is not None:
#             B   = self.h_short.size(1)
#             dev = self.h_short.device
#             self.h_short = torch.zeros(2, B, self.short_units // 2, device=dev)
#             self.c_short = torch.zeros(2, B, self.short_units // 2, device=dev)

#     def reset_long(self):
#         """
#         Re-initialize only the long-LSTM buffers.
#         Safe no-op if use_long_lstm=False.
#         """
#         if self.h_long is not None:
#             B   = self.h_long.size(1)
#             dev = self.h_long.device
#             self.h_long = torch.zeros(2, B, self.long_units // 2, device=dev)
#             self.c_long = torch.zeros(2, B, self.long_units // 2, device=dev)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for inputs x of shape
#         (…, S, F) or (S, F). Returns (B, S, 1) time-distributed outputs.
#         """
#         # collapse leading dims → (B, S, F)
#         if x.dim() > 3:
#             *lead, S, F = x.shape
#             x = x.view(-1, S, F)
#         if x.dim() == 2:
#             x = x.unsqueeze(0)
#         B, S, _ = x.shape

#         # 0) Conv1d + BN + ReLU (or identity)
#         xc = x.transpose(1, 2)           # (B, F, S)
#         xc = self.conv(xc)
#         xc = self.bn(xc)
#         x  = Funct.relu(xc).transpose(1, 2)  # back to (B, S, F)

#         # 1) Short Bi-LSTM (stateful) or bypass
#         if self.use_short_lstm:
#             if self.h_short is None or self.h_short.size(1) != B:
#                 self._init_states(B)
#             out_s, (h_s, c_s) = self.short_lstm(x, (self.h_short, self.c_short))
#             h_s.detach_(); c_s.detach_()
#             self.h_short, self.c_short = h_s, c_s
#             out_s = self.ln_short(out_s)
#             out_s = self.do_short(out_s)
#         else:
#             out_s = x

#         # 2) Projection → LayerNorm → Dropout
#         out_p = self.short2long(out_s)
#         out_p = self.ln_proj(out_p)
#         out_p = self.do_proj(out_p)

#         # 3) Long Bi-LSTM (stateful) or bypass
#         if self.use_long_lstm:
#             if self.h_long is None or self.h_long.size(1) != B:
#                 # allocate only long buffers, preserving short state
#                 dev = next(self.parameters()).device
#                 self.h_long = torch.zeros(2, B, self.long_units // 2, device=dev)
#                 self.c_long = torch.zeros(2, B, self.long_units // 2, device=dev)
#             out_l, (h_l, c_l) = self.long_lstm(out_p, (self.h_long, self.c_long))
#             h_l.detach_(); c_l.detach_()
#             self.h_long, self.c_long = h_l, c_l
#             out_l = self.ln_long(out_l)
#             out_l = self.do_long(out_l)
#         else:
#             out_l = out_p

#         # 4) Time-distributed regression head
#         raw_reg = self.pred(out_l)  # → (B, S, 1)
#         return raw_reg


# class ModelClass(nn.Module):
#     """
#     Stateful CNN → BiLSTM (short) → projection → (optional) BiLSTM (long)
#       → flattened-window MLP head + gated linear residual skip,
#       with flattened‐vector LayerNorm and weight‐norm on head & skip.

#     Architecture (when enabled):
#       0) Conv1d + BatchNorm1d + ReLU
#       1) Stateful BiLSTM(short) → LayerNorm → Dropout
#       2) Linear projection → LayerNorm → Dropout
#       3) Stateful BiLSTM(long) → LayerNorm → Dropout
#       4) Flatten look-back window →
#            • ln_flat: LayerNorm(flat_dim)
#            • head_flat:
#                weight_norm(Linear(flat_dim → pred_hidden)) → ReLU
#                weight_norm(Linear(pred_hidden → 1))
#            • skip_proj: weight_norm(Linear(flat_dim → 1))
#            • skip gate α: nn.Parameter(init = SKIP_ALPHA)
#              gated via sigmoid → initial gate ≃ sigmoid(SKIP_ALPHA)
#            output = head_flat(flat_normed) + sigmoid(α) * skip_proj(flat_normed)

#     Gating flags:
#       use_conv, use_short_lstm, use_long_lstm

#     Output shape: (batch_size, 1, 1)
#     """
#     def __init__(
#         self,
#         n_feats:       int,
#         short_units:   int,
#         long_units:    int,
#         dropout_short: float,
#         dropout_long:  float,
#         conv_k:        int,
#         conv_dilation: int,
#         pred_hidden:   int,
#         window_len:    int,
#         use_conv:       bool = True,
#         use_short_lstm: bool = True,
#         use_long_lstm:  bool = False,
#     ):
#         super().__init__()
#         self.n_feats        = n_feats
#         self.short_units    = short_units
#         self.long_units     = long_units
#         self.pred_hidden    = pred_hidden
#         self.window_len     = window_len
#         self.use_conv       = use_conv
#         self.use_short_lstm = use_short_lstm
#         self.use_long_lstm  = use_long_lstm

#         # 0) Conv1d + BatchNorm1d or identity
#         if use_conv:
#             pad = (conv_k // 2) * conv_dilation
#             self.conv = nn.Conv1d(n_feats, n_feats, conv_k,
#                                   dilation=conv_dilation, padding=pad)
#             self.bn   = nn.BatchNorm1d(n_feats)
#         else:
#             self.conv = nn.Identity()
#             self.bn   = nn.Identity()

#         # 1) Short Bi-LSTM or bypass
#         if use_short_lstm:
#             assert short_units % 2 == 0, "short_units must be even"
#             self.short_lstm = nn.LSTM(
#                 input_size    = n_feats,
#                 hidden_size   = short_units // 2,
#                 batch_first   = True,
#                 bidirectional = True,
#             )
#             self.ln_short = nn.LayerNorm(short_units)
#             self.do_short = nn.Dropout(dropout_short)
#         else:
#             self.short_lstm = None
#             self.ln_short   = nn.Identity()
#             self.do_short   = nn.Identity()

#         # 2) Projection → LayerNorm → Dropout
#         proj_in = short_units if use_short_lstm else n_feats
#         self.short2long = nn.Linear(proj_in, long_units)
#         self.ln_proj    = nn.LayerNorm(long_units)
#         self.do_proj    = nn.Dropout(dropout_long)

#         # 3) Long Bi-LSTM or bypass
#         if use_long_lstm:
#             assert long_units % 2 == 0, "long_units must be even"
#             self.long_lstm = nn.LSTM(
#                 input_size    = long_units,
#                 hidden_size   = long_units // 2,
#                 batch_first   = True,
#                 bidirectional = True,
#             )
#             self.ln_long = nn.LayerNorm(long_units)
#             self.do_long = nn.Dropout(dropout_long)
#         else:
#             self.long_lstm = None
#             self.ln_long   = nn.Identity()
#             self.do_long   = nn.Identity()

#         # compute flattened dimension
#         flat_dim = window_len * long_units

#         # 4a) LayerNorm on flattened vector
#         self.ln_flat = nn.LayerNorm(flat_dim)

#         # 4b) Flattened-window head: single hidden layer with weight‐norm
#         self.head_flat = nn.Sequential(
#             weight_norm(nn.Linear(flat_dim,    pred_hidden)),
#             nn.ReLU(inplace=True),
#             weight_norm(nn.Linear(pred_hidden, 1))
#         )

#         # 5) Linear residual skip + learnable gate α, with weight‐norm
#         self.skip_proj  = weight_norm(nn.Linear(flat_dim, 1))
#         init_alpha = params.hparams["SKIP_ALPHA"]
#         self.skip_alpha = nn.Parameter(
#             torch.tensor(init_alpha, dtype=torch.float32)
#         )

#         # 6) Stateful LSTM buffers (initially empty)
#         self.h_short = None
#         self.c_short = None
#         self.h_long  = None
#         self.c_long  = None

#     def _init_states(self, batch_size: int):
#         """Allocate hidden/cell buffers for short & long LSTMs if enabled."""
#         device = next(self.parameters()).device
#         if self.use_short_lstm:
#             self.h_short = torch.zeros(2, batch_size,
#                                        self.short_units // 2,
#                                        device=device)
#             self.c_short = torch.zeros(2, batch_size,
#                                        self.short_units // 2,
#                                        device=device)
#         else:
#             self.h_short = self.c_short = None

#         if self.use_long_lstm:
#             self.h_long = torch.zeros(2, batch_size,
#                                       self.long_units // 2,
#                                       device=device)
#             self.c_long = torch.zeros(2, batch_size,
#                                       self.long_units // 2,
#                                       device=device)
#         else:
#             self.h_long = self.c_long = None

#     def reset_short(self):
#         """Re-init short LSTM buffers (no-op if unused)."""
#         if self.h_short is not None:
#             batch_size, device = self.h_short.size(1), self.h_short.device
#             self.h_short = torch.zeros(2, batch_size,
#                                        self.short_units // 2,
#                                        device=device)
#             self.c_short = torch.zeros(2, batch_size,
#                                        self.short_units // 2,
#                                        device=device)

#     def reset_long(self):
#         """Re-init long LSTM buffers (no-op if unused)."""
#         if self.h_long is not None:
#             batch_size, device = self.h_long.size(1), self.h_long.device
#             self.h_long = torch.zeros(2, batch_size,
#                                       self.long_units // 2,
#                                       device=device)
#             self.c_long = torch.zeros(2, batch_size,
#                                       self.long_units // 2,
#                                       device=device)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for x of shape (…, time_steps, n_feats) or (time_steps, n_feats).
#         Returns (batch_size, 1, 1): one prediction per window.
#         """
#         # collapse leading dims → (batch_size, time_steps, feature_dim)
#         if x.dim() > 3:
#             *lead, time_steps, feature_dim = x.shape
#             x = x.view(-1, time_steps, feature_dim)
#         if x.dim() == 2:
#             x = x.unsqueeze(0)
#         batch_size, time_steps, feature_dim = x.shape

#         # 1) Conv1d or identity
#         xc = x.transpose(1, 2)              # → (batch_size, feature_dim, time_steps)
#         xc = self.conv(xc); xc = self.bn(xc)
#         x  = F.relu(xc).transpose(1, 2)     # → (batch_size, time_steps, feature_dim)

#         # 2) Short LSTM or bypass
#         if self.use_short_lstm:
#             if (self.h_short is None
#                 or self.h_short.size(1) != batch_size):
#                 self._init_states(batch_size)
#             out_s, (h_s, c_s) = self.short_lstm(
#                 x, (self.h_short, self.c_short)
#             )
#             h_s.detach_(); c_s.detach_()
#             self.h_short, self.c_short = h_s, c_s
#             out_s = self.ln_short(out_s); out_s = self.do_short(out_s)
#         else:
#             out_s = x

#         # 3) Projection → LayerNorm → Dropout
#         out_p = self.short2long(out_s)
#         out_p = self.ln_proj(out_p); out_p = self.do_proj(out_p)

#         # 4) Long LSTM or bypass
#         if self.use_long_lstm:
#             if (self.h_long is None
#                 or self.h_long.size(1) != batch_size):
#                 self._init_states(batch_size)
#             out_l, (h_l, c_l) = self.long_lstm(
#                 out_p, (self.h_long, self.c_long)
#             )
#             h_l.detach_(); c_l.detach_()
#             self.h_long, self.c_long = h_l, c_l
#             out_l = self.ln_long(out_l); out_l = self.do_long(out_l)
#         else:
#             out_l = out_p

#         # 5) Flatten + LayerNorm + gated residual head
#         assert time_steps == self.window_len, (
#             f"Expected window_len={self.window_len}, got {time_steps}"
#         )
#         flat       = out_l.reshape(batch_size, -1)  # (batch_size, flat_dim)
#         flat_norm  = self.ln_flat(flat)             # normalize flattened vector
#         main       = self.head_flat(flat_norm)      # (batch_size, 1)
#         gate       = torch.sigmoid(self.skip_alpha) # in (0,1)
#         skip       = self.skip_proj(flat_norm) * gate  # (batch_size, 1)
#         output_raw = main + skip                       # residual sum
#         return output_raw.unsqueeze(-1)                # (batch_size, 1, 1)


def _allocate_lstm_states(batch_size: int,
                          hidden_size: int,
                          bidirectional: bool,
                          device: torch.device):
    """
    Allocate hidden (h) and cell (c) buffers for an LSTM.
    Returns two tensors shaped (num_directions, batch_size, hidden_size).
    """
    num_directions = 2 if bidirectional else 1
    h = torch.zeros(num_directions, batch_size, hidden_size, device=device)
    c = torch.zeros(num_directions, batch_size, hidden_size, device=device)
    return h, c



###############


class ModelClass(nn.Module):
    """
    Stateful CNN → optional TCN → BiLSTM(short) → optional Transformer
      → projection → optional BiLSTM(long) → flattened-window MLP head + gated skip.

    Gating flags (in __init__):
      use_conv, use_tcn, use_short_lstm, use_transformer, use_long_lstm

    Layer summary:
      0) Conv1d + BatchNorm1d + ReLU
      1) TCN: dilated Conv1d stack (params.TCN_LAYERS, TCN_KERNEL)
      2) Short BiLSTM → LayerNorm → Dropout
      3) TransformerEncoder (params.TRANSFORMER_*)
      4) Linear projection → LayerNorm → Dropout
      5) Long BiLSTM → LayerNorm → Dropout
      6) Flatten look-back window →
           • ln_flat: LayerNorm(window_len×long_units)
           • head_flat: WeightNorm(Linear→ReLU→Linear)
           • skip_proj + gate α (SKIP_ALPHA): gated linear skip
    Output: shape (batch_size, 1, 1)
    """
    def __init__(self,
                 n_feats: int,
                 short_units: int,
                 long_units: int,
                 dropout_short: float,
                 dropout_long: float,
                 conv_k: int,
                 conv_dilation: int,
                 pred_hidden: int,
                 window_len: int,
                 use_conv: bool,
                 use_tcn: bool,
                 use_short_lstm: bool,
                 use_transformer: bool,
                 use_long_lstm: bool,
                 flatten_mode: str,  
                ):
        super().__init__()

        # Save dimensions & flags
        self.window_len      = window_len
        self.short_units     = short_units
        self.long_units      = long_units
        self.use_conv        = use_conv
        self.use_tcn         = use_tcn
        self.use_short_lstm  = use_short_lstm
        self.use_transformer = use_transformer
        self.use_long_lstm   = use_long_lstm

        # 0) Conv1d + BatchNorm1d or identity
        if use_conv:
            pad = (conv_k // 2) * conv_dilation
            self.conv = nn.Conv1d(n_feats, n_feats, conv_k,
                                  dilation=conv_dilation, padding=pad)
            self.bn   = nn.BatchNorm1d(n_feats)
        else:
            self.conv = nn.Identity()
            self.bn   = nn.Identity()

        # 1) TCN: dilated Conv1d stack or identity
        if use_tcn:
            tcn_layers = params.hparams["TCN_LAYERS"]
            tcn_kernel = params.hparams["TCN_KERNEL"]
            blocks = []
            in_ch = n_feats
            for i in range(tcn_layers):
                dil = 2 ** i
                pad = (tcn_kernel // 2) * dil
                blocks += [
                    nn.Conv1d(in_ch, n_feats, tcn_kernel,
                              dilation=dil, padding=pad),
                    nn.BatchNorm1d(n_feats),
                    nn.ReLU(inplace=True)
                ]
                in_ch = n_feats
            self.tcn = nn.Sequential(*blocks)
        else:
            self.tcn = nn.Identity()

        # 2) Short BiLSTM or identity
        if use_short_lstm:
            assert short_units % 2 == 0
            self.short_lstm = nn.LSTM(input_size=n_feats,
                                      hidden_size=short_units // 2,
                                      batch_first=True,
                                      bidirectional=True)
            self.ln_short = nn.LayerNorm(short_units)
            self.do_short = nn.Dropout(dropout_short)
        else:
            self.short_lstm = None
            self.ln_short   = nn.Identity()
            self.do_short   = nn.Identity()

        # buffers for short‐LSTM states
        self.h_short = None
        self.c_short = None

        # 3) TransformerEncoder or identity
        if use_transformer:
            assert use_short_lstm, "Transformer requires short_lstm output"
            layers  = params.hparams["TRANSFORMER_LAYERS"]
            heads   = params.hparams["TRANSFORMER_HEADS"]
            ff_mult = params.hparams["TRANSFORMER_FF_MULT"]
            d_model = short_units
            ff_dim  = d_model * ff_mult
            layer   = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=ff_dim,
                dropout=dropout_short
            )
            self.transformer = nn.TransformerEncoder(layer,
                                                     num_layers=layers)
        else:
            self.transformer = nn.Identity()

        # 4) Projection → LayerNorm → Dropout
        proj_in       = short_units if use_short_lstm else n_feats
        self.short2long = nn.Linear(proj_in, long_units)
        self.ln_proj    = nn.LayerNorm(long_units)
        self.do_proj    = nn.Dropout(dropout_long)

        # 5) Long BiLSTM or identity
        if use_long_lstm:
            assert long_units % 2 == 0
            self.long_lstm = nn.LSTM(input_size=long_units,
                                     hidden_size=long_units // 2,
                                     batch_first=True,
                                     bidirectional=True)
            self.ln_long = nn.LayerNorm(long_units)
            self.do_long = nn.Dropout(dropout_long)
        else:
            self.long_lstm = None
            self.ln_long   = nn.Identity()
            self.do_long   = nn.Identity()

        # buffers for long‐LSTM states
        self.h_long = None
        self.c_long = None

        # 6) Flatten + gated MLP head
        assert flatten_mode in ("flatten","last","pool")
        self.flatten_mode = flatten_mode
         # compute head input dim for full flatten only
        if flatten_mode == "flatten":
            flat_dim = window_len * long_units
        else:
            flat_dim = long_units
        self.ln_flat = nn.LayerNorm(flat_dim)
        self.head_flat = nn.Sequential(
            weight_norm(nn.Linear(flat_dim,    pred_hidden)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Linear(pred_hidden, 1))
        )
        self.skip_proj  = weight_norm(nn.Linear(flat_dim, 1))
        init_alpha      = params.hparams["SKIP_ALPHA"]
        self.skip_alpha = nn.Parameter(
            torch.tensor(init_alpha, dtype=torch.float32)
        )

    def reset_short(self):
        """
        Called by _reset_states(): zeroes out short‐LSTM buffers at day change.
        """
        if self.h_short is not None:
            batch_sz        = self.h_short.size(1)
            device          = self.h_short.device
            self.h_short, self.c_short = _allocate_lstm_states(
                batch_sz, self.short_units // 2, True, device
            )

    def reset_long(self):
        """
        Called by _reset_states(): zeroes out long‐LSTM buffers at week wrap.
        """
        if self.h_long is not None:
            batch_sz       = self.h_long.size(1)
            device         = self.h_long.device
            self.h_long, self.c_long = _allocate_lstm_states(
                batch_sz, self.long_units // 2, True, device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., time_steps, feature_dim) or (time_steps, feature_dim)
        returns (batch_size, 1, 1)
        """
        # collapse to (batch_size, time_steps, feature_dim)
        if x.dim() > 3:
            *lead, time_steps, feature_dim = x.shape
            x = x.view(-1, time_steps, feature_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, time_steps, feature_dim = x.shape

        # 0) Conv1d + BN + ReLU
        xc = x.transpose(1, 2)  
        xc = self.conv(xc); xc = self.bn(xc)
        xc = F.relu(xc)            
        x  = xc.transpose(1, 2)    

        # 1) TCN
        x  = self.tcn(x.transpose(1,2)).transpose(1,2)

        # 2) Short LSTM
        if self.use_short_lstm:
            if (self.h_short is None
                or self.h_short.size(1) != batch_size):
                self.h_short, self.c_short = _allocate_lstm_states(
                    batch_size, self.short_units // 2, True, x.device
                )
            out_s, (h_s, c_s) = self.short_lstm(
                x, (self.h_short, self.c_short))
            self.h_short, self.c_short = h_s.detach(), c_s.detach()
            out_s = self.ln_short(out_s); out_s = self.do_short(out_s)
        else:
            out_s = x

        # 3) Transformer
        if self.use_transformer:
            tr_in  = out_s.transpose(0, 1)
            tr_out = self.transformer(tr_in)
            out_s  = tr_out.transpose(0, 1)

        # 4) Projection → LN → Dropout
        out_p = self.short2long(out_s)
        out_p = self.ln_proj(out_p); out_p = self.do_proj(out_p)

        # 5) Long LSTM
        if self.use_long_lstm:
            if (self.h_long is None
                or self.h_long.size(1) != batch_size):
                self.h_long, self.c_long = _allocate_lstm_states(
                    batch_size, self.long_units // 2, True, out_p.device
                )
            out_l, (h_l, c_l) = self.long_lstm(
                out_p, (self.h_long, self.c_long))
            self.h_long, self.c_long = h_l.detach(), c_l.detach()
            out_l = self.ln_long(out_l); out_l = self.do_long(out_l)
        else:
            out_l = out_p

        # 6) Collapse time → (batch_size, flat_dim)
        assert time_steps == self.window_len

        if self.flatten_mode == "flatten":
            flat = out_l.reshape(batch_size, -1)
        elif self.flatten_mode == "last":
            flat = out_l[:, -1, :]               # (B, long_units)
        else:  # "pool"
            flat = out_l.mean(dim=1)             # (B, long_units)

        norm_flat = self.ln_flat(flat)
        main_out  = self.head_flat(norm_flat)
        gate      = torch.sigmoid(self.skip_alpha)
        skip_out  = self.skip_proj(norm_flat) * gate
        output    = main_out + skip_out

        return output.unsqueeze(-1)  # (batch_size, 1, 1)




######################################################################################################

class SmoothMSELoss(nn.Module):
    """
    Combines pointwise MSE with a penalty on one-step differences to enforce
    that your model matches both the level and the slope of the target.

    Args:
      alpha (float): Weight of the slope-matching term.  
        If alpha <= 0, this loss reduces to standard MSELoss.

    Attributes:
      level_loss (nn.MSELoss): standard pointwise MSE.  
      alpha      (float)    : slope‐penalty multiplier.
    """
    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.level_loss = nn.MSELoss()
        self.alpha      = alpha

    def forward(self, preds: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Inputs:
          preds, targs : tensors of shape (B,) or (B, W)
            – B if you’re computing on final‐window predictions
            – (B, W) if you pass full sequences of window‐end preds

        Workflow:
          1. L1 = MSE(preds, targs)
          2. If alpha <= 0, return L1 (pure MSE).
          3. Otherwise, compute one‐step diffs along dim=–1:
             dp = preds[...,1:] – preds[..., :-1]
             dt =  targs[...,1:] –  targs[..., :-1]
          4. L2 = MSE(dp, dt)
          5. return L1 + alpha * L2

        Returns:
          loss (torch.Tensor): scalar combining level + slope penalties.
        """
        # 1) pointwise MSE
        L1 = self.level_loss(preds, targs)

        # 2) if no smoothing penalty, exit early
        if self.alpha <= 0:
            return L1

        # 3) compute one-step differences
        if preds.dim() == 1:
            dp = preds[1:]    - preds[:-1]
            dt = targs[1:]    - targs[:-1]
        else:
            dp = preds[:, 1:] - preds[:, :-1]
            dt = targs[:, 1:] - targs[:, :-1]

        # 4) slope‐matching MSE
        L2 = self.level_loss(dp, dt)

        # 5) combined loss
        return L1 + self.alpha * L2

        
######################################################################################################

def compute_baselines(loader) -> tuple[float, float]:
    """
    Compute two static, one‐step‐per‐window baselines over all windows in `loader`:
      1) mean_rmse       – RMSE if you always predict μ = average y_sig per window
      2) persistence_rmse – RMSE if you always predict last‐seen (yₜ = yₜ₋₁)
    """
    nexts, last_deltas = [], []
    
    for x_pad, y_sig, _y_bin, _y_ret, _y_ter, _rc, wd, ts_list, lengths in loader:
        for i, L in enumerate(lengths):
            if L < 1:
                continue
            arr = y_sig[i, :L].view(-1).cpu().numpy()
            nexts.append(arr[-1])
            if L > 1:
                last_deltas.append(arr[-1] - arr[-2])
    
    nexts = np.array(nexts, dtype=float)
    mean_rmse = float(np.sqrt(((nexts - nexts.mean()) ** 2).mean()))
    
    if last_deltas:
        last_deltas = np.array(last_deltas, dtype=float)
        persistence_rmse = float(np.sqrt((last_deltas ** 2).mean()))
    else:
        persistence_rmse = float("nan")
    
    # print(f"###### BASELINE CHECK on y_sig: nexts mean±std: "f"{nexts.mean():.5f} ± {nexts.std():.5f}") ################################
    
    return mean_rmse, persistence_rmse

###############

def _compute_metrics(preds: np.ndarray, targs: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, and R² on flat NumPy arrays of predictions vs. targets.
    """
    if preds.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    diff = preds - targs
    mse  = float((diff ** 2).mean())
    mae  = float(np.abs(diff).mean())
    rmse = float(np.sqrt(mse))

    # R² = 1 − RSS/TSS
    if preds.size < 2 or np.isclose(targs.var(), 0.0):
        r2 = float("nan")
    else:
        tss = float(((targs - targs.mean()) ** 2).sum())
        rss = float((diff ** 2).sum())
        r2  = float("nan") if tss == 0.0 else 1.0 - (rss / tss)

    return {"rmse": rmse, "mae": mae, "r2": r2}

############### 

def _reset_states(
    model:    nn.Module,
    wd_i:     torch.Tensor,
    prev_day: int | None
) -> int:
    """
    Reset short-term on any day change, and long-term weekly, ie when the day index
    wraps around (i.e. new_day < prev_day).

    Args:
      model     : ModelClass instance (implements reset_short/reset_long)
      wd_i      : scalar tensor with day-of-week (0=Mon,…,6=Sun)
      prev_day  : last seen day-of-week or None

    Returns:
      the new day-of-week for next call
    """
    day = int(wd_i.item())

    # daily reset if the day changed
    if prev_day is None or day != prev_day:
        model.reset_short()

        # weekly reset if we've wrapped past the end of the week
        if prev_day is not None and day < prev_day:
            model.reset_long()

    return day


############### 

    # def eval_on_loader(loader, model: nn.Module, clamp_preds: bool = True) -> tuple[dict,np.ndarray]:
    #     """
    #     One‐step‐per‐window evaluation.
    
    #     For each sliding window:
    #       1. Reset LSTM short‐term state on day rollover.
    #       2. Skip zero‐length windows.
    #       3. Run model(x_windows) → raw_out.
    #       4. Unpack & head‐apply → raw_reg of shape (W, look_back, 1).
    #       5. Squeeze → (W, look_back).
    #       6. Take preds = raw_reg[:, -1]; targs = y_sig[i, :W].
    #       7. Accumulate all preds & targs.
    
    #     Returns:
    #       metrics: dict with keys "rmse", "mae", "r2" computed once over
    #                the flat arrays of all predictions vs. all targets.
    #     """
    #     device = next(model.parameters()).device
    #     model.to(device).eval()
    #     model.h_short = model.h_long = None
    #     prev_day = None
    
    #     all_preds, all_targs = [], []
    #     with torch.no_grad():
    #         for x_pad, y_sig, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths in tqdm(loader, desc="eval", leave=False):
    
    #             x_pad = x_pad.to(device)
    #             y_sig = y_sig.to(device)
    #             wd    = wd.to(device)
    
    #             B = x_pad.size(0)
    #             for i in range(B):
    #                 prev_day = _reset_states(model, wd[i], prev_day)
    #                 W = int(lengths[i])
    #                 if W == 0:
    #                     continue
    
    #                 seqs    = x_pad[i, :W]                              # (W, look_back, feats)
    #                 raw_out = model(seqs)
    #                 raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
    
    #                 # force through regression head if needed
    #                 if raw_reg.dim() == 3 and raw_reg.size(-1) != 1:
    #                     raw_reg = model.pred(raw_reg)
    #                 elif raw_reg.dim() == 2:
    #                     raw_reg = model.pred(raw_reg.unsqueeze(0)).squeeze(0)
    
    #                 raw_reg   = raw_reg.squeeze(-1)                  # → (W, look_back)
    #                 preds_win = raw_reg[:, -1]                       # (W,)
    #                 targs_win = y_sig[i, :W].view(-1)                # (W,)
    
    #                 all_preds.extend(preds_win.cpu().tolist())
    #                 all_targs.extend(targs_win.cpu().tolist())
    
    #     preds = np.array(all_preds, dtype=float)
    #     targs = np.array(all_targs, dtype=float)
        
    #     # if clamp_preds and preds.size:
    #     #     preds = np.clip(preds, 0.0, 1.0)
    
    #     # attach Torch tensors to model for logger
    #     model.last_val_preds  = torch.from_numpy(preds).float()
    #     model.last_val_targs  = torch.from_numpy(targs).float()
    
    #     return _compute_metrics(preds, targs), preds
    

def eval_on_loader(loader, model: nn.Module, clamp_preds: bool = True) -> tuple[dict, np.ndarray]:
    """
    One-step-per-window evaluation.

    For each sliding window in the loader:
      1. Reset short-term state on day rollover.
      2. Skip zero-length windows.
      3. Forward the sequence through the model.
      4. Unwrap and apply any extra regression head.
      5. Squeeze to (seq_len, window_len).
      6. Take the final time-step prediction and true target.
      7. Accumulate all preds & targs into flat arrays.

    Returns:
      metrics: dict with global "rmse", "mae", "r2".
      preds:   np.ndarray of all predictions.
    """
    device = next(model.parameters()).device
    model.to(device).eval()
    model.h_short = model.h_long = None
    prev_day = None

    all_preds, all_targs = [], []
    with torch.no_grad():
        for x_batch, y_signal, y_bin, y_ret, y_ter, rc, wd, ts_list, seq_lengths in \
                tqdm(loader, desc="eval", leave=False):

            x_batch = x_batch.to(device)
            y_signal = y_signal.to(device)
            wd = wd.to(device)

            batch_size = x_batch.size(0)
            for example_idx in range(batch_size):
                prev_day = _reset_states(model, wd[example_idx], prev_day)

                seq_len = int(seq_lengths[example_idx])
                if seq_len == 0:
                    continue

                # (seq_len, window_len, feature_dim)
                daily_windows = x_batch[example_idx, :seq_len]
                raw_out = model(daily_windows)
                raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

                # ensure regression head if needed
                if raw_reg.dim() == 3 and raw_reg.size(-1) != 1:
                    raw_reg = model.pred(raw_reg)
                elif raw_reg.dim() == 2:
                    raw_reg = model.pred(raw_reg.unsqueeze(0)).squeeze(0)

                # (seq_len, window_len)
                seq_reg = raw_reg.squeeze(-1)
                # final look-back prediction
                preds_seq = seq_reg[:, -1]
                # true sig target flattened
                targs_seq = y_signal[example_idx, :seq_len].reshape(-1)

                all_preds.extend(preds_seq.cpu().tolist())
                all_targs.extend(targs_seq.cpu().tolist())

    preds = np.array(all_preds, dtype=float)
    targs = np.array(all_targs, dtype=float)

    model.last_val_preds = torch.from_numpy(preds).float()
    model.last_val_targs = torch.from_numpy(targs).float()

    return _compute_metrics(preds, targs), preds


    
    ############### 
    
    
    # def model_training_loop(
    #     model:           nn.Module,
    #     optimizer:       torch.optim.Optimizer,
    #     scheduler:       torch.optim.lr_scheduler.OneCycleLR,
    #     scaler:          GradScaler,
    #     train_loader, val_loader,
    #     *, max_epochs:         int,
    #        early_stop_patience: int,
    #        clipnorm:           float,
    #        alpha_smooth:       int,
    # ) -> float:
    #     """
    #     Stateful training + per‐epoch validation + baseline logging,
    #     with linear warmup followed by cosine restarts, and console printing
    #     of TRAIN/VALID RMSE, R², and current LR each epoch.
    
    #     1. Precompute four static baselines via compute_baselines().
    #     2. For each epoch:
    #        • Linear warmup of LR over the first LR_EPOCHS_WARMUP epochs.
    #        • After warmup, step the CosineAnnealingWarmRestarts scheduler.
    #        • TRAIN loop: iterate days, reset day‐state, batch all windows,
    #          accumulate per‐window MSELoss and collect preds/targs.
    #        • Compute train_metrics via _compute_metrics.
    #        • Compute val_metrics via eval_on_loader.
    #        • Call log_epoch_summary with all metrics and baselines.
    #        • Print a one‐line summary: TRAIN RMSE/R², VALID RMSE/R², LR.
    #        • Checkpoint, update early‐stop, and optionally break.
    #     Returns:
    #       best_val_rmse: float
    #     """
    #     device = next(model.parameters()).device
    #     model.to(device)
    
    #     # 1) Precompute static baselines
    #     base_tr_mean, base_tr_pers = compute_baselines(train_loader)
    #     base_vl_mean, base_vl_pers = compute_baselines(val_loader)
    
    #     smooth_loss = SmoothMSELoss(alpha_smooth) 
    #     live_plot = plots.LiveRMSEPlot()
    #     best_val, best_state, patience = float("inf"), None, 0
    
    #     for epoch in range(1, max_epochs + 1):
    
    #         # --- TRAIN PHASE ---
    #         model.train()
    #         # model.h_short = model.h_long = None
    #         train_preds, train_targs = [], []
    #         prev_day = None
    
    #         for x_pad, y_sig, y_bin, y_ret, y_ter, rc, wd, ts_list, lengths in tqdm(     
    #             train_loader, desc=f"Epoch {epoch} ▶ Train", leave=False
    #         ):
    #             x_pad, y_sig, wd = x_pad.to(device), y_sig.to(device), wd.to(device)
    #             optimizer.zero_grad(set_to_none=True)
    
    #             batch_loss, windows = 0.0, 0
    #             B = x_pad.size(0)
    
    #             # Iterate each day in the batch
    #             for i in range(B):
    #                 prev_day = _reset_states(model, wd[i], prev_day)
    #                 W = int(lengths[i])
    #                 if W == 0:
    #                     continue
    
    #                 # Forward all windows of day i
    #                 x_win   = x_pad[i, :W]                        
    #                 raw_out = model(x_win)
    #                 raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
    
    #                 # Ensure regression head applied
    #                 if raw_reg.dim() == 3 and raw_reg.size(-1) != 1:
    #                     raw_reg = model.pred(raw_reg)
    #                 elif raw_reg.dim() == 2:
    #                     raw_reg = model.pred(raw_reg.unsqueeze(0)).squeeze(0)
    
    #                 seq_preds = raw_reg.squeeze(-1)      # (W,)
    #                 preds_win = seq_preds[:, -1]         # final window-end preds
    #                 targs_win = y_sig[i, :W].view(-1)    # true window-end targets
    
    #                 # Accumulate loss & collect preds/targs 
    #                 loss = smooth_loss(preds_win, targs_win)
    #                 batch_loss += loss
    #                 windows   += 1
    
    #                 train_preds.extend(preds_win.detach().cpu().tolist())
    #                 train_targs.extend(targs_win.detach().cpu().tolist())
    
    #             # Backprop once per batch
    #             batch_loss = batch_loss / max(1, windows)
    #             scaler.scale(batch_loss).backward()
    #             scaler.unscale_(optimizer)
    #             nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
    #             scaler.step(optimizer)
    #             scaler.update()
    #             scheduler.step()
    
    #         # --- METRICS & VALIDATION ---
    #         train_metrics = _compute_metrics(
    #             np.array(train_preds, dtype=float),
    #             np.array(train_targs, dtype=float),
    #         )
    #         val_metrics, _ = eval_on_loader(val_loader, model)
    
    #         # --- LOG & CHECKPOINT ---
    #         tr_rmse, tr_mae, tr_r2 = (
    #             train_metrics["rmse"],
    #             train_metrics["mae"],
    #             train_metrics["r2"],
    #         )
    #         vl_rmse, vl_mae, vl_r2 = (
    #             val_metrics["rmse"],
    #             val_metrics["mae"],
    #             val_metrics["r2"],
    #         )
    #         lr = optimizer.param_groups[0]["lr"]
    
    #         # Append to log file and update live plot
    #         models_core.log_epoch_summary(
    #             epoch,
    #             model,
    #             optimizer,
    #             train_metrics   = train_metrics,
    #             val_metrics     = val_metrics,
    #             base_tr_mean    = base_tr_mean,
    #             base_tr_pers    = base_tr_pers,
    #             base_vl_mean    = base_vl_mean,
    #             base_vl_pers    = base_vl_pers,
    #             slip_thresh     = 1e-6,
    #             log_file        = params.log_file,
    #             top_k           = 999,
    #             hparams         = params.hparams,
    #         )
    #         live_plot.update(tr_rmse, vl_rmse)
    
    #         # Console summary: TRAIN/VALID RMSE, R², and LR
    #         print(
    #             f"Epoch {epoch:02d}  "
    #             f"TRAIN → RMSE={tr_rmse:.5f}, R²={tr_r2:.3f} |  "
    #             f"VALID → RMSE={vl_rmse:.5f}, R²={vl_r2:.3f} |  "
    #             f"lr={lr:.2e}"
    #         )
    
    #         # Checkpointing & early stopping
    #         models_dir = Path(params.models_folder)
    #         best_val, improved, *_ = models_core.maybe_save_chkpt(
    #             models_dir, model, vl_rmse, best_val,
    #             {"rmse": tr_rmse}, {"rmse": vl_rmse},
    #             live_plot, params
    #         )
    
    #         if improved:
    #             best_state, patience = {k: v.cpu() for k, v in model.state_dict().items()}, 0
    #         else:
    #             patience += 1
    #             if patience >= early_stop_patience:
    #                 print(f"Early stopping at epoch {epoch}")
    #                 break
    
    #     # --- FINAL CHECKPOINT RESTORE ---
    #     if best_state is not None:
    #         model.load_state_dict(best_state)
    #         models_core.save_final_chkpt(
    #             models_dir, best_state, best_val, params,
    #             {}, {}, live_plot, suffix="_fin"
    #         )
    
    #     return best_val


def model_training_loop(
    model:                nn.Module,
    optimizer:            torch.optim.Optimizer,
    scheduler:            torch.optim.lr_scheduler.OneCycleLR,
    scaler:               torch.cuda.amp.GradScaler,
    train_loader,
    val_loader,
    *,
    max_epochs:          int,
    early_stop_patience: int,
    clipnorm:            float,
    alpha_smooth:        int,
) -> float:
    """
    Stateful training + per-epoch validation + logging.
    Uses OneCycleLR for linear warmup + cosine annealing.

    1. Precompute static baselines.
    2. For each epoch:
       • TRAIN: reset per-day state, forward each day's windows,
         accumulate SmoothMSELoss, collect preds/targs, step optimizer.
       • VALID: call `eval_on_loader` to get metrics.
       • Log, print summary, checkpoint, and early-stop.
    Returns:
      best validation RMSE across all epochs.
    """
    device = next(model.parameters()).device
    model.to(device)

    base_tr_mean, base_tr_pers = compute_baselines(train_loader)
    base_vl_mean, base_vl_pers = compute_baselines(val_loader)

    smooth_loss = SmoothMSELoss(alpha_smooth)
    live_plot  = plots.LiveRMSEPlot()
    best_val, best_state, patience = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        # --- TRAIN PHASE ---
        model.train()
        model.h_short = model.h_long = None
        train_preds, train_targs = [], []
        prev_day = None

        for x_batch, y_signal, y_bin, y_ret, y_ter, rc, wd, ts_list, seq_lengths in \
                tqdm(train_loader, desc=f"Epoch {epoch} ▶ Train", leave=False):

            x_batch, y_signal, wd = (
                x_batch.to(device), y_signal.to(device), wd.to(device)
            )
            optimizer.zero_grad(set_to_none=True)

            batch_loss = 0.0
            window_count = 0
            batch_size = x_batch.size(0)

            for example_idx in range(batch_size):
                prev_day = _reset_states(model, wd[example_idx], prev_day)

                seq_len = int(seq_lengths[example_idx])
                if seq_len == 0:
                    continue

                daily_windows = x_batch[example_idx, :seq_len]
                raw_out = model(daily_windows)
                raw_reg = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

                if raw_reg.dim() == 3 and raw_reg.size(-1) != 1:
                    raw_reg = model.pred(raw_reg)
                elif raw_reg.dim() == 2:
                    raw_reg = model.pred(raw_reg.unsqueeze(0)).squeeze(0)

                seq_reg    = raw_reg.squeeze(-1)
                preds_seq  = seq_reg[:, -1]
                targs_seq  = y_signal[example_idx, :seq_len].reshape(-1)

                loss = smooth_loss(preds_seq, targs_seq)
                batch_loss += loss
                window_count += 1

                train_preds.extend(preds_seq.detach().cpu().tolist())
                train_targs.extend(targs_seq.detach().cpu().tolist())

            if window_count > 0:
                batch_loss = batch_loss / window_count
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        # --- METRICS & VALIDATION ---
        tr_metrics = _compute_metrics(
            np.array(train_preds, dtype=float),
            np.array(train_targs, dtype=float),
        )
        vl_metrics, _ = eval_on_loader(val_loader, model)

        # --- LOG & CHECKPOINT ---
        tr_rmse, tr_mae, tr_r2 = tr_metrics["rmse"], tr_metrics["mae"], tr_metrics["r2"]
        vl_rmse, vl_mae, vl_r2 = vl_metrics["rmse"], vl_metrics["mae"], vl_metrics["r2"]
        current_lr = optimizer.param_groups[0]["lr"]

        models_core.log_epoch_summary(
            epoch,
            model,
            optimizer,
            train_metrics   = tr_metrics,
            val_metrics     = vl_metrics,
            base_tr_mean    = base_tr_mean,
            base_tr_pers    = base_tr_pers,
            base_vl_mean    = base_vl_mean,
            base_vl_pers    = base_vl_pers,
            slip_thresh     = 1e-6,
            log_file        = params.log_file,
            top_k           = 999,
            hparams         = params.hparams,
        )
        live_plot.update(tr_rmse, vl_rmse)

        print(
            f"Epoch {epoch:02d}  "
            f"TRAIN→ RMSE={tr_rmse:.5f}, R²={tr_r2:.3f} |  "
            f"VALID→ RMSE={vl_rmse:.5f}, R²={vl_r2:.3f} |  "
            f"lr={current_lr:.2e}"
        )

        models_dir = Path(params.models_folder)
        best_val, improved, *_ = models_core.maybe_save_chkpt(
            models_dir, model, vl_rmse, best_val,
            {"rmse": tr_rmse}, {"rmse": vl_rmse},
            live_plot, params
        )
        if improved:
            best_state, patience = {
                k: v.cpu() for k, v in model.state_dict().items()
            }, 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        models_core.save_final_chkpt(
            models_dir, best_state, best_val, params,
            {}, {}, live_plot, suffix="_fin"
        )

    return best_val
