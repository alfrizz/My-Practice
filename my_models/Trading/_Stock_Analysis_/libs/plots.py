from libs import params, trades

from typing import Sequence, List, Tuple, Optional, Union, Dict

import pandas as pd
import numpy  as np
import datetime

import gc
import re
import textwrap
import math
import os
import json
import glob

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import plotly.graph_objects as go
from IPython.display import display, update_display, HTML
import seaborn as sns
sns.set_style("white")

from tqdm.auto import tqdm
import optuna
from optuna.trial import TrialState
import torch

###############################################################################

def plot_close_volume(df, title="Close Price and Volume"):
    """
    Quickly plots the 'close' and 'volume' columns from the DataFrame
    using a secondary y-axis for volume.
    """
    ax = df[['close', 'volume']].plot(secondary_y=['volume'], figsize=(10, 5), title=title, alpha=0.7)
    ax.set_xlabel("Date")
    plt.show()
    
#################################################################################

class LiveRMSEPlot:
    """
    LiveRMSEPlot updates a single figure to show training progress without spawning
    a new image for each epoch. It works with different matplotlib backends, e.g.,
    %matplotlib inline, widget, or notebook.

    The plot displays:
      - Blue line and dot: training RMSE history.
      - Orange line and dot: validation RMSE history.

    If the latest validation RMSE is not a number (NaN), the corresponding dot is
    hidden by setting its offsets to an empty 2D array.
    """

    def __init__(self):
        # Retrieve the current matplotlib backend and convert it to lowercase.
        self.backend = matplotlib.get_backend().lower()
        # Build the figure and axes.
        self._build_figure()
        # Display the figure once and keep a reference to the display_id so that we can
        # update the same output cell on subsequent calls instead of spawning a new figure.
        self.disp_id = display(self.fig, display_id=True)
        # Initialize empty lists to store epoch numbers and RMSE metrics.
        self.e, self.tr, self.va = [], [], []      # e = epochs, tr = train RMSE, va = validation RMSE

    # ------------------------------------------------------------------ #
    def _build_figure(self):
        """
        Constructs and configures the matplotlib figure and axes.
        - Creates empty line plots for training (blue) and validation (orange).
        - Creates scatter plot objects (dots) for the latest RMSE values.
        - Sets up grid, labels, title, and legend.
        """
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=110)
        self.ax.set(xlabel="epoch", ylabel="RMSE", title="Training progress")
        self.ax.grid(True)
        
        # Create a blue line for training RMSE.
        (self.tr_line,) = self.ax.plot([], [], c="#1f77b4", lw=1.5)
        # Create an orange line for validation RMSE.
        (self.va_line,) = self.ax.plot([], [], c="#ff7f0e", lw=1.5)
        # Create scatter objects for the latest training and validation points.
        self.tr_dot = self.ax.scatter([], [], c="#1f77b4", s=30)
        self.va_dot = self.ax.scatter([], [], c="#ff7f0e", s=30)
        
        # Add a legend to differentiate between training and validation RMSE.
        self.ax.legend(["train", "val"])

    # ------------------------------------------------------------------ #
    def update(self, train_rmse: float, val_rmse: float):
        """
        Updates the live plot with new training and validation RMSE values.

        Steps:
         1. Append the new epoch and metric values.
         2. Update the line plots with the full RMSE history.
         3. Update the latest dot position for both training and validation.
            - If the validation RMSE is NaN, hide its dot by setting an empty 2D array.
         4. Recalculate and update axis limits.
         5. Redraw the figure using the appropriate method for the backend.
        """
        # 1. Append new data:
        #    - Epochs are automatically numbered starting from 1.
        self.e.append(len(self.e) + 1)
        self.tr.append(train_rmse)
        self.va.append(val_rmse)
        
        # 2. Update line plots:
        #    - For the training line, simply use all available data.
        self.tr_line.set_data(self.e, self.tr)
        
        #    - For the validation line, filter out non-finite values (e.g., NaN).
        finite = np.isfinite(self.va)
        self.va_line.set_data(np.asarray(self.e)[finite],
                              np.asarray(self.va)[finite])
        
        # 3. Update the latest dots:
        #    - Always update the training dot with the most recent training RMSE.
        self.tr_dot.set_offsets([[self.e[-1], self.tr[-1]]])
        
        #    - For the validation dot, only update if the latest value is finite.
        if np.isfinite(self.va[-1]):
            self.va_dot.set_offsets([[self.e[-1], self.va[-1]]])
        else:
            # Instead of an empty list, we pass an empty 2D NumPy array with shape (0,2)
            # to properly hide the dot when the validation RMSE is NaN.
            self.va_dot.set_offsets(np.empty((0, 2)))
        
        # 4. Rescale the axes:
        #    - This ensures all data is visible in the plot.
        self.ax.relim()
        self.ax.autoscale_view()
        
        # 5. Redraw the figure:
        #    - For widget backends, use draw_idle to schedule a redraw.
        #    - For inline / notebook backends, force a redraw and update the output cell.
        if "widget" in self.backend or "ipympl" in self.backend:
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw()
            self.disp_id.update(self.fig)


#########################################################################################################


class LiveFeatGuBars:
    """
    LiveFeatGuBars

    Two side-by-side horizontal bar plots: feature importance (left) and grad norms (right).
    - Feature/layer names and optional in-bar labels unchanged.
    - Rank numbers 1..N with the bar value in parentheses (2 decimals) are printed outside
      each subplot on the left margin (axes fraction x = -0.02), right-aligned so they do
      not overlap the bars.
    - Removes internal y-axis numeric ticks so only the outside rank/value text is visible.
    - Usage: instantiate and call update(feat_dict, g_dict, epoch).
    """

    def __init__(self, top_feats=30, top_params=30, figsize=(14, 10), dpi=110, max_display=80):
        self.top_feats = top_feats
        self.top_params = top_params
        self.dpi = dpi
        self.base_figsize = figsize
        self.max_display = max_display

        self.fig = None
        self.ax_feat = None
        self.ax_param = None
        self.disp_id = None
        self.eps = []

    def _ellipsize(self, s: str, max_chars: int) -> str:
        if len(s) <= max_chars:
            return s
        if max_chars <= 3:
            return s[:max_chars]
        return s[: max_chars - 3] + "..."

    def _init_fig(self, width: float, height: float, left: float, wspace: float = 0.44):
        self.fig, (self.ax_feat, self.ax_param) = plt.subplots(
            1, 2, figsize=(width, height), dpi=self.dpi, constrained_layout=False
        )
        self.fig.subplots_adjust(left=left, right=0.98, wspace=wspace)
        for ax in (self.ax_feat, self.ax_param):
            ax.cla()
            # ensure no internal numeric y-ticks or labels
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.45)
        self.ax_feat.set_xlabel("score")
        self.ax_param.set_xlabel("g norm")

    def update(self, feat_dict: dict, g_dict: dict, epoch: int):
        s_feat = pd.Series(feat_dict).fillna(0.0)
        s_g = pd.Series(g_dict).fillna(0.0)

        feat_cols = s_feat.sort_values(ascending=False).index[: self.top_feats].tolist()[: self.max_display]
        g_cols = s_g.sort_values(ascending=False).index[: self.top_params].tolist()[: self.max_display]

        labels = [str(x) for x in feat_cols + g_cols]
        max_label_len = int(max((len(l) for l in labels), default=0))

        # reserve left area based on longest label
        char_inch = 0.11
        label_area_inch = min(max_label_len * char_inch, 28.0)

        base_width = max(self.base_figsize[0], 10.0)
        width = min(base_width + label_area_inch, 80.0)

        n_rows = max(len(feat_cols), len(g_cols), 1)
        per_row = 0.36
        extra_h = 1.2
        height = min(max(5.0, n_rows * per_row + extra_h), 48.0)

        left_margin = max(0.18, min(0.50, label_area_inch / width))

        should_init = (
            self.fig is None
            or (abs(self.fig.get_size_inches()[0] - width) > 0.1)
            or (abs(self.fig.get_size_inches()[1] - height) > 0.1)
        )
        if should_init:
            self._init_fig(width=width, height=height, left=left_margin, wspace=0.44)

        self.eps.append(epoch)

        # fontsize estimation
        avail_label_inch = left_margin * width * 0.88
        if max_label_len <= 0:
            fontsize = 11
        else:
            max_font_by_width = int((avail_label_inch * 72) / (max_label_len * 0.6))
            fontsize = max(9, min(16, max_font_by_width))

        def bar_height(rows):
            if rows <= 8:
                return 1.0
            if rows <= 20:
                return 0.78
            if rows <= 40:
                return 0.62
            return 0.50

        bh_feat = bar_height(len(feat_cols))
        bh_param = bar_height(len(g_cols))

        left_color = "#66C2FF"   # light blue
        right_color = "#FF8C42"  # orange
        edge_color = "#FFFFFF"
        edge_lw = 0.6

        max_chars_fit = int((avail_label_inch * 72) / (fontsize * 0.6)) if fontsize > 0 else max_label_len
        max_chars_fit = max(6, max_chars_fit)

        feat_labels = [self._ellipsize(str(l), max_chars_fit) for l in feat_cols]
        param_labels = [self._ellipsize(str(l), max_chars_fit) for l in g_cols]

        # transforms: we'll place left-side rank/value using Axes fraction coordinates
        # with x_out < 0 so text is outside left; use ha='right' to align against axis edge
        left_x_out = -0.02  # axes fraction (negative -> outside left)
        rank_font = max(9, fontsize - 1)

        # LEFT panel
        self.ax_feat.cla()
        # ensure no internal numeric y-ticks or labels (re-assert after cla)
        self.ax_feat.set_yticks([])
        self.ax_feat.set_yticklabels([])
        self.ax_feat.tick_params(axis='y', which='both', left=False, labelleft=False)

        if feat_cols:
            vals = s_feat[feat_cols].values
            y = np.arange(len(feat_cols))[::-1]
            bars = self.ax_feat.barh(
                y, vals, color=left_color, align="center",
                height=bh_feat, edgecolor=edge_color, linewidth=edge_lw, zorder=2
            )
            # no numeric ticks
            self.ax_feat.set_yticks([])
            self.ax_feat.set_yticklabels([])
            self.ax_feat.tick_params(axis='y', which='both', left=False, labelleft=False)

            vmax = max(vals.max(), 1e-6)
            widths = np.array([rect.get_width() for rect in bars])
            wide_enough = widths >= (0.14 * vmax)

            # --- draw rank+value outside the axis on the left ---
            trans_out_feat = blended_transform_factory(self.ax_feat.transAxes, self.ax_feat.transData)
            for idx_zero, val in enumerate(vals):
                idx = idx_zero + 1  # rank ascending: 1 = top
                yv = y[idx_zero]
                txt = f"{idx} ({val:.2e})"
                self.ax_feat.text(
                    left_x_out, yv, txt,
                    transform=trans_out_feat, va="center", ha="right",
                    fontsize=rank_font, color="gray", zorder=6, clip_on=False
                )

            # draw labels as before (in-bar or left-fixed just inside the axes)
            for idx, (lbl, yv) in enumerate(zip(feat_labels, y)):
                if wide_enough[idx]:
                    x_text = widths[idx] * 0.02
                    self.ax_feat.text(
                        x_text, yv, lbl, va="center", ha="left",
                        fontsize=fontsize, color="black", zorder=3, clip_on=True
                    )
                else:
                    # left-fixed label inside axis area a bit to the right of the axis left edge
                    # use axes-fraction->data transform so it's always aligned
                    trans_inside = blended_transform_factory(self.ax_feat.transAxes, self.ax_feat.transData)
                    self.ax_feat.text(
                        0.01, yv, lbl,
                        transform=trans_inside, va="center", ha="left",
                        fontsize=fontsize, color="black", zorder=4, clip_on=False
                    )
            self.ax_feat.set_xlim(left=0)
        else:
            self.ax_feat.set_yticks([])

        self.ax_feat.set_title(f"FEAT_TOP (epoch {epoch})", fontsize=max(10, fontsize + 1))

        # RIGHT panel
        self.ax_param.cla()
        # re-assert no numeric y-ticks or labels
        self.ax_param.set_yticks([])
        self.ax_param.set_yticklabels([])
        self.ax_param.tick_params(axis='y', which='both', left=False, labelleft=False)

        if g_cols:
            vals_p = s_g[g_cols].values
            y = np.arange(len(g_cols))[::-1]
            bars_p = self.ax_param.barh(
                y, vals_p, color=right_color, align="center",
                height=bh_param, edgecolor=edge_color, linewidth=edge_lw, zorder=2
            )
            # ensure no internal numeric ticks
            self.ax_param.set_yticks([])
            self.ax_param.set_yticklabels([])
            self.ax_param.tick_params(axis='y', which='both', left=False, labelleft=False)

            vmax_p = max(vals_p.max(), 1e-6)
            widths_p = np.array([rect.get_width() for rect in bars_p])
            wide_enough_p = widths_p >= (0.14 * vmax_p)

            trans_out_param = blended_transform_factory(self.ax_param.transAxes, self.ax_param.transData)
            for idx_zero, val in enumerate(vals_p):
                idx = idx_zero + 1
                yv = y[idx_zero]
                txt = f"{idx} ({val:.2e})"
                self.ax_param.text(
                    left_x_out, yv, txt,
                    transform=trans_out_param, va="center", ha="right",
                    fontsize=rank_font, color="gray", zorder=6, clip_on=False
                )

            for idx, (lbl, yv) in enumerate(zip(param_labels, y)):
                if wide_enough_p[idx]:
                    x_text = widths_p[idx] * 0.02
                    self.ax_param.text(
                        x_text, yv, lbl, va="center", ha="left",
                        fontsize=fontsize, color="black", zorder=3, clip_on=True
                    )
                else:
                    trans_inside = blended_transform_factory(self.ax_param.transAxes, self.ax_param.transData)
                    self.ax_param.text(
                        0.01, yv, lbl,
                        transform=trans_inside, va="center", ha="left",
                        fontsize=fontsize, color="black", zorder=4, clip_on=False
                    )
            self.ax_param.set_xlim(left=0)
        else:
            self.ax_param.set_yticks([])

        self.ax_param.set_title(f"G (grad norm) (epoch {epoch})", fontsize=max(10, fontsize + 1))

        # draw/update
        self.fig.canvas.draw_idle()
        if self.disp_id is None:
            self.disp_id = display(self.fig, display_id=True)
        else:
            self.disp_id.update(self.fig)



#########################################################################################################


def plot_trades(
    df,
    *,
    col_close: str='close',
    start_plot: 'datetime.time'=None,
    features: list[str]=None,
    col_signal1: str=None,
    col_signal2: str=None,
    sign_thresh: float=None,
    trades: list[tuple]=None,
    performance_stats: dict=None,
    autoscale: bool=False
):
    """
    Interactive Plotly view of intraday trades and signals.

    Shows price, primary signal, optional predicted signal and features,
    highlights trade intervals, optionally draws threshold, and displays
    bid/ask/trailstop in the unified hover.
    """
    # filter by start time
    if start_plot is not None:
        df = df.loc[df.index.time >= start_plot]

    fig = go.Figure()

    # derive trade intervals either from trades list or from action series
    intervals = []

    if trades is not None:
        intervals = [(b, s) for ((b,s),_,_,_,_) in trades]
        lines_src = performance_stats["TRADES"]
    
        for i, ((b_dt, s_dt), _, ret_pc, _, _) in enumerate(trades, start=1):
            seg = df.loc[b_dt:s_dt, col_close]
            perf_line = lines_src[i-1]
            rhs = perf_line.split('=')[-1].strip()
            hover = f"Return: {float(rhs):.3f}<br>Return%: {ret_pc:.3f}%<extra></extra>"
        
            fig.add_trace(go.Scatter(
                x=seg.index, y=seg.values,
                mode='lines+markers',
                line=dict(color='green', width=1),
                marker=dict(size=3, color='green'),
                legendgroup='Trades', showlegend=False,
                hovertemplate=hover
            ))

    # shaded bands for trade intervals
    if intervals:
        fig.update_layout(yaxis3=dict(domain=[0,1], anchor='x', overlaying='y', visible=False))
        for i,(b0,b1) in enumerate(intervals):
            fig.add_trace(go.Scatter(
                x=[b0,b1,b1,b0,b0],
                y=[0,0,1,1,0],
                mode='none',
                fill='toself',
                fillcolor='rgba(255,165,0,0.25)',
                legendgroup='Trades',
                name='Trades' if i==0 else None,
                showlegend=(i==0),
                yaxis='y3',
                hoverinfo='skip'
            ))

    # prepare customdata for unified hover: (bid,ask,trailstop)
    n = len(df)
    def _col_vals(name):
        return df[name].to_numpy() if (name and name in df) else [float('nan')]*n
    customdata = list(zip(_col_vals("bid"), _col_vals("ask"), _col_vals("trailstop_price")))

    # price line with hover showing bid/ask/trailstop
    fig.add_trace(go.Scatter(
        x=df.index, y=df[col_close],
        mode='lines', line=dict(color='grey', width=1),
        name='Close',
        customdata=customdata,
        hovertemplate='Close: %{y:.3f}<br>Bid: %{customdata[0]:.3f}<br>Ask: %{customdata[1]:.3f}<br>Trail: %{customdata[2]:.3f}<extra></extra>'
    ))

    # primary signal 
    if col_signal1 and col_signal1 in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_signal1],
            mode='lines', line=dict(color='blue', dash='dot', width=2),
            name='Target Signal', yaxis='y2',
            hovertemplate='Signal: %{y:.3f}<extra></extra>'
        ))

    # secondary signal
    if col_signal2 and col_signal2 in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_signal2],
            mode='lines', line=dict(color='crimson', dash='dot', width=2),
            name='Pred Signal', yaxis='y2',
            hovertemplate='Pred: %{y:.3f}<extra></extra>'
        ))

    # optional feature lines (hidden by default)
    if features is None:
        features = sorted([c for c in df.columns if c not in {"action", col_signal1, col_signal2, col_close,
                                                            "Position", "Cash", "NetValue", "Action", "TradedAmount",
                                                             "signal_raw", "trailstop_price",
                                                             }])
        
    # minimal in-place autoscale (map each feature into close range)
    if autoscale:
        rmin, rmax = df[col_close].min(), df[col_close].max()
        span = (rmax - rmin) or 1.0
        for f in features:
            if f in df:
                a, b = df[f].min(), df[f].max()
                df.loc[:, f] = ((df[f] - a) / ((b - a) or 1e-9)) * span + rmin

    for feat in sorted(features):
        if feat in df:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[feat],
                mode='lines', line=dict(width=1),
                name=feat, yaxis='y2', visible='legendonly',
                hovertemplate=f'{feat}: %{{y:.3f}}<extra></extra>'
            ))

    # threshold line on signal axis (supports scalar or per-row Series/array)
    if sign_thresh is not None:
        sign_thr = df[sign_thresh]
        # if array-like, convert to numpy and validate length
        if np.ndim(sign_thr) == 0:
            # scalar threshold
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]],
                y=[float(sign_thr), float(sign_thr)],
                mode='lines',
                line=dict(color='purple', dash='dot', width=1),
                name='Threshold', yaxis='y2',
                hovertemplate=f"Thresh: {float(sign_thr):.3f}<extra></extra>"
            ))
        else:
            buy_arr = np.asarray(sign_thr)
            if buy_arr.shape[0] != len(df):
                raise ValueError("sign_thr length must match number of rows in df")
            # plot per-row threshold (one value per timestamp)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=buy_arr,
                mode='lines',
                line=dict(color='purple', dash='dot', width=1),
                name='Threshold', yaxis='y2',
                hovertemplate='Thresh: %{y:.3f}<extra></extra>'
            ))

    # layout and display
    fig.update_layout(
        hovermode='x unified',
        template='plotly_white',
        height=800,
        xaxis_title='Time',
        yaxis_title='Price',
        yaxis2=dict(overlaying='y', side='right', title='Signal', showgrid=False),
        legend=dict(font=dict(size=12), tracegroupgap=4)
    )

    fig.show()


#########################################################################################################


def aggregate_performance(
    perf_list: list,
    df: pd.DataFrame,
) -> None:
    """
    Aggregate and print summary when per-day perf dicts contain only formatted strings.

    - Parses perf entries created by simulate_trading (lines ending " = <number>").
    - Computes:
        * one-time buy&hold across dataset (start ask -> end bid)
        * sum of per-day buy&hold (parsed)
        * canonical Strategy = sum(all per-trade PnL parsed from TRADES lines)
        * Trades Count, Strategy per trade, Strategy per day
    - Prints concise table and a small bar plot (unchanged layout).
    """
    def _parse_eq_value(s: str) -> float:
        # expects formatted lines ending with " = <number>"
        return float(s.strip().split(" = ")[-1])

    # first/last trading day and one-time B&H legs (simplified)
    first_day = df.index.normalize().min()
    last_day  = df.index.normalize().max()

    start_ask = df.loc[df.index.normalize() == first_day, "ask"].iloc[0]
    end_bid   = df.loc[df.index.normalize() == last_day,  "bid"].iloc[-1]

    print("\n" + "=" * 115)
    print(f"Overall Summary ({first_day.date()} = {start_ask:.3f} → {last_day.date()} = {end_bid:.3f})")
    num_days = df.index.normalize().nunique()
    print(f"Num. trading days: {num_days}")
    trades_count = sum(len(perf.get("TRADES", [])) for perf in perf_list if perf)
    print(f"Trades Count: {trades_count}")

    # collect buy&hold per-day values and all trade PnL values
    bh_per_day_vals = [_parse_eq_value(perf.get("BUY&HOLD")) for perf in perf_list]
    all_trade_vals = [_parse_eq_value(perf.get("STRATEGY")) for perf in perf_list]

    one_time_bh = end_bid - start_ask
    print(f"\nOne-Time B&H gain: {one_time_bh:.{3}f}")
    strategy_sum = sum(all_trade_vals)
    print(f"Sum Strategy gain: {strategy_sum:.{3}f}")
    intraday_bh_sum = sum(bh_per_day_vals)
    print(f"Sum Intraday B&H gain: {intraday_bh_sum:.{3}f}")

    one_time_bh_per_day = one_time_bh / num_days if num_days else 0.0
    print(f"\nOne-Time B&H gain per day: {one_time_bh_per_day:.{4}f}")
    strategy_per_day = strategy_sum / num_days
    print(f"Strategy gain per day: {strategy_per_day:.{4}f}")
    strategy_per_trade = strategy_sum / trades_count if trades_count > 0 else 0
    print(f"Strategy gain per trade: {strategy_per_trade:.{4}f}")

    # small bar plot (same layout, rounded annotations)
    primary = {
        "One-Time B&H gain": one_time_bh,
        "Sum Strategy gain": strategy_sum,
        "Sum Intraday B&H gain": intraday_bh_sum
    }
    secondary = {
        "One-Time B&H per day": one_time_bh_per_day,
        "Strategy gain per day": strategy_per_day,
        "Strategy gain per trade": strategy_per_trade
    }

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    names1 = list(primary.keys())
    names2 = list(secondary.keys())
    x1 = np.arange(len(names1))
    x2 = np.arange(len(names2)) + len(names1)

    width = 0.6
    bars1 = ax1.bar(x1, list(primary.values()), width, color="#4C72B0", label="Absolute")
    bars2 = ax2.bar(x2, list(secondary.values()), width, color="#C44E52", label="Relative")

    all_names = names1 + names2
    ax1.set_xticks(np.concatenate([x1, x2]))
    ax1.set_xticklabels(all_names, rotation=30, ha="right")
    ax1.set_ylabel("USD (big sums)")
    ax2.set_ylabel("USD (per trade/day)")
    ax1.set_title(f"Performance Summary ({first_day.date()} → {last_day.date()})")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")

    plt.tight_layout()
    plt.show()


#########################################################################################################


def plot_dual_histograms(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    sample: int | None = 50000,
    bins: int = 40,
    clip_pct: tuple[float, float] = (0.02, 0.98),
):
    """
    For each feature column, plot how its
    distribution changes before vs. after scaling/transformation.

    Functionality:
      1) Auto-discover common feat_… columns in df_before & df_after.
      2) Optionally sample up to `sample` rows for speed.
      3) For numeric, high-cardinality features:
         - Compute clipped [pct lower, pct upper] ranges.
         - Overlay “before” histogram in blue (bottom x / left y).
         - Overlay “after” histogram in orange (top x / right y).
      4) For categorical or low-card features:
         - Draw side-by-side bar charts of value frequencies.
      5) Arrange subplots in a grid and clean up unused axes.
      6) Show a tqdm progress bar per feature.
    """
    # 1) identify all feat_… columns present in both DataFrames
    feat_cols = [
        col for col in df_before.columns
        if col in df_after.columns
    ]
    if not feat_cols:
        raise ValueError("No overlapping features columns found in the two DataFrames.")

    # 2) optional random sampling to limit plotting cost
    if sample:
        n = min(len(df_before), len(df_after), sample)
        dfb = df_before.sample(n, random_state=0)
        dfa = df_after.sample(n, random_state=0)
    else:
        dfb, dfa = df_before, df_after

    # 3) prepare a grid of subplots
    cols = 2
    rows = math.ceil(len(feat_cols) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.flatten()

    # 4) loop over each feature with a tqdm progress bar
    for ax, feat in tqdm(zip(axes, feat_cols),
                         total=len(feat_cols),
                         desc="Plotting features"):
        before = dfb[feat].dropna()
        after  = dfa[feat].dropna()

        # numeric & high-cardinality: dual overlaid histograms
        if pd.api.types.is_numeric_dtype(before) and before.nunique() > 10:
            # compute clipping bounds for each
            lo_b, hi_b = np.quantile(before, clip_pct)
            lo_a, hi_a = np.quantile(after,  clip_pct)

            edges_b = np.linspace(lo_b, hi_b, bins + 1)
            edges_a = np.linspace(lo_a, hi_a, bins + 1)

            # blue histogram on bottom x / left y
            ax.hist(before, bins=edges_b, color="C0", alpha=0.6, edgecolor="C0")
            ax.set_xlim(lo_b, hi_b)
            ax.margins(x=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("C0")
            ax.spines["left"].set_color("C0")
            ax.tick_params(axis="x", colors="C0", rotation=45)
            ax.tick_params(axis="y", colors="C0")

            # orange histogram on top x / right y
            ax_top    = ax.twiny()
            ax_orange = ax_top.twinx()

            ax_orange.hist(after, bins=edges_a, color="C1", alpha=0.6, edgecolor="C1")
            ax_top.set_xlim(lo_a, hi_a)
            ax_top.margins(x=0)

            # style top x-axis
            ax_top.spines["bottom"].set_visible(False)
            ax_top.spines["left"].set_visible(False)
            ax_top.spines["right"].set_visible(False)
            ax_top.spines["top"].set_color("C1")
            ax_top.xaxis.set_ticks_position("top")
            ax_top.xaxis.set_label_position("top")
            ax_top.tick_params(axis="x", colors="C1", rotation=45)

            # style right y-axis
            ax_orange.spines["bottom"].set_visible(False)
            ax_orange.spines["left"].set_visible(False)
            ax_orange.spines["top"].set_visible(False)
            ax_orange.spines["right"].set_color("C1")
            ax_orange.yaxis.set_ticks_position("right")
            ax_orange.yaxis.set_label_position("right")
            ax_orange.tick_params(axis="y", colors="C1", labelright=True)

        else:
            # categorical or low-cardinality: side-by-side frequency bars
            bd = before.value_counts(normalize=True).sort_index()
            ad = after .value_counts(normalize=True).sort_index()
            cats = sorted(set(bd.index) | set(ad.index))
            x = np.arange(len(cats))
            w = 0.4

            ax.bar(x - w/2, [bd.get(c, 0) for c in cats], w,
                   color="C0", alpha=0.6)
            ax.bar(x + w/2, [ad.get(c, 0) for c in cats], w,
                   color="C1", alpha=0.6)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_color("C0")
            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=45)
            ax.tick_params(axis="y", colors="C0")

        ax.set_title(feat, color="black")

    # 5) hide any unused subplots
    for extra_ax in axes[len(feat_cols):]:
        extra_ax.axis("off")

    plt.tight_layout()
    plt.show()


#########################################################################################################


def make_live_plot_callback(fig, ax, line, handle):
    """
    Build an Optuna callback that will update *this* fig/ax/line/handle.
    Returns: callback(study, frozen_trial)
    """
    def live_plot_callback(study, _trial):
        # only use fully completed trials, sorted by index
        complete = sorted(
            (t for t in study.trials if t.state == TrialState.COMPLETE),
            key=lambda t: t.number
        )
        if not complete:
            return

        xs = [t.number for t in complete]
        ys = [t.value  for t in complete]

        # update line data
        line.set_data(xs, ys)

        # recompute axis limits + small padding
        x_min, x_max = xs[0], xs[-1]
        x_pad = max(1, (x_max - x_min) * 0.05)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

        y_min, y_max = min(ys), max(ys)
        y_pad = (y_max - y_min) * 0.1 or 0.1
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # re-draw in place
        update_display(fig, display_id=handle.display_id)

        # free all of the Figure’s memory on disk/UI
        plt.close(fig)

    return live_plot_callback


#############################################


def plot_callback(study, trial):
    """
    Live-update a small Matplotlib line chart of trial.value vs. trial.number.
    `state` lives across calls and holds the figure, axes and data lists.
    """
    if trial.state != TrialState.COMPLETE:
        return    # skip pruned or errored trials
        
    # 1) Initialize a single persistent state dict
    if not hasattr(plot_callback, "state"):
        plot_callback.state = {
            "initialized": False,
            "fig": None, "ax": None, "line": None, "handle": None,
            "x": [], "y": []
        }
    state = plot_callback.state

    # 2) Skip pruned or errored trials
    if trial.value is None:
        return state

    # 3) One-time figure setup
    if not state["initialized"]:
        import matplotlib.pyplot as plt
        plt.ioff()
        fig, ax = plt.subplots(figsize=(7, 3))
        line, = ax.plot([], [], "bo-", markersize=3, linewidth=1)
        ax.set(xlabel="Trial #", ylabel="Avg Daily P&L", title="Optuna Progress")
        ax.grid(True)
        handle = display(fig, display_id=True)
        state.update(fig=fig, ax=ax, line=line, handle=handle, initialized=True)

    # 4) Append new point and redraw
    state["x"].append(trial.number)
    state["y"].append(float(trial.value))
    state["line"].set_data(state["x"], state["y"])
    state["ax"].relim()
    state["ax"].autoscale_view()
    state["handle"].update(state["fig"])

    # 5) Close the figure to free memory—but keep the display alive
    import matplotlib.pyplot as plt
    plt.close(state["fig"])

    return state

##########################

def save_best_trial_callback(study, trial):
    # only act when this trial just became the study’s best
    if study.best_trial != trial:
        return

    best_value  = trial.value
    best_params = trial.params

    # scan the folder for existing JSONs for this ticker
    pattern = os.path.join(params.optuna_folder, f"{params.ticker}_*_target.json")
    files   = glob.glob(pattern)

    # extract the float values out of the filenames
    existing = []
    # regex matches: <TICKER>_<float>_target.json
    rx = re.compile(rf'^{re.escape(params.ticker)}_(?P<val>-?\d+(?:\.\d+)?)_target\.json$')
    for fn in files:
        name = os.path.basename(fn)
        m = rx.match(name)
        if not m:
            continue
        try:
            existing.append(float(m.group("val")))
        except ValueError:
            continue

    # only save if our new best_value beats all on disk
    max_existing = max(existing) if existing else float("-inf")
    if best_value <= max_existing:
        return

    # dump to a new file
    fname = f"{params.ticker}_{best_value:.4f}_target.json"
    path  = os.path.join(params.optuna_folder, fname)
    with open(path, "w") as fp:
        json.dump(
            {"value":  best_value,
             "params": best_params},
            fp,
            indent=2
        )

########################## 

# in-memory accumulator for completed trial results
_results: list[dict] = []

def save_results_callback(study, trial):
    """
    Optuna callback to persist completed trials to a CSV whose name
    matches the JSON filename produced by a different Optuna run.

    What this function does:
    1) Skips any trial that was pruned or errored.
    2) Extracts the three hyperparameters plus the average daily PnL.
    3) Appends that data as a dict into the module‐level `_results` list.
    4) Builds a pandas DataFrame from `_results`, sorts it descending by avg_daily_pnl.
    """

    # 1) Only process trials that ran to completion
    if trial.state != TrialState.COMPLETE:
        return

    # 2) Extract trial number, params, and objective value
    entry = {"trial": trial.number}
    for k, v in trial.params.items():
        entry[k] = round(v, 5) if isinstance(v, (int, float)) else v
    entry["avg_daily_pnl"] = round(trial.value, 5)

    _results.append(entry)

    # 3) Build & sort DataFrame of all completed trials so far
    df = pd.DataFrame(_results)
    df = df.sort_values("avg_daily_pnl", ascending=False)

    # 4) Construct the CSV filename & write it
    csv_name = f"{params.ticker}_live_predicted.csv"
    out_path = os.path.join(params.optuna_folder, csv_name)
    df.to_csv(out_path, index=False)


####################################################################################################################


def plot_correlation_before_after(
    corr_full: pd.DataFrame,
    corr_pruned: pd.DataFrame,
    figsize: Tuple[int, int] = (18, 8),
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm"
) -> None:
    """
    Plot two side-by-side heatmaps:
      - corr_full  : correlation matrix before pruning
      - corr_pruned: correlation matrix after pruning

    Both inputs are absolute-valued correlation DataFrames (0..1).
    Uses a short tqdm progress bar while rendering each heatmap.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    panels = [
        (corr_full, axes[0], "Correlation Before Pruning"),
        (corr_pruned, axes[1], "Correlation After Pruning"),
    ]

    for corr_df, ax, title in panels:
        sns.heatmap(
            corr_df,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            cbar_kws={"shrink": 0.7}
        )
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


    
####################################################################################################################


def compute_psd(signal: np.ndarray, dt: float):
    """Return freqs (1/min) and power spectral density for 1D signal."""
    y = signal - np.nanmean(signal)
    fft = np.fft.rfft(y)
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(y), d=dt)
    return freqs, psd


def analyze_signal_psd(
    df,
    day,
    signal_col="signal",
    pred_col="pred_signal",
    dt: float = 1.0,
    high_freq_thresh: float = 0.10,
    tail_bins: int = 10
):
    """
    Receive a random day, compute and plot PSDs for true vs predicted signals,
    print the highest-frequency bins and return avg high-frequency power ratio.
    """
    day_mask = df.index.normalize() == day
    df_day = df.loc[day_mask, [signal_col, pred_col]].dropna()
    if df_day.empty:
        raise ValueError("no data for selected day")

    y_true = df_day[signal_col].to_numpy()
    y_pred = df_day[pred_col].to_numpy()

    f_t, psd_t = compute_psd(y_true, dt)
    f_p, psd_p = compute_psd(y_pred, dt)

    plt.figure(figsize=(6,4))
    plt.loglog(f_t, psd_t, label="true")
    plt.loglog(f_p, psd_p, label="pred")
    plt.xlabel("Frequency (1/min)"); plt.ylabel("Power"); plt.legend()
    plt.title(f"PSD on {pd.to_datetime(day).date()}"); plt.show()

    print(f"\nTop {tail_bins} frequency bins:")
    for ft, tt, tp in zip(f_t[-tail_bins:], psd_t[-tail_bins:], psd_p[-tail_bins:]):
        print(f"{ft:8.4f}  {tt:12.3e}  {tp:12.3e}")

    hf = f_t > high_freq_thresh
    ratio = psd_p[hf].mean() / psd_t[hf].mean()
    print(f"\nHigh-freq power ratio (pred/true) >{high_freq_thresh}: {ratio:.3f}")
    return ratio
