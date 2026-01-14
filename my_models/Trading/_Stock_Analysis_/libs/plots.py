from libs import params

from typing import Sequence, List, Tuple, Optional, Union, Dict

import pandas as pd 
import numpy  as np
import datetime

import re
import math
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import plotly.graph_objects as go
from IPython.display import display
import seaborn as sns
sns.set_style("white")

from tqdm.auto import tqdm


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
    col_close: str = "close",
    features: list[str] = None,
    col_signal1: str = None,
    col_signal2: str = None,
    sign_thresh: float = None,
    axis_sig_thresh: str = "first",
    autoscale: bool = False,
):

    df = df.copy()
    df.index = pd.to_datetime(df.index).floor("min")
    fig = go.Figure()

    def _arr_for(x):
        if isinstance(x, str):
            return df.get(x, pd.Series(np.nan, index=df.index)).to_numpy()
        val = float(x) if x is not None else np.nan
        return np.full(len(df), val, dtype=float)

    # choose y-axis for signals/threshold: "first" -> y1, "second" -> y2 
    sig_yaxis = "y2" if str(axis_sig_thresh).lower().startswith("second") else "y1"
    
    # signals
    if col_signal1 and col_signal1 in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_signal1],
            mode="lines", line=dict(color="blue", dash="dot", width=2),
            name="Target Signal", yaxis=sig_yaxis,
            hovertemplate="Signal: %{y:.3f}<extra></extra>",
        ))
        
    if sign_thresh is not None:
        sign_arr = _arr_for(sign_thresh)
        if np.ndim(sign_arr) == 0:
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]],
                y=[float(sign_arr), float(sign_arr)],
                mode="lines",
                line=dict(color="purple", dash="dot", width=1),
                name="Threshold", yaxis=sig_yaxis,
                hovertemplate=f"Thresh: {float(sign_arr):.3f}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df.index, y=sign_arr,
                mode="lines",
                line=dict(color="purple", dash="dot", width=1),
                name="Threshold", yaxis=sig_yaxis,
                hovertemplate="Thresh: %{y:.3f}<extra></extra>",
            ))
            
    if col_signal2 and col_signal2 in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_signal2],
            mode="lines", line=dict(color="crimson", dash="dot", width=2),
            name="Pred Signal", yaxis=sig_yaxis,
            hovertemplate="Pred: %{y:.3f}<extra></extra>",
        ))

    if autoscale:
        rmin, rmax = df[col_close].min(), df[col_close].max()
        span = (rmax - rmin) or 1.0

    for feat in sorted(features):
        if feat in df:
            y_orig = df[feat].astype(float)
            if autoscale:
                a, b = y_orig.min(), y_orig.max()
                y_scaled = ((y_orig - a) / ((b - a) or 1e-9)) * span + rmin
                fig.add_trace(go.Scatter(
                    x=df.index, y=y_scaled,
                    mode="lines", line=dict(width=1),
                    name=feat, yaxis="y2", visible="legendonly",
                    customdata=y_orig.to_numpy(),  # original values
                    hovertemplate=f"{feat}: %{{customdata:.3f}}<extra></extra>",
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=df.index, y=y_orig,
                    mode="lines", line=dict(width=1),
                    name=feat, yaxis="y2", visible="legendonly",
                    hovertemplate=f"{feat}: %{{y:.3f}}<extra></extra>",
                ))

    # base customdata
    bid_arr  = df.get("bid",  pd.Series(np.nan, index=df.index)).to_numpy()
    ask_arr  = df.get("ask",  pd.Series(np.nan, index=df.index)).to_numpy()
    trail_arr= df.get("trail_stop_price", pd.Series(np.nan, index=df.index)).to_numpy()
    atr_arr  = df.get("atr_stop_price",   pd.Series(np.nan, index=df.index)).to_numpy()
    vwap_arr = df.get("vwap_stop_price",  pd.Series(np.nan, index=df.index)).to_numpy()

    act_arr  = df.get("Action", df.get("action", pd.Series(np.nan, index=df.index))).astype(str).to_numpy()
    shar_arr = df.get("Shares", pd.Series(np.nan, index=df.index)).to_numpy()
    pos_arr  = df.get("Position", pd.Series(np.nan, index=df.index)).to_numpy()
    posamt_arr  = df.get("Posamt", pd.Series(np.nan, index=df.index)).to_numpy()
    cash_arr = df.get("Cash", pd.Series(np.nan, index=df.index)).to_numpy()
    net_arr  = df.get("Pnl", pd.Series(np.nan, index=df.index)).to_numpy()
    # trade_arr= df.get("TradeID", pd.Series(np.nan, index=df.index)).to_numpy()
    trade_arr = np.where((t := df.get("TradeID", pd.Series(np.nan, index=df.index)).fillna(0).astype(int).astype(str)) == "0", "", t)

    # customdata ordering
    base_fields = [bid_arr, ask_arr, trail_arr, atr_arr, vwap_arr, trade_arr, act_arr, shar_arr, pos_arr, posamt_arr, cash_arr, net_arr]
    customdata = list(zip(*base_fields))

    market_parts = [
        "Close: %{y:.3f}",
        "Bid: %{customdata[0]:.3f}",
        "Ask: %{customdata[1]:.3f}",
        "Trail_run: %{customdata[2]:.3f}",
        "Atr_run: %{customdata[3]:.3f}",
        "Vwap_run: %{customdata[4]:.3f}",
    ]

    state_parts = [
        "",
        "TrID: %{customdata[5]:.0f}",
        "Action: %{customdata[6]}",
        "Shares: %{customdata[7]:.0f}",
        "Position: %{customdata[8]:.0f}", 
        "PosAmt: %{customdata[9]:.3f}",
        "Cash: %{customdata[10]:.3f}",
        "Pnl: %{customdata[11]:.3f}",
    ]
    hover_template = "<br>".join(state_parts + [""] + market_parts)  + "<extra></extra>"

    # invisible trace for unified hover
    fig.add_trace(go.Scatter(
        x=df.index, y=df[col_close],
        mode="markers",
        marker=dict(size=0, opacity=0),
        hovertemplate=hover_template,
        customdata=customdata,
        showlegend=False,
        hoverinfo="text",
    ))

    # base price line (grey) — hover skipped
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[col_close],
        mode="lines",
        line=dict(color="grey", width=1.2),
        name="Close",
        hoverinfo="skip",
        showlegend=True,
    ))

    buy_mask = df["Action"] == "Buy"
    sell_mask = df["Action"] == "Sell"

    def _sizes(shares_arr): # sell and buy markers sizes proportional to the shares number
        return np.clip(np.nan_to_num(np.abs(shares_arr), nan=0.0) * 0.3 + 6, 6, 18)

    if buy_mask.any():
        fig.add_trace(go.Scatter(
            x=df.index[buy_mask],
            y=df.loc[buy_mask, col_close],
            mode="markers",
            marker=dict(color="green", size=_sizes(shar_arr[buy_mask]), opacity=0.9),
            name="Buy",
            hoverinfo="skip",
            showlegend=True,
        ))
    if sell_mask.any():
        fig.add_trace(go.Scatter(
            x=df.index[sell_mask], 
            y=df.loc[sell_mask, col_close],
            mode="markers",
            marker=dict(color="red", size=_sizes(shar_arr[sell_mask]), opacity=0.9),
            name="Sell",
            hoverinfo="skip",
            showlegend=True,
        ))

    # minimal: two vertical lines at regular session start and session end (same base day)
    base = df.index[0].normalize()
    fig.add_vline(x=base + pd.Timedelta(hours=params.sess_start_reg.hour, minutes=params.sess_start_reg.minute),
                  line=dict(color="black", dash="dot", width=1), opacity=0.6)
    fig.add_vline(x=base + pd.Timedelta(hours=params.sess_end.hour, minutes=params.sess_end.minute),
                  line=dict(color="black", dash="dot", width=1), opacity=0.6)

    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        height=800,
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis2=dict(overlaying="y", side="right", title="Signal", showgrid=False),
        legend=dict(font=dict(size=12), tracegroupgap=4),
    )

    fig.show()


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
