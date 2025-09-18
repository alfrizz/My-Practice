from libs import params, trades

import pandas as pd
import numpy  as np
import gc

import math
import os
import json
import glob

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import display, update_display, HTML
import seaborn as sns
sns.set_style("white")

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


def plot_trades(
    df,
    col_signal1: str,
    *,
    col_close: str='close',
    start_plot: 'datetime.time'=None,
    features: list[str]=None,
    col_signal2: str=None,
    col_action: str=None,
    trades: list[tuple]=None,
    buy_threshold: float=None,
    performance_stats: dict=None
):
    """
    Plots:
      - price (col_close)
      - target signal (col_signal1)
      - optional pred. signal (col_signal2)
      - optional extra feature lines
      - optional buy/sell intervals from (trades,col_action)
      - optional threshold line
      - unified hover, cleaned legend, tall figure
    
    All of col_signal2, col_action, trades, buy_threshold and performance_stats
    are optional. Pass only what you need.
    """
    # 1) filter by time if requested
    if start_plot is not None:
        df = df.loc[df.index.time >= start_plot]

    fig = go.Figure()

    # 2) draw trade‐interval bands if col_action + trades given
    intervals = []
    if col_action and trades is None:
        # infer intervals from col_action
        events, last_buy = df[col_action], None
        for ts, act in events.items():
            if act == 1:
                last_buy = ts
            elif act == -1 and last_buy is not None:
                intervals.append((last_buy, ts))
                last_buy = None
    elif trades:
        # assume trades is list of ((b_dt,s_dt),...,ret_pc)
        intervals = [(b, s) for ((b,s),_,_) in trades]

    if intervals:
        # full‐height axis for shading
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

    # 3) price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df[col_close],
        mode='lines', line=dict(color='grey',width=1),
        name='Close', hovertemplate='Price: %{y:.3f}<extra></extra>'
    ))

    # 4) target signal
    fig.add_trace(go.Scatter(
        x=df.index, y=df[col_signal1],
        mode='lines', line=dict(color='blue',dash='dot',width=2),
        name='Target Signal', yaxis='y2',
        hovertemplate='Signal: %{y:.3f}<extra></extra>'
    ))

    # 5) optional pred. signal
    if col_signal2 and col_signal2 in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_signal2],
            mode='lines', line=dict(color='crimson',dash='dot',width=2),
            name='Pred Signal', yaxis='y2',
            hovertemplate='Pred: %{y:.3f}<extra></extra>'
        ))

    # 6) optional features
    if features:
        for feat in features:
            if feat in df:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[feat],
                    mode='lines', line=dict(width=1),
                    name=feat, yaxis='y2', visible='legendonly',
                    hovertemplate=f'{feat}: %{{y:.3f}}<extra></extra>'
                ))

    # 7) overlay individual trade legs (green) if trades list given
    if trades:
        for i,((b_dt,s_dt),_,ret_pc) in enumerate(trades, start=1):
            seg = df.loc[b_dt:s_dt, col_close]
            abs_gain = None
            if performance_stats and 'Trades Returns ($)' in performance_stats:
                abs_gain = performance_stats['Trades Returns ($)'][i-1]
            hover = (
                f"Return$:{abs_gain:.3f}<br>Return%:{ret_pc:.3f}%<extra></extra>"
                if abs_gain is not None
                else f"Return%:{ret_pc:.3f}%<extra></extra>"
            )
            fig.add_trace(go.Scatter(
                x=seg.index, y=seg.values,
                mode='lines+markers',
                line=dict(color='green',width=1),
                marker=dict(size=3,color='green'),
                legendgroup='Trades', showlegend=False,
                hovertemplate=hover
            ))

    # 8) optional threshold
    if buy_threshold is not None:
        fig.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[buy_threshold, buy_threshold],
            mode='lines', line=dict(color='purple',dash='dot',width=1),
            name='Threshold', yaxis='y2',
            hovertemplate=f"Thresh: {buy_threshold:.3f}<extra></extra>"
        ))

    # 9) layout tweaks
    fig.update_layout(
        hovermode='x unified',
        template='plotly_white',
        height=800,
        xaxis_title='Time',
        yaxis_title='Price',
        yaxis2=dict(overlaying='y',side='right',title='Signal',showgrid=False),
        legend=dict(font=dict(size=12),tracegroupgap=4)
    )

    fig.show()


#########################################################################################################

def aggregate_performance(
    perf_list: list,
    df: pd.DataFrame,
    round_digits: int = 3
) -> None:
    """
    Given per-day performance dicts and the full-minute-bar DataFrame,
    print a clean summary:
      • One-time buy&-hold gain over the entire period
      • Sum of daily Buy & Hold returns (intraday)
      • Sum of Strategy returns
      • Total number of trades
      • Strategy return per trade
      • Number of trading days
    """

    # 1) Collect all keys present in daily dicts
    all_keys = set().union(*(perf.keys() for perf in perf_list if perf))

    # 2) Sum numeric fields except 'Trades Returns ($)'
    aggregated = {}
    for key in all_keys:
        if key != "Trades Returns ($)":
            total = sum(
                perf.get(key, 0)
                for perf in perf_list
                if isinstance(perf.get(key), (int, float))
            )
            aggregated[key] = round(total, round_digits)

    # 3) Count total trades
    trades_count = sum(
        len(perf.get("Trades Returns ($)", []))
        for perf in perf_list
        if isinstance(perf.get("Trades Returns ($)"), list)
    )
    aggregated["Trades Count"] = trades_count

    # 4) Rename the per-day Buy & Hold key
    aggregated["Buy & Hold – each day ($)"] = aggregated.pop("Buy & Hold Return ($)", 0.0)

    # 5) Determine first and last trading days from df
    session_df = df.between_time(params.sess_start, params.sess_end)
    if not session_df.empty:
        first_day = session_df.index.normalize().min()
        last_day  = session_df.index.normalize().max()
    else:
        all_days  = df.index.normalize().unique()
        first_day = all_days.min()
        last_day  = all_days.max()

    # 6) One-time buy & hold legs
    mask_start = (
        (df.index.normalize() == first_day) &
        (df.index.time >= params.sess_start)
    )
    if df.loc[mask_start, "ask"].empty:
        mask_start = df.index.normalize() == first_day
    start_ask = df.loc[mask_start, "ask"].iloc[0]

    mask_end = (
        (df.index.normalize() == last_day) &
        (df.index.time <= params.sess_end)
    )
    if df.loc[mask_end, "bid"].empty:
        mask_end = df.index.normalize() == last_day
    end_bid = df.loc[mask_end, "bid"].iloc[-1]

    # Print overall summary
    print(f"\n===========================================================================================================================================================")
    print(f"Overall Summary ({first_day.date()} = {start_ask:.4f} → {last_day.date()} = {end_bid:.4f})")
    print(f"\nOne-time buy&hold gain: {end_bid - start_ask:.3f}")
    strategy_sum = aggregated.get("Strategy Return ($)", 0.0)
    buyhold_sum  = aggregated.get("Buy & Hold – each day ($)", 0.0)

    # — use all actual trading days for the Num. trading days —
    # count calendar days with any bar in the raw df
    num_days = df.index.normalize().nunique()

    print(f"Buy & Hold – each day ($): {buyhold_sum:.3f}")
    print(f"Strategy Return ($): {strategy_sum:.3f}")
    print(f"Trades Count: {aggregated['Trades Count']}")
    if aggregated["Trades Count"] > 0:
        per_trade = strategy_sum / aggregated["Trades Count"]
        print(f"Strategy return per trade: {per_trade:.3f}")
    print(f"Num. trading days: {num_days}")
    if num_days > 0:
        per_day = strategy_sum / num_days
        print(f"Strategy return per trading day: {per_day:.3f}")

    ####################### simple plots ############################

    # 1) Pack your metrics into two groups
    one_time_bh  = end_bid - start_ask
    buyhold_sum  = aggregated.get("Buy & Hold – each day ($)", 0.0)
    strategy_sum = aggregated.get("Strategy Return ($)", 0.0)
    trades_cnt   = aggregated["Trades Count"]
    per_trade    = strategy_sum / trades_cnt if trades_cnt else 0.0
    per_day      = strategy_sum / num_days if num_days else 0.0

    # Only keep non-zero metrics (optional)
    primary = {
        "One-time B&H": one_time_bh,
        "Sum intraday B&H": buyhold_sum,
        "Sum Strategy": strategy_sum
    }
    secondary = {
        "Per trade": per_trade,
        "Per day": per_day
    }

    # 2) Set up figure + twin axes
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # 3) Build x-locations
    names1 = list(primary.keys())
    names2 = list(secondary.keys())
    x1 = np.arange(len(names1))
    x2 = np.arange(len(names2)) + len(names1)  # shift right of primary

    # 4) Plot bars
    width = 0.6
    bars1 = ax1.bar(x1, list(primary.values()), width, color="#4C72B0", label="Primary metrics")
    bars2 = ax2.bar(x2, list(secondary.values()), width, color="#C44E52", label="Secondary metrics")

    # 5) Ticks / labels
    all_names = names1 + names2
    ax1.set_xticks(np.concatenate([x1, x2]))
    ax1.set_xticklabels(all_names, rotation=30, ha="right")
    ax1.set_ylabel("USD (big sums)")
    ax2.set_ylabel("USD (per trade/day)")
    ax1.set_title(f"Performance Summary ({first_day.date()} → {last_day.date()})")

    # 6) Grid & annotation
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}",
                     xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}",
                     xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    # 7) Legend & layout
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

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

    # 4) loop over each feature and draw its before/after visualization
    for ax, feat in zip(axes, feat_cols):
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

##########################

def lightweight_plot_callback(study, trial):
    """
    Live-update a small Matplotlib line chart of trial.value vs. trial.number.
    `state` lives across calls and holds the figure, axes and data lists.
    """
    if trial.state != TrialState.COMPLETE:
        return    # skip pruned or errored trials
        
    # 1) Initialize a single persistent state dict
    if not hasattr(lightweight_plot_callback, "state"):
        lightweight_plot_callback.state = {
            "initialized": False,
            "fig": None, "ax": None, "line": None, "handle": None,
            "x": [], "y": []
        }
    state = lightweight_plot_callback.state

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
    pattern = os.path.join(params.optuna_folder, f"{params.ticker}_*.json")
    files   = glob.glob(pattern)

    # extract the float values out of the filenames
    existing = []
    prefix   = f"{params.ticker}_"
    for fn in files:
        name = os.path.basename(fn)
        # name looks like "AAPL_0.6036.json"
        try:
            val = float(name[len(prefix):-5])
            existing.append(val)
        except ValueError:
            continue

    # only save if our new best_value beats all on disk
    min_existing = min(existing) if existing else float("-inf")
    if best_value <= min_existing:
        return

    # dump to a new file
    fname = f"{params.ticker}_{best_value:.4f}.json"
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
    5) Finds the most recent JSON file in params.optuna_folder matching
       '{ticker}_*.json', pulls its numeric suffix.
    6) Writes out the sorted DataFrame to:
         '{ticker}_{that_same_suffix}_pred_sign_params.csv'
    """

    # 1) Only process trials that ran to completion
    if trial.state != TrialState.COMPLETE:
        return

    # 2) Extract trial number, params, and objective value
    entry = {
        "trial"              : trial.number,
        "pred_threshold"     : round(trial.params["pred_threshold"], 5),
        "trailing_stop_pred" : round(trial.params["trailing_stop_pred"], 5),
        "smoothing_window"   : round(trial.params["smoothing_window"], 5),
        "avg_daily_pnl"      : trial.value,
    }
    _results.append(entry)

    # 3) Build & sort DataFrame of all completed trials so far
    df = pd.DataFrame(_results)
    df = df.sort_values("avg_daily_pnl", ascending=False)

    # 4) Locate the matching JSON file to copy its suffix
    pattern = os.path.join(
        params.optuna_folder,
        f"{params.ticker}_*.json"
    )
    json_files = glob.glob(pattern)
    if not json_files:
        raise FileNotFoundError(
            f"No JSON found matching pattern: {pattern}"
        )

    # Pick the most recently modified JSON
    latest_json = max(json_files, key=os.path.getmtime)
    base = os.path.splitext(os.path.basename(latest_json))[0]
    # base looks like 'AAPL_0.4329'; split off the ticker_ prefix
    suffix = base.split(f"{params.ticker}_", 1)[-1]

    # 5) Construct the CSV filename & write it
    csv_name = f"{params.ticker}_{suffix}_pred_sign_params.csv"
    out_path = os.path.join(params.optuna_folder, csv_name)
    df.to_csv(out_path, index=False)

    
########################## 

def cleanup_callback(study, trial):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



