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

from optuna.trial import TrialState

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


def plot_trades(df, col_signal1, col_signal2, col_action, trades, buy_threshold, performance_stats, regular_start_pred, trade_color="green"):
    """
    Plots the overall close-price series plus trade intervals and two continuous signals,
    with the signals shown on a secondary y-axis.

    • The base trace (grey) plots the close-price series on the primary y-axis.
    • Trade traces (green by default) indicate the intervals for each trade from the original trade list.
    • A dotted blue line shows the"signal" on a secondary y-axis.
    • A dashed red line shows the predicted signal on the secondary y-axis.
    • A horizontal dotted line is drawn at the buy_threshold.
    • Additionally, areas between each buy and sell event determined by the new 
      col_action field (buy=+1, sell=-1) are highlighted (in orange).
    • An update menu is added with two buttons:
         - "Hide Trades": Hides only the trade-specific traces.
         - "Show Trades": Makes all traces visible.

    Parameters:
      df : pd.DataFrame
          DataFrame with a datetime index and at least the columns "close", col_signal1, col_signal2, col_action,.
      trades : list
          A list of tuples, each in the form:
            ((buy_date, sell_date), (buy_price, sell_price), profit_pc).
      buy_threshold : float
          The threshold used for candidate buy detection (shown as a horizontal dotted line on the 
          secondary y-axis).
      performance_stats : dict, optional
          Dictionary containing performance metrics. If provided and if it contains keys
          "Trade Gains ($)" (each a list), they will be added to the trade annotations. 
      trade_color : str, optional
          The color to use for the original trade traces.
    """
    fig = go.Figure()

    # only plot from regular_start_pred
    df = df.loc[df.index.time >= regular_start_pred]
    
    # Trace 0: Base close-price trace.
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        mode='lines',
        line=dict(color='grey', width=1),
        name='Close Price',
        hoverinfo='x+y',
        hovertemplate="Date: %{x}<br>Close: %{y:.2f}<extra></extra>",
    ))
    
    # Trade traces: one per original trade.
    for i, trade in enumerate(trades):
        # Unpack the trade tuple: ((buy_date, sell_date), (buy_price, sell_price), profit_pc)
        (buy_date, sell_date), (_, _), trade_return = trade
        trade_df = df.loc[buy_date:sell_date]
        fig.add_trace(go.Scatter(
            x=trade_df.index,
            y=trade_df['close'],
            mode='lines+markers',
            line=dict(color=trade_color, width=3),
            marker=dict(size=4, color=trade_color),
            name=f"Trade {i+1}",
            hoveron='points',
            hovertemplate=f"Trade {i+1}: Return: {trade_return:.2f}%<extra></extra>",
            visible=True
        ))
        
    # --------------------------------------------------------------------
    # New Trade Action Highlights: using the col_action field.
    # Extract rows where col_action is not zero.
    trade_events = df[df[col_action] != 0][col_action]
    pairs = []
    prev_buy = None
    for timestamp, action in trade_events.items():
        if action == 1:   # Buy signal
            prev_buy = timestamp
        elif action == -1 and prev_buy is not None:
            pairs.append((prev_buy, timestamp))
            prev_buy = None

    # For each buy-sell pair, add a vertical shaded region with annotation.
    for j, (buy_ts, sell_ts) in enumerate(pairs):
        if (performance_stats is not None and 
            "Trades Returns ($)" in performance_stats and 
            len(performance_stats["Trades Returns ($)"]) > j):  
            ann_text = (f"TA Trade {j+1}<br>$: {performance_stats['Trades Returns ($)'][j]}<br>")
        else:
            ann_text = f"TA Trade {j+1}"
            
        fig.add_vrect(
            x0=buy_ts, x1=sell_ts,
            fillcolor="orange", opacity=0.25,
            line_width=0,
            annotation_text=ann_text,
            annotation_position="top left",
            annotation_font_color="orange"
        )
    # --------------------------------------------------------------------
    
    # Signal1 trace on a secondary y-axis.
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[col_signal1],
        mode='lines',
        line=dict(color='blue', width=2, dash='dot'),
        name=col_signal1,
        hovertemplate="Date: %{x}<br>Signal: %{y:.2f}<extra></extra>",
        visible=True,
        yaxis="y2"
    ))

    if col_signal2:
        # Signal2 trace on a secondary y-axis.
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col_signal2],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=col_signal2,
            hovertemplate="Date: %{x}<br>Pred Signal: %{y:.2f}<extra></extra>",
            visible=True,
            yaxis="y2"
        ))
    
    # Add a horizontal dotted line for the buy_threshold (on secondary y-axis).
    fig.add_hline(y=buy_threshold, line=dict(color="purple", dash="dot"),
                  annotation_text="Buy Threshold", annotation_position="top left", yref="y2")
    
    # Total traces: 1 Base + n_trades (original trades) + 2 (for the signal traces).
    n_trades = len(trades)
    total_traces = 1 + n_trades + 2
    vis_show = [True] * total_traces  
    vis_hide = [True] + ["legendonly"] * n_trades + [True, True]
    
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "buttons": [
                    {
                        "label": "Hide Trades",
                        "method": "update",
                        "args": [{"visible": vis_hide}],
                    },
                    {
                        "label": "Show Trades",
                        "method": "update",
                        "args": [{"visible": vis_show}],
                    },
                ],
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.9,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "top",
            }
        ],
        hovermode="x unified",
        template="plotly_white",
        title="Close Price, Trade Intervals, and Signals",
        xaxis_title="Datetime",
        yaxis_title="Close Price",
        height=700,
        yaxis2=dict(
            title="Signals",
            overlaying="y",
            side="right",
            showgrid=False,
        )
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
    session_df = df.between_time(params.regular_start, params.regular_end)
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
        (df.index.time >= params.regular_start)
    )
    if df.loc[mask_start, "ask"].empty:
        mask_start = df.index.normalize() == first_day
    start_ask = df.loc[mask_start, "ask"].iloc[0]

    mask_end = (
        (df.index.normalize() == last_day) &
        (df.index.time <= params.regular_end)
    )
    if df.loc[mask_end, "bid"].empty:
        mask_end = df.index.normalize() == last_day
    end_bid = df.loc[mask_end, "bid"].iloc[-1]

    # Print overall summary
    print(f"\n=============================================================================================================")
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

    return live_plot_callback



def lightweight_plot_callback(
    study,
    trial,
    state={
        "initialized": False,
        "fig": None, "ax": None, "line": None, "handle": None,
        "x": [], "y": [],
    }
):
    # Ignore trials without a numeric value (e.g., pruned)
    if trial.value is None:
        return

    # One-time init: create one figure + one display handle
    if not state["initialized"]:
        import matplotlib.pyplot as plt
        plt.ioff()  # prevent interactive backend from opening extra windows
        fig, ax = plt.subplots(figsize=(7, 3))
        (line,) = ax.plot([], [], "bo-", markersize=3, linewidth=1)
        ax.set(xlabel="Trial #", ylabel="Avg Daily P&L", title="Optuna Progress")
        ax.grid(True)
        handle = display(fig, display_id=True)
        # DO NOT close the figure here if you want continuous updates
        state.update({
            "initialized": True,
            "fig": fig, "ax": ax, "line": line, "handle": handle
        })

    # Append and update
    state["x"].append(trial.number)
    state["y"].append(float(trial.value))
    state["line"].set_data(state["x"], state["y"])
    state["ax"].relim()
    state["ax"].autoscale_view()
    state["handle"].update(state["fig"])



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
    max_existing = max(existing) if existing else float("-inf")
    if best_value <= max_existing:
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




def cleanup_callback(study, trial):
    gc.collect()
    
#########################################################################################################


sns.set_style("white")

def plot_dual_histograms(
    df_before: pd.DataFrame,
    df_after:  pd.DataFrame,
    features:  list[str],
    sample:    int | None        = 50000,
    bins:      int               = 40,
    clip_pct:  tuple[float,float] = (0.02, 0.98),
):
    # 1) figure out which features we can plot
    common = [f for f in features if f in df_before and f in df_after]
    if not common:
        raise ValueError("No overlapping features.")

    # 2) optional sampling
    if sample:
        dfb = df_before.sample(min(len(df_before), sample), random_state=0)
        dfa = df_after .sample(min(len(df_after),  sample), random_state=0)
    else:
        dfb, dfa = df_before, df_after

    # 3) layout
    cols = 2
    rows = math.ceil(len(common) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.flatten()

    for ax, feat in zip(axes, common):
        before = dfb[feat].dropna()
        after  = dfa[feat].dropna()

        # only numeric, high‐cardinality fields get overlaid histograms
        if pd.api.types.is_numeric_dtype(before) and before.nunique() > 10:
            # compute clipped ranges
            lo_b, hi_b = np.quantile(before, clip_pct)
            lo_a, hi_a = np.quantile(after,  clip_pct)

            # bin edges for each
            edges_b = np.linspace(lo_b, hi_b, bins + 1)
            edges_a = np.linspace(lo_a, hi_a, bins + 1)

            # --- BLUE histogram on primary ax ---
            ax.hist(before, bins=edges_b, color="C0", alpha=0.6, edgecolor="C0")
            ax.set_xlim(lo_b, hi_b)
            ax.margins(x=0)

            # style blue spines & ticks
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("C0")
            ax.spines["left"].set_color("C0")
            ax.tick_params(axis="x", colors="C0", rotation=45)
            ax.tick_params(axis="y", colors="C0")

            # --- ORANGE histogram on top‐x + right‐y twin axes ---
            ax_top = ax.twiny()        # new x-axis on top, shares y with ax
            ax_orange = ax_top.twinx() # new y-axis on right, shares x with ax_top

            # draw orange bars onto the right‐y axis
            ax_orange.hist(after, bins=edges_a, color="C1", alpha=0.6, edgecolor="C1")

            # set the orange x‐range (this also applies to ax_orange, since x is shared)
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
            # fallback: side-by-side bars for categorical or low-cardinality
            bd = before.value_counts(normalize=True).sort_index()
            ad = after .value_counts(normalize=True).sort_index()
            cats = sorted(set(bd.index) | set(ad.index))
            x = np.arange(len(cats))
            w = 0.4

            ax.bar(x - w/2, [bd.get(c, 0) for c in cats],
                   w, color="C0", alpha=0.6)
            ax.bar(x + w/2, [ad.get(c, 0) for c in cats],
                   w, color="C1", alpha=0.6)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_color("C0")

            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=45)
            ax.tick_params(axis="y", colors="C0")

        ax.set_title(feat, color="black")

    # disable any unused subplots
    for extra in axes[len(common):]:
        extra.axis("off")

    plt.tight_layout()
    plt.show()

