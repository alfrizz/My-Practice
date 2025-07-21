from libs import params
import pandas as pd
import numpy  as np

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import display, HTML

from optuna.trial import TrialState

###############################################################################

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


def plot_trades(df, col_signal1, col_signal2, col_action, trades, buy_threshold, performance_stats, trade_color="green"):
    """
    Plots the overall close-price series plus trade intervals and two continuous signals,
    with the signals shown on a secondary y-axis.

    • The base trace (grey) plots the close-price series on the primary y-axis.
    • Trade traces (green by default) indicate the intervals for each trade from the original trade list.
    • A dotted blue line shows the raw normalized "signal" on a secondary y-axis.
    • A dashed red line shows the smooth normalized signal on the secondary y-axis.
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
    df = df.loc[df.index.time >= params.regular_start_pred]
    
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
    
    # Signal2 trace on a secondary y-axis.
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[col_signal2],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name=col_signal2,
        hovertemplate="Date: %{x}<br>Smooth Signal: %{y:.2f}<extra></extra>",
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
) -> dict:
    """
    Given a list of daily performance dicts and the full-minute-bar
    DataFrame, print & return one summary dict.  Buy&hold uses the
    first ask after regular_start on the first day and the last bid
    before regular_end on the last day.
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
    aggregated["Trades Returns ($)"] = f"{trades_count} trades"

    # 4) Rename the per-day Buy & Hold key
    aggregated["Buy & Hold – each day ($)"] = aggregated.pop("Buy & Hold Return ($)")

    # 5) Restrict to bars inside your regular hours, to find the true trading days
    session_df = df.between_time(params.regular_start, params.regular_end)
    if not session_df.empty:
        first_day = session_df.index.normalize().min()
        last_day  = session_df.index.normalize().max()
    else:
        # fallback if somehow no bars in session hours at all
        all_days  = df.index.normalize().unique()
        first_day = all_days.min()
        last_day  = all_days.max()

    # 6) Grab the first ask ≥ regular_start on first_day
    mask_start = (
        (df.index.normalize() == first_day) &
        (df.index.time     >= params.regular_start)
    )
    if df.loc[mask_start, "ask"].empty:
        # if the day's first session-minute bar is missing, just take the day's first ask
        mask_start = df.index.normalize() == first_day
    start_ask = df.loc[mask_start, "ask"].iloc[0]

    # 7) Grab the last bid ≤ regular_end on last_day
    mask_end = (
        (df.index.normalize() == last_day) &
        (df.index.time     <= params.regular_end)
    )
    if df.loc[mask_end, "bid"].empty:
        # if the day's last session-minute bar is missing, just take the day's last bid
        mask_end = df.index.normalize() == last_day
    end_bid = df.loc[mask_end, "bid"].iloc[-1]

    # 8) Print the clean summary
    print(f"\n=== Overall Summary ({first_day.date()} → {last_day.date()}) ===")
    print(f"Start date price: {first_day.date()} = {start_ask:.4f}")
    print(f"  End date price:  {last_day.date()} = {end_bid:.4f}")
    print(f"One-time buy&hold gain: {end_bid - start_ask:.3f}\n")


    return aggregated

#########################################################################################################

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

    # no tight_layout() needed—constrained_layout will handle margins
    update_display(fig, display_id=handle.display_id)

