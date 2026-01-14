from libs import params

from typing import Sequence, List, Tuple, Optional, Union, Dict

import pandas as pd 
import numpy  as np

import re
import os
import json
import glob

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import display
import seaborn as sns
sns.set_style("white")

import optuna
from optuna.trial import TrialState
from optuna.importance import get_param_importances
# from optuna.visualization import plot_optimization_history
from optuna.visualization.matplotlib import plot_optimization_history  # <- MPL version


##########################################################################################


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


##########################################################################################



# def save_best_trial_callback(study, trial):
#     """
#     Optuna callback that saves the trial parameters and objective value to disk
#     when the trial becomes the study's new best.
#     """

#     if study.best_trial != trial:
#         return

#     best_value  = trial.value
#     best_params = trial.params

#     # scan the folder for existing JSONs for this ticker
#     pattern = os.path.join(params.optuna_folder, f"{params.ticker}_*_target.json")
#     files   = glob.glob(pattern)

#     # extract the float values out of the filenames
#     existing = []
#     # regex matches: <TICKER>_<float>_target.json
#     rx = re.compile(rf'^{re.escape(params.ticker)}_(?P<val>-?\d+(?:\.\d+)?)_target\.json$')
#     for fn in files:
#         name = os.path.basename(fn)
#         m = rx.match(name)
#         if not m:
#             continue
#         try:
#             existing.append(float(m.group("val")))
#         except ValueError:
#             continue

#     # only save if our new best_value beats all on disk
#     max_existing = max(existing) if existing else float("-inf")
#     if best_value <= max_existing:
#         return

#     # dump to a new file
#     fname = f"{params.ticker}_{best_value:.4f}_target.json"
#     path  = os.path.join(params.optuna_folder, fname)
#     with open(path, "w") as fp:
#         json.dump(
#             {"value":  best_value,
#              "params": best_params},
#             fp,
#             indent=2
#         )


#######################################################################


# shared display handles
_display_state = {"progress": None, "best": None}

def init_optuna_displays():
    """
    Create/refresh display slots for progress and best-history plots.
    Call this once after printing the baseline, before study.optimize.
    """
    import matplotlib.pyplot as plt
    plt.ioff()  # avoid interactive popups
    _display_state["progress"] = display(display_id=True)
    _display_state["best"]     = display(display_id=True)


#######################################################################


def make_save_results_callback(suffix: str):
    """
    Return a callback that saves a CSV of all completed trials whenever a new best (rounded) appears.
    Filename pattern: {params.ticker}_{best_rounded}_{suffix}.csv
    """
    state = {"results": [], "last_best": None, "last_csv": None}

    def _callback(study, trial):
        # skip non-complete trials
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # build entry, expanding tc_id if mapping exists in __main__ or globals()
        entry = {"trial": trial.number}
        tc_map = getattr(__import__("__main__"), "trading_combinations", None) or globals().get("trading_combinations")
        for k, v in trial.params.items():
            if k == "tc_id" and tc_map and v in tc_map:
                combo = tc_map[v]
                entry["col_signal"] = combo.get("col_signal")
                entry["sign_thresh"] = combo.get("sign_thresh")
            else:
                entry[k] = round(v, 5) if isinstance(v, (int, float)) else v

        entry["avg_daily_pnl"] = round(trial.value, 5)
        state["results"].append(entry)

        # sort by P&L
        df = pd.DataFrame(state["results"]).sort_values("avg_daily_pnl", ascending=False)

        # check for new best (rounded)
        best_val = getattr(study, "best_value", None)
        if best_val is None:
            return
        best_rounded = round(float(best_val), 4)
        if state["last_best"] is not None and best_rounded == state["last_best"]:
            return

        # replace previous CSV if any
        if state["last_csv"] and os.path.exists(state["last_csv"]):
            os.remove(state["last_csv"])

        csv_name = f"{params.ticker}_{best_rounded}_{suffix}.csv"
        out_path = os.path.join(params.optuna_folder, csv_name)
        df.to_csv(out_path, index=False)

        state["last_best"] = best_rounded
        state["last_csv"] = out_path
        print(f"[save_results] wrote {out_path}")

    return _callback


#############################################################################


def make_save_best_json_callback(suffix: str):
    """
    Save JSON summary when a new best (rounded) appears (one file kept).
    Also updates a Matplotlib optimization-history plot in the `best` display slot.
    Skips importances until >=2 completed trials.
    """
    state = {"last_best": None, "last_json": None}

    def _callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            return

        best_val = float(study.best_value)
        best_rounded = round(best_val, 4)
        if state["last_best"] is not None and best_rounded == state["last_best"]:
            return

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        importances = get_param_importances(study) if len(completed) >= 2 else {}

        summary = {
            "best_params": study.best_params,
            "best_value": best_val,
            "importances": importances,
            "trials": [
                {"number": t.number, "value": t.value, "params": t.params, "state": t.state.name}
                for t in study.trials
            ],
        }

        # replace previous JSON
        if state["last_json"] and os.path.exists(state["last_json"]):
            os.remove(state["last_json"])

        file_name = f"{params.ticker}_{best_rounded}_{suffix}.json"
        file_path = os.path.join(params.optuna_folder, file_name)
        with open(file_path, "w") as f:
            json.dump(summary, f, indent=4)

        state["last_best"] = best_rounded
        state["last_json"] = file_path
        print(f"[save_best_json] wrote {file_path}")

        # Optional best-history plot (MPL)
        if len(completed) >= 2:
            ax = plot_optimization_history(study)
            ax.figure.set_size_inches(12, 3)
            handle = _display_state.get("best")
            if handle is not None:
                handle.update(ax.figure)
            import matplotlib.pyplot as plt
            plt.close(ax.figure)

    return _callback
    

#############################################################################


def short_log_callback(study, trial):
    """
    Lightweight logging callback for Optuna trials that prints a concise summary.
    """

    if trial.value is None:
        return

    mean_excess  = trial.value                     # mean_pnl - mean_bh
    mean_pnl     = trial.user_attrs.get("mean_pnl", float("nan"))
    mean_bh      = trial.user_attrs.get("mean_bh_pnls", float("nan"))
    action_counts= trial.user_attrs.get("action_counts", {})

    pct_improv = (mean_pnl - mean_bh) / abs(mean_bh) * 100 

    print(
        f"[Results] mean_pnl:{mean_pnl:.4f} mean_bh:{mean_bh:.4f} "
        f"mean_excess:{mean_excess:.4f} improv_vs_bh:{pct_improv:.2f}%\n"
        f"Action counts: {action_counts}\n"
        f"Best trial is: {study.best_trial.number} with best_val: {study.best_value:.4f}"
    )


#############################################################################


# def plot_callback(study, trial):
#     """
#     Live-update a small Matplotlib line chart of trial.value vs. trial.number.
#     `state` lives across calls and holds the figure, axes and data lists.
#     """
#     if trial.state != TrialState.COMPLETE:
#         return    # skip pruned or errored trials
        
#     # 1) Initialize a single persistent state dict
#     if not hasattr(plot_callback, "state"):
#         plot_callback.state = {
#             "initialized": False,
#             "fig": None, "ax": None, "line": None, "handle": None,
#             "x": [], "y": []
#         }
#     state = plot_callback.state

#     # 2) Skip pruned or errored trials
#     if trial.value is None:
#         return state

#     # 3) One-time figure setup
#     if not state["initialized"]:
#         import matplotlib.pyplot as plt
#         plt.ioff()
#         fig, ax = plt.subplots(figsize=(7, 3))
#         line, = ax.plot([], [], "bo-", markersize=3, linewidth=1)
#         ax.set(xlabel="Trial #", ylabel="Avg Daily P&L", title="Optuna Progress")
#         ax.grid(True)
#         handle = display(fig, display_id=True)
#         state.update(fig=fig, ax=ax, line=line, handle=handle, initialized=True)

#     # 4) Append new point and redraw
#     state["x"].append(trial.number)
#     state["y"].append(float(trial.value))
#     state["line"].set_data(state["x"], state["y"])
#     state["ax"].relim()
#     state["ax"].autoscale_view()
#     state["handle"].update(state["fig"])

#     # 5) Close the figure to free memory—but keep the display alive
#     import matplotlib.pyplot as plt
#     plt.close(state["fig"])

#     return state


def plot_callback(study, trial):
    """
    Live-update Matplotlib line chart of trial.value vs trial.number.
    Uses the shared display handle `_display_state["progress"]`.
    """
    if trial.state != TrialState.COMPLETE or trial.value is None:
        return

    # init state on first use
    if not hasattr(plot_callback, "state"):
        plot_callback.state = {"initialized": False, "fig": None, "ax": None, "line": None, "x": [], "y": []}
    st = plot_callback.state

    if not st["initialized"]:
        import matplotlib.pyplot as plt
        plt.ioff()
        fig, ax = plt.subplots(figsize=(12, 3))
        line, = ax.plot([], [], "bo-", markersize=3, linewidth=1)
        ax.set(xlabel="Trial #", ylabel="Avg Daily P&L", title="Optuna Progress")
        ax.grid(True)
        st.update(fig=fig, ax=ax, line=line, initialized=True)

    st["x"].append(trial.number)
    st["y"].append(float(trial.value))
    st["line"].set_data(st["x"], st["y"])
    st["ax"].relim()
    st["ax"].autoscale_view()

    # update the display in-place
    handle = _display_state.get("progress")
    if handle is not None:
        handle.update(st["fig"])

    import matplotlib.pyplot as plt
    plt.close(st["fig"])  # keep memory low
    

####################################################################################################################


def propose_ranges_from_top(
    csv_path: str,
    top_n: int = 20,
    spread: float = 1.0,          # how many stds around center
    agg: str = "median",          # or "mean"
    cat_top_k: int = 3,
    floors: dict[str, float] | None = None,
    ceils: dict[str, float] | None = None,
):
    """
    Propose parameter search ranges from the top N Optuna trial results in a CSV.

    Behavior:
      - Loads `csv_path`, sorts by `avg_daily_pnl` if present, and keeps the top_n rows.
      - For each column (except skipped keys), returns either:
          * a categorical list of the top `cat_top_k` values (for object dtype), or
          * a numeric range (lo, hi) computed as center ± spread * std,
            where center is the median or mean depending on `agg`.
      - Integer columns are rounded to integer bounds.
    """

    df = pd.read_csv(csv_path)
    if "avg_daily_pnl" in df:
        df = df.sort_values("avg_daily_pnl", ascending=False)
    df = df.head(top_n)

    skip = {"trial", "avg_daily_pnl", "col_signal", "sign_thresh"}
    ranges = {}

    for col in df.columns:
        if col in skip:
            continue

        s = df[col].dropna()
        if s.empty:
            continue

        # categorical/object
        if s.dtype == object:
            top_cats = s.value_counts().head(cat_top_k).index.tolist()
            ranges[col] = {"categorical": top_cats}
            continue

        # numeric
        x = s.astype(float)
        is_int_dtype = np.issubdtype(s.dtype, np.integer)  # only true int dtype counts as int

        center = x.median() if agg == "median" else x.mean()
        spread_val = x.std(ddof=0) * spread
        lo, hi = center - spread_val, center + spread_val

        # only round if original dtype was integer
        if is_int_dtype:
            lo, hi = int(np.floor(lo)), int(np.ceil(hi))

        ranges[col] = (lo, hi)

    return ranges