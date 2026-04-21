"""
Plot Double Descent Learning Curves (Figure 7 style from Hernandez et al.)

W&B project: jchud-stanford-university/hernandez-replication

Sweep structure:
  - Models: auto-detected from run configs (e.g. 48M, 63M, 93M)
  - data_config.num_repeats: 1, 3, 13, 51, 193, 719, 2682, 10000
  - data_config.repetition_budget: auto-detected
  - data_config.direction: "top" or "bot"
  - trainer_config.overtrain_multiplier: auto-detected

Usage:
    # Quick: plot all models (finished + running) with output in plots/ dir:
    python analyze_wandb/plot_double_descent.py --output-dir analyze_wandb/plots

    # List all runs with auto-detected config:
    python analyze_wandb/plot_double_descent.py --list-runs

    # Filter to specific sweeps:
    python analyze_wandb/plot_double_descent.py --sweeps urnekydr 3p1olv16

    # Plot only "bot" direction:
    python analyze_wandb/plot_double_descent.py --direction bot

    # Include crashed runs (partial data):
    python analyze_wandb/plot_double_descent.py --include-crashed
"""

import argparse
import re
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    import wandb
except ImportError:
    raise ImportError("Install wandb: pip install wandb")


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_ENTITY = "jchud-stanford-university"
DEFAULT_PROJECT = "hernandez-replication"

TEST_LOSS_KEY = "eval/loss"
TOKENS_KEY = "train/num_input_tokens_seen"
TRAIN_LOSS_KEY = "train/loss"


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_runs(entity, project, required_tags=None, name_contains=None,
               run_ids=None, sweep_ids=None, include_states=("finished",)):
    """Fetch relevant runs from W&B, optionally filtered by sweep IDs."""
    api = wandb.Api()

    if run_ids:
        runs = []
        for rid in run_ids:
            try:
                run = api.run(f"{entity}/{project}/{rid}")
                runs.append(run)
            except Exception as e:
                print(f"  WARNING: Could not fetch run {rid}: {e}")
        return runs

    filters = {}
    if required_tags:
        filters["tags"] = {"$all": required_tags}

    all_runs = api.runs(f"{entity}/{project}", filters=filters or None)

    filtered = []
    for run in all_runs:
        if run.state not in include_states:
            continue
        if name_contains and name_contains not in run.name:
            continue
        if sweep_ids:
            run_sweep = run.sweep.id if run.sweep else None
            if run_sweep not in sweep_ids:
                continue
        filtered.append(run)

    n_by_state = defaultdict(int)
    for r in filtered:
        n_by_state[r.state] += 1
    state_summary = ", ".join(f"{s}: {c}" for s, c in sorted(n_by_state.items()))
    print(f"Found {len(filtered)} matching runs ({state_summary})")
    return filtered


def _parse_args_config(run):
    """Parse data_config/model_config/trainer_config from run.metadata['args'] strings."""
    import ast
    result = {}
    args = (run.metadata or {}).get("args", [])
    for arg in args:
        for key in ("data_config", "model_config", "trainer_config"):
            prefix = f"--{key}="
            if arg.startswith(prefix):
                try:
                    result[key] = ast.literal_eval(arg[len(prefix):])
                except Exception:
                    pass
    return result


def get_run_info(run):
    """Extract metadata from run config."""
    info = {
        "num_repeats": None,
        "repetition_budget": None,
        "model_params": None,
        "model_name": "unknown",
        "overtrain_multiplier": None,
        "direction": None,
        "sweep_id": run.sweep.id if run.sweep else None,
    }

    # Try run.config first, then fall back to metadata args
    data_config = run.config.get("data_config", {})
    model_config = run.config.get("model_config", {})
    trainer_config = run.config.get("trainer_config", {})

    if not data_config and not model_config:
        parsed = _parse_args_config(run)
        data_config = parsed.get("data_config", {})
        model_config = parsed.get("model_config", {})
        trainer_config = parsed.get("trainer_config", {})

    if isinstance(data_config, dict):
        info["num_repeats"] = data_config.get("num_repeats")
        info["repetition_budget"] = data_config.get("repetition_budget")
        info["direction"] = data_config.get("direction")

    info["model_params"] = run.config.get("model/num_parameters")
    if isinstance(model_config, dict):
        info["model_name"] = model_config.get("model_name", "unknown")
        if info["model_params"] is None:
            info["model_params"] = model_config.get("num_parameters")
        # Parse param count from model name like "Qwen3/Qwen3-93M" or "gpt2-48M"
        if info["model_params"] is None and info["model_name"] != "unknown":
            m = re.search(r'(\d+)M', str(info["model_name"]))
            if m:
                info["model_params"] = int(m.group(1)) * 1_000_000

    if isinstance(trainer_config, dict):
        info["overtrain_multiplier"] = trainer_config.get("overtrain_multiplier")

    # Fallback: parse from hub_model_id
    hub_id = run.config.get("hub_model_id", "")
    if info["num_repeats"] is None:
        match = re.search(r'_nr_(\d+)', str(hub_id))
        if match:
            info["num_repeats"] = int(match.group(1))
    if info["repetition_budget"] is None:
        match = re.search(r'_rb_([\d.]+)', str(hub_id))
        if match:
            info["repetition_budget"] = float(match.group(1))

    return info


def fetch_run_history(run, samples=1500):
    """Fetch eval/loss and tokens for a run using scan_history (no pandas needed)."""
    keys = [TEST_LOSS_KEY, TOKENS_KEY]

    x_list, y_list = [], []
    last_token = None
    for row in run.scan_history(keys=keys):
        loss = row.get(TEST_LOSS_KEY)
        tokens = row.get(TOKENS_KEY)
        if tokens is not None:
            last_token = tokens
        if loss is not None and last_token is not None:
            x_list.append(float(last_token))
            y_list.append(float(loss))

    if not x_list:
        print(f"    WARNING: No eval data for run '{run.name}'")
        return None, None

    x_arr, y_arr = np.array(x_list), np.array(y_list)

    # Downsample if too many points (keep first, last, and evenly spaced)
    if len(x_arr) > samples:
        indices = np.linspace(0, len(x_arr) - 1, samples, dtype=int)
        x_arr, y_arr = x_arr[indices], y_arr[indices]

    return x_arr, y_arr


# =============================================================================
# HELPERS
# =============================================================================

def format_number(n):
    if n is None:
        return "?"
    n = float(n)
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    elif n >= 1_000:
        return f"{n/1e3:.0f}k" if n % 1000 == 0 else f"{n:,.0f}"
    elif n == int(n):
        return f"{int(n)}"
    else:
        return f"{n:.1f}"


def _format_token_axis(ax):
    import matplotlib.ticker as ticker

    def token_formatter(x, pos):
        if x >= 1e12:
            return f'{x/1e12:.1f}T'
        elif x >= 1e9:
            return f'{x/1e9:.1f}B' if x % 1e9 != 0 else f'{x/1e9:.0f}B'
        elif x >= 1e6:
            return f'{x/1e6:.0f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}k'
        else:
            return f'{x:.0f}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(token_formatter))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50))
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(token_formatter))
    ax.tick_params(axis='x', which='minor', labelsize=8, rotation=45)
    ax.tick_params(axis='x', which='major', labelsize=10, rotation=45)


def timestamp_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_timestamp(fig):
    fig.text(0.99, 0.01, f"Generated: {timestamp_str()}", fontsize=8,
             color='gray', ha='right', va='bottom', style='italic')


def make_model_label(params, model_name=None):
    if params:
        label = f"{params/1e6:.0f}M Parameter Model"
        if model_name and model_name != "unknown":
            label += f" ({model_name})"
        return label
    if model_name and model_name != "unknown":
        return model_name
    return "Unknown Model"


# =============================================================================
# PLOTTING (identical logic to original, accepts model_label as arg)
# =============================================================================

def plot_fig7(data_by_nr, direction, model_label, rep_budget_label, output_path,
              running_nrs=None):
    running_nrs = running_nrs or set()
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    sorted_nrs = sorted(data_by_nr.keys())
    n = len(sorted_nrs)
    cmap = plt.cm.plasma
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for i, nr in enumerate(sorted_nrs):
        x_vals, test_loss = data_by_nr[nr]
        is_running = nr in running_nrs
        ls = '--' if is_running else '-'
        lbl = f"{format_number(nr)}" + (" [running]" if is_running else "")
        ax.plot(x_vals, test_loss, color=colors[i], linewidth=1.8,
                label=lbl, alpha=0.6 if is_running else 0.9,
                linestyle=ls, marker='o', markersize=2)

    ax.set_xscale('log')
    _format_token_axis(ax)
    ax.set_xlabel('Tokens Trained', fontsize=14)
    ax.set_ylabel('Test Loss (eval/loss)', fontsize=14)
    ax.set_title(
        f'Learning Curves — {model_label}\n'
        f'{rep_budget_label} Repeated Data — Split "{direction}"',
        fontsize=15)
    ax.legend(title="num_repeats", fontsize=9, title_fontsize=10,
              loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    add_timestamp(fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_fig7_zoomed(data_by_nr, direction, model_label, rep_budget_label, output_path,
                     running_nrs=None):
    running_nrs = running_nrs or set()
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    sorted_nrs = sorted(data_by_nr.keys())
    n = len(sorted_nrs)
    cmap = plt.cm.plasma
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    all_max_x = max(x[-1] for x, _ in data_by_nr.values() if len(x) > 0)
    zoom_start = all_max_x * 0.3
    all_y_in_zoom = []

    for i, nr in enumerate(sorted_nrs):
        x_vals, test_loss = data_by_nr[nr]
        mask = x_vals >= zoom_start
        if mask.any():
            all_y_in_zoom.extend(test_loss[mask])
        is_running = nr in running_nrs
        ls = '--' if is_running else '-'
        lbl = f"{format_number(nr)}" + (" [running]" if is_running else "")
        ax.plot(x_vals, test_loss, color=colors[i], linewidth=2.2,
                label=lbl, alpha=0.6 if is_running else 0.9,
                linestyle=ls, marker='o', markersize=3.5)

    if all_y_in_zoom:
        y_min, y_max = min(all_y_in_zoom), max(all_y_in_zoom)
        y_pad = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(left=zoom_start)
    ax.set_xscale('log')
    _format_token_axis(ax)
    ax.set_xlabel('Tokens Trained', fontsize=14)
    ax.set_ylabel('Test Loss (eval/loss)', fontsize=14)
    ax.set_title(
        f'Learning Curves (Zoomed) — {model_label}\n'
        f'{rep_budget_label} Repeated Data — Split "{direction}"\n'
        f'Effect is small at Chinchilla-optimal compute (CE=1)',
        fontsize=14)
    ax.legend(title="num_repeats", fontsize=9, title_fontsize=10,
              loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    add_timestamp(fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_fig2(data_by_nr, direction, model_label, rep_budget_label, output_path,
              running_nrs=None):
    running_nrs = running_nrs or set()
    nrs, final_losses, is_running_list = [], [], []
    for nr in sorted(data_by_nr.keys()):
        _, test_loss = data_by_nr[nr]
        if len(test_loss) > 0:
            nrs.append(nr)
            final_losses.append(test_loss[-1])
            is_running_list.append(nr in running_nrs)

    fig, ax = plt.subplots(figsize=(10, 7))
    # Plot finished points as solid, running as open/dashed
    fin_nrs = [n for n, r in zip(nrs, is_running_list) if not r]
    fin_loss = [l for l, r in zip(final_losses, is_running_list) if not r]
    run_nrs = [n for n, r in zip(nrs, is_running_list) if r]
    run_loss = [l for l, r in zip(final_losses, is_running_list) if r]
    ax.plot(nrs, final_losses, '-', color='steelblue', linewidth=2.5, alpha=0.5)
    if fin_nrs:
        ax.scatter(fin_nrs, fin_loss, color='steelblue', s=100, zorder=5, label='finished')
    if run_nrs:
        ax.scatter(run_nrs, run_loss, color='steelblue', s=100, zorder=5,
                   facecolors='none', edgecolors='steelblue', linewidths=2, label='running')
    for nr, loss, running in zip(nrs, final_losses, is_running_list):
        suffix = " *" if running else ""
        ax.annotate(f'{loss:.3f}{suffix}', (nr, loss), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('num_repeats (epochs on repeated tokens)', fontsize=14)
    ax.set_ylabel('Test Loss (final)', fontsize=14)
    ax.set_title(
        f'Final Test Loss vs Repetition Frequency\n'
        f'{model_label}, {rep_budget_label} Repeated — Split "{direction}"',
        fontsize=15)
    if run_nrs:
        ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=11)
    add_timestamp(fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_fig2_repeated_tokens(data_by_nr, direction, model_label, rep_budget_label,
                               rep_budget, output_path, running_nrs=None):
    running_nrs = running_nrs or set()
    nrs, final_losses, total_tokens_list, is_running_list = [], [], [], []
    for nr in sorted(data_by_nr.keys()):
        x_vals, test_loss = data_by_nr[nr]
        if len(test_loss) > 0 and len(x_vals) > 0:
            nrs.append(nr)
            final_losses.append(test_loss[-1])
            total_tokens_list.append(x_vals[-1])
            is_running_list.append(nr in running_nrs)

    total_tokens = np.median(total_tokens_list)
    repeated_unique_tokens = [(rep_budget * total_tokens) / nr for nr in nrs]

    fig, ax = plt.subplots(figsize=(10, 7))
    # Line connecting all points
    ax.plot(repeated_unique_tokens, final_losses, '-', color='steelblue',
            linewidth=2.5, alpha=0.5)
    # Finished = solid, running = open
    for tokens, loss, running in zip(repeated_unique_tokens, final_losses, is_running_list):
        if running:
            ax.scatter([tokens], [loss], s=100, facecolors='none',
                       edgecolors='steelblue', linewidths=2, zorder=5)
        else:
            ax.scatter([tokens], [loss], s=100, color='steelblue', zorder=5)

    for tokens, loss, nr, running in zip(repeated_unique_tokens, final_losses, nrs, is_running_list):
        tok_label = (f"{tokens/1e6:.1f}M" if tokens >= 1e6
                     else f"{tokens/1e3:.0f}k" if tokens >= 1e3
                     else f"{tokens:.0f}")
        suffix = " *" if running else ""
        ax.annotate(f'{tok_label}\n({nr}x){suffix}', (tokens, loss),
                    textcoords="offset points", xytext=(0, 14),
                    ha='center', fontsize=8, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Unique Tokens Repeated (fewer = more repeats)', fontsize=13)
    ax.set_ylabel('Test Loss (final)', fontsize=14)
    ax.set_title(
        f'Test Loss vs Size of Repeated Subset\n'
        f'{model_label}, {rep_budget_label} Repeated — Split "{direction}"\n'
        f'(Total tokens: {total_tokens/1e9:.2f}B, compute held constant)',
        fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=11)
    ax.invert_xaxis()
    add_timestamp(fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_combined_fig2(data_top, data_bot, model_label, rep_budget_label, output_path,
                       running_keys=None):
    running_keys = running_keys or set()
    fig, ax = plt.subplots(figsize=(11, 7))
    for data, direction, color, marker in [
        (data_top, "top", "tab:blue", "o"),
        (data_bot, "bot", "tab:red", "s"),
    ]:
        if not data:
            continue
        nrs, final_losses, is_running_list = [], [], []
        for nr in sorted(data.keys()):
            _, test_loss = data[nr]
            if len(test_loss) > 0:
                nrs.append(nr)
                final_losses.append(test_loss[-1])
                is_running_list.append((direction, nr) in running_keys)
        ax.plot(nrs, final_losses, '-', color=color, linewidth=2.5, alpha=0.5)
        fin_n = [n for n, r in zip(nrs, is_running_list) if not r]
        fin_l = [l for l, r in zip(final_losses, is_running_list) if not r]
        run_n = [n for n, r in zip(nrs, is_running_list) if r]
        run_l = [l for l, r in zip(final_losses, is_running_list) if r]
        if fin_n:
            ax.scatter(fin_n, fin_l, color=color, marker=marker, s=100, zorder=5,
                       label=f"split: {direction}")
        if run_n:
            ax.scatter(run_n, run_l, facecolors='none', edgecolors=color, marker=marker,
                       s=100, linewidths=2, zorder=5, label=f"split: {direction} [running]")

    ax.set_xscale('log')
    ax.set_xlabel('num_repeats (epochs on repeated tokens)', fontsize=14)
    ax.set_ylabel('Test Loss (final)', fontsize=14)
    ax.set_title(
        f'Final Test Loss vs Repetition — Top vs Bot Split\n'
        f'{model_label}, {rep_budget_label} Repeated Data',
        fontsize=15)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=11)
    add_timestamp(fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_all_models_fig2(all_models, output_path):
    """Plot final test loss vs num_repeats — ALL model sizes, both directions, one chart.
    Solid lines = top, dashed lines = bot. Color = model size."""
    fig, ax = plt.subplots(figsize=(14, 8))

    numeric_keys = sorted([k for k in all_models if isinstance(k, (int, float))])
    str_keys = sorted([k for k in all_models if isinstance(k, str)])
    sorted_keys = numeric_keys + str_keys

    n = len(sorted_keys)
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '<', '>'] * 3

    dir_styles = {"top": ("-", "o"), "bot": ("--", "^")}

    for i, param_key in enumerate(sorted_keys):
        m = all_models[param_key]
        running_keys = m["running_keys"]
        short_label = f"{param_key}M" if isinstance(param_key, (int, float)) else (m["model_label"] or f"{param_key}")

        for direction in ["top", "bot"]:
            data = m[direction]
            if not data:
                continue
            ls, marker = dir_styles[direction]

            nrs, final_losses, is_running_list = [], [], []
            for nr in sorted(data.keys()):
                _, test_loss = data[nr]
                if len(test_loss) > 0:
                    nrs.append(nr)
                    final_losses.append(test_loss[-1])
                    is_running_list.append((direction, nr) in running_keys)

            if not nrs:
                continue

            ax.plot(nrs, final_losses, linestyle=ls, color=colors[i], linewidth=2, alpha=0.7,
                    marker=marker, markersize=7, label=f"{short_label} ({direction})")

            # Mark running points with open markers
            run_nrs = [rn for rn, r in zip(nrs, is_running_list) if r]
            run_losses = [rl for rl, r in zip(final_losses, is_running_list) if r]
            if run_nrs:
                ax.scatter(run_nrs, run_losses, facecolors='none', edgecolors=colors[i],
                           marker=marker, s=120, linewidths=2.5, zorder=6)

    ax.set_xscale('log')
    ax.set_xlabel('num_repeats (epochs on repeated tokens)', fontsize=14)
    ax.set_ylabel('Test Loss (final)', fontsize=14)
    ax.set_title(
        f'Final Test Loss vs Repetition — All Model Sizes\n'
        f'Solid = top split, Dashed = bot split | 10% Repeated Data',
        fontsize=15)
    ax.legend(title="Model (split)", fontsize=9, title_fontsize=10,
              loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=11)
    add_timestamp(fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_all_models_fig7(all_models, output_path):
    """Plot learning curves — ALL models, both directions, one chart.
    Color = model size, solid = top, dashed = bot, alpha varies by num_repeats."""
    fig, ax = plt.subplots(figsize=(16, 9))

    numeric_keys = sorted([k for k in all_models if isinstance(k, (int, float))])
    str_keys = sorted([k for k in all_models if isinstance(k, str)])
    sorted_keys = numeric_keys + str_keys

    n = len(sorted_keys)
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # Collect all unique num_repeats across all models and directions
    all_nrs = set()
    for param_key in sorted_keys:
        for direction in ["top", "bot"]:
            all_nrs.update(all_models[param_key][direction].keys())
    sorted_all_nrs = sorted(all_nrs)
    # Map num_repeats to alpha (more repeats = more opaque)
    nr_to_alpha = {nr: 0.4 + 0.5 * (j / max(len(sorted_all_nrs) - 1, 1))
                   for j, nr in enumerate(sorted_all_nrs)}

    dir_styles = {"top": "-", "bot": "--"}

    model_legend_added = set()
    for i, param_key in enumerate(sorted_keys):
        m = all_models[param_key]
        short_label = f"{param_key}M" if isinstance(param_key, (int, float)) else (m["model_label"] or f"{param_key}")

        for direction in ["top", "bot"]:
            data = m[direction]
            if not data:
                continue

            for nr in sorted(data.keys()):
                x_vals, test_loss = data[nr]
                if len(x_vals) == 0:
                    continue
                label = f"{short_label} ({direction})" if (param_key, direction) not in model_legend_added else None
                model_legend_added.add((param_key, direction))
                ax.plot(x_vals, test_loss, color=colors[i], linewidth=1.5,
                        linestyle=dir_styles[direction],
                        alpha=nr_to_alpha.get(nr, 0.7), label=label)

    ax.set_xscale('log')
    _format_token_axis(ax)
    ax.set_xlabel('Tokens Trained', fontsize=14)
    ax.set_ylabel('Test Loss (eval/loss)', fontsize=14)
    ax.set_title(
        f'Learning Curves — All Model Sizes\n'
        f'Solid = top, Dashed = bot | Alpha varies by num_repeats',
        fontsize=15)

    # Model/direction legend
    legend1 = ax.legend(title="Model (split)", fontsize=8, title_fontsize=9,
                        loc='upper right', framealpha=0.9, ncol=2)
    ax.add_artist(legend1)

    # Second legend: num_repeats alpha scale
    from matplotlib.lines import Line2D
    nr_handles = [Line2D([0], [0], color='gray', linewidth=2,
                         alpha=nr_to_alpha[nr],
                         label=f'nr={format_number(nr)}') for nr in sorted_all_nrs]
    ax.legend(handles=nr_handles, title="num_repeats", fontsize=8, title_fontsize=9,
              loc='lower left', framealpha=0.9, ncol=2 if len(sorted_all_nrs) > 4 else 1)
    ax.add_artist(legend1)

    ax.grid(True, alpha=0.3, which='both')
    add_timestamp(fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot double descent from W&B (Hernandez replication) — multi-model")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--direction", default=None, choices=["top", "bot"])
    parser.add_argument("--sweeps", nargs="*", default=None,
                        help="Filter to specific sweep IDs (e.g. --sweeps urnekydr 3p1olv16)")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--name-contains", default=None)
    parser.add_argument("--run-ids", nargs="*", default=None)
    parser.add_argument("--include-crashed", action="store_true")
    parser.add_argument("--exclude-running", action="store_true",
                        help="Exclude currently running (incomplete) runs")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--output-dir", default="analyze_wandb/plots", help="Directory for output plots")
    args = parser.parse_args()

    states = ["finished", "running"]
    if args.exclude_running:
        states.remove("running")
    if args.include_crashed:
        states.append("crashed")
    states = tuple(states)

    print(f"Fetching runs from {args.entity}/{args.project}...")
    runs = fetch_runs(
        args.entity, args.project,
        required_tags=args.tags if args.tags else None,
        name_contains=args.name_contains,
        run_ids=args.run_ids,
        sweep_ids=args.sweeps,
        include_states=states,
    )

    if not runs:
        print("No runs found!")
        return

    # === LIST MODE ===
    if args.list_runs:
        print(f"\n{'='*120}")
        print(f"{'Run Name':<28} {'State':<10} {'Sweep':<12} {'Model':<20} {'Params':<10} "
              f"{'num_repeats':<13} {'rep_budget':<12} {'OT mult':<10} {'direction'}")
        print(f"{'='*120}")
        for run in runs:
            info = get_run_info(run)
            print(f"{run.name:<28} {run.state:<10} "
                  f"{str(info['sweep_id'] or '?'):<12} "
                  f"{str(info['model_name']):<20} "
                  f"{str(info['model_params'] or '?'):<10} "
                  f"{str(info['num_repeats'] or '?'):<13} "
                  f"{str(info['repetition_budget'] or '?'):<12} "
                  f"{str(info['overtrain_multiplier'] or '?'):<10} "
                  f"{str(info['direction'] or '?')}")

        # Print config of one run per sweep for inspection
        seen_sweeps = set()
        for run in runs:
            sweep_id = run.sweep.id if run.sweep else "no_sweep"
            if sweep_id not in seen_sweeps:
                seen_sweeps.add(sweep_id)
                print(f"\n--- Config sample: sweep={sweep_id}, run='{run.name}' ---")
                for k, v in sorted(run.config.items()):
                    print(f"  {k}: {v}")
                print(f"\n--- Available metric keys ---")
                hist = run.history(samples=1)
                if not hist.empty:
                    for col in sorted(hist.columns):
                        print(f"  {col}")
        return

    # === FETCH AND ORGANIZE DATA ===
    # Group by (param_count_bucket, direction) -> {num_repeats: (tokens, loss)}
    # param_bucket key: rounded to nearest 1M for grouping
    #   {param_bucket: {"model_label": str, "rep_budget": float, "top": {...}, "bot": {...}}}
    models = defaultdict(lambda: {
        "model_label": None,
        "rep_budget": None,
        "sweep_id": None,
        "top": {},
        "bot": {},
        "running_keys": set(),  # tracks (direction, nr) pairs that are still running
    })

    for run in runs:
        info = get_run_info(run)
        nr = info["num_repeats"]
        direction = info["direction"]
        params = info["model_params"]

        if nr is None or direction is None:
            print(f"  SKIP '{run.name}': missing num_repeats ({nr}) or direction ({direction})")
            continue

        if args.direction and direction != args.direction:
            continue

        # Build a stable key for this model size
        if params:
            # Round to nearest million for grouping (handles minor float differences)
            param_key = round(params / 1e6)
        else:
            # Fall back to sweep ID if no params found
            param_key = f"sweep_{info['sweep_id'] or 'unknown'}"

        m = models[param_key]
        if m["model_label"] is None:
            m["model_label"] = make_model_label(params, info["model_name"])
            m["rep_budget"] = info["repetition_budget"]
            m["sweep_id"] = info["sweep_id"]
        # Update rep_budget if not set yet
        if m["rep_budget"] is None and info["repetition_budget"] is not None:
            m["rep_budget"] = info["repetition_budget"]

        print(f"  Fetching '{run.name}' "
              f"(model={param_key}M, sweep={info['sweep_id']}, "
              f"nr={nr}, dir={direction})...")
        x_vals, test_loss = fetch_run_history(run)

        if x_vals is None or len(x_vals) == 0:
            print(f"    No eval data — skipping")
            continue

        # Keep longer run if duplicate nr exists
        if nr in m[direction]:
            existing_len = len(m[direction][nr][0])
            if len(x_vals) <= existing_len:
                print(f"    Duplicate nr={nr} dir={direction}, keeping longer ({existing_len} pts)")
                continue
            else:
                print(f"    Duplicate nr={nr} dir={direction}, replacing with longer ({len(x_vals)} pts)")

        m[direction][nr] = (x_vals, test_loss)
        if run.state == "running":
            m["running_keys"].add((direction, nr))
        print(f"    {len(test_loss)} eval pts, "
              f"tokens: {x_vals[0]:.0f} — {x_vals[-1]:.0f}"
              f"{' [RUNNING]' if run.state == 'running' else ''}")

    if not models:
        print("No usable data found!")
        return

    print(f"\nFound {len(models)} distinct model size(s): {sorted(models.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)

    # === GENERATE PLOTS — one set per model ===
    for param_key, m in sorted(models.items()):
        model_label = m["model_label"] or f"{param_key}M Model"
        rep_budget = m["rep_budget"] or 0.1
        rep_budget_label = f"{int(rep_budget * 100)}%"
        # Safe filename prefix (no spaces)
        safe_key = str(param_key).replace(" ", "_")
        odir = args.output_dir

        print(f"\n{'='*60}")
        print(f"Model: {model_label}  (key={param_key}, sweep={m['sweep_id']})")
        print(f"{'='*60}")

        running_keys = m["running_keys"]

        for direction in ["top", "bot"]:
            if args.direction and direction != args.direction:
                continue
            data = m[direction]
            if not data:
                print(f"  No data for direction='{direction}' — skipping")
                continue

            # running nr values for this direction
            dir_running_nrs = {nr for (d, nr) in running_keys if d == direction}

            print(f"\n  direction='{direction}': "
                  f"{len(data)} num_repeats values: {sorted(data.keys())}"
                  f"{f' (running: {sorted(dir_running_nrs)})' if dir_running_nrs else ''}")

            plot_fig7(data, direction, model_label, rep_budget_label,
                      f"{odir}/fig7_{safe_key}_{direction}.png",
                      running_nrs=dir_running_nrs)

            plot_fig7_zoomed(data, direction, model_label, rep_budget_label,
                             f"{odir}/fig7_{safe_key}_{direction}_zoomed.png",
                             running_nrs=dir_running_nrs)

            if len(data) >= 2:
                plot_fig2(data, direction, model_label, rep_budget_label,
                          f"{odir}/fig2_{safe_key}_{direction}.png",
                          running_nrs=dir_running_nrs)

                plot_fig2_repeated_tokens(
                    data, direction, model_label, rep_budget_label, rep_budget,
                    f"{odir}/fig2_{safe_key}_{direction}_tokens.png",
                    running_nrs=dir_running_nrs)

        # Combined top vs bot
        if m["top"] and m["bot"] and not args.direction:
            print(f"\n  Plotting combined top vs bot for {model_label}")
            plot_combined_fig2(m["top"], m["bot"], model_label, rep_budget_label,
                               f"{odir}/fig2_{safe_key}_combined.png",
                               running_keys=running_keys)

    # === ALL-MODELS COMBINED CHARTS (both directions in one chart) ===
    if len(models) >= 2:
        print(f"\n  Plotting all-models combined charts (top + bot)")
        plot_all_models_fig2(models, f"{args.output_dir}/fig2_all_models.png")
        plot_all_models_fig7(models, f"{args.output_dir}/fig7_all_models.png")

    print("\nDone!")


if __name__ == "__main__":
    main()