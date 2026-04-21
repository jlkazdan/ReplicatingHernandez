"""
Combined status chart: Test Loss vs Size of Repeated Subset — all runs,
including running ones with elapsed time and dates.

Same style as hernandez_all_runs.png but with:
  - Running runs shown (open markers, dashed connecting line)
  - Duration annotation on each point (elapsed for running, total for finished)
  - Start date on each point

Usage:
    python analyze_wandb/plot_combined_status.py
    python analyze_wandb/plot_combined_status.py --output analyze_wandb/plots/combined_status.png
    python analyze_wandb/plot_combined_status.py --include-crashed
"""

import os
import re
import argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    raise ImportError("pip install wandb")

ENTITY = "jchud-stanford-university"
PROJECT = "hernandez-replication"
TEST_LOSS_KEY = "eval/loss"
TOKENS_KEY = "train/num_input_tokens_seen"

COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<"]


def human_readable(n):
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}k"
    return str(int(n))


def duration_str(seconds):
    if seconds is None:
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, _ = divmod(rem, 60)
    if h >= 24:
        return f"{h // 24}d{h % 24}h"
    return f"{h}h{m}m"


def extract_run_info(run):
    info = {}
    data_config = run.config.get("data_config", {})
    info["num_repeats"] = data_config.get("num_repeats")
    info["repetition_budget"] = data_config.get("repetition_budget", 0.1)
    info["direction"] = data_config.get("direction", "unknown")

    info["model_params"] = run.config.get("model/num_parameters")
    model_config = run.config.get("model_config", {})
    info["model_name"] = model_config.get("model_name", "unknown")

    trainer_config = run.config.get("trainer_config", {})
    info["overtrain_multiplier"] = trainer_config.get("overtrain_multiplier", 1.0)

    hub_id = run.config.get("hub_model_id", "")
    if info["num_repeats"] is None:
        m = re.search(r'_nr_(\d+)', hub_id)
        if m:
            info["num_repeats"] = int(m.group(1))
    if info["direction"] == "unknown":
        m = re.search(r'_dir_(\w+)', hub_id)
        if m:
            info["direction"] = m.group(1)
    if info["model_name"] == "unknown":
        m = re.search(r'(Qwen3-\d+M)', hub_id)
        if m:
            info["model_name"] = m.group(1)

    return info


def model_size_label(info):
    name = info.get("model_name", "")
    m = re.search(r'(\d+)M', name)
    if m:
        return f"{m.group(1)}M"
    params = info.get("model_params")
    if params:
        return f"{round(params / 1e6)}M"
    return "unknown"


def get_final_eval_loss(run):
    summary_loss = run.summary.get("eval/loss")
    if summary_loss is not None:
        return summary_loss
    try:
        history = run.history(keys=[TEST_LOSS_KEY], samples=2000)
        if history.empty:
            return None
        losses = history[TEST_LOSS_KEY].dropna()
        return float(losses.iloc[-1]) if not losses.empty else None
    except Exception:
        return None


def get_total_tokens(run):
    summary_tokens = run.summary.get("train/num_input_tokens_seen")
    if summary_tokens is not None:
        return int(summary_tokens)
    try:
        history = run.history(keys=[TOKENS_KEY], samples=2000)
        if history.empty:
            return None
        tokens = history[TOKENS_KEY].dropna()
        return int(tokens.iloc[-1]) if not tokens.empty else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Combined status chart — all runs, one plot")
    parser.add_argument("--output", default="analyze_wandb/plots/combined_status.png")
    parser.add_argument("--include-crashed", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    now = datetime.now(timezone.utc)

    states = ["finished", "running"]
    if args.include_crashed:
        states += ["crashed", "failed"]

    print(f"Fetching runs from {ENTITY}/{PROJECT}...")
    all_runs = list(api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"state": {"$in": states}},
        per_page=300,
    ))
    print(f"  Found {len(all_runs)} runs")

    # Collect data, grouped by (model_size, overtrain_multiplier)
    groups = defaultdict(list)
    skipped = 0

    for i, run in enumerate(all_runs):
        info = extract_run_info(run)
        nr = info.get("num_repeats")
        if nr is None:
            skipped += 1
            continue

        final_loss = get_final_eval_loss(run)
        if final_loss is None:
            skipped += 1
            continue

        total_tokens = get_total_tokens(run)
        if total_tokens is None:
            params = info.get("model_params")
            if params:
                total_tokens = int(20 * params)
            else:
                skipped += 1
                continue

        rep_budget = info.get("repetition_budget", 0.1)
        unique_tokens = (rep_budget * total_tokens) / nr if nr > 0 else total_tokens

        msize = model_size_label(info)
        ot = info.get("overtrain_multiplier", 1.0)

        # Compute duration and date
        start = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
        date_str = start.strftime("%b %d")

        if run.state == "finished":
            runtime_sec = (run.summary or {}).get("_wandb", {}).get("runtime")
            dur = duration_str(runtime_sec)
        elif run.state == "running":
            elapsed = (now - start).total_seconds()
            dur = duration_str(elapsed)
        else:
            runtime_sec = (run.summary or {}).get("_wandb", {}).get("runtime")
            dur = duration_str(runtime_sec) if runtime_sec else "?"

        key = (msize, ot)
        groups[key].append({
            "unique_tokens": unique_tokens,
            "num_repeats": nr,
            "final_loss": final_loss,
            "total_tokens": total_tokens,
            "rep_budget": rep_budget,
            "run_name": run.name,
            "state": run.state,
            "date_str": date_str,
            "duration": dur,
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(all_runs)} runs...")

    total_pts = sum(len(v) for v in groups.values())
    print(f"\n  Collected {total_pts} runs across {len(groups)} groups")
    if skipped:
        print(f"  Skipped {skipped} runs (missing config or metrics)")

    if not groups:
        print("No data to plot!")
        return

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 10))

    sorted_groups = sorted(groups.items())
    n_finished = sum(1 for pts in groups.values() for p in pts if p["state"] == "finished")
    n_running = sum(1 for pts in groups.values() for p in pts if p["state"] == "running")

    for idx, ((msize, ot), pts) in enumerate(sorted_groups):
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        label = f"{msize} (OT={ot})"

        # Group by num_repeats and average (top+bot)
        nr_groups = defaultdict(list)
        for p in pts:
            nr_groups[p["num_repeats"]].append(p)

        # Individual points (transparent)
        all_xs = [p["unique_tokens"] for p in pts]
        all_ys = [p["final_loss"] for p in pts]
        # Use open markers for running, filled for finished
        fin_xs = [p["unique_tokens"] for p in pts if p["state"] == "finished"]
        fin_ys = [p["final_loss"] for p in pts if p["state"] == "finished"]
        run_xs = [p["unique_tokens"] for p in pts if p["state"] == "running"]
        run_ys = [p["final_loss"] for p in pts if p["state"] == "running"]

        if fin_xs:
            ax.scatter(fin_xs, fin_ys, color=color, marker=marker, s=60, alpha=0.35, zorder=3)
        if run_xs:
            ax.scatter(run_xs, run_ys, facecolors="none", edgecolors=color, marker=marker,
                       s=80, linewidths=1.5, alpha=0.5, zorder=3)

        # Averaged line
        avg_xs, avg_ys, avg_states, avg_dates, avg_durs = [], [], [], [], []
        for nr in sorted(nr_groups.keys()):
            group = nr_groups[nr]
            avg_xs.append(np.mean([p["unique_tokens"] for p in group]))
            avg_ys.append(np.mean([p["final_loss"] for p in group]))
            # If any run in this nr group is running, mark it
            any_running = any(p["state"] == "running" for p in group)
            avg_states.append("running" if any_running else "finished")
            # Pick most recent date and longest duration for annotation
            avg_dates.append(group[-1]["date_str"])
            avg_durs.append(group[-1]["duration"])

        order = np.argsort(avg_xs)[::-1]
        avg_xs = [avg_xs[i] for i in order]
        avg_ys = [avg_ys[i] for i in order]
        avg_states = [avg_states[i] for i in order]
        avg_dates = [avg_dates[i] for i in order]
        avg_durs = [avg_durs[i] for i in order]
        ordered_nrs = [sorted(nr_groups.keys())[i] for i in order]

        # Split line into finished segments (solid) and segments touching running points (dashed)
        ax.plot(avg_xs, avg_ys, "-", color=color, linewidth=2.5, markersize=10,
                marker=marker, label=label, zorder=4)

        # Mark running averaged points with open markers on top
        for x, y, st in zip(avg_xs, avg_ys, avg_states):
            if st == "running":
                ax.scatter([x], [y], facecolors="none", edgecolors=color, marker=marker,
                           s=150, linewidths=2.5, zorder=5)

        # Annotations: unique_tokens, (Nx), date, duration
        for x, y, nr, st, date, dur in zip(avg_xs, avg_ys, ordered_nrs, avg_states, avg_dates, avg_durs):
            running_tag = "\n[running]" if st == "running" else ""
            annotation = f"{human_readable(x)}\n({nr}x){running_tag}"
            ax.annotate(
                annotation, (x, y),
                textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=7, color=color, alpha=0.85,
                fontweight="bold" if st == "running" else "normal",
            )

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(3.2, 5.0)
    ax.set_xlabel("Number of Unique Tokens Repeated (fewer = more repeats)", fontsize=14)
    ax.set_ylabel("Test Loss (final)", fontsize=14)
    ax.set_title(
        f"Test Loss vs Size of Repeated Subset — All Runs\n"
        f"Hernandez Replication ({total_pts} runs: {n_finished} finished, {n_running} running)  |  "
        f"{now.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=14,
    )
    ax.legend(fontsize=11, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=11)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
