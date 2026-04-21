"""
List all running and finished runs, then plot a combined
Test Loss vs Size of Repeated Subset chart (same style as plot_2.py).

Usage:
    python scripts/list_runs.py
    python scripts/list_runs.py --output-dir analyze_wandb/plots
"""

import argparse
import re
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import wandb

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


def _plot_group(ax, pts, color, marker, label):
    pts_sorted = sorted(pts, key=lambda p: p["unique_tokens"], reverse=True)

    # Individual points (transparent)
    fin_xs = [p["unique_tokens"] for p in pts_sorted if p["state"] == "finished"]
    fin_ys = [p["final_loss"] for p in pts_sorted if p["state"] == "finished"]
    run_xs = [p["unique_tokens"] for p in pts_sorted if p["state"] == "running"]
    run_ys = [p["final_loss"] for p in pts_sorted if p["state"] == "running"]

    if fin_xs:
        ax.scatter(fin_xs, fin_ys, color=color, marker=marker, s=60, alpha=0.4, zorder=3)
    if run_xs:
        ax.scatter(run_xs, run_ys, facecolors="none", edgecolors=color,
                   marker=marker, s=80, linewidths=1.5, alpha=0.5, zorder=3)

    # Average by num_repeats
    nr_groups = defaultdict(list)
    for p in pts_sorted:
        nr_groups[p["num_repeats"]].append(p)

    avg_xs, avg_ys, avg_running = [], [], []
    for nr in sorted(nr_groups.keys()):
        group = nr_groups[nr]
        avg_xs.append(np.mean([p["unique_tokens"] for p in group]))
        avg_ys.append(np.mean([p["final_loss"] for p in group]))
        avg_running.append(any(p["state"] == "running" for p in group))

    order = np.argsort(avg_xs)[::-1]
    avg_xs = [avg_xs[i] for i in order]
    avg_ys = [avg_ys[i] for i in order]
    avg_running = [avg_running[i] for i in order]
    ordered_nrs = [sorted(nr_groups.keys())[i] for i in order]

    ax.plot(avg_xs, avg_ys, "-", color=color, linewidth=2.5, markersize=10,
            marker=marker, label=label, zorder=4)

    # Open markers on running averaged points
    for x, y, is_run in zip(avg_xs, avg_ys, avg_running):
        if is_run:
            ax.scatter([x], [y], facecolors="none", edgecolors=color,
                       marker=marker, s=150, linewidths=2.5, zorder=5)

    # Labels
    for x, y, nr, is_run in zip(avg_xs, avg_ys, ordered_nrs, avg_running):
        tag = "\n[running]" if is_run else ""
        ax.annotate(
            f"{human_readable(x)}\n({nr}x){tag}", (x, y),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=8, color=color, alpha=0.85,
            fontweight="bold" if is_run else "normal",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="analyze_wandb/plots")
    args = parser.parse_args()

    api = wandb.Api()

    print(f"Fetching runs from {ENTITY}/{PROJECT}...")
    all_runs = list(api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"state": {"$in": ["finished", "running"]}},
        per_page=300,
    ))

    # ---- Terminal listing ----
    running = [r for r in all_runs if r.state == "running"]
    finished = [r for r in all_runs if r.state == "finished"]

    print(f"\n=== RUNNING ({len(running)}) ===")
    for r in sorted(running, key=lambda r: r.name):
        print(f"  {r.name}")

    print(f"\n=== FINISHED ({len(finished)}) ===")
    for r in sorted(finished, key=lambda r: r.name):
        print(f"  {r.name}")

    print(f"\nTotal: {len(running)} running, {len(finished)} finished")

    # ---- Collect data for chart ----
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

        groups[(msize, ot)].append({
            "unique_tokens": unique_tokens,
            "num_repeats": nr,
            "final_loss": final_loss,
            "total_tokens": total_tokens,
            "rep_budget": rep_budget,
            "run_name": run.name,
            "state": run.state,
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(all_runs)} runs...")

    total_pts = sum(len(v) for v in groups.values())
    n_fin = sum(1 for pts in groups.values() for p in pts if p["state"] == "finished")
    n_run = sum(1 for pts in groups.values() for p in pts if p["state"] == "running")
    print(f"\n  Collected {total_pts} runs across {len(groups)} groups")
    if skipped:
        print(f"  Skipped {skipped} runs (missing config or metrics)")

    if not groups:
        print("No data to plot!")
        return

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14, 9))

    for idx, ((msize, ot), pts) in enumerate(sorted(groups.items())):
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        _plot_group(ax, pts, color, marker, f"{msize} (OT={ot})")

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(3.2, 5.0)
    ax.set_xlabel("Number of Unique Tokens Repeated (fewer = more repeats)", fontsize=14)
    ax.set_ylabel("Test Loss (final)", fontsize=14)
    ax.set_title(
        f"Test Loss vs Size of Repeated Subset — All Runs\n"
        f"Hernandez Replication ({total_pts} runs: {n_fin} finished, {n_run} running)",
        fontsize=15,
    )
    ax.legend(fontsize=11, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=11)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "list_runs_chart.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved chart: {out}")
    plt.close()


if __name__ == "__main__":
    main()
