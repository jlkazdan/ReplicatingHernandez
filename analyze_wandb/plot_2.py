"""
Plot ALL Hernandez Replication Runs — Test Loss vs Size of Repeated Subset

Pulls every run from jchud-stanford-university/hernandez-replication on W&B,
groups by model size (averaging top and bot directions together), and creates
a chart similar to the "Test Loss vs Size of Repeated Subset" figure.

X-axis: Number of Unique Tokens Repeated (log scale, inverted — fewer = more repeats)
Y-axis: Test Loss (final eval/loss)
Each point labeled with subset size and repetition count (Nx)

Usage:
    python plot_all_hernandez_runs.py
    python plot_all_hernandez_runs.py --include-crashed
    python plot_all_hernandez_runs.py --one-chart        # force everything onto one chart
    python plot_all_hernandez_runs.py --separate          # one chart per model×direction
"""

import argparse
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    import wandb
except ImportError:
    raise ImportError("pip install wandb")


# =============================================================================
# CONFIG
# =============================================================================

ENTITY = "jchud-stanford-university"
PROJECT = "hernandez-replication"

TEST_LOSS_KEY = "eval/loss"
TOKENS_KEY = "train/num_input_tokens_seen"

# Nice color cycle for different model×direction combos
COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<"]


# =============================================================================
# HELPERS
# =============================================================================

def extract_run_info(run):
    """Pull config fields from a W&B run."""
    info = {}

    # --- data config ---
    data_config = run.config.get("data_config", {})
    info["num_repeats"] = data_config.get("num_repeats")
    info["repetition_budget"] = data_config.get("repetition_budget", 0.1)
    info["direction"] = data_config.get("direction", "unknown")

    # --- model ---
    info["model_params"] = run.config.get("model/num_parameters")
    model_config = run.config.get("model_config", {})
    info["model_name"] = model_config.get("model_name", "unknown")

    # --- training ---
    trainer_config = run.config.get("trainer_config", {})
    info["overtrain_multiplier"] = trainer_config.get("overtrain_multiplier", 1.0)

    # --- fallback: parse hub_model_id ---
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


def get_final_eval_loss(run, samples=2000):
    """Get the last eval/loss value from a run's history."""
    # Try summary first (fastest)
    summary_loss = run.summary.get("eval/loss")
    if summary_loss is not None:
        return summary_loss

    # Fall back to history
    try:
        history = run.history(keys=[TEST_LOSS_KEY], samples=samples)
        if history.empty:
            return None
        losses = history[TEST_LOSS_KEY].dropna()
        if losses.empty:
            return None
        return float(losses.iloc[-1])
    except Exception:
        return None


def get_total_tokens(run, samples=2000):
    """Get total tokens trained from run summary or history."""
    # Try summary
    summary_tokens = run.summary.get("train/num_input_tokens_seen")
    if summary_tokens is not None:
        return int(summary_tokens)

    # Fall back
    try:
        history = run.history(keys=[TOKENS_KEY], samples=samples)
        if history.empty:
            return None
        tokens = history[TOKENS_KEY].dropna()
        if tokens.empty:
            return None
        return int(tokens.iloc[-1])
    except Exception:
        return None


def human_readable(n):
    """Convert token count to human-readable string."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}k"
    else:
        return str(int(n))


def model_size_label(info):
    """Create a readable model label like '63M' from config."""
    name = info.get("model_name", "")
    m = re.search(r'(\d+)M', name)
    if m:
        return f"{m.group(1)}M"
    params = info.get("model_params")
    if params:
        return f"{round(params / 1e6)}M"
    return "unknown"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot all Hernandez replication runs")
    parser.add_argument("--include-crashed", action="store_true",
                        help="Include crashed runs (uses partial data)")
    parser.add_argument("--one-chart", action="store_true",
                        help="Force all model×direction combos onto one chart")
    parser.add_argument("--separate", action="store_true",
                        help="One chart per model×direction")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--sweeps", nargs="*", default=None,
                        help="Filter to specific sweep IDs")
    args = parser.parse_args()

    api = wandb.Api()

    # Build filters
    states = ["finished"]
    if args.include_crashed:
        states.append("crashed")

    filters = {"state": {"$in": states}}
    if args.sweeps:
        filters["sweep"] = {"$in": args.sweeps}

    print(f"Fetching runs from {ENTITY}/{PROJECT}...")
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters=filters, per_page=300)
    runs = list(runs)
    print(f"  Found {len(runs)} runs")

    # -------------------------------------------------------------------------
    # Collect data
    # -------------------------------------------------------------------------
    # Group: (model_size_label, ot) -> list of runs (top+bot pooled together)
    groups = defaultdict(list)
    skipped = 0

    for i, run in enumerate(runs):
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
            # Estimate: 20 tokens/param * model_params for CE=1
            params = info.get("model_params")
            if params:
                total_tokens = int(20 * params)
            else:
                skipped += 1
                continue

        rep_budget = info.get("repetition_budget", 0.1)
        unique_tokens = (rep_budget * total_tokens) / nr if nr > 0 else total_tokens

        msize = model_size_label(info)
        direction = info.get("direction", "unknown")
        ot = info.get("overtrain_multiplier", 1.0)

        # Average across top/bot — group only by model size and OT
        key = (msize, ot)
        groups[key].append({
            "unique_tokens": unique_tokens,
            "num_repeats": nr,
            "final_loss": final_loss,
            "total_tokens": total_tokens,
            "rep_budget": rep_budget,
            "run_name": run.name,
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(runs)} runs...")

    print(f"\n  Collected data for {sum(len(v) for v in groups.values())} runs across {len(groups)} groups")
    if skipped:
        print(f"  Skipped {skipped} runs (missing config or metrics)")

    # Print group summary
    print("\n  Groups found:")
    for (msize, ot), pts in sorted(groups.items()):
        print(f"    {msize} / OT={ot}: {len(pts)} runs (top+bot averaged)")

    if not groups:
        print("No data to plot!")
        return

    # -------------------------------------------------------------------------
    # Plot: one big chart with all groups
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    if args.separate:
        # One chart per group
        for idx, ((msize, ot), pts) in enumerate(sorted(groups.items())):
            fig, ax = plt.subplots(figsize=(12, 8))
            _plot_group(ax, pts, COLORS[0], MARKERS[0], f"{msize}")

            total_tok_example = pts[0]["total_tokens"]
            rep_b = pts[0]["rep_budget"]
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xlabel("Number of Unique Tokens Repeated (fewer = more repeats)", fontsize=13)
            ax.set_ylabel("Test Loss (final)", fontsize=13)
            ax.set_title(
                f"Test Loss vs Size of Repeated Subset\n"
                f"{msize} Parameter Model (Qwen3/{msize}), {int(rep_b*100)}% Repeated — top+bot averaged\n"
                f"(Total tokens: {human_readable(total_tok_example)}, compute held constant)",
                fontsize=14,
            )
            ax.grid(True, alpha=0.3, which="both")
            ax.tick_params(labelsize=11)
            plt.tight_layout()
            fname = f"hernandez_{msize}_avg_ot{ot}.png"
            out = os.path.join(args.output_dir, fname)
            plt.savefig(out, dpi=200, bbox_inches="tight")
            print(f"  Saved: {out}")
            plt.close()

    else:
        # One combined chart
        fig, ax = plt.subplots(figsize=(14, 9))

        sorted_groups = sorted(groups.items())
        for idx, ((msize, ot), pts) in enumerate(sorted_groups):
            color = COLORS[idx % len(COLORS)]
            marker = MARKERS[idx % len(MARKERS)]
            label = f"{msize} (OT={ot})"
            _plot_group(ax, pts, color, marker, label)

        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Number of Unique Tokens Repeated (fewer = more repeats)", fontsize=14)
        ax.set_ylabel("Test Loss (final)", fontsize=14)
        ax.set_title(
            f"Test Loss vs Size of Repeated Subset — All Runs\n"
            f"Hernandez Replication ({sum(len(v) for v in groups.values())} runs, compute held constant per group)",
            fontsize=15,
        )
        ax.legend(fontsize=11, framealpha=0.9, loc="best")
        ax.grid(True, alpha=0.3, which="both")
        ax.tick_params(labelsize=11)
        plt.tight_layout()

        out = os.path.join(args.output_dir, "hernandez_all_runs.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"\n  Saved: {out}")
        plt.close()

        # Also save a per-model version (subplots)
        model_sizes = sorted(set(k[0] for k in groups.keys()))
        if len(model_sizes) > 1:
            fig, axes = plt.subplots(1, len(model_sizes), figsize=(7 * len(model_sizes), 7), sharey=True)
            if len(model_sizes) == 1:
                axes = [axes]

            for ax_i, msize in enumerate(model_sizes):
                ax = axes[ax_i]
                color_idx = 0
                for (ms, ot), pts in sorted_groups:
                    if ms != msize:
                        continue
                    color = COLORS[color_idx % len(COLORS)]
                    marker = MARKERS[color_idx % len(MARKERS)]
                    _plot_group(ax, pts, color, marker, f"OT={ot}")
                    color_idx += 1

                ax.set_xscale("log")
                ax.invert_xaxis()
                ax.set_xlabel("Unique Tokens Repeated", fontsize=12)
                if ax_i == 0:
                    ax.set_ylabel("Test Loss (final)", fontsize=12)
                ax.set_title(f"{msize} (Qwen3)", fontsize=14)
                ax.legend(fontsize=10, framealpha=0.9)
                ax.grid(True, alpha=0.3, which="both")
                ax.tick_params(labelsize=10)

            fig.suptitle(
                "Test Loss vs Size of Repeated Subset — by Model Size",
                fontsize=16, y=1.02,
            )
            plt.tight_layout()
            out2 = os.path.join(args.output_dir, "hernandez_by_model.png")
            plt.savefig(out2, dpi=200, bbox_inches="tight")
            print(f"  Saved: {out2}")
            plt.close()


def _plot_group(ax, pts, color, marker, label):
    """Plot one model×direction group on an axes."""
    # Sort by unique_tokens descending (left = many tokens, right = few)
    pts_sorted = sorted(pts, key=lambda p: p["unique_tokens"], reverse=True)

    xs = [p["unique_tokens"] for p in pts_sorted]
    ys = [p["final_loss"] for p in pts_sorted]

    # If there are duplicate num_repeats, average them
    from collections import defaultdict as dd
    nr_groups = dd(list)
    for p in pts_sorted:
        nr_groups[p["num_repeats"]].append(p)

    # Plot individual points with some transparency
    ax.scatter(xs, ys, color=color, marker=marker, s=60, alpha=0.4, zorder=3)

    # Plot averaged line
    avg_xs, avg_ys = [], []
    for nr in sorted(nr_groups.keys()):
        group = nr_groups[nr]
        avg_unique = np.mean([p["unique_tokens"] for p in group])
        avg_loss = np.mean([p["final_loss"] for p in group])
        avg_xs.append(avg_unique)
        avg_ys.append(avg_loss)

    # Sort for line plot
    order = np.argsort(avg_xs)[::-1]
    avg_xs = [avg_xs[i] for i in order]
    avg_ys = [avg_ys[i] for i in order]

    ax.plot(avg_xs, avg_ys, f"-", color=color, linewidth=2.5, markersize=10,
            marker=marker, label=label, zorder=4)

    # Label each averaged point
    for x, y, nr in zip(avg_xs, avg_ys, [sorted(nr_groups.keys())[i] for i in order]):
        ax.annotate(
            f"{human_readable(x)}\n({nr}x)",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8,
            color=color,
            alpha=0.85,
        )


if __name__ == "__main__":
    main()