import ast
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import wandb

# Point at the project src/ (the actual repo root on the data mount).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import src.analyze
import src.plot

# src/plot.py enables LaTeX rendering; disable it if latex is not installed.
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "sans-serif"

refresh = False
# refresh = True

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=NOTEBOOK_DIR,
    refresh=False,
)

# Sweeps from jchud-stanford-university/hernandez-replication with finished runs.
# Each sweep trains one model size across multiple num_repeats values (grid search).
sweep_ids = [
    "20mrz98t",  # Qwen3  93M  top  (14 finished runs)
    "apj92xvb",  # Qwen3  93M  top  ( 4 finished runs – partial)
    "3p1olv16",  # Qwen3  63M  top  (16 finished runs)
    "urnekydr",  # Qwen3  48M  top  (16 finished runs)
    "9w4r4pk8",  # Qwen3 153M  top  (10 finished runs)
]

# ── Download run configs ───────────────────────────────────────────────────────
pretrain_run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="hernandez-replication",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username="jchud-stanford-university",
    finished_only=True,
)

# ── Extract fields from nested config dicts ───────────────────────────────────
def _parse(cfg_str):
    if isinstance(cfg_str, str):
        return ast.literal_eval(cfg_str)
    return cfg_str if isinstance(cfg_str, dict) else {}

pretrain_run_configs_df["Num. Repeats"] = pretrain_run_configs_df["data_config"].apply(
    lambda s: _parse(s).get("num_repeats", 1)
)
pretrain_run_configs_df["Repetition Budget"] = pretrain_run_configs_df["data_config"].apply(
    lambda s: _parse(s).get("repetition_budget", 0.0)
)
pretrain_run_configs_df["Direction"] = pretrain_run_configs_df["data_config"].apply(
    lambda s: _parse(s).get("direction", "?")
)
pretrain_run_configs_df["Model"] = pretrain_run_configs_df["model_config"].apply(
    lambda s: _parse(s).get("model_name", "?").split("/")[-1]
)
pretrain_run_configs_df["Num. Parameters"] = pretrain_run_configs_df["model/num_parameters"]
pretrain_run_configs_df["Parameters"] = pretrain_run_configs_df["Num. Parameters"].apply(
    lambda n: f"{n / 1_000_000:.0f}M"
)
pretrain_run_configs_df["Final Eval Loss"] = pretrain_run_configs_df["eval_after/eval_loss"]

print(f"Loaded {len(pretrain_run_configs_df)} finished runs.")
print(
    pretrain_run_configs_df[
        ["Model", "Parameters", "Direction", "Num. Repeats", "Repetition Budget", "Final Eval Loss"]
    ]
    .sort_values(["Parameters", "Num. Repeats"])
    .to_string(index=False)
)

# ── Plot 1: single panel, one line per model size ──────────────────────────────
plt.close()
fig, ax = plt.subplots(figsize=(12, 6))
g = sns.lineplot(
    data=pretrain_run_configs_df.sort_values("Num. Repeats"),
    x="Num. Repeats",
    y="Final Eval Loss",
    hue="Parameters",
    style="Direction",
    marker="o",
    err_style="bars",
    errorbar="sd",
    ax=ax,
)
ax.set(
    xlabel="Num. Repeats",
    xscale="log",
    ylabel="Cross Entropy (Final Eval Loss)",
    title="Effect of Repetition on Final Eval Loss",
)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=eval-loss_x=num-repeats_hue=model-size_style=direction",
)

# ── Plot 2: faceted by model size ──────────────────────────────────────────────
plt.close()
g = sns.relplot(
    data=pretrain_run_configs_df.sort_values("Num. Repeats"),
    kind="line",
    x="Num. Repeats",
    y="Final Eval Loss",
    hue="Direction",
    style="Direction",
    col="Parameters",
    col_wrap=4,
    marker="o",
    facet_kws={"sharex": True, "sharey": False},
)
g.set(
    xlabel="Num. Repeats",
    xscale="log",
    ylabel="Cross Entropy (Final Eval Loss)",
)
g.set_titles(col_template="{col_name}")
sns.move_legend(g, loc="upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=eval-loss_x=num-repeats_hue=direction_col=model-size",
)

print("Finished 02_hernandez_num_repeats_vs_eval_loss.py!")
print(f"Plots saved to: {results_dir}")
