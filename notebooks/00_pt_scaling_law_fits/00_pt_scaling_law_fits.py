from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wandb

import src.analyze
import src.plot

refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    "dc54nn8d",  # Qwen 3   34M 1xOT
    "9lzr0v43",  # Qwen 3   48M 1xOT
    "r7tj4ki1",  # Qwen 3   63M 1xOT
    "go2fzcue",  # Qwen 3   93M 1xOT
    "stzt1epz",  # Qwen 3  153M 1xOT
    "vw7a3nt4",  # Qwen 3  344M 1xOT
    "wr21ll5w",  # Qwen 3  499M 1xOT
    "xl54i94h",  # Qwen 3  660M 1xOT
    # "",  # Qwen 3  M 1xOT
]

pretrain_run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="scaling-memorization-pt",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)

pretrain_run_configs_df = (
    src.analyze.add_pretraining_quantities_to_pretrain_run_configs_df(
        pretrain_run_configs_df=pretrain_run_configs_df
    )
)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=pretrain_run_configs_df,
    x="Num. FLOP (6ND)",
    y="eval_after/eval_loss",
    marker="o",
    hue="Direction",
    style="Shuffle Seed",
)
g.set(
    xlabel="Pretraining FLOP (6ND)",
    xscale="log",
    yscale="log",
    ylabel="Cross Entropy (Test)",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="y=eval-loss_x=flop_hue=dir_style=shuffle-seed"
)
# plt.show()

# Fit power law scaling to each model's inference scaling w.r.t. k.
power_law_fits_df = pd.DataFrame(
    [
        src.analyze.fit_neural_scaling_law(
            subset_df,
            x_col="Num. FLOP (6ND)",
            y_col="eval_after/eval_loss",
            additional_columns_to_add=["Direction", "Shuffle Seed"],
        )
        for (direction, shuffle_seed), subset_df in pretrain_run_configs_df.groupby(
            ["Direction", "Shuffle Seed"]
        )
    ]
)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=pretrain_run_configs_df,
    x="Num. FLOP (6ND)",
    y="eval_after/eval_loss",
    palette="viridis",
    marker="o",
    s=100,
    # legend="full",
    legend=False,
)
g.set(
    xlabel="Pretraining FLOP (6ND)",
    xscale="log",
    yscale="log",
    ylabel="Cross Entropy (Test)",
)
x_vals = np.geomspace(
    start=pretrain_run_configs_df["Num. FLOP (6ND)"].min() / 1.1,
    stop=pretrain_run_configs_df["Num. FLOP (6ND)"].max() * 1.1,
    num=100,
)
for row_idx, row in power_law_fits_df.iterrows():
    yhat_vals = row["fit_param_E_0"] + row["fit_param_C_0"] * np.power(
        x_vals, -row["fit_param_alpha"]
    )
    sns.lineplot(
        x=x_vals,
        y=yhat_vals,
        ax=g,
        color="black",
        linestyle="--",
        # legend=False,
    )
    # Add the irreducible error
    sns.lineplot(
        x=x_vals,
        y=np.full_like(
            x_vals,
            fill_value=row["fit_param_E_0"],
        ),
        ax=g,
        linestyle="--",
        legend=False,
        color="black",
    )
plt.text(1e18, 2.4, "Estimated Irreducible Errors", size="x-small")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=eval-loss_x=flop_overlay=fit-scaling-laws",
)
# plt.show()

print("Finished notebook/00_pt_scaling_law_fits.py!")
