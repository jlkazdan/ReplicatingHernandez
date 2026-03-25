import ast
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

# refresh = False
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    "6w78yino",  # Qwen 3   34M 1xOT
    "w0gj0s7v",  # Qwen 3   48M 1xOT
    "rj7wb1qp",  # Qwen 3   63M 1xOT
    "gekq38pk",  # Qwen 3   93M 1xOT
    # "",  # Qwen 3  153M 1xOT
    # "",  # Qwen 3  344M 1xOT
    # "",  # Qwen 3  499M 1xOT
    # "",  # Qwen 3  M 1xOT
    # "",  # Qwen 3  M 1xOT
]

per_seq_scaling_law_fits_df = src.analyze.create_or_load_per_seq_scaling_laws(
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    num_to_subsample=5000,
)

plt.close()
vars_to_plot = ["fit_loss", "fit_param_E_0", "fit_param_C_0", "fit_param_alpha"]
g = sns.pairplot(
    data=per_seq_scaling_law_fits_df[vars_to_plot],
    kind="scatter",
    corner=True,
)
# Define your custom limits for each variable
custom_limits = {}
for var_to_plot in vars_to_plot:
    custom_limits[var_to_plot] = (
        per_seq_scaling_law_fits_df[var_to_plot].quantile(q=0.01),
        per_seq_scaling_law_fits_df[var_to_plot].quantile(q=0.99),
    )

# Apply limits to each subplot
for i, y_var in enumerate(vars_to_plot):
    for j, x_var in enumerate(vars_to_plot):
        ax = g.axes[i, j]
        if ax is not None:  # ax is None for the hidden upper triangle when corner=True
            if x_var in custom_limits:
                ax.set_xlim(custom_limits[x_var])
                ax.set_xscale("log")
            if y_var in custom_limits:
                ax.set_ylim(custom_limits[y_var])
                ax.set_yscale("log")
plt.show()

per_seq_nll_runs_histories_df: pd.DataFrame = (
    src.analyze.create_or_load_per_seq_nll_runs_histories(
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
    )
)


# Sanity check the correctness.
num_model_sizes = per_seq_nll_runs_histories_df["Num. Parameters"].nunique()
num_document_counts_by_pretraining_dataset_seq_id_split_df = (
    per_seq_nll_runs_histories_df.groupby(["Pretraining Dataset", "seq_id", "split"])
    .size()
    .reset_index()
)
print(
    "Fraction of test documents with the correct number of documents per dataset per seq_id per split: ",
    np.mean(
        num_document_counts_by_pretraining_dataset_seq_id_split_df[
            num_document_counts_by_pretraining_dataset_seq_id_split_df["split"]
            == "train"
        ][0]
        == num_model_sizes
    ),
)
print(
    "Fraction of eval documents with the correct number of documents per dataset per seq_id per split: ",
    np.mean(
        num_document_counts_by_pretraining_dataset_seq_id_split_df[
            num_document_counts_by_pretraining_dataset_seq_id_split_df["split"]
            == "eval"
        ][0]
        == (
            4 * num_model_sizes
        )  # For each model size, we evaluate against all four conditions.
    ),
)

train_per_seq_nll_runs_histories_df = per_seq_nll_runs_histories_df[
    per_seq_nll_runs_histories_df["split"] == "train"
]

eval_per_seq_nll_runs_histories_df = per_seq_nll_runs_histories_df[
    per_seq_nll_runs_histories_df["split"] == "eval"
]

rand_subset_run_id_seq_id = np.random.choice(
    per_seq_nll_runs_histories_df["Pretraining Dataset+seq_id"].unique(),
    replace=False,
    size=1000,
)
subset_per_seq_nll_runs_histories_df = per_seq_nll_runs_histories_df[
    per_seq_nll_runs_histories_df["Pretraining Dataset+seq_id"].isin(
        rand_subset_run_id_seq_id
    )
]

plt.close()
g = sns.relplot(
    data=subset_per_seq_nll_runs_histories_df,
    kind="line",
    x="Num. FLOP (6ND)",
    y="avg_nll",
    units="Pretraining Dataset+seq_id",
    estimator=None,
    col="split",
    col_order=["train", "eval"],
    alpha=0.01,
)
g.set(
    xscale="log",
    xlabel="Pretraining Compute (6ND)",
    yscale="log",
    ylabel="Cross Entropy",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="y=loss_x=flop_col=split_unit=dataset+seq"
)
# plt.show()

plt.close()
g = sns.relplot(
    data=train_per_seq_nll_runs_histories_df,
    kind="line",
    x="Num. FLOP (6ND)",
    y="avg_nll",
    col="Pretraining Dataset",
    col_wrap=2,
    hue="Eval Dataset",
    facet_kws={"margin_titles": True, "sharex": True, "sharey": True},
)
g.set(
    xscale="log",
    yscale="log",
)
plt.show()


# plt.close()
# g = sns.displot(
#     data=train_per_seq_nll_runs_histories_df,
#     kind="line",
#     x="Pretraining Dataset",
#     y="Eval Dataset",
#     hue="Eval Dataset",
# )

print("Finished notebook/01_pt_per_seq_scaling_law_fits.py!")
