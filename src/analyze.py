import ast
from functools import partial
import concurrent.futures
import hashlib
import numpy as np
import os
import pandas as pd
import pyarrow
import re
import requests
import scipy.optimize
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import wandb
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import src.globals
import src.neural_scaling_laws


def add_pretraining_quantities_to_pretrain_run_configs_df(
    pretrain_run_configs_df: pd.DataFrame,
) -> pd.DataFrame:
    pretrain_run_configs_df["Model"] = pretrain_run_configs_df["model_config"].apply(
        lambda model_config: ast.literal_eval(model_config)["model_name"]
    )
    pretrain_run_configs_df["Num. Parameters"] = pretrain_run_configs_df[
        "model/num_parameters"
    ]
    pretrain_run_configs_df["Parameters"] = pretrain_run_configs_df[
        "Num. Parameters"
    ].apply(lambda n: f"{n / 1_000_000:.0f}M")
    pretrain_run_configs_df["Overtrain Multiplier"] = pretrain_run_configs_df[
        "trainer_config"
    ].apply(
        lambda trainer_config: ast.literal_eval(trainer_config)["overtrain_multiplier"]
    )
    pretrain_run_configs_df["Num. Tokens"] = (
        20.0
        * pretrain_run_configs_df["Overtrain Multiplier"]
        * pretrain_run_configs_df["Num. Parameters"]
    )
    pretrain_run_configs_df["Num. FLOP (6ND)"] = (
        6
        * pretrain_run_configs_df["Num. Parameters"]
        * pretrain_run_configs_df["Num. Tokens"]
    )
    pretrain_run_configs_df["Num. Epochs"] = pretrain_run_configs_df[
        "trainer_config"
    ].apply(lambda trainer_config: ast.literal_eval(trainer_config)["num_train_epochs"])

    pretrain_run_configs_df["Corpus"] = pretrain_run_configs_df["data_config"].apply(
        lambda data_config: ast.literal_eval(data_config)["corpus"]
    )
    pretrain_run_configs_df["Direction"] = pretrain_run_configs_df["data_config"].apply(
        lambda data_config: ast.literal_eval(data_config)["direction"]
    )
    pretrain_run_configs_df["Direction"] = pretrain_run_configs_df["data_config"].apply(
        lambda data_config: ast.literal_eval(data_config)["direction"]
    )
    pretrain_run_configs_df["Shuffle Seed"] = pretrain_run_configs_df[
        "data_config"
    ].apply(lambda data_config: ast.literal_eval(data_config)["shuffle_seed"])

    return pretrain_run_configs_df


def calculate_compute_contamination_exchange_rate(
    loss: float, irreducible_error: float, prefactor: float, exponent: float
) -> float:
    return np.power((loss - irreducible_error) / prefactor, -1.0 / exponent)


def construct_dataset_name_for_eval_analysis(row: pd.Series):
    trainer_config = ast.literal_eval(row["trainer_config"])
    data_config = ast.literal_eval(row["data_config"])
    num_train_epochs = trainer_config["num_train_epochs"]
    overtrain_multiplier = 1
    seed = 0
    direction = data_config["direction"]
    shuffle_seed = data_config["shuffle_seed"]
    train_test_split_seed = data_config["train_test_split_seed"]
    dataset_name = f"epch_{num_train_epochs}_ot_{overtrain_multiplier}_s_{seed}_dir_{direction}_shfs_{shuffle_seed}_ttss_{train_test_split_seed}"
    return dataset_name


def convert_float_to_latex_scientific_notation(num: float, precision: int = 2) -> str:
    """Converts a float to a LaTeX scientific notation string."""
    # Format the number in scientific notation (e.g., '1.23e+04')
    sci_notation_str = f"{num:.{precision}e}"

    # Split into mantissa and exponent
    mantissa, exponent = sci_notation_str.split("e")

    # Convert exponent to an integer
    exponent_val = int(exponent)

    # Return just the mantissa if the exponent is 0
    if exponent_val == 0:
        return mantissa

    # Construct the LaTeX string
    return rf"{mantissa}e{{{exponent_val}}}"


def construct_latex_power_law_equation_from_num_reference_models(
    E_0_of_k: float,
    C_0_of_k: float,
    alpha_of_k: float,
    precision: int = 2,
):
    latex_E_0_of_k = convert_float_to_latex_scientific_notation(
        E_0_of_k, precision=precision
    )
    latex_C_0_of_k = convert_float_to_latex_scientific_notation(
        C_0_of_k, precision=precision
    )
    latex_alpha_of_k = convert_float_to_latex_scientific_notation(
        alpha_of_k, precision=precision
    )

    latex_equation = (
        rf"${latex_E_0_of_k} + {latex_C_0_of_k} \cdot N^{{{latex_alpha_of_k}}}$"
    )
    return latex_equation


def create_or_load_per_seq_nll_runs_histories(
    data_dir: str,
    sweep_ids: List[str],
    refresh: bool = False,
) -> pd.DataFrame:
    per_seq_nll_runs_configs_df: pd.DataFrame = (
        src.analyze.download_wandb_project_runs_configs(
            wandb_project_path="scaling-memorization-eval",
            data_dir=data_dir,
            sweep_ids=sweep_ids,
            refresh=refresh,
            wandb_username=wandb.api.default_entity,
            finished_only=True,
        )
    )
    per_seq_nll_runs_configs_df["Model Name"] = per_seq_nll_runs_configs_df[
        "model_config"
    ].apply(lambda model_config: ast.literal_eval(model_config)["model_name"])
    per_seq_nll_runs_configs_df["Pretraining Dataset"] = per_seq_nll_runs_configs_df[
        "Model Name"
    ].apply(src.analyze.extract_pretraining_dataset_name_for_eval_analysis)
    per_seq_nll_runs_configs_df["Eval Dataset"] = per_seq_nll_runs_configs_df.apply(
        src.analyze.construct_dataset_name_for_eval_analysis, axis=1
    )
    per_seq_nll_runs_configs_df["Num. Parameters"] = per_seq_nll_runs_configs_df[
        "Model Name"
    ].apply(src.analyze.extract_num_model_parameters)
    per_seq_nll_runs_configs_df["Num. Tokens"] = (
        20.0 * per_seq_nll_runs_configs_df["Num. Parameters"]
    )
    per_seq_nll_runs_configs_df["Num. FLOP (6ND)"] = 120 * np.square(
        per_seq_nll_runs_configs_df["Num. Parameters"]
    )

    per_seq_nll_runs_histories_df: pd.DataFrame = (
        src.analyze.download_wandb_project_runs_histories(
            wandb_project_path="scaling-memorization-eval",
            data_dir=data_dir,
            sweep_ids=sweep_ids,
            refresh=refresh,
            wandb_username=wandb.api.default_entity,
            max_workers=32,
            wandb_run_history_num_samples=10_000_000,
            filetype="parquet",
        )
    )

    per_seq_nll_runs_histories_df = per_seq_nll_runs_histories_df.merge(
        per_seq_nll_runs_configs_df[
            [
                "run_id",
                "Model Name",
                "Num. Parameters",
                "Num. FLOP (6ND)",
                "Pretraining Dataset",
                "Eval Dataset",
            ]
        ],
        on="run_id",
        how="left",
    )
    per_seq_nll_runs_histories_df[
        "Pretraining Dataset+seq_id"
    ] = per_seq_nll_runs_histories_df.apply(
        lambda row: f"{row['Pretraining Dataset']}_{row['seq_id']}", axis=1
    )
    return per_seq_nll_runs_histories_df


def create_or_load_per_seq_scaling_laws(
    data_dir: str,
    sweep_ids: List[str],
    refresh: bool = False,
    num_outer_workers: int = 32,
    num_to_subsample: Optional[int] = None,
) -> pd.DataFrame:
    filename = "sweeps=" + ",".join(sweep_ids)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    scaling_laws_per_seq_path = os.path.join(
        data_dir, f"{hashed_filename}_scaling_laws_per_seq_path.parquet"
    )
    if not os.path.exists(scaling_laws_per_seq_path) or refresh:
        per_seq_nll_runs_histories_df = create_or_load_per_seq_nll_runs_histories(
            data_dir=data_dir,
            sweep_ids=sweep_ids,
            refresh=refresh,
        )

        num_model_sizes = per_seq_nll_runs_histories_df["Num. Parameters"].nunique()

        grouped_data = [
            subset_df
            for _, subset_df in per_seq_nll_runs_histories_df.groupby(
                ["Pretraining Dataset", "Eval Dataset", "seq_id", "split"]
            )
            if len(subset_df) == num_model_sizes
        ]
        if num_to_subsample is not None:
            import random

            grouped_data = random.sample(grouped_data, num_to_subsample)

        # # Prevent a massive increase of processes due to nested Pool.
        # num_outer_workers = min(num_outer_workers, os.cpu_count())
        # num_inner_workers = os.cpu_count() // num_outer_workers
        #
        # fit_func = partial(
        #     src.analyze.fit_neural_scaling_law,
        #     x_col="Num. FLOP (6ND)",
        #     y_col="avg_nll",
        #     additional_columns_to_add=[
        #         "Pretraining Dataset",
        #         "Eval Dataset",
        #         "seq_id",
        #         "split",
        #     ],
        #     n_workers=num_inner_workers,
        # )
        #
        # results: List[Dict[str, float]] = process_map(
        #     fit_func,
        #     grouped_data,
        #     max_workers=num_outer_workers,
        #     chunksize=len(grouped_data) // num_outer_workers,
        # )
        #
        # per_seq_scaling_law_fits_df = pd.DataFrame(results)

        # Sequential implementation.
        per_seq_scaling_law_fits_df = pd.DataFrame(
            [
                src.analyze.fit_neural_scaling_law(
                    subset_df,
                    x_col="Num. FLOP (6ND)",
                    y_col="avg_nll",
                    additional_columns_to_add=[
                        "Pretraining Dataset",
                        "seq_id",
                        "split",
                    ],
                )
                # for (
                #     pt_dataset,
                #     seq_id,
                #     split,
                # ), subset_df in per_seq_nll_runs_histories_df.groupby(
                #     ["Pretraining Dataset", "seq_id", "split"]
                # )
                for subset_df in grouped_data
            ]
        )

        per_seq_scaling_law_fits_df.to_parquet(
            path=scaling_laws_per_seq_path, index=False, engine="pyarrow"
        )
        del per_seq_scaling_law_fits_df

    per_seq_scaling_law_fits_df = pd.read_parquet(scaling_laws_per_seq_path)
    return per_seq_scaling_law_fits_df


def create_or_load_strong_membership_inference_attack_data(
    data_dir: str, refresh: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    auc_models_parquet_path = os.path.join(data_dir, "auc_models.parquet")
    tpr_fpr_models_data_path = os.path.join(data_dir, "tpr_fpr_models.parquet")

    if not os.path.exists(tpr_fpr_models_data_path) or refresh:
        auc_models_data = []
        tpr_fpr_models_data = []
        num_reference_models = np.power(2, np.arange(0, 9), dtype=int)
        for num_models in num_reference_models:
            with open(os.path.join(data_dir, f"numrefs_{num_models}.txt"), "r") as f:
                data = f.readlines()
                fpr = ast.literal_eval(
                    data[0][5:-1]
                )  # Strip off leading "fpr: " and trailing "\n".
                tpr = ast.literal_eval(
                    data[1][5:-1]
                )  # Strip off leading "tpr: " and trailing "\n".
                assert len(tpr) == len(fpr)
                auc = float(
                    data[2][5:-1]
                )  # Strip off leading "auc: " and trailing "\n".
            auc_models_data.append(
                pd.Series({"Num. Reference Models": num_models, "AUC": auc})
            )
            tpr_fpt_model_df = pd.DataFrame(
                {
                    "Num. Reference Models": [num_models] * len(tpr),
                    "TPR": tpr,
                    "FPR": fpr,
                }
            )
            # There are duplicate FPRs. I assume that this is due to a loss of numerical precision.
            # Let's take the first.
            tpr_fpr_models_data.append(
                tpr_fpt_model_df.groupby("FPR").mean().reset_index()
            )

        auc_models_df = pd.DataFrame(auc_models_data).reset_index(drop=True)
        auc_models_df["Neg. Log AUC"] = -np.log(auc_models_df["AUC"])
        auc_models_df.to_parquet(auc_models_parquet_path, engine="pyarrow")
        tpr_fpr_models_df = pd.concat(tpr_fpr_models_data).reset_index(drop=True)
        tpr_fpr_models_df["Neg. Log TPR"] = -np.log(tpr_fpr_models_df["TPR"])
        tpr_fpr_models_df.to_parquet(tpr_fpr_models_data_path, engine="pyarrow")

        del auc_models_df, tpr_fpr_models_df

    auc_models_df = pd.read_parquet(auc_models_parquet_path)
    tpr_fpr_models_df = pd.read_parquet(tpr_fpr_models_data_path)
    return auc_models_df, tpr_fpr_models_df


def download_wandb_project_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    max_workers: int = 10,  # New parameter to control the number of parallel workers
) -> pd.DataFrame:
    assert filetype in {"csv", "feather", "parquet"}

    filename = "sweeps=" + ",".join(sweep_ids)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    runs_configs_df_path = os.path.join(
        data_dir, hashed_filename + f"_runs_configs.{filetype}"
    )

    if refresh or not os.path.isfile(runs_configs_df_path):
        print(f"Creating {runs_configs_df_path} anew.")

        api = wandb.Api(timeout=600)

        if wandb_username is None:
            wandb_username = api.viewer.username

        sweep_results_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {}

            for sweep_id in sweep_ids:
                try:
                    sweep = api.sweep(
                        f"{wandb_username}/{wandb_project_path}/{sweep_id}"
                    )
                    for run in sweep.runs:
                        future_to_run[
                            executor.submit(
                                download_wandb_project_runs_configs_helper, run
                            )
                        ] = run
                except Exception as e:
                    print(f"Error processing sweep {sweep_id}: {str(e)}")

            for future in tqdm(
                concurrent.futures.as_completed(future_to_run), total=len(future_to_run)
            ):
                result = future.result()
                if result is not None:
                    sweep_results_list.append(result)

        runs_configs_df = pd.DataFrame(sweep_results_list)
        runs_configs_df.reset_index(inplace=True, drop=True)

        # Cast massive FLOP counts to float to avoid C long overflow
        if "total_flos" in runs_configs_df.columns:
            runs_configs_df["total_flos"] = pd.to_numeric(
                runs_configs_df["total_flos"], errors="coerce"
            ).astype(float)

        # Save to disk
        runs_configs_df.to_csv(
            runs_configs_df_path.replace(filetype, "csv"), index=False
        )
        try:
            runs_configs_df.to_feather(
                runs_configs_df_path.replace(filetype, "feather")
            )
        except Exception as e:
            print(f"Error saving to feather: {str(e)}")

        try:
            runs_configs_df.to_parquet(
                runs_configs_df_path.replace(filetype, "parquet"),
                index=False,
                engine="pyarrow",
            )
        except Exception as e:
            print(f"Error saving to parquet: {str(e)}")

        print(f"Regenerated and wrote {runs_configs_df_path} to disk.")
        del runs_configs_df

    print(f"Reading {runs_configs_df_path} from disk.")
    if filetype == "csv":
        runs_configs_df = pd.read_csv(runs_configs_df_path)
    elif filetype == "feather":
        runs_configs_df = pd.read_feather(runs_configs_df_path)
    elif filetype == "parquet":
        runs_configs_df = pd.read_parquet(runs_configs_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    print(f"Loaded {runs_configs_df_path} from disk.")

    # Keep only finished runs
    finished_runs = runs_configs_df["State"] == "finished"
    print(
        f"% of successfully finished runs: {100.0 * finished_runs.mean():.2f}% ({finished_runs.sum()} / {len(finished_runs)})"
    )

    if finished_only:
        runs_configs_df = runs_configs_df[finished_runs]

        # Check that we don't have an empty data frame.
        assert len(runs_configs_df) > 0

        # Ensure we aren't working with a slice.
        runs_configs_df = runs_configs_df.copy()

    return runs_configs_df


def download_wandb_pretraining_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    max_workers: int = 10,  # New parameter to control the number of parallel workers
):
    pt_runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
        wandb_project_path=wandb_project_path,
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
        wandb_username=wandb_username,
        finished_only=finished_only,
        filetype=filetype,
        max_workers=max_workers,
    )

    # Extract information to make analyzing and visualizing easier.
    pt_runs_configs_df["Num. Replicas Per Epoch"] = pt_runs_configs_df[
        "data_config"
    ].apply(
        lambda data_config: ast.literal_eval(data_config)[
            "num_benchmark_replicas_per_epoch"
        ]
    )
    pt_runs_configs_df["Benchmark"] = pt_runs_configs_df["data_config"].apply(
        lambda data_config: ast.literal_eval(data_config)["benchmark"]
    )
    pt_runs_configs_df["Benchmark Subset Fraction"] = pt_runs_configs_df[
        "data_config"
    ].apply(
        lambda data_config: ast.literal_eval(data_config)["benchmark_subset_fraction"]
    )

    def compute_number_of_benchmark_tokens_per_replica(row: pd.Series):
        if row["Benchmark"] == "EleutherAI/minerva_math":
            num_tokens_in_benchmark = 1.5e6
        else:
            raise NotImplementedError
        return num_tokens_in_benchmark * row["Benchmark Subset Fraction"]

    pt_runs_configs_df["Benchmark Subset Num. Tokens"] = pt_runs_configs_df.apply(
        compute_number_of_benchmark_tokens_per_replica,
        axis=1,
    )

    pt_runs_configs_df["Num. Epochs"] = pt_runs_configs_df["train/epoch"]
    pt_runs_configs_df["Num. Replicas"] = (
        pt_runs_configs_df["Num. Replicas Per Epoch"]
        * pt_runs_configs_df["Num. Epochs"]
    )
    pt_runs_configs_df["Model"] = pt_runs_configs_df["model_config"].apply(
        lambda model_config: ast.literal_eval(model_config)["model_name"]
    )
    # pt_runs_configs_df["Num. Parameters"] = pt_runs_configs_df["Model"].apply(
    #     src.analyze.extract_num_model_parameters
    # )
    pt_runs_configs_df["Num. Parameters"] = pt_runs_configs_df["model/num_parameters"]
    pt_runs_configs_df["Num. Tokens"] = pt_runs_configs_df[
        "eval_after/num_input_tokens_seen"
    ]
    pt_runs_configs_df["FLOP (6ND)"] = (
        6 * pt_runs_configs_df["Num. Parameters"] * pt_runs_configs_df["Num. Tokens"]
    )

    # Use slightly nicer column names.
    pt_runs_configs_df["benchmark_loss"] = pt_runs_configs_df[
        "eval_after/eval_benchmark_loss"
    ]
    pt_runs_configs_df["eval_loss"] = pt_runs_configs_df["eval_after/eval_eval_loss"]

    return pt_runs_configs_df


def download_wandb_project_runs_configs_helper(run):
    try:
        summary = run.summary._json_dict
        summary.update({k: v for k, v in run.config.items() if not k.startswith("_")})
        summary.update(
            {
                "State": run.state,
                "Sweep": run.sweep.id if run.sweep is not None else None,
                "run_id": run.id,
                "run_name": run.name,
            }
        )
        return summary
    except Exception as e:
        print(f"Error processing run {run.id}: {str(e)}")
        return None


def download_wandb_project_runs_histories(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    wandb_run_history_num_samples: int = 10000,
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    nrows_to_read: Optional[int] = None,
    cols_to_drop: Optional[List[str]] = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    assert filetype in {"csv", "feather", "parquet"}

    # Hash because otherwise too long.
    filename = "sweeps=" + ",".join(sweep_ids)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    runs_histories_df_path = os.path.join(
        data_dir, hashed_filename + f"_runs_histories.{filetype}"
    )
    if refresh or not os.path.isfile(runs_histories_df_path):
        # Download sweep results
        api = wandb.Api(timeout=6000)

        if wandb_username is None:
            wandb_username = api.viewer.username

        runs_histories_list = []
        print("Downloading runs' histories...")
        for sweep_id in sweep_ids:
            sweep = api.sweep(f"{wandb_username}/{wandb_project_path}/{sweep_id}")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_run = {
                    executor.submit(
                        download_wandb_project_runs_histories_helper,
                        run,
                        wandb_run_history_num_samples,
                        cols_to_drop,
                    ): run
                    for run in sweep.runs
                }

                for future in tqdm(
                    concurrent.futures.as_completed(future_to_run),
                    total=len(future_to_run),
                ):
                    run = future_to_run[future]
                    try:
                        history = future.result()
                        if history is not None:
                            runs_histories_list.append(history)
                    except Exception as exc:
                        print(f"{run.id} generated an exception: {exc}")

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        runs_histories_df.sort_values(["run_id"], ascending=True, inplace=True)
        runs_histories_df.reset_index(inplace=True, drop=True)

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        runs_histories_df.sort_values(["run_id"], ascending=True, inplace=True)
        runs_histories_df.reset_index(inplace=True, drop=True)

        runs_histories_df.rename(columns={"id": "seq_id"}, inplace=True)
        runs_histories_df["run_id+seq_id"] = runs_histories_df.apply(
            lambda row: f"{row['run_id']}_{row['seq_id']}", axis=1
        )

        # Save all three because otherwise this is a pain in the ass.
        # runs_histories_df.to_csv(
        #     runs_histories_df_path.replace(filetype, "csv"), index=False
        # )
        try:
            runs_histories_df.to_feather(
                runs_histories_df_path.replace(filetype, "feather")
            )
        except BaseException:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        try:
            runs_histories_df.to_parquet(
                runs_histories_df_path.replace(filetype, "parquet"),
                index=False,
                engine="pyarrow",
            )
        except pyarrow.lib.ArrowInvalid:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        print(f"Wrote {runs_histories_df_path} to disk")
        del runs_histories_df

    print(f"Loading {runs_histories_df_path} from disk.")
    if filetype == "csv":
        runs_histories_df = pd.read_csv(runs_histories_df_path, nrows=nrows_to_read)
    elif filetype == "feather":
        runs_histories_df = pd.read_feather(runs_histories_df_path)
    elif filetype == "parquet":
        runs_histories_df = pd.read_parquet(runs_histories_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    print(f"Loaded {runs_histories_df_path} from disk.")

    return runs_histories_df


def download_wandb_project_runs_histories_helper(
    run,
    wandb_run_history_num_samples: int,
    cols_to_drop: Optional[List[str]] = None,
):
    history = None
    for num_attempts in range(5):
        try:
            history = run.history(samples=wandb_run_history_num_samples)
            break
        except (requests.exceptions.HTTPError, wandb.errors.CommError):
            print(f"Retrying run {run.id}...")
            time.sleep(3)

    if history is None or history.empty:
        return None

    if cols_to_drop is not None:
        history.drop(columns=cols_to_drop, inplace=True)
    history["run_id"] = run.id
    return history


def extract_hf_model_name_or_path(model_config: str) -> str:
    hf_model_name_or_path = ast.literal_eval(model_config)["model"]
    return hf_model_name_or_path


def extract_num_model_parameters(model_name: str) -> int:
    if model_name.startswith("RylanSchaeffer"):
        # "RylanSchaeffer/mem_Qwen3-34M_minerva_math_replicas_0_epch_1_ot_1_pt"
        # will become "Qwen3-34M".
        base_model_name = model_name.replace(
            "RylanSchaeffer/scale_mem_Qwen3-", ""
        ).split("_")[0]
    else:
        match = re.search(r"model_(.*)_dataset", model_name)
        if match:
            base_model_name = match.group(1)
        else:
            # Assume base model. Drop the organization name and take only the base model name.
            # Example: "Qwen/Qwen2.5-1.5B" becomes "Qwen2.5-1.5B"
            base_model_name = model_name.split("/")[-1]
    num_model_parameters = src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[base_model_name]
    return num_model_parameters


def extract_num_train_epochs(model_name: str) -> int:
    if "RylanSchaeffer" not in model_name:
        # Base model. We assume 0 epochs.
        return 0
    match = re.search(r"_epochs_(\d+)_seed_", model_name)
    if not match:
        raise ValueError
    num_epochs = int(match.group(1))
    return num_epochs


def extract_pretraining_dataset_name_for_eval_analysis(model_name: str) -> str:
    pattern = r"epch_(?P<epoch>\d+)_ot_(?P<overtraining>\d+)_s_(?P<seed>\d+)_dir_(?P<direction>[^_]+)_shfs_(?P<shuffle_seed>\d+)_ttss_(?P<split_seed>\d+)"
    match = re.search(pattern, model_name)
    if match:
        dataset_name = match.group(0)
    # # if match:
    # #     config = match.groupdict()
    # #     # Convert numeric strings to integers where appropriate
    # #     for key in ["epoch", "overtraining", "seed", "shuffle_seed", "split_seed"]:
    # #         config[key] = int(config[key])
    # dataset_name = f"epch_{num_train_epochs}_ot_{overtrain_multiplier}_s_{seed}_dir_{direction}_shfs_{shuffle_seed}_ttss_{train_test_split_seed}"
    else:
        raise ValueError(f"Model name '{model_name}' does not match expected pattern.")
    return dataset_name


def fit_neural_scaling_law(
    df: pd.DataFrame,
    x_col: str = "Pretraining Compute",
    y_col: str = "neg_log_",
    exclude_nans: bool = True,
    additional_columns_to_add: List[str] | None = None,
    n_workers: int = 10,
) -> Dict[str, float]:
    x_vals_all = df[x_col].values
    y_vals_all = df[y_col].values
    if exclude_nans:
        nan_mask = np.isnan(x_vals_all) | np.isnan(y_vals_all)
        print(f"Excluding {np.sum(nan_mask)} NaNs out of {len(nan_mask)} entries")
        x_vals = x_vals_all[~nan_mask]
        y_vals = y_vals_all[~nan_mask]
    else:
        x_vals = np.copy(x_vals_all)
        y_vals = np.copy(y_vals_all)

    if len(x_vals) >= 3:
        # Fit a power law loss = E + A * FLOPS^(-alpha) via linear regression in log space.
        best_fit_result, y_all_pred = src.neural_scaling_laws.fit_chinchilla_scaling(
            x_all=x_vals,
            y_all=y_vals,
            functional_form="compute",
            n_workers=n_workers,
        )
        fit_results_dict = dict(
            covariate_cols=x_col,
            target_col=y_col,
            fit_loss=best_fit_result.fit_loss,
            fit_converged=best_fit_result.converged,
        )
        for k, v in best_fit_result.fit_params.items():
            fit_results_dict[f"fit_param_{k}"] = v

        # Convert the log-space parameters to the original space.
        fit_results_dict["fit_param_C_0"] = np.exp(fit_results_dict["fit_param_c_0"])
        fit_results_dict["fit_param_E_0"] = np.exp(fit_results_dict["fit_param_e_0"])

    else:
        # Create a mock fit result with NaNs.
        fit_results_dict = dict(
            covariate_cols=x_col,
            target_col=y_col,
            fit_loss=np.nan,
            fit_converged=False,
        )

    if additional_columns_to_add is not None:
        for col in additional_columns_to_add:
            fit_results_dict[col] = df[col].values[0]

    return fit_results_dict


def setup_notebook_dir(
    notebook_dir: str,
    refresh: bool = False,
) -> Tuple[str, str]:
    # Declare paths.
    data_dir = os.path.join(notebook_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    results_dir = os.path.join(notebook_dir, "results")
    if refresh:
        import shutil

        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir
