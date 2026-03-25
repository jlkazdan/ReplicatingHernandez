from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
import itertools
import logging
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scipy.special
from typing import Dict, List, Optional, Tuple


@dataclass
class FitResult:
    """Results from fitting the scaling law"""

    fit_params: OrderedDict[str, float]
    fit_loss: float
    initial_params: OrderedDict[str, float]
    converged: bool


class PowerLawScalingFitter:
    def __init__(self, functional_form: str, n_workers: int = 30):
        assert functional_form in ["compute", "parameters_and_tokens"]
        self.functional_form = functional_form
        self.n_workers = n_workers
        if self.functional_form == "compute":
            self.grid_search_points = self.create_grid_search_points_compute()
            self.compute_log_pred_fn = self.compute_log_pred_from_compute
        elif self.functional_form == "parameters_and_tokens":
            self.grid_search_points = (
                self.create_grid_search_points_parameters_and_tokens()
            )
            self.compute_log_pred_fn = self.compute_log_pred_from_parameters_and_tokens
        self.best_fit_result = None

    def compute_huber_loss_of_diff_of_logs(
        self,
        grid_search_point: Dict[str, float],
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        log_pred = self.compute_log_pred_fn(
            grid_search_point=grid_search_point,
            x=x,
        )
        log_pred_minus_log_target = log_pred - np.log(y)
        losses = huber_loss(log_pred_minus_log_target)
        return np.mean(losses)

    @staticmethod
    def compute_log_pred_from_compute(
        grid_search_point: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        # Shape: (2, x.shape[0])
        concatenated = np.stack(
            [
                grid_search_point[0] - grid_search_point[1] * np.log(x[:, 0]),
                np.broadcast_to(np.array([grid_search_point[2]]), shape=(x.shape[0],)),
            ]
        )
        # Shape: x.shape[0]
        log_pred = scipy.special.logsumexp(concatenated, axis=0)
        return log_pred

    @staticmethod
    def compute_log_pred_from_parameters_and_tokens(
        grid_search_point: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        concatenated_terms = np.stack(
            [
                grid_search_point[0] - grid_search_point[1] * np.log(x[:, 0]),
                grid_search_point[2] - grid_search_point[3] * np.log(x[:, 1]),
                np.broadcast_to(
                    np.array([grid_search_point[4]]), shape=(x.shape[0],)
                ),  # manual broadcast. annoying.
            ]
        )
        log_pred = scipy.special.logsumexp(concatenated_terms, axis=0)
        return log_pred

    def create_grid_search_points_compute(self):
        # -log(pass@1) = A / C^{alpha} + E
        c_0_range = np.arange(0.0, 30.0, 2.5)
        alpha_range = np.arange(0.0, 2.0, 0.1)
        e_0_range = np.arange(-3.0, 3.0, 0.25)
        grid_search_points = [
            OrderedDict([("c_0", a), ("alpha", alpha), ("e_0", e)])
            for alpha, a, e in itertools.product(alpha_range, c_0_range, e_0_range)
        ]
        return grid_search_points

    def create_grid_search_points_parameters_and_tokens(self):
        alpha_range = np.arange(0, 5.0, 0.5)
        a_range = (0.0, 30.0, 5.0)
        beta_range = (0.0, 5.0, 0.5)
        b_range = (0.0, 30.0, 5.0)
        e_range = (-3.0, 3.0, 0.5)
        grid_search_points = [
            OrderedDict(
                [("a_0", a), ("alpha", alpha), ("b_0", b), ("beta", beta), ("e_0", e)]
            )
            for alpha, a, beta, b, e in itertools.product(
                alpha_range, a_range, beta_range, b_range, e_range
            )
        ]

        return grid_search_points

    def fit(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        # # Create a mask to exclude non-finite values.
        # # This can occur for at least 2 reasons:
        # #   1. The pretraining compute is 0.
        # #   2. The average score is 0 and thus the negative log average score is inf.
        # finite_mask = np.logical_and(
        #     np.logical_and.reduce(np.isfinite(x), axis=0), np.isfinite(y)
        # )
        # if finite_mask.sum() == 0:
        #     raise ValueError("No finite values in x and y.")
        x_finite = x  # [finite_mask]
        y_finite = y  # [finite_mask]

        # Parallelize fitting.
        # Create a partial function with fixed x and y arguments.
        optimize_point = partial(self.optimize_single_point, x=x_finite, y=y_finite)

        # Create a process pool - by default, we use the number of CPU cores.
        with Pool(processes=self.n_workers) as pool:
            # Map the optimization function across all grid points
            results = pool.map(optimize_point, self.grid_search_points)

        # Exclude all fits that did not converge.
        num_results = len(results)
        converged_results = [result for result in results if result.converged]
        num_converged_results = len(converged_results)
        fraction_converged_results = float(num_converged_results) / float(num_results)
        logging.info(
            f"Fraction of {num_results} fits that converged: {fraction_converged_results}"
        )

        # If no results were found, raise an error.
        if len(converged_results) == 0:
            # import matplotlib.pyplot as plt
            #
            # plt.close()
            # plt.plot(x, y)
            # plt.xscale("log")
            # plt.yscale("log")
            # plt.show()

            raise ValueError("No converged results found.")

        self.best_fit_result = min(converged_results, key=lambda r: r.fit_loss)
        return self.best_fit_result

    def optimize_single_point(
        self,
        grid_search_point: OrderedDict[str, float],
        x: np.ndarray,
        y: np.ndarray,
    ) -> FitResult:
        # If using "compute", then grid_search_point will have three (ordered) keys: a, alpha, e.
        result = scipy.optimize.minimize(
            self.compute_huber_loss_of_diff_of_logs,
            x0=np.array(list(grid_search_point.values())),
            args=(x, y),
            method="L-BFGS-B",
            options=dict(maxiter=50000),
        )

        # Ensure that the fit params have the same ordering as the initial params.
        fit_params = OrderedDict(
            [(k, v) for (k, v) in zip(grid_search_point.keys(), result.x)]
        )

        return FitResult(
            fit_params=fit_params,
            fit_loss=result.fun,
            initial_params=grid_search_point,
            converged=result.success,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.best_fit_result is None:
            raise ValueError("No fit has been performed yet.")
        return np.exp(
            self.compute_log_pred_fn(
                grid_search_point=np.array(
                    list(self.best_fit_result.fit_params.values())
                ),
                x=x,
            )
        )


def huber_loss(diffs: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    """
    Compute Huber loss without reduction.

    Args:
        diffs: Array of differences
        delta: Threshold for switching between L1 and L2 loss

    Returns:
        Element-wise Huber loss values
    """
    loss = 0.5 * np.square(diffs)
    return loss


def fit_chinchilla_scaling(
    x_all: np.ndarray,
    y_all: np.ndarray,
    functional_form: str = "compute",
    n_workers: int = 10,
) -> Tuple[FitResult, np.ndarray]:
    """
    Fit Chinchilla scaling laws using parallel grid search with L-BFGS-B optimization.
    """

    assert (
        x_all.shape[0] == y_all.shape[0]
    ), "x and y must have the same number of samples"

    # Make sure that x is a 2D array for shape consistency. If it isn't, expand.
    if len(x_all.shape) == 1:
        x_all = x_all[:, None]

    scaling_law_fitter = PowerLawScalingFitter(
        functional_form=functional_form,
        n_workers=n_workers,
    )
    best_fit_result = scaling_law_fitter.fit(x=x_all, y=y_all)
    logging.info(f"Initial params: {best_fit_result.initial_params}")
    logging.info(f"Best fit params: {best_fit_result.fit_params}")
    logging.info(f"Best fit loss: {best_fit_result.fit_loss}")

    y_all_pred = np.exp(
        scaling_law_fitter.compute_log_pred_fn(
            grid_search_point=np.array(list(best_fit_result.fit_params.values())),
            x=x_all,
        )
    )

    return best_fit_result, y_all_pred
