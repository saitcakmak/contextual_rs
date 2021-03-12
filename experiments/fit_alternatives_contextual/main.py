"""
Analyzing the effect of multiple fit tries on the performance of LCEGP under
the contextual setting.
"""
import json
import math
import sys
from os import path
from time import time
from typing import Union, List

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.sampling import SobolQMCNormalSampler
from torch import Tensor

from contextual_rs.models.custom_fit import custom_fit_gpytorch_model
from gpytorch import ExactMarginalLogLikelihood

from contextual_rs.models.lce_gp import LCEGP
from contextual_rs.generalized_pcs import (
    estimate_lookahead_generalized_pcs,
    estimate_current_generalized_pcs,
)


class GroundTruthModel:
    def __init__(
        self,
        num_arms: int,
        context_map: Tensor,
        num_init_samples: int = None,
        init_scale: float = 50.0,
        observation_noise: float = 3.0,
    ):
        r"""
        Generate a GP model for use as the ground truth for function evaluations.
        The dimension is inferred from context_map, as context_map.shape[-1] + 1.
        """
        self.num_arms = num_arms
        self.arm_map = torch.linspace(0, 1, num_arms).view(-1, 1).to(context_map)
        self.context_map = context_map
        self.dim = context_map.shape[-1] + 1
        num_init_samples = num_init_samples or self.dim * 10
        train_X = torch.rand(num_init_samples, self.dim).to(context_map)
        train_Y = torch.randn(num_init_samples, 1).to(context_map) * init_scale
        self.model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        self.observation_noise = observation_noise

    def evaluate_true(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        r"""
        Evaluate the posterior mean at the given arm and context.
        If arm is int, context should be `1 x d` tensor.
        If arm is a list of size n, context should be `n x d` tensor.
        Returns a `n x 1`-dim tensor.
        """
        arms = self.arm_map[arm_idx].view(-1, 1)
        X = torch.cat([arms, context], dim=-1)
        return self.model.posterior(X).mean.view(-1, 1).detach()

    def evaluate(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        true_evals = self.evaluate_true(arm_idx, context)
        return true_evals + torch.randn_like(true_evals) * self.observation_noise

    def evaluate_w_index(
        self, arm_idx: Union[List, int], context_idx: Union[List, int]
    ) -> Tensor:
        context = torch.atleast_2d(self.context_map[context_idx])
        return self.evaluate(arm_idx, context)

    def evaluate_all_true(self):
        r"""
        Evaluates all arm-context pairs without noise.
        """
        X = torch.cat(
            [
                self.arm_map.view(-1, 1, 1).repeat(1, self.context_map.shape[0], 1),
                self.context_map.repeat(self.num_arms, 1, 1),
            ],
            dim=-1,
        ).view(-1, self.dim)
        return self.model.posterior(X).mean.view(-1, 1).detach()

    def evaluate_all(self):
        true_evals = self.evaluate_all_true()
        return true_evals + torch.randn_like(true_evals) * self.observation_noise


def fit_lcegp(X, Y, fit_tries):
    model = LCEGP(
        X,
        Y,
        categorical_cols=[0],
        embs_dim_list=[1],
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    custom_fit_gpytorch_model(mll, num_retries=fit_tries)
    return model


labels = ["5-1", "10-1", "25-1", "10-5", "25-5", "25-10"]
fit_frequency_list = [5, 10, 25, 10, 25, 25]
fit_tries_list = [1, 1, 1, 5, 5, 10]
num_labels = len(labels)


def main(
    iterations: int,
    seed: int,
    subset_labels: List = None,
    num_pcs_samples: int = 64,
    num_arms: int = 10,
    num_contexts: int = 10,
    context_dim: int = 1,
    num_full_train: int = 3,
    ground_truth_kwargs: dict = None,
    batch_size: int = 20,
    num_fantasies: int = 0,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
) -> dict:
    global labels, num_labels
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckwargs = {"dtype": dtype, "device": device}
    context_map = torch.rand(num_contexts, context_dim, **ckwargs)
    ground_truth_kwargs = ground_truth_kwargs or dict()
    ground_truth = GroundTruthModel(num_arms, context_map, **ground_truth_kwargs)
    if subset_labels:
        labels = [labels[i] for i in subset_labels]
        num_labels = len(labels)
        raise NotImplementedError(
            "The current implementation with subset_labels is buggy!"
        )

    true_means = ground_truth.evaluate_all_true()
    tm_maximizers = true_means.view(num_arms, num_contexts).argmax(dim=0)

    arm_set = torch.arange(0, num_arms, **ckwargs).view(-1, 1)
    train_X = (
        torch.cat(
            [
                arm_set.view(-1, 1, 1).expand(-1, num_contexts, -1),
                context_map.expand(num_arms, -1, -1),
            ],
            dim=-1,
        )
        .view(-1, context_dim + 1)
        .repeat(num_full_train, 1)
    )
    train_Y = torch.cat(
        [ground_truth.evaluate_all() for _ in range(num_full_train)], dim=0
    )

    start = time()
    all_alternatives = train_X[: num_arms * num_contexts].clone()
    pcs_estimates = [torch.zeros(iterations, **ckwargs) for _ in range(num_labels)]
    correct_selection = [
        torch.zeros(iterations, num_contexts, **ckwargs) for _ in range(num_labels)
    ]
    X_list = [train_X.clone() for _ in range(num_labels)]
    Y_list = [train_Y.clone() for _ in range(num_labels)]
    old_models = [None for _ in range(num_labels)]
    for i in range(iterations):
        if i % 10 == 0:
            print(f"Starting seed {seed}, iteration {i}, time: {time()-start}")
        for j in range(num_labels):
            if i % fit_frequency_list[j] != 0:
                # in this case, we will not re-train LCEGP.
                # we will just condition on observations.
                model = old_models[j].condition_on_observations(
                    X=X_list[j][-1].view(1, -1),
                    Y=Y_list[j][-1].view(1, 1),
                )
            else:
                model = fit_lcegp(X_list[j], Y_list[j], fit_tries_list[j])

            # LCEGP
            old_models[j] = model
            pcs_vals = torch.zeros(all_alternatives.shape[0], **ckwargs)
            num_batches = math.ceil(all_alternatives.shape[0] / float(batch_size))
            for k in range(num_batches):
                l_idx = k * batch_size
                r_idx = min(l_idx + batch_size, all_alternatives.shape[0])
                pcs_vals[l_idx:r_idx] = estimate_lookahead_generalized_pcs(
                    candidate=all_alternatives[l_idx:r_idx].view(
                        -1, 1, context_dim + 1
                    ),
                    model=model,
                    model_sampler=SobolQMCNormalSampler(num_samples=num_fantasies)
                    if num_fantasies
                    else None,
                    arm_set=arm_set,
                    context_set=context_map,
                    num_samples=64,
                    base_samples=None,
                    func_I=lambda X: (X > 0).to(**ckwargs),
                    rho=lambda X: X.mean(dim=-2),
                )
            maximizer = pcs_vals.argmax()
            next_arm = maximizer // num_contexts
            next_context = maximizer % num_contexts
            next_point = torch.cat(
                [torch.tensor([next_arm], **ckwargs), context_map[next_context]]
            ).view(1, -1)

            next_eval = ground_truth.evaluate_w_index(next_arm, next_context)

            X_list[j] = torch.cat([X_list[j], next_point], dim=0)
            Y_list[j] = torch.cat([Y_list[j], next_eval], dim=0)

            # report current PCS estimate
            pcs_estimates[j][i] = estimate_current_generalized_pcs(
                model=model,
                arm_set=arm_set,
                context_set=context_map,
                num_samples=num_pcs_samples,
                base_samples=None,
                func_I=lambda X: (X > 0).to(**ckwargs),
                rho=lambda X: X.mean(dim=-2),
            )

            # check for correct selection for empirical PCS
            post_mean = model.posterior(all_alternatives).mean.view(
                num_arms, num_contexts
            )

            maximizers = post_mean.argmax(dim=0)

            correct_selection[j][i] = tm_maximizers == maximizers

    output_dict = {
        "labels": labels,
        "X_list": X_list,
        "Y_list": Y_list,
        "true_means": true_means,
        "pcs_estimates": pcs_estimates,
        "correct_selection": correct_selection,
    }
    return output_dict


if __name__ == "__main__":
    current_dir = path.dirname(path.abspath(__file__))
    exp_dir = path.join(current_dir, sys.argv[1])
    config_path = path.join(exp_dir, "config.json")
    seed = int(sys.argv[2])
    output_path = path.join(exp_dir, f"{str(seed).zfill(4)}.pt")
    if path.exists(output_path):
        if len(sys.argv) > 3 and sys.argv[3] == "-f":
            print("Overwriting the existing output!")
        else:
            print(
                "The output file exists for this experiment & seed!"
                "Pass -f as the 3rd argument to overwrite!"
            )
            quit()
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    output = main(seed=seed, **kwargs)
    torch.save(output, output_path)
