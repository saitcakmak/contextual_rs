"""
Comparing the selection strategies of Li and Gao with LCEGP w/ lookahead contextual PCS,
where the objective is expected PCS over contexts, and no covariance / distance
structure over arms and contexts is known beforehand, i.e., LCEGP treats all variables
as categorical.

The ground truth is the Example 1 of Li, in which each arm-context pair has a reward
with true mean sampled from N(50, 9). To simplify things, we will go with homoscedastic
observation noise with noise standard deviation of 10. (all can be overwritten)
"""
import json
import math
import sys
from os import path
from time import time

import numpy as np
import torch
from botorch.models.transforms import Standardize
from botorch.sampling import SobolQMCNormalSampler

from contextual_rs.models.custom_fit import custom_fit_gpytorch_model
from gpytorch import ExactMarginalLogLikelihood

from contextual_rs.models.lce_gp import LCEGP
from contextual_rs.models.contextual_independent_model import ContextualIndependentModel
from contextual_rs.generalized_pcs import (
    estimate_lookahead_generalized_pcs,
    estimate_current_generalized_pcs,
)
from contextual_rs.contextual_rs_strategies import (
    li_sampling_strategy,
    gao_sampling_strategy,
)


def model_constructor(model_type: str):
    def model_create(X, Y, emb_dim_list, fit_tries, standardize):
        if model_type == "LCEGP":
            model = LCEGP(
                X,
                Y,
                categorical_cols=[0, 1],
                embs_dim_list=emb_dim_list,
                outcome_transform=Standardize(m=1) if standardize else None,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            custom_fit_gpytorch_model(mll, num_retries=fit_tries)
            return model
        else:
            model = ContextualIndependentModel(X, Y.squeeze(-1))
            return model

    return model_create


labels = ["LCEGP", "Li", "Gao"]
num_labels = len(labels)

model_constructor_list = [model_constructor(m_type) for m_type in labels]


def main(
    iterations: int,
    seed: int,
    num_pcs_samples: int = 64,
    num_arms: int = 10,
    num_contexts: int = 10,
    num_full_train: int = 5,
    fit_frequency: int = 1,  # set > 1 if we don't want to fit LCEGP at each iteration.
    fit_tries: int = 1,
    standardize: int = 0,  # 0 or 1
    emb_dim_list: list = None,
    obs_noise: float = 10.0,
    true_mean_loc: float = 50.0,
    true_mean_scale: float = 3.0,
    batch_size: int = 20,
    skip_lcegp: bool = False,  # if True, run only the benchmarks for sanity check.
    input_dict: dict = None,  # this is for adding more iterations
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    emb_dim_list = emb_dim_list or [1, 1]
    ckwargs = {"dtype": dtype, "device": device}
    skip_lcegp = int(skip_lcegp)

    true_means = (
        torch.randn(num_arms, num_contexts, **ckwargs) * true_mean_scale + true_mean_loc
    )
    tm_maximizers = true_means.view(num_arms, num_contexts).argmax(dim=0)

    def get_single_observation(arm, context):
        return torch.randn(1, **ckwargs) * obs_noise + true_means[arm, context]

    def get_initial_observations(num_samples: int):
        return torch.randn(
            num_samples, num_arms, num_contexts, **ckwargs
        ) * obs_noise + true_means.expand(num_samples, -1, -1)

    arm_set = torch.arange(0, num_arms, **ckwargs).view(-1, 1)
    context_set = torch.arange(0, num_contexts, **ckwargs).view(-1, 1)
    train_X = (
        torch.cat(
            [
                arm_set.view(-1, 1, 1).expand(-1, num_contexts, -1),
                context_set.view(1, -1, 1).expand(num_arms, -1, -1),
            ],
            dim=-1,
        )
        .view(-1, 2)
        .repeat(num_full_train, 1)
    )
    train_Y = get_initial_observations(num_full_train).view(-1, 1)

    start = time()
    all_alternatives = train_X[: num_arms * num_contexts].clone()
    existing_iterations = 0
    pcs_estimates = [torch.zeros(iterations, **ckwargs) for _ in range(num_labels)]
    correct_selection = [
        torch.zeros(iterations, num_contexts, **ckwargs) for _ in range(num_labels)
    ]
    X_list = [train_X.clone() for _ in range(num_labels)]
    Y_list = [train_Y.clone() for _ in range(num_labels)]
    if input_dict is not None:
        existing_iterations = input_dict["pcs_estimates"][0].shape[0]
        if existing_iterations >= iterations:
            raise ValueError("Existing output has as many or more iterations!")
        for j in range(num_labels):
            pcs_estimates[j][:existing_iterations] = input_dict["pcs_estimates"][j]
            correct_selection[j][:existing_iterations] = input_dict[
                "correct_selection"
            ][j]
            X_list[j] = input_dict["X_list"][j]
            Y_list[j] = input_dict["Y_list"][j]
    old_lcegp = None
    for i in range(existing_iterations, iterations):
        if i % 10 == 0:
            print(f"Starting seed {seed}, iteration {i}, time: {time()-start}")
        for j in range(skip_lcegp, num_labels):
            constructor = model_constructor_list[j]
            if j == 0 and (i - existing_iterations) % fit_frequency != 0:
                # in this case, we will not re-train LCEGP.
                # we will just condition on observations.
                model = old_lcegp.condition_on_observations(
                    X=X_list[j][-1].view(1, 2),
                    Y=Y_list[j][-1].view(1, 1),
                )
            else:
                model = constructor(
                    X_list[j], Y_list[j], emb_dim_list, fit_tries, standardize
                )

            if j == 0:
                # LCEGP
                old_lcegp = model
                pcs_vals = torch.zeros(all_alternatives.shape[0], **ckwargs)
                num_batches = math.ceil(all_alternatives.shape[0] / float(batch_size))
                for k in range(num_batches):
                    l_idx = k * batch_size
                    r_idx = min(l_idx + batch_size, all_alternatives.shape[0])
                    pcs_vals[l_idx:r_idx] = estimate_lookahead_generalized_pcs(
                        candidate=all_alternatives[l_idx:r_idx].view(-1, 1, 2),
                        model=model,
                        model_sampler=SobolQMCNormalSampler(num_samples=16),
                        arm_set=arm_set,
                        context_set=context_set,
                        num_samples=64,
                        base_samples=None,
                        func_I=lambda X: (X > 0).to(**ckwargs),
                        rho=lambda X: X.mean(dim=-2),
                    )
                maximizer = pcs_vals.argmax()
                next_arm = maximizer // num_contexts
                next_context = maximizer % num_contexts
            elif j == 1:
                # Li
                next_arm, next_context = li_sampling_strategy(model)
                pass
            else:
                # Gao
                next_arm, next_context = gao_sampling_strategy(model)

            next_point = torch.tensor([[next_arm, next_context]], **ckwargs)
            next_eval = get_single_observation(next_arm, next_context).view(-1, 1)

            X_list[j] = torch.cat([X_list[j], next_point], dim=0)
            Y_list[j] = torch.cat([Y_list[j], next_eval], dim=0)

            # report current PCS estimate
            pcs_estimates[j][i] = estimate_current_generalized_pcs(
                model=model,
                arm_set=arm_set,
                context_set=context_set,
                num_samples=num_pcs_samples,
                base_samples=None,
                func_I=lambda X: (X > 0).to(**ckwargs),
                rho=lambda X: X.mean(dim=-2),
            )

            # check for correct selection for empirical PCS
            if j == 0:
                post_mean = model.posterior(
                    all_alternatives.view(num_arms, num_contexts, 2)
                ).mean.squeeze(-1)
            else:
                post_mean = model.means

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
    input_dict = None
    if path.exists(output_path):
        if len(sys.argv) > 3 and sys.argv[3] in ["-a", "-f"]:
            if sys.argv[3] == "-f":
                print("Overwriting the existing output!")
            else:
                print(
                    "Appending iterations to existing output!"
                    "Warning: If parameters other than `iterations` have "
                    "been changed, this will corrupt the output!"
                )
                input_dict = torch.load(output_path)
        else:
            print(
                "The output file exists for this experiment & seed!"
                "Pass -f as the 3rd argument to overwrite!"
                "Pass -a as the 3rd argument to add more iterations!"
            )
            quit()
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    output = main(seed=seed, input_dict=input_dict, **kwargs)
    torch.save(output, output_path)
