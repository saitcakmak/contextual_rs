"""
This piece of code adds empirical PCS to the existing output.
In doing so, we will be re-fitting the models, which may lead to
slightly different output for LCEGP than what would be reported from
the original experiment.
"""
import json
import sys
from os import path
from time import time
from typing import List

import numpy as np
import torch
from botorch.models.transforms import Standardize
from gpytorch import ExactMarginalLogLikelihood

from contextual_rs.models.contextual_independent_model import ContextualIndependentModel
from contextual_rs.models.custom_fit import custom_fit_gpytorch_model
from contextual_rs.models.lce_gp import LCEGP


def fix_output(
    iterations: int,
    input_dict: dict,
    seed: int,
    subset_labels: List = None,
    num_arms: int = 10,
    num_contexts: int = 10,
    context_dim: int = 1,
    num_full_train: int = 3,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckwargs = {"dtype": dtype, "device": device}

    if subset_labels is not None:
        raise NotImplementedError

    labels = input_dict["labels"]
    assert labels == ["1-1", "5-1" "10-1", "5-5", "10-10"]
    fit_frequency_list = [1, 5, 10, 5, 10]
    fit_tries_list = [1, 1, 1, 5, 10]
    num_labels = len(labels)
    pcs_estimates = input_dict["pcs_estimates"]

    train_data_size = num_arms * num_contexts * num_full_train

    true_means = input_dict["true_means"]
    tm_maximizers = true_means.argmax(dim=0)

    start = time()
    correct_selection = [
        torch.zeros(iterations, num_contexts, **ckwargs) for _ in range(num_labels)
    ]
    train_X = input_dict["train_X"]
    train_Y = input_dict["train_Y"]
    all_alternatives = train_X[: num_arms * num_contexts].view(
        num_arms, num_contexts, context_dim + 1
    )
    old_models = [None for _ in range(num_labels)]
    for i in range(iterations):
        if i % 10 == 0:
            print(f"Starting seed {seed}, iteration {i}, time: {time()-start}")
        for j in range(num_labels):
            if i % fit_frequency_list[j] != 0:
                model = old_models[j].condition_on_observations(
                    X=train_X[
                        train_data_size
                        + (i - 1) * num_contexts : train_data_size
                        + i * num_contexts
                    ].view(-1, context_dim + 1),
                    Y=train_Y[
                        train_data_size
                        + (i - 1) * num_contexts : train_data_size
                        + i * num_contexts
                    ].view(-1, 1),
                )
            else:
                model = LCEGP(
                    train_X[: train_data_size + i * num_contexts],
                    train_Y[: train_data_size + i * num_contexts],
                    categorical_cols=[0],
                    embs_dim_list=[1],
                    outcome_transform=Standardize(m=1),
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                custom_fit_gpytorch_model(mll, num_retries=fit_tries_list[j])
            old_models[j] = model

            post_mean = model.posterior(all_alternatives).mean.squeeze(-1)

            maximizers = post_mean.argmax(dim=0)

            correct_selection[j][i] = tm_maximizers == maximizers

    output_dict = {
        "labels": labels,
        "train_X": train_X,
        "train_Y": train_Y,
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
    input_dict = torch.load(output_path)
    if "correct_selection" in input_dict:
        print("Output already processed, skipping!")
        quit()
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    output = fix_output(input_dict=input_dict, seed=seed, **kwargs)
    torch.save(output, output_path)
