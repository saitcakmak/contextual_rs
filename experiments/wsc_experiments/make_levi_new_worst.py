"""
Used for making worst PCS versions of LEVI new for Branin and Greiwank experiments.
The candidate generation is identical between mean and worst for these two problems.
"""
import json
import sys
from os import path
from time import time
from typing import Union, List, Optional

import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.test_functions import Branin, Hartmann, Griewank, Cosine8, Powell, Levy
from botorch.utils.transforms import unnormalize
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor

from contextual_rs.contextual_rs_strategies import (
    li_sampling_strategy,
    gao_sampling_strategy,
    gao_modellist,
    sur_modellist,
)
from contextual_rs.finite_ikg import (
    finite_ikg_maximizer_modellist,
)
from contextual_rs.generalized_pcs import (
    estimate_current_generalized_pcs,
)
from contextual_rs.levi import discrete_levi
from contextual_rs.models.contextual_independent_model import ContextualIndependentModel

labels = [
    "LEVI_new",
]


def main(
    iterations: int,
    seed: int,
    label: str,
    use_full_train: bool,
    output_path: str,
    rho_key: str = "mean",
    num_pcs_samples: int = 64,
    num_arms: int = 10,
    num_contexts: int = 10,
    context_dim: int = 1,
    weights: Optional[Tensor] = None,
    num_full_train: Optional[int] = None,
    num_train_per_arm: Optional[int] = None,
    fit_frequency: int = 1,  # set > 1 if we don't want to fit the GP at each iteration.
    ground_truth_kwargs: dict = None,
    input_dict: dict = None,  # this is for adding more iterations
    randomize_ties: bool = True,
    mode: Optional[str] = None,
    dtype: torch.dtype = torch.double,
    device: Optional[str] = None,
) -> dict:
    r"""
    Run a single replication of the experiment with the specified algorithm.
    This is typically used with a config file and the helpers found at the bottom
    to run a set of experiments on a cluster / server.

    Args:
        iterations: Number of iterations to run the experiment for.
        seed: The seed to use for this replication.
        label: This is the algorithm label, select from the list above.
        use_full_train: If True, we evaluate all arm context pairs for initialization.
        output_path: The file path for saving the experiment output.
        rho_key: "mean" or "worst". Specifies the contextual PCS to use.
        num_pcs_samples: Number of posterior samples used to estimate the current
            contextual PCS. This is just an estimate, not the actual reported PCS.
        num_arms: Number of arms to use.
        num_contexts: Number of contexts to use.
        context_dim: The dimension of the context tensors. This should be
            `function.dim - 1`.
        weights: The optional weights if using mean PCS.
        num_full_train: Number of "full" training samples. Number of samples drawn
            from each arm-context pair.
        num_train_per_arm: If not using samples for all arm-context pairs for training,
            this is the number of samples drawn for each arm. This many contexts are
            randomly selected for each arm and those are evaluated.
        fit_frequency: How often to fit the hyper-parameters of the GP models. Larger
            values will reduce the computational cost but may lead to poorer model
            predictions. 5-10 are safe choices.
        ground_truth_kwargs: Arguments for the ground truth model / function.
        input_dict: This is used if the experiment was terminated early and we want to
            continue from where we left off. The experiment will warm-start from here
            and continue up to a total number of `iterations`.
        randomize_ties: If there are multiple minimizers / maximizers, this determines
            whether one of them is picked randomly or as the one that is simply returned
            by the min/max operator.
        mode: If `input_dict` is specified, mode must be "-a", as in append, to add
            more iterations. No other modes are currently supported.
        dtype: Tensor data type to use.
        device: The device to use, "cpu" / "cuda".
    """
    if device is None:
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
    assert label in labels, "label not supported!"
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckwargs = {"dtype": dtype, "device": device}
    if weights is not None:
        weights = torch.as_tensor(weights, **ckwargs).view(-1)

    def rho(X: Tensor) -> Tensor:
        r"""Operates on -2 dimension. Dimension gets reduced by 1."""
        if rho_key == "mean":
            if weights is not None:
                X = X * weights.view(-1, 1).expand_as(X)
            return X.mean(dim=-2)
        elif rho_key == "worst":
            min_, _ = X.min(dim=-2)
            return min_
        else:
            raise NotImplementedError

    if input_dict is None:
        raise RuntimeError("Input dict is required!")
    # read the given output file and continue from there.
    assert input_dict["label"] == label
    existing_iterations = input_dict["pcs_estimates"].shape[0]
    if existing_iterations != iterations:
        raise ValueError("Not enough iterations in the output!")
    input_dict["pcs_estimates"] = None
    input_dict["rho_cs"] = rho(input_dict["correct_selection"].unsqueeze(-1)).squeeze(-1)
    torch.save(input_dict, output_path)
    return input_dict


def submitit_main(
    config: str, base_config: str, label: str, seed: Union[int, str], last_arg=None
) -> None:
    r"""
    This is used with `submit.py` to submit jobs to a slurm cluster.

    Args:
        config: The name of the config folder.
        base_config: The source experiment output to transfer.
        label: The algorithm label.
        seed: The seed to run.
        last_arg: Used for force re-running an experiment, continuing
            an incomplete one etc.
    """
    current_dir = path.dirname(path.abspath(__file__))
    exp_dir = path.join(current_dir, config)
    config_path = path.join(exp_dir, "config.json")
    seed = int(seed)
    output_path = path.join(exp_dir, f"{str(seed).zfill(4)}_{label}.pt")
    if path.exists(output_path):
        raise RuntimeError("Output file exists!")
    input_dir = path.join(current_dir, base_config)
    input_path = path.join(input_dir, f"{str(seed).zfill(4)}_{label}.pt")
    input_dict = torch.load(input_path)
    with open(config_path, "r") as f:
        kwargs = json.load(f)
        if (
            kwargs["ground_truth_kwargs"]["function"] in ["cosine8", "hartmann"]
            and label == "ML_IKG"
        ):
            kwargs["iterations"] = min(kwargs["iterations"], 1000)
    output = main(
        seed=seed,
        label=label,
        input_dict=input_dict,
        mode=None,
        output_path=output_path,
        **kwargs,
    )
    torch.save(output, output_path)


if __name__ == "__main__":
    config = sys.argv[1]
    base_config = sys.argv[2]
    label = sys.argv[3]
    seed = sys.argv[4]
    submitit_main(config, base_config, label, seed)
