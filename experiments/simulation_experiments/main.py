"""
Comparing the GP-C-OCBA with IKG and DSCO and C-OCBA.
This is for using the Covid & Cancer simulators.
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
from botorch.models.transforms import Standardize, Normalize
from botorch.utils.transforms import unnormalize
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor

from contextual_rs.contextual_rs_strategies import (
    li_sampling_strategy,
    gao_sampling_strategy,
    gao_modellist,
)
from contextual_rs.finite_ikg import (
    finite_ikg_maximizer_modellist,
)
from contextual_rs.models.contextual_independent_model import ContextualIndependentModel
from contextual_rs.test_functions.covid_exp_class import CovidSim, CovidEval
from contextual_rs.test_functions.esophageal_cancer import EsophagealCancer


class SimulatorWrapper:
    """A class that wraps the simulators and implements some helper methods."""

    covid_arms = torch.tensor(
        [
            [0.2, 0.2],
            [0.2, 0.3],
            [0.2, 0.4],
            [0.2, 0.5],
            [0.2, 0.6],
            [0.3, 0.2],
            [0.3, 0.3],
            [0.3, 0.4],
            [0.3, 0.5],
            [0.4, 0.2],
            [0.4, 0.3],
            [0.4, 0.4],
            [0.5, 0.2],
            [0.5, 0.3],
            [0.6, 0.2],
        ]
    )

    def __init__(
        self,
        function: str,
        **ckwargs,
    ) -> None:
        r"""
        Construct the wrapper for sampling from the simulators.

        Args:
            context_map: A tensor denoting the locations of the contexts in the
                function input space. This should be `num_contexts x (function.dim - 1)`
                dimensional.
            function: The name of the base test function.
            ckwargs: Common tensor arguments, dtype and device.
        """
        assert function in ["covid", "cancer"]
        if function == "covid":
            # Arguments for Covid simulator.
            self.arm_map = self.covid_arms
            self.num_arms = self.arm_map.shape[0]
            # TODO: unnormalize
            self.context_map = unnormalize(CovidSim.w_samples, CovidSim.bounds[:, 2:])
            self.dim = CovidSim.dim
            self.function = CovidSim()
            self.true_function = CovidEval()
        else:
            # Arguments for cancer simulator.
            self.arm_map = torch.tensor([0, 1, 2])
            self.num_arms = 3
            self.context_map = 0  # TODO:
            self.dim = 5
            self.function = EsophagealCancer()
            self.true_function = lambda X: 0  # TODO:
            # TODO: We need the true_function to be cheap to evalute.
            #  Probably simulate and store some value.
            raise NotImplementedError
        self.ckwargs = ckwargs

    def get_X(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        r"""
        Construct the function input X from the arm index and context.

        Args:
            arm_idx: An int or a list of ints.
            context: `n x d_c`-dim tensor of contexts.

        Returns:
            `n x dim`-dim input tensor.
        """
        arms = self.arm_map[arm_idx].view(-1, 1)
        return torch.cat([arms, context], dim=-1)

    def evaluate_true(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        r"""
        Evaluate the true reward at the given arm and context.
        If arm is int, context should be a `1 x d` tensor.
        If arm is a list of size n, context should be an `n x d` tensor.
        Returns an `n x 1`-dim tensor.
        """
        X = self.get_X(arm_idx, context)
        # TODO: unnormalize??
        # TODO: this is supposed to be "true" evaluation
        return self.true_function(X).view(-1, 1).to(**self.ckwargs)

    def evaluate(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        X = self.get_X(arm_idx, context)
        return self.function(X).view(-1, 1).to(**self.ckwargs)

    def evaluate_w_index(
        self, arm_idx: Union[List, int], context_idx: Union[List, int]
    ) -> Tensor:
        r"""
        Evaluate using the index for both the arm and context.
        """
        context = torch.atleast_2d(self.context_map[context_idx])
        return self.evaluate(arm_idx, context)

    def evaluate_all_true(self):
        r"""
        Evaluates all arm-context pairs "without" noise.
        """
        X = torch.cat(
            [
                self.arm_map.unsqueeze(-2).repeat(1, self.context_map.shape[0], 1),
                self.context_map.repeat(self.num_arms, 1, 1),
            ],
            dim=-1,
        ).view(-1, self.dim)
        # TODO: unnormalize?
        return self.true_function(X).view(-1, 1).to(**self.ckwargs)

    def evaluate_all(self):
        X = torch.cat(
            [
                self.arm_map.unsqueeze(-2).repeat(1, self.context_map.shape[0], 1),
                self.context_map.repeat(self.num_arms, 1, 1),
            ],
            dim=-1,
        ).view(-1, self.dim)
        return self.function(X).view(-1, 1).to(**self.ckwargs)


def fit_modellist(X: Tensor, Y: Tensor, num_arms: int) -> ModelListGP:
    r"""
    Fit a ModelListGP with a SingleTaskGP model for each arm.

    Args:
        X: A tensor representing all arm-context pairs that have been evaluated.
            First column represents the arm.
        Y: A tensor representing the corresponding evaluations.
        num_arms: An integer denoting the number of arms.

    Returns:
        A fitted ModelListGP.
    """
    mask_list = [X[..., 0] == i for i in range(num_arms)]
    model = ModelListGP(
        *[
            SingleTaskGP(
                X[mask_list[i]][..., 1:],
                Y[mask_list[i]],
                outcome_transform=Standardize(m=1),
            )
            for i in range(num_arms)
        ]
    )
    for m in model.models:
        mll = ExactMarginalLogLikelihood(m.likelihood, m)
        fit_gpytorch_model(mll)
    return model


def fit_single_gp(X: Tensor, Y: Tensor) -> SingleTaskGP:
    r"""
    Fit a SingleTaskGP on all data.
    """
    model = SingleTaskGP(
        X, Y, outcome_transform=Standardize(m=1), input_transform=Normalize(d=X.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


# These are the allowed algorithm names.
labels = [
    "IKG",
    "GP-C-OCBA",
    "DSCO",
    "C-OCBA",
]


def main(
    iterations: int,
    seed: int,
    label: str,
    use_full_train: bool,
    output_path: str,
    simulator_name: str,
    rho_key: str = "mean",
    weights: Optional[Tensor] = None,
    num_full_train: Optional[int] = None,
    num_train_per_arm: Optional[int] = None,
    fit_frequency: int = 1,  # set > 1 if we don't want to fit the GP at each iteration.
    input_dict: dict = None,  # this is for adding more iterations
    randomize_ties: bool = True,
    mode: Optional[str] = None,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
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
        simulator_name: The name of the simulator to use, "covid" or "cancer".
        rho_key: "mean" or "worst". Specifies the contextual PCS to use.
            contextual PCS. This is just an estimate, not the actual reported PCS.
        weights: The optional weights if using mean PCS.
        num_full_train: Number of "full" training samples. Number of samples drawn
            from each arm-context pair.
        num_train_per_arm: If not using samples for all arm-context pairs for training,
            this is the number of samples drawn for each arm. This many contexts are
            randomly selected for each arm and those are evaluated.
        fit_frequency: How often to fit the hyper-parameters of the GP models. Larger
            values will reduce the computational cost but may lead to poorer model
            predictions. 5-10 are safe choices.
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
    assert label in labels, "label not supported!"
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckwargs = {"dtype": dtype, "device": device}
    # get some defaults
    num_full_train = num_full_train or 2
    num_train_per_arm = num_train_per_arm or 10
    if weights is not None:
        weights = torch.as_tensor(weights).view(-1)
    use_full_train = bool(use_full_train)

    def rho(X: Tensor) -> Tensor:
        r"""Operates on -2 dimension. Dimension gets reduced by 1."""
        if rho_key == "mean":
            if weights is not None:
                # TODO: is this fixed? There was some issue here.
                X = X * weights.view(-1, 1).expand_as(X)
            return X.mean(dim=-2)
        elif rho_key == "worst":
            min_, _ = X.min(dim=-2)
            return min_
        else:
            raise NotImplementedError

    ground_truth = SimulatorWrapper(function=simulator_name, **ckwargs)
    num_arms = ground_truth.num_arms
    context_map = ground_truth.context_map
    num_contexts = context_map.shape[0]
    context_dim = context_map.shape[-1]

    # find the true best arms.
    true_means = ground_truth.evaluate_all_true()
    tm_maximizers = true_means.view(num_arms, num_contexts).argmax(dim=0)

    # get the evaluations for initialization.
    # We use the indices for the arms to avoid issues with having different
    # dimensions for the arms for different problems.
    arm_idcs = torch.arange(0, num_arms, **ckwargs).view(-1, 1)
    if use_full_train:
        # sample from all arm-context pairs
        if label not in ["DSCO", "C-OCBA"]:
            X = (
                torch.cat(
                    [
                        arm_idcs.view(-1, 1, 1).expand(-1, num_contexts, -1),
                        context_map.expand(num_arms, -1, -1),
                    ],
                    dim=-1,
                )
                .view(-1, context_dim + 1)
                .repeat(num_full_train, 1)
            )
        else:
            context_idx_set = torch.arange(0, num_contexts, **ckwargs).view(-1, 1)
            X = (
                torch.cat(
                    [
                        arm_idcs.view(-1, 1, 1).expand(-1, num_contexts, -1),
                        context_idx_set.expand(num_arms, -1, -1),
                    ],
                    dim=-1,
                )
                .view(-1, 2)
                .repeat(num_full_train, 1)
            )
        Y = torch.cat(
            [ground_truth.evaluate_all() for _ in range(num_full_train)], dim=0
        )
    else:
        # sample only a given number of contexts for each arm.
        if label in ["DSCO", "C-OCBA"]:
            raise NotImplementedError
        init_context_idcs = torch.randint(
            num_contexts, (num_arms, num_train_per_arm), device=device
        )
        X = torch.cat(
            [
                arm_idcs.view(-1, 1, 1).expand(-1, num_train_per_arm, -1),
                context_map[init_context_idcs],
            ],
            dim=-1,
        ).view(-1, context_dim + 1)
        Y = ground_truth.evaluate(X[..., 0].long().tolist(), X[..., 1:])
    num_total_train = X.shape[0]

    # set some counters etc for keeping track of things
    start = time()
    existing_iterations = 0
    pcs_estimates = torch.zeros(iterations, **ckwargs)
    correct_selection = torch.zeros(iterations, num_contexts, **ckwargs)
    wall_time = torch.zeros(iterations, **ckwargs)
    if input_dict is not None:
        # read the given output file and continue from there.
        assert torch.allclose(true_means, input_dict["true_means"])
        if mode == "-a":
            # adding iterations to existing output
            assert input_dict["label"] == label
            existing_iterations = input_dict["pcs_estimates"].shape[0]
            if existing_iterations >= iterations:
                raise ValueError("Existing output has as many or more iterations!")
            pcs_estimates[:existing_iterations] = input_dict["pcs_estimates"]
            correct_selection[:existing_iterations] = input_dict["correct_selection"]
            wall_time[:existing_iterations] = input_dict["wall_time"]
            start -= float(input_dict["wall_time"][-1])
            X = input_dict["X"]
            Y = input_dict["Y"]
        else:
            # This should never happen!
            raise RuntimeError("Mode unsupported!")
    old_model = None
    for i in range(existing_iterations, iterations):
        if i % 10 == 0:
            print(
                f"Starting label {label}, seed {seed}, iteration {i}, time: {time()-start}"
            )
        if label in ["IKG", "GP-C-OCBA"]:
            # using a ModelListGP
            if (i - existing_iterations) % fit_frequency != 0:
                # append the last evaluations to the model with low cost updates.
                # the hyper-parameters are not re-trained here.
                try:
                    last_X = X[-1].view(1, -1)
                    last_idx = int(last_X[0, 0])
                    models = old_model.models
                    models[last_idx] = models[last_idx].condition_on_observations(
                        X=last_X[:, 1:], Y=Y[-1].view(-1, 1)
                    )
                    model = ModelListGP(*models)
                except RuntimeError:
                    # Default to fitting a fresh model in case of an error.
                    model = fit_modellist(X, Y, num_arms)
            else:
                # Fit and train a new ModelListGP.
                model = fit_modellist(X, Y, num_arms)
            old_model = model
        elif label in ["DSCO", "C-OCBA"]:
            # Use the independent normally distributed model.
            model = ContextualIndependentModel(X, Y.squeeze(-1))
        else:
            raise NotImplementedError

        if label in ["IKG", "GP-C-OCBA"]:
            # Algorithms for ModelListGP
            if "IKG" in label:
                with torch.no_grad():
                    next_arm, next_context = finite_ikg_maximizer_modellist(
                        model,
                        context_map,
                        weights,
                        randomize_ties,
                        rho=rho if "rho" in label else None,
                    )
                next_point = torch.cat(
                    [torch.tensor([next_arm], **ckwargs), context_map[next_context]]
                ).view(1, -1)
            elif "GP-C-OCBA" in label:
                # This is GP-C-OCBA
                with torch.no_grad():
                    next_arm, next_context = gao_modellist(
                        model,
                        context_map,
                        randomize_ties,
                        infer_p="infer_p" in label,
                    )
                next_point = torch.cat(
                    [torch.tensor([next_arm], **ckwargs), next_context]
                ).view(1, -1)
            else:
                raise NotImplementedError
        else:
            if label == "DSCO":
                next_arm, next_context = li_sampling_strategy(model)
            elif label == "C-OCBA":
                next_arm, next_context = gao_sampling_strategy(model)
            else:
                raise NotImplementedError
            next_point = torch.tensor([[next_arm, next_context]], **ckwargs)

        # get the next evaluation
        if label == "GP-C-OCBA":
            next_eval = ground_truth.evaluate(next_arm, next_context.view(1, -1))
        else:
            next_eval = ground_truth.evaluate_w_index(next_arm, next_context)

        # update the training data
        X = torch.cat([X, next_point], dim=0)
        Y = torch.cat([Y, next_eval], dim=0)

        # check for correct selection for empirical PCS
        # This is for the actual reported PCS.
        if label in ["IKG", "GP-C-OCBA"]:
            post_mean = model.posterior(context_map).mean.t()
        else:
            post_mean = model.means

        maximizers = post_mean.argmax(dim=0)

        correct_selection[i] = tm_maximizers == maximizers

        wall_time[i] = time() - start

        if (i + 1) % fit_frequency == 0:
            # save the output periodically.
            # can be used to restart in case of an error.
            rho_cs = rho(correct_selection.unsqueeze(-1)).squeeze(-1)
            output_dict = {
                "label": label,
                "X": X[: num_total_train + i + 1],
                "Y": Y[: num_total_train + i + 1],
                "true_means": true_means,
                "correct_selection": correct_selection[: i + 1],
                "wall_time": wall_time[: i + 1],
                "rho_cs": rho_cs[: i + 1],
            }
            torch.save(output_dict, output_path)

    # apply rho to correct selection
    rho_cs = rho(correct_selection.unsqueeze(-1)).squeeze(-1)

    output_dict = {
        "label": label,
        "X": X,
        "Y": Y,
        "true_means": true_means,
        "correct_selection": correct_selection,
        "wall_time": wall_time,
        "rho_cs": rho_cs,
    }
    return output_dict


def submitit_main(
    config: str, label: str, seed: Union[int, str], last_arg=None
) -> None:
    r"""
    This is used with `submit.py` to submit jobs to a slurm cluster.

    Args:
        config: The name of the config folder.
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
    input_dict = None
    mode = None
    if path.exists(output_path):
        if last_arg and last_arg in ["-a", "-f"]:
            mode = last_arg
            if last_arg == "-f":
                print("Overwriting the existing output!")
            elif last_arg == "-a":
                print(
                    "Appending iterations to existing output!"
                    "Warning: If parameters other than `iterations` have "
                    "been changed, this will corrupt the output!"
                )
                input_dict = torch.load(output_path)
            else:
                raise RuntimeError
        else:
            print(
                "The output file exists for this experiment & seed!"
                "Pass -f as the 4th argument to overwrite!"
                "Pass -a as the 4th argument to add more iterations!"
            )
            quit()
    with open(config_path, "r") as f:
        kwargs = json.load(f)
        # if label == "ML_IKG":
        #     kwargs["iterations"] = min(kwargs["iterations"], 1000)
    output = main(
        seed=seed,
        label=label,
        input_dict=input_dict,
        mode=mode,
        output_path=output_path,
        **kwargs,
    )
    torch.save(output, output_path)


if __name__ == "__main__":
    config = sys.argv[1]
    label = sys.argv[2]
    seed = sys.argv[3]
    last_arg = sys.argv[4] if len(sys.argv) > 4 else None
    submitit_main(config, label, seed, last_arg)
