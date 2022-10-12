"""
For running the experiments with the modified TS and TS+ policies.
This is for using the Branin & Greiwank functions.
"""
import json
import sys
from os import path
from time import time
from typing import Union, List, Optional

import numpy as np
import torch
from botorch.test_functions import Branin, Griewank, Hartmann
from botorch.utils.transforms import unnormalize
from torch import Tensor

from contextual_rs.modified_ts_policies import modified_ts, modified_ts_plus, get_beta_hat
from contextual_rs.test_functions.covid_exp_class import CovidSim, CovidEval
from contextual_rs.test_functions.esophageal_cancer import EsophagealCancer


class GroundTruthModel:
    """A class representing the true reward function."""

    def __init__(
        self,
        num_arms: int,
        context_map: Tensor,
        function: str,
        init_scale: float = 50.0,
        observation_noise: float = 3.0,
        **ckwargs,
    ) -> None:
        r"""Create the ground truth model from the given function.
        The first input dimension of the function is used for the arms, i.e.,
        a specific value of `x_0` corresponds to an arm, and the remaining dimensions
        are used for the contexts, which are filled from `context_map`.

        Args:
            num_arms: Number of arms to use.
            context_map: A tensor denoting the locations of the contexts in the
                function input space. This should be `num_contexts x (function.dim - 1)`
                dimensional.
            function: The name of the base test function.
            init_scale: Used to scale the observation noise level. The scaled
                observation noise has a standard deviation of
                `2 * init_scale * (f_max - f_min) / observation_noise` where `f_max`
                and `f_min` are calculated as `scale_y.max/min` below.
                In implementation, we actually scale the function values and add
                noise with standard deviation `observation_noise` but these are
                equivalent for all practical purposes. (relics from an earlier version)
            observation_noise: See `init_scale`.
            ckwargs: Common tensor arguments, dtype and device.
        """
        assert function in ["branin", "greiwank"]
        self.num_arms = num_arms
        self.context_map = context_map
        self.dim = context_map.shape[-1] + 1
        if function in ["branin", "greiwank"]:
            self.extreme_design = torch.stack(
                [context_map.min(dim=0).values, context_map.max(dim=0).values], dim=0
            )
        elif function == "hartmann":
            self.extreme_design = context_map[[0, 1, 11, 16]]
        # Get the extreme desings: This takes the largest and smallest
        #   This will not work larger than 1D context spaces!!
        self.arm_map = torch.linspace(0, 1, num_arms, **ckwargs).view(-1, 1)
        if function == "branin":
            self.function = Branin(negate=True)
        elif function == "greiwank":
            self.function = Griewank(dim=2, negate=True)
            # Modify the arm map to avoid arms having the same value, leading to multiple optimizers.
            self.arm_map[:-1] += 0.01
            self.function.bounds[0, :].fill_(-10)
            self.function.bounds[1, :].fill_(10)
        elif function == "hartmann":
            self.function = Hartmann(dim=3, negate=True)
        scale_x = unnormalize(
            torch.rand(1000, self.dim, **ckwargs), self.function.bounds
        )
        scale_y = self.function(scale_x)
        self.func_scale = init_scale * 2 / (scale_y.max() - scale_y.min())
        self.observation_noise = observation_noise
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
        arms = self.arm_map[arm_idx].view(-1, self.arm_map.shape[-1])
        return torch.cat([arms, context], dim=-1)

    def evaluate_true(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        r"""
        Evaluate the true reward at the given arm and context.
        If arm is int, context should be a `1 x d` tensor.
        If arm is a list of size n, context should be an `n x d` tensor.
        Returns an `n x 1`-dim tensor.
        """
        X = self.get_X(arm_idx, context)
        return (
            self.function(unnormalize(X, self.function.bounds)).view(-1, 1)
            * self.func_scale
        )

    def evaluate(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        true_evals = self.evaluate_true(arm_idx, context)
        return true_evals + torch.randn_like(true_evals) * self.observation_noise

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
        return (
            self.function(unnormalize(X, self.function.bounds)).view(-1, 1)
            * self.func_scale
        )

    def evaluate_all(self):
        true_evals = self.evaluate_all_true()
        return true_evals + torch.randn_like(true_evals) * self.observation_noise

    def evaluate_extreme_design(self):
        X = torch.cat(
            [
                self.arm_map.unsqueeze(-2).repeat(1, self.extreme_design.shape[0], 1),
                self.extreme_design.repeat(self.num_arms, 1, 1),
            ],
            dim=-1,
        ).view(-1, self.dim)
        true_evals = (
            self.function(unnormalize(X, self.function.bounds)).view(-1, 1)
            * self.func_scale
        )
        return true_evals + torch.randn_like(true_evals) * self.observation_noise


# These are the allowed algorithm names.
labels = [
    "TS",
    "TS+",
]


def main(
    iterations: int,
    seed: int,
    label: str,
    output_path: str,
    rho_key: str = "mean",
    num_arms: int = 10,
    num_contexts: int = 10,
    context_dim: int = 1,
    weights: Optional[Tensor] = None,
    num_full_train: Optional[int] = None,
    ground_truth_kwargs: dict = None,
    input_dict: dict = None,  # this is for adding more iterations
    mode: Optional[str] = None,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
) -> dict:
    r"""
    Run a single replication of the experiment with the specified algorithm.
    This is typically used with a config file and the helpers found at the bottom
    to run a set of experiments on a cluster / server.

    Args:
        iterations: Total number of iterations to run the experiment for. This is
            a parameter for other algorithms, and is used to infer the total number of
            samples to use here.
        seed: The seed to use for this replication.
        label: This is the algorithm label, select from the list above. It also specifies
            whether to use the full or extreme design, as well as the number of samples
            to use for initialization.
            structure as: <alg_name>_<f/e>_<num_train>
        output_path: The file path for saving the experiment output.
        rho_key: "mean" or "worst". Specifies the contextual PCS to use.
            contextual PCS. This is just an estimate, not the actual reported PCS.
        num_arms: Number of arms to use.
        num_contexts: Number of contexts to use.
        context_dim: The dimension of the context tensors. This should be
            `function.dim - 1`.
        weights: The optional weights if using mean PCS.
        num_full_train: Number of "full" training samples. Number of samples drawn
            from each arm-context pair.
        ground_truth_kwargs: Arguments for the ground truth model / function.
        input_dict: This is used if the experiment was terminated early and we want to
            continue from where we left off. The experiment will warm-start from here
            and continue up to a total number of `iterations`.
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
    # fixing context_map for consistency across replications
    old_state = torch.get_rng_state()
    torch.manual_seed(0)
    context_map = torch.rand(num_contexts, context_dim, **ckwargs)
    torch.set_rng_state(old_state)
    ground_truth_kwargs = ground_truth_kwargs or dict()
    if weights is not None:
        weights = torch.as_tensor(weights).view(-1)

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

    ground_truth = GroundTruthModel(num_arms, context_map, **ground_truth_kwargs)

    # find the true best arms.
    true_means = ground_truth.evaluate_all_true()
    tm_maximizers = true_means.view(num_arms, num_contexts).argmax(dim=0)

    # get the evaluations for initialization.
    # sample from all designs / extreme designs
    active_design = ground_truth.extreme_design
    # Make it use the same number of initial samples as other algorithms
    assert num_contexts % active_design.shape[0] == 0
    num_train = int(num_full_train * num_contexts / active_design.shape[0])
    samples_count = torch.full((num_arms, active_design.shape[0]), num_train, **ckwargs)
    Y = torch.stack(
        [
            ground_truth.evaluate_extreme_design().view_as(samples_count) for _ in range(num_train)
        ]
    )
    # NOTE: Use Y.sum(dim=0) / samples_count to get the sample mean.
    # The regression estimators require the "X" to have the intercept term.
    design_w_intercept = torch.cat(
        [torch.ones(active_design.shape[0], 1, **ckwargs), active_design], dim=-1
    )
    all_designs_w_intercept = torch.cat(
        [torch.ones(context_map.shape[0], 1, **ckwargs), context_map], dim=-1
    )

    # Convert iterations to num_total_samples
    num_total_samples = num_full_train * num_arms * num_contexts + iterations

    if "TS+" in label:
        fractions = modified_ts_plus(
            design_w_intercept, Y, total_budget=num_total_samples, ratios_only=True,
        )
    else:
        fractions = modified_ts(
            design_w_intercept, Y, total_budget=num_total_samples, ratios_only=True,
        )

    # set some counters etc for keeping track of things
    start = time()
    correct_selection = torch.zeros(0, num_contexts, **ckwargs)
    samples_count_all = samples_count.unsqueeze(0).clone()
    wall_time = torch.zeros(0, **ckwargs)
    if input_dict is not None:
        # read the given output file and continue from there.
        assert torch.allclose(true_means, input_dict["true_means"])
        if mode == "-a":
            # adding iterations to existing output
            assert input_dict["label"] == label
            samples_count_all = input_dict["samples_count_all"]
            samples_count = samples_count_all[-1]
            if samples_count.sum() >= num_total_samples:
                raise ValueError("Existing output has as many or more samples!")
            correct_selection = input_dict["correct_selection"]
            wall_time = input_dict["wall_time"]
            start -= float(input_dict["wall_time"][-1])
            Y = input_dict["Y"]
        else:
            # This should never happen!
            raise RuntimeError("Mode unsupported!")

    for i in range(int(samples_count.sum()), num_total_samples):
        if i % 10 == 0:
            print(
                f"Starting label {label}, seed {seed}, iteration {i}, time: {time()-start}"
            )

        proposed_allocation = (fractions * i).round()
        new_allocation = torch.max(samples_count, proposed_allocation)
        additional_samples = new_allocation - samples_count
        if additional_samples.sum() == 0:
            continue

        # Get the new samples and append to Y.
        new_Y = torch.zeros(int(additional_samples.max()), *samples_count.shape, **ckwargs)
        for i, add_ in enumerate(additional_samples):
            for j, count in enumerate(add_):
                if count:
                    arms = [i] * int(count)
                    contexts = active_design[j].view(1, -1).repeat(int(count), 1)
                    new_Y[:int(count), i, j] = ground_truth.evaluate(arms, contexts).view(-1)
        Y = torch.cat([Y, new_Y], dim=0)
        samples_count = new_allocation

        # Use sum / count to get Y_mean. We may have samples missing (0 values).
        Y_mean = Y.sum(dim=0) / samples_count

        # Get the beta and update the predictions.
        beta = get_beta_hat(design_w_intercept, Y_mean)
        # The linear predictions for arms x contexts
        predictions = beta.matmul(all_designs_w_intercept.transpose(-2, -1))
        maximizers = predictions.argmax(dim=0)

        correct_selection = torch.cat([correct_selection, (tm_maximizers == maximizers).unsqueeze(0)])
        wall_time = torch.cat([wall_time, torch.tensor([time() - start], **ckwargs)])

        samples_count_all = torch.cat(
            [samples_count_all, samples_count.unsqueeze(0)], dim=0,
        )

        # save the output periodically.
        # can be used to restart in case of an error.
        rho_cs = rho(correct_selection.unsqueeze(-1)).squeeze(-1)
        output_dict = {
            "label": label,
            "samples_count_all": samples_count_all,
            "Y": Y,
            "true_means": true_means,
            "correct_selection": correct_selection,
            "wall_time": wall_time,
            "rho_cs": rho_cs,
        }
        torch.save(output_dict, output_path)

    # apply rho to correct selection
    rho_cs = rho(correct_selection.unsqueeze(-1)).squeeze(-1)

    output_dict = {
        "label": label,
        "samples_count_all": samples_count_all,
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
    output_path = path.join(exp_dir, label.split("_")[0], f"{str(seed).zfill(4)}_{label}.pt")
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
    for key in ["fit_frequency", "use_full_train"]:
        kwargs.pop(key, None)
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
