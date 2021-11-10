"""
For running the experiments with the modified TS and TS+ policies.
This is for using the Covid & Cancer simulators.
"""
import json
import sys
from os import path
from time import time
from typing import Union, List, Optional

import numpy as np
import torch
from botorch.utils.transforms import unnormalize
from torch import Tensor

from contextual_rs.modified_ts_policies import modified_ts, modified_ts_plus, get_beta_hat
from contextual_rs.test_functions.covid_exp_class import CovidSim, CovidEval, CovidSimV2, CovidEvalV2
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
        assert function in ["covid", "covid_v2", "cancer"]
        if function == "covid":
            # Arguments for Covid simulator.
            self.arm_map = self.covid_arms
            self.num_arms = self.arm_map.shape[0]
            self.context_map = unnormalize(CovidSim.w_samples, CovidSim.bounds[:, 2:])
            # Extreme design is the edges of the cube.
            self.extreme_design = self.context_map[[0, 2, 6, 8, 18, 20, 24, 26]]
            self.dim = CovidSim.dim
            self.function = CovidSim(negate=True)
            self.true_function = CovidEval(negate=True)
        elif function == "covid_v2":
            # Arguments for Covid simulator with updated parameterization
            self.arm_map = self.covid_arms
            self.num_arms = self.arm_map.shape[0]
            self.context_map = CovidSimV2().context_samples
            # Extreme design is the edges of the cube.
            self.extreme_design = self.context_map[[0, 3, -4, -1]]
            self.dim = CovidSimV2.dim
            self.function = CovidSimV2(negate=True)
            self.true_function = CovidEvalV2(negate=True)
        else:
            # Arguments for cancer simulator.
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

    def evaluate_extreme_design(self):
        X = torch.cat(
            [
                self.arm_map.unsqueeze(-2).repeat(1, self.extreme_design.shape[0], 1),
                self.extreme_design.repeat(self.num_arms, 1, 1),
            ],
            dim=-1,
        ).view(-1, self.dim)
        return self.function(X).view(-1, 1).to(**self.ckwargs)


# These are the allowed algorithm names.
labels = [
    "TS",
    "TS+",
]


def main(
    num_total_samples: int,
    seed: int,
    label: str,
    output_path: str,
    simulator_name: str,
    rho_key: str = "mean",
    weights: Optional[Tensor] = None,
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
        num_total_samples: Total number of samples to use. Includes the initialization.
        seed: The seed to use for this replication.
        label: This is the algorithm label, select from the list above. It also specifies
            whether to use the full or extreme design, as well as the number of samples
            to use for initialization.
            structure as: <alg_name>_<f/e>_<num_train>
        output_path: The file path for saving the experiment output.
        simulator_name: The name of the simulator to use, "covid" or "cancer".
        rho_key: "mean" or "worst". Specifies the contextual PCS to use.
            contextual PCS. This is just an estimate, not the actual reported PCS.
        weights: The optional weights if using mean PCS.
        input_dict: This is used if the experiment was terminated early and we want to
            continue from where we left off. The experiment will warm-start from here
            and continue up to a total number of `iterations`.
        mode: If `input_dict` is specified, mode must be "-a", as in append, to add
            more iterations. No other modes are currently supported.
        dtype: Tensor data type to use.
        device: The device to use, "cpu" / "cuda".
    """
    split_label = label.split("_")
    assert split_label[0] in labels, "label not supported!"
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckwargs = {"dtype": dtype, "device": device}
    # get some defaults
    if weights is not None:
        weights = torch.as_tensor(weights).view(-1)
    use_full_design = split_label[1] == "f"
    num_train = int(split_label[2])

    def rho(X: Tensor) -> Tensor:
        r"""Operates on -2 dimension. Dimension gets reduced by 1."""
        if rho_key == "mean":
            if weights is not None:
                X = X * weights.view(-1, 1).expand_as(X)
            return X.sum(dim=-2)
        elif rho_key == "worst":
            min_, _ = X.min(dim=-2)
            return min_
        else:
            raise NotImplementedError

    simulator = SimulatorWrapper(function=simulator_name, **ckwargs)
    num_arms = simulator.num_arms
    context_map = simulator.context_map
    num_contexts = context_map.shape[0]

    # find the true best arms.
    true_means = simulator.evaluate_all_true()
    tm_maximizers = true_means.view(num_arms, num_contexts).argmax(dim=0)

    # get the evaluations for initialization.
    # sample from all designs / extreme designs
    active_design = context_map if use_full_design else simulator.extreme_design
    samples_count = torch.full((num_arms, active_design.shape[0]), num_train, **ckwargs)
    Y = torch.stack(
        [
            (simulator.evaluate_all() if use_full_design else simulator.evaluate_extreme_design()).view_as(samples_count) for _ in range(num_train)
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
                    new_Y[:int(count), i, j] = simulator.evaluate(arms, contexts).view(-1)
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
    for key in ["fit_frequency"]:
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
