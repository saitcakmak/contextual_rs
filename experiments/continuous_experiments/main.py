"""
Comparing the GP-C-OCBA with IKG.

This refers to the algorithms with different names.
GP-C-OCBA is continuous extension of GP-C-OCBA.
"""
import json
import sys
from os import path
from time import time
from typing import Union, List, Optional

import gpytorch
import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin, Hartmann, Griewank, Cosine8, Powell, Levy
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor

from contextual_rs.continuous_context import MinZeta, find_next_arm_given_context
from contextual_rs.levi import PredictiveEI, discrete_levi
from contextual_rs.experiment_utils import fit_modellist_with_reuse


class GroundTruthModel:
    """A class representing the true reward function."""

    def __init__(
        self,
        num_arms: int,
        function: str,
        init_scale: float = 50.0,
        observation_noise: float = 3.0,
        **ckwargs,
    ) -> None:
        r"""Create the ground truth model from the given function.
        The first input dimension of the function is used for the arms, i.e.,
        a specific value of `x_0` corresponds to an arm, and the remaining dimensions
        are used for the contexts.

        Args:
            num_arms: Number of arms to use.
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
        self.num_arms = num_arms
        self.arm_map = torch.linspace(0, 1, num_arms, **ckwargs).view(-1, 1)
        if function == "branin":
            self.function = Branin(negate=True)
        elif function == "hartmann":
            self.function = Hartmann(dim=3, negate=True)
        elif function == "greiwank":
            self.function = Griewank(dim=2, negate=True)
            self.function.bounds[0, :].fill_(-10)
            self.function.bounds[0, :].fill_(10)
        elif function == "cosine8":
            self.function = Cosine8()
        elif function == "powell":
            self.function = Powell(negate=True)
        elif function == "levy":
            self.function = Levy(dim=4, negate=True)
        else:
            raise NotImplementedError
        scale_x = unnormalize(
            torch.rand(1000, self.function.dim, **ckwargs), self.function.bounds
        )
        scale_y = self.function(scale_x)
        self.func_scale = init_scale * 2 / (scale_y.max() - scale_y.min())
        self.observation_noise = observation_noise

    def evaluate_true(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        r"""
        Evaluate the true reward at the given arm and context.
        If arm is int, context should be a `1 x d` tensor.
        If arm is a list of size n, context should be an `n x d` tensor.
        Returns an `n x 1`-dim tensor.
        """
        arms = self.arm_map[arm_idx].view(-1, 1)
        X = torch.cat([arms, context], dim=-1)
        return (
            self.function(unnormalize(X, self.function.bounds)).view(-1, 1)
            * self.func_scale
        )

    def evaluate(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        true_evals = self.evaluate_true(arm_idx, context)
        return true_evals + torch.randn_like(true_evals) * self.observation_noise


# These are the allowed algorithm names. See top of the file for what these are.
labels = [
    "GP-C-OCBA",
    "GP-C-OCBA-1.0",  # KDE scale 1.0.
    "GP-C-OCBA-0.5",  # KDE scale 0.5.
    "random",
    # "LEVI",
    "LEVI-new",
]


def main(
    iterations: int,
    seed: int,
    label: str,
    output_path: str,
    num_arms: int = 10,
    num_eval_contexts: int = 1024,
    context_dim: int = 1,
    num_train_per_arm: Optional[int] = None,
    fit_frequency: int = 1,  # Set > 1 if we don't want to fit the GP at each iteration.
    ground_truth_kwargs: dict = None,
    input_dict: dict = None,  # This is for adding more iterations.
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
        output_path: The file path for saving the experiment output.
        num_arms: Number of arms to use.
        num_eval_contexts: Number of Sobol contexts used to evaluate the reported PCS.
        context_dim: The dimension of the context tensors. This should be
            `function.dim - 1`.
        num_train_per_arm: This is the number of samples drawn for each arm for
            initialization. This many contexts are randomly selected for each arm
            and those are evaluated.
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
    assert label in labels, "label not supported!"
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckwargs = {"dtype": dtype, "device": device}
    # get some defaults
    num_train_per_arm = num_train_per_arm or 2 * context_dim + 2

    ground_truth_kwargs = ground_truth_kwargs or dict()
    ground_truth = GroundTruthModel(num_arms, **ground_truth_kwargs)
    normalized_context_bounds = torch.tensor([[0], [1]], **ckwargs).expand(2, context_dim)

    # Sobol samples the contexts to evaluate the PCS at.
    eval_contexts = draw_sobol_samples(
        bounds=normalized_context_bounds, n=num_eval_contexts, q=1, seed=0
    ).squeeze(-2).to(**ckwargs)

    # Find the true best arms.
    true_means = ground_truth.evaluate_true(
        arm_idx=list(range(num_arms)) * num_eval_contexts,
        context=eval_contexts.repeat_interleave(num_arms, dim=0),
    ).view(num_eval_contexts, num_arms)
    tm_maximizers = true_means.argmax(dim=1)

    # Get the evaluations for initialization.
    arm_set = torch.arange(0, num_arms, **ckwargs).view(-1, 1)
    # Sample a given number of contexts for each arm.
    # Fix the seed to ensure identical initial samples for each seed & method.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        init_contexts = torch.rand(num_arms, num_train_per_arm, context_dim, **ckwargs)
        X = torch.cat(
            [
                arm_set.view(-1, 1, 1).expand(-1, num_train_per_arm, -1),
                init_contexts,
            ],
            dim=-1,
        ).view(-1, context_dim + 1)
        Y = ground_truth.evaluate(X[..., 0].long().tolist(), X[..., 1:])
    num_total_train = X.shape[0]

    # Set some counters etc for keeping track of things.
    start = time()
    existing_iterations = 0
    correct_selection = torch.zeros(iterations, num_eval_contexts, **ckwargs)
    wall_time = torch.zeros(iterations, **ckwargs)
    if input_dict is not None:
        # read the given output file and continue from there.
        assert torch.allclose(true_means, input_dict["true_means"])
        if mode == "-a":
            # adding iterations to existing output
            assert input_dict["label"] == label
            existing_iterations = input_dict["existing_iterations"]
            if existing_iterations >= iterations:
                raise ValueError("Existing output has as many or more iterations!")
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
        # Using a ModelListGP.
        if (i - existing_iterations) % fit_frequency != 0:
            # Append the last evaluations to the model with low cost updates.
            # The hyper-parameters are not re-trained here.
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
                model = fit_modellist_with_reuse(X, Y, num_arms, old_model)
        else:
            # Fit and train a new ModelListGP.
            model = fit_modellist_with_reuse(X, Y, num_arms, old_model)
        old_model = model

        if "GP-C-OCBA" in label:
            acqf = MinZeta(
                model=model,
            )
            with gpytorch.settings.cholesky_max_tries(6):
                next_context, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=normalized_context_bounds,
                    q=1,
                    num_restarts=20,
                    raw_samples=1024,
                )
            next_arm = find_next_arm_given_context(
                next_context=next_context,
                model=model,
                kernel_scale=float(label[10:]) if len(label) > 9 else 2.0,
            )
        elif "LEVI" in label:
            acqf = PredictiveEI(
                model=model
            )
            with gpytorch.settings.cholesky_max_tries(6):
                next_context, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=normalized_context_bounds,
                    q=1,
                    num_restarts=20,
                    raw_samples=1024,
                )
            next_arm, _ = discrete_levi(
                model=model,
                context_set=next_context,
            )
        elif label == "random":
            next_arm = torch.randint(low=0, high=num_arms, size=(1,), device=device)
            next_context = torch.rand(1, context_dim, **ckwargs)
        else:
            raise NotImplementedError

        # Get the next evaluation.
        next_eval = ground_truth.evaluate(next_arm, next_context.view(1, -1))
        next_point = torch.cat(
            [torch.tensor([[next_arm]], **ckwargs), next_context], dim=-1
        )

        # Update the training data.
        X = torch.cat([X, next_point], dim=0)
        Y = torch.cat([Y, next_eval], dim=0)

        # Check for correct selection for empirical PCS.
        # This is for the actual reported PCS.
        # TODO: is this faster if we unsqueeze & squeeze?
        post_mean = model.posterior(eval_contexts).mean
        maximizers = post_mean.argmax(dim=1)
        correct_selection[i] = tm_maximizers == maximizers

        wall_time[i] = time() - start

        if (i + 1) % fit_frequency == 0:
            # Save the output periodically.
            # This can be used to restart in case of an error.
            pcs = correct_selection.mean(dim=-1)
            output_dict = {
                "label": label,
                "X": X[: num_total_train + i + 1],
                "Y": Y[: num_total_train + i + 1],
                "true_means": true_means,
                "correct_selection": correct_selection[: i + 1],
                "wall_time": wall_time[: i + 1],
                "pcs": pcs[: i + 1],
                "existing_iterations": i + 1,
            }
            torch.save(output_dict, output_path)

    # Get the mean PCS.
    pcs = correct_selection.mean(dim=-1)

    output_dict = {
        "label": label,
        "X": X,
        "Y": Y,
        "true_means": true_means,
        "correct_selection": correct_selection,
        "wall_time": wall_time,
        "pcs": pcs,
        "existing_iterations": iterations,
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
