"""
Comparing the GP-C-OCBA with IKG and DSCO and C-OCBA.

This refers to the algorithms with different names.
GP-C-OCBA is referred to as ML_Gao;
IKG is referred to as ML_IKG;
DSCO is referred to as Li;
C-OCBA is referred to as Gao.
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
from contextual_rs.experiment_utils import fit_modellist_with_reuse


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
        self.num_arms = num_arms
        self.context_map = context_map.to(**ckwargs)
        self.dim = context_map.shape[-1] + 1
        if function is None:
            raise NotImplementedError
        else:
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
            self.function.to(**ckwargs)
            scale_x = unnormalize(
                torch.rand(1000, self.dim).to(**ckwargs), self.function.bounds
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
        return (
            self.function(unnormalize(X, self.function.bounds)).view(-1, 1)
            * self.func_scale
        )

    def evaluate_all(self):
        true_evals = self.evaluate_all_true()
        return true_evals + torch.randn_like(true_evals) * self.observation_noise


# These are the allowed algorithm names. See top of the file for what these are.
labels = [
    "ML_IKG",
    "ML_Gao",
    "ML_Gao_kde",
    "Li",
    "Gao",
    "SUR_simple",
    "SUR_fantasy",
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
    # get some defaults
    num_full_train = num_full_train or 2
    num_train_per_arm = num_train_per_arm or 2 * context_dim + 2
    # fixing context_map for consistency across replications
    old_state = torch.get_rng_state()
    torch.manual_seed(0)
    context_map = torch.rand(num_contexts, context_dim, dtype=dtype).to(device=device)
    torch.set_rng_state(old_state)
    ground_truth_kwargs = ground_truth_kwargs or {}
    ground_truth_kwargs.update(ckwargs)
    if weights is not None:
        weights = torch.as_tensor(weights, **ckwargs).view(-1)
    use_full_train = bool(use_full_train)

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
    arm_set = torch.arange(0, num_arms, **ckwargs).view(-1, 1)
    if use_full_train:
        # sample from all arm-context pairs
        if label not in ["Li", "Gao"]:
            X = (
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
        else:
            context_idx_set = torch.arange(0, num_contexts, **ckwargs).view(-1, 1)
            X = (
                torch.cat(
                    [
                        arm_set.view(-1, 1, 1).expand(-1, num_contexts, -1),
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
        if label in ["Li", "Gao"]:
            raise NotImplementedError
        init_context_idcs = torch.randint(
            num_contexts, (num_arms, num_train_per_arm), device=device
        )
        X = torch.cat(
            [
                arm_set.view(-1, 1, 1).expand(-1, num_train_per_arm, -1),
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
    all_maximizers = torch.zeros(iterations, num_contexts, **ckwargs)
    if input_dict is not None:
        # read the given output file and continue from there.
        if not torch.allclose(true_means, input_dict["true_means"].to(**ckwargs)):
            raise RuntimeError(f"True means are different. Got {true_means.flatten()[:10]} and {input_dict['true_means'].flatten()[:10]} for first 10.")
        if mode == "-a":
            # adding iterations to existing output
            assert input_dict["label"] == label
            existing_iterations = input_dict["pcs_estimates"].shape[0]
            if existing_iterations >= iterations:
                raise ValueError("Existing output has as many or more iterations!")
            pcs_estimates[:existing_iterations] = input_dict["pcs_estimates"].to(**ckwargs)
            correct_selection[:existing_iterations] = input_dict["correct_selection"].to(**ckwargs)
            wall_time[:existing_iterations] = input_dict["wall_time"].to(**ckwargs)
            if "all_maximizers" in input_dict:
                all_maximizers[:existing_iterations] = input_dict["all_maximizers"][:existing_iterations].to(**ckwargs)
            else:
                all_maximizers = None
            start -= float(input_dict["wall_time"][-1])
            X = input_dict["X"].to(**ckwargs)
            Y = input_dict["Y"].to(**ckwargs)
        else:
            # This should never happen!
            raise RuntimeError("Mode unsupported!")
    old_model = None
    for i in range(existing_iterations, iterations):
        if i % 10 == 0:
            print(
                f"Starting label {label}, seed {seed}, iteration {i}, time: {time()-start}"
            )
        if "ML" in label or "SUR" in label or "LEVI" in label:
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
                    model = fit_modellist_with_reuse(X, Y, num_arms, old_model)
            else:
                # Fit and train a new ModelListGP.
                model = fit_modellist_with_reuse(X, Y, num_arms, old_model)
            old_model = model
        elif label in ["Li", "Gao"]:
            # Use the independent normally distributed model.
            model = ContextualIndependentModel(X, Y.squeeze(-1))
        else:
            raise NotImplementedError

        if "ML" in label:
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
            elif "Gao" in label:
                # This is GP-C-OCBA
                with torch.no_grad():
                    next_arm, next_context = gao_modellist(
                        model,
                        context_map,
                        randomize_ties,
                        infer_p="infer_p" in label,
                        use_kde="kde" in label,
                        kernel_scale=1.0,
                    )
                next_point = torch.cat(
                    [torch.tensor([next_arm], **ckwargs), next_context]
                ).view(1, -1)
            else:
                raise NotImplementedError
        elif "LEVI" in label:
            next_arm, next_context = discrete_levi(
                model=model,
                context_set=context_map,
                weights=weights,
            )
            next_point = torch.cat(
                [torch.tensor([next_arm], **ckwargs), next_context]
            ).view(1, -1)
        elif "SUR" in label:
            with torch.no_grad():
                next_arm, next_context = sur_modellist(
                    model=model,
                    context_set=context_map,
                    use_cheap_mean_apx="simple" in label,
                )
            next_point = torch.cat(
                [torch.tensor([next_arm], **ckwargs), next_context]
            ).view(1, -1)
        else:
            if label == "Li":
                # Li / DSCO
                next_arm, next_context = li_sampling_strategy(model)
            elif label == "Gao":
                # Gao / C-OCBA
                next_arm, next_context = gao_sampling_strategy(model)
            else:
                raise NotImplementedError
            next_point = torch.tensor([[next_arm, next_context]], **ckwargs)

        # get the next evaluation
        if "ML_Gao" in label or "SUR" in label or "LEVI" in label:
            next_eval = ground_truth.evaluate(next_arm, next_context.view(1, -1))
        else:
            next_eval = ground_truth.evaluate_w_index(next_arm, next_context)

        # update the training data
        X = torch.cat([X, next_point], dim=0)
        Y = torch.cat([Y, next_eval], dim=0)

        # report current PCS estimate
        pcs_estimates[i] = estimate_current_generalized_pcs(
            model=model,
            arm_set=arm_set,
            context_set=context_idx_set if label in ["Li", "Gao"] else context_map,
            num_samples=num_pcs_samples,
            base_samples=None,
            func_I=lambda X: (X > 0).to(**ckwargs),
            rho=rho,
        )

        # check for correct selection for empirical PCS
        # This is for the actual reported PCS.
        if "ML" in label or "SUR" in label or "LEVI" in label:
            post_mean = model.posterior(context_map).mean.t()
        else:
            post_mean = model.means

        maximizers = post_mean.argmax(dim=0)
        if all_maximizers is not None:
            all_maximizers[i] = maximizers

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
                "pcs_estimates": pcs_estimates[: i + 1],
                "correct_selection": correct_selection[: i + 1],
                "wall_time": wall_time[: i + 1],
                "rho_cs": rho_cs[: i + 1],
            }
            if all_maximizers is not None:
                output_dict["all_maximizers"] = all_maximizers
            torch.save(output_dict, output_path)

    # apply rho to correct selection
    rho_cs = rho(correct_selection.unsqueeze(-1)).squeeze(-1)

    output_dict = {
        "label": label,
        "X": X,
        "Y": Y,
        "true_means": true_means,
        "pcs_estimates": pcs_estimates,
        "correct_selection": correct_selection,
        "wall_time": wall_time,
        "rho_cs": rho_cs,
    }
    if all_maximizers is not None:
        output_dict["all_maximizers"] = all_maximizers
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
        if last_arg and last_arg in ["-a", "-f", "-reeval", "-reeval-f"]:
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
        if (
            kwargs["ground_truth_kwargs"]["function"] in ["cosine8", "hartmann"]
            and label == "ML_IKG"
        ):
            kwargs["iterations"] = min(kwargs["iterations"], 1000)
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
