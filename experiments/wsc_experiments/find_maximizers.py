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
from contextual_rs.models.contextual_independent_model import ContextualIndependentModel


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


# # of inputs used to train the model the last time. Used to skip re-fitting if not necessary.
num_last_train_inputs = []


def fit_modellist(X: Tensor, Y: Tensor, num_arms: int, old_model: Optional[ModelListGP] = None) -> ModelListGP:
    r"""
    Fit a ModelListGP with a SingleTaskGP model for each arm.

    Args:
        X: A tensor representing all arm-context pairs that have been evaluated.
            First column represents the arm.
        Y: A tensor representing the corresponding evaluations.
        num_arms: An integer denoting the number of arms.
        old_model:

    Returns:
        A fitted ModelListGP.
    """
    global num_last_train_inputs
    mask_list = [X[..., 0] == i for i in range(num_arms)]
    models = []
    for i in range(num_arms):
        num_train = len(Y[mask_list[i]])
        if old_model is not None and len(old_model.models) == len(num_last_train_inputs):
            # If the model has the same inputs, we can reuse it.
            if num_train == num_last_train_inputs[i]:
                models.append(old_model.models[i])
                continue
        # If the model inputs changed, re-fit.
        m = SingleTaskGP(
            X[mask_list[i]][..., 1:],
            Y[mask_list[i]],
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(m.likelihood, m)
        fit_gpytorch_model(mll)
        models.append(m)
        try:
            num_last_train_inputs[i] = num_train
        except IndexError:
            assert len(num_last_train_inputs) == i
            num_last_train_inputs.append(num_train)
    return ModelListGP(*models)


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
    # fixing context_map for consistency across replications
    old_state = torch.get_rng_state()
    torch.manual_seed(0)
    context_map = torch.rand(num_contexts, context_dim, dtype=dtype).to(device=device)
    torch.set_rng_state(old_state)
    arm_set = torch.arange(0, num_arms, **ckwargs).view(-1, 1)

    if input_dict is None:
        raise RuntimeError("Input dict is required!")
    assert input_dict["label"] == label

    existing_iterations = input_dict["pcs_estimates"].shape[0]
    if existing_iterations != iterations:
        raise RuntimeError(f"Not enough observations {existing_iterations=}")
    if "maximizers" in input_dict and mode != "-f":
        raise RuntimeError("Maximizers were already computed. Skipping!")
    if mode not in ["-f", "-a"]:
        raise RuntimeError("Mode unsupported!")
    start = time()
    all_X = input_dict["X"].to(**ckwargs)
    all_Y = input_dict["Y"].to(**ckwargs)
    num_init_obs = len(all_X) - iterations
    all_maximizers = torch.zeros(iterations, num_contexts, **ckwargs)
    old_model = None
    for i in range(0, iterations):
        X = all_X[num_init_obs + i:]
        Y = all_Y[num_init_obs + i:]
        if i % 10 == 0:
            print(
                f"Starting label {label}, seed {seed}, iteration {i}, time: {time()-start}"
            )
        if "ML" in label or "SUR" in label or "LEVI" in label:
            # using a ModelListGP
            if i % fit_frequency != 0:
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
                    model = fit_modellist(X, Y, num_arms, old_model)
            else:
                # Fit and train a new ModelListGP.
                model = fit_modellist(X, Y, num_arms, old_model)
            old_model = model
        elif label in ["Li", "Gao"]:
            # Use the independent normally distributed model.
            model = ContextualIndependentModel(X, Y.squeeze(-1))
        else:
            raise NotImplementedError

        # check for correct selection for empirical PCS
        # This is for the actual reported PCS.
        if "ML" in label or "SUR" in label or "LEVI" in label:
            post_mean = model.posterior(context_map).mean.t()
        else:
            post_mean = model.means

        maximizers = post_mean.argmax(dim=0)
        all_maximizers[i] = maximizers

    input_dict["all_maximizers"] = all_maximizers
    return input_dict


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
    mode = None
    if not path.exists(output_path):
        raise FileNotFoundError("Expected output file!")
    input_dict = torch.load(output_path)
    if last_arg:
        mode = last_arg
        if last_arg == "-f":
            print("Overwriting the existing maximizers in the output!")
        elif last_arg != "-a":
            raise RuntimeError
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
