"""
Comparing the lookahead PCS with MLGP with IKG and Li and Gao.
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
)
from contextual_rs.finite_ikg import (
    finite_ikg_maximizer_modellist,
)
from contextual_rs.generalized_pcs import (
    estimate_current_generalized_pcs,
)
from contextual_rs.models.contextual_independent_model import ContextualIndependentModel


class GroundTruthModel:
    def __init__(
        self,
        num_arms: int,
        context_map: Tensor,
        init_scale: float = 50.0,
        observation_noise: float = 3.0,
        function: Optional[str] = None,
        **ckwargs,
    ):
        r"""
        If function is specified, the GP model is replaced with that function,
        where the first dimension is used for the arms, and the remaining dimensions
        are used for contexts. The function should be defined over d_c + 1 dimensions.
        Arm indices are randomly mapped to the underlying space to avoid giving
        SingleTaskGP an advantage.
        In this case, the `num_init_samples` argument is ignored. The `init_scale`
        argument is used to approximately scale the function values to match the scale.
        """
        self.num_arms = num_arms
        self.context_map = context_map
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
            scale_x = unnormalize(
                torch.rand(1000, self.dim, **ckwargs), self.function.bounds
            )
            scale_y = self.function(scale_x)
            self.func_scale = init_scale * 2 / (scale_y.max() - scale_y.min())
            self.model = None
        self.observation_noise = observation_noise

    def evaluate_true(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        r"""
        Evaluate the posterior mean at the given arm and context.
        If using true_function, this evaluates that function.
        If arm is int, context should be `1 x d` tensor.
        If arm is a list of size n, context should be `n x d` tensor.
        Returns a `n x 1`-dim tensor.
        """
        arms = self.arm_map[arm_idx].view(-1, 1)
        X = torch.cat([arms, context], dim=-1)
        if self.model:
            return self.model.posterior(X).mean.view(-1, 1).detach()
        else:
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
        if self.model:
            return self.model.posterior(X).mean.view(-1, 1).detach()
        else:
            return (
                self.function(unnormalize(X, self.function.bounds)).view(-1, 1)
                * self.func_scale
            )

    def evaluate_all(self):
        true_evals = self.evaluate_all_true()
        return true_evals + torch.randn_like(true_evals) * self.observation_noise


def fit_modellist(X, Y, num_arms):
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


labels = [
    "ML_IKG",
    "ML_Gao",
    "Li",
    "Gao",
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
    device: str = "cpu",
) -> dict:
    assert label in labels, "label not supported!"
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckwargs = {"dtype": dtype, "device": device}
    num_full_train = num_full_train or 2
    num_train_per_arm = num_train_per_arm or 2 * context_dim + 2
    # fixing context_map for consistency across replications
    old_state = torch.get_rng_state()
    torch.manual_seed(0)
    context_map = torch.rand(num_contexts, context_dim, **ckwargs)
    torch.set_rng_state(old_state)
    ground_truth_kwargs = ground_truth_kwargs or dict()
    if weights is not None:
        weights = torch.as_tensor(weights).view(-1)
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

    true_means = ground_truth.evaluate_all_true()
    tm_maximizers = true_means.view(num_arms, num_contexts).argmax(dim=0)

    arm_set = torch.arange(0, num_arms, **ckwargs).view(-1, 1)
    if use_full_train:
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
        Y = torch.cat([ground_truth.evaluate_all() for _ in range(num_full_train)], dim=0)
    else:
        if label in ["Li", "Gao"]:
            raise NotImplementedError
        init_context_idcs = torch.randint(
            num_contexts, (num_arms, num_train_per_arm), device=device
        )
        X = (
            torch.cat(
                [
                    arm_set.view(-1, 1, 1).expand(-1, num_train_per_arm, -1),
                    context_map[init_context_idcs],
                ],
                dim=-1,
            )
            .view(-1, context_dim + 1)
        )
        Y = ground_truth.evaluate(
            X[..., 0].long().tolist(), X[..., 1:]
        )
    num_total_train = X.shape[0]

    start = time()
    existing_iterations = 0
    pcs_estimates = torch.zeros(iterations, **ckwargs)
    correct_selection = torch.zeros(iterations, num_contexts, **ckwargs)
    wall_time = torch.zeros(iterations, **ckwargs)
    if input_dict is not None:
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
        if "ML" in label:
            if (i - existing_iterations) % fit_frequency != 0:
                try:
                    last_X = X[-1].view(1, -1)
                    last_idx = int(last_X[0, 0])
                    models = old_model.models
                    models[last_idx] = models[last_idx].condition_on_observations(
                        X=last_X[:, 1:], Y=Y[-1].view(-1, 1)
                    )
                    model = ModelListGP(*models)
                except RuntimeError:
                    model = fit_modellist(X, Y, num_arms)
            else:
                model = fit_modellist(X, Y, num_arms)
            old_model = model
        elif label in ["Li", "Gao"]:
            model = ContextualIndependentModel(X, Y.squeeze(-1))
        else:
            raise NotImplementedError

        if "ML" in label:
            # ModelListGP
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
            if label == "Li":
                # Li
                next_arm, next_context = li_sampling_strategy(model)
            elif label == "Gao":
                # Gao
                next_arm, next_context = gao_sampling_strategy(model)
            else:
                raise NotImplementedError
            next_point = torch.tensor([[next_arm, next_context]], **ckwargs)

        if label == "ML_Gao":
            next_eval = ground_truth.evaluate(next_arm, next_context.view(1, -1))
        else:
            next_eval = ground_truth.evaluate_w_index(next_arm, next_context)

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
        if "ML" in label:
            post_mean = model.posterior(context_map).mean.t()
        else:
            post_mean = model.means

        maximizers = post_mean.argmax(dim=0)

        correct_selection[i] = tm_maximizers == maximizers

        wall_time[i] = time() - start

        if (i + 1) % fit_frequency == 0:
            # save the output periodically.
            rho_cs = rho(correct_selection.unsqueeze(-1)).squeeze(-1)
            output_dict = {
                "label": label,
                "X": X[:num_total_train + i + 1],
                "Y": Y[:num_total_train + i + 1],
                "true_means": true_means,
                "pcs_estimates": pcs_estimates[:i + 1],
                "correct_selection": correct_selection[:i + 1],
                "wall_time": wall_time[:i + 1],
                "rho_cs": rho_cs[:i + 1],
            }
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
    return output_dict


def submitit_main(config, label, seed, last_arg=None):
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
        if kwargs["ground_truth_kwargs"]["function"] in ["cosine8", "hartmann"] and label == "ML_IKG":
            kwargs["iterations"] = min(kwargs["iterations"], 1000)
    output = main(
        seed=seed,
        label=label,
        input_dict=input_dict,
        mode=mode,
        output_path=output_path,
        **kwargs
    )
    torch.save(output, output_path)


if __name__ == "__main__":
    config = sys.argv[1]
    label = sys.argv[2]
    seed = sys.argv[3]
    last_arg = sys.argv[4] if len(sys.argv) > 4 else None
    submitit_main(config, label, seed, last_arg)
