"""
Comparing the selection strategies of Li and Gao with LCEGP w/ lookahead contextual PCS,
where the objective is expected PCS over contexts, and the arms are categorical but the
contexts are represented by a vector of chosen dimension. The context vector is known
to LCEGP, so it only treats arms as categorical.

The ground truth is a GP model fitted on random data of appropriate dimension.
"""
import json
import math
import sys
from os import path
from time import time
from typing import Union, List, Optional

import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize, Normalize
from botorch.optim.fit import fit_gpytorch_torch
from botorch.sampling import SobolQMCNormalSampler
from torch import Tensor

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
from contextual_rs.finite_ikg import (
    finite_ikg_maximizer,
    finite_ikg_maximizer_modellist,
)


class GroundTruthModel:
    def __init__(
        self,
        num_arms: int,
        context_map: Tensor,
        num_init_samples: int = None,
        init_scale: float = 50.0,
        observation_noise: float = 3.0,
    ):
        r"""
        Generate a GP model for use as the ground truth for function evaluations.
        The dimension is inferred from context_map, as context_map.shape[-1] + 1.
        """
        self.num_arms = num_arms
        self.arm_map = torch.linspace(0, 1, num_arms).view(-1, 1).to(context_map)
        self.context_map = context_map
        self.dim = context_map.shape[-1] + 1
        num_init_samples = num_init_samples or self.dim * 10
        train_X = torch.rand(num_init_samples, self.dim).to(context_map)
        train_Y = torch.randn(num_init_samples, 1).to(context_map) * init_scale
        self.model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        self.observation_noise = observation_noise

    def evaluate_true(self, arm_idx: Union[List, int], context: Tensor) -> Tensor:
        r"""
        Evaluate the posterior mean at the given arm and context.
        If arm is int, context should be `1 x d` tensor.
        If arm is a list of size n, context should be `n x d` tensor.
        Returns a `n x 1`-dim tensor.
        """
        arms = self.arm_map[arm_idx].view(-1, 1)
        X = torch.cat([arms, context], dim=-1)
        return self.model.posterior(X).mean.view(-1, 1).detach()

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
        return self.model.posterior(X).mean.view(-1, 1).detach()

    def evaluate_all(self):
        true_evals = self.evaluate_all_true()
        return true_evals + torch.randn_like(true_evals) * self.observation_noise


def fit_lcegp(
    X,
    Y,
    emb_dim,
    fit_tries,
    old_model: LCEGP = None,
    adam: bool = False,
    use_matern: bool = False,
    use_outputscale: bool = False,
) -> LCEGP:
    model = LCEGP(
        X,
        Y,
        categorical_cols=[0],
        embs_dim_list=[emb_dim],
        outcome_transform=Standardize(m=1),
        use_matern=use_matern,
        use_outputscale=use_outputscale,
    )
    if old_model:
        # initialize new model with old_model's state dict
        old_state_dict = old_model.state_dict()
        model.load_state_dict(old_state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if adam:
        custom_fit_gpytorch_model(
            mll,
            optimizer=fit_gpytorch_torch,
            num_retries=fit_tries,
            options={"disp": False},
        )
    else:
        custom_fit_gpytorch_model(mll, num_retries=fit_tries)
    return model


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


def fit_singletask(X, Y):
    model = SingleTaskGP(
        X,
        Y,
        input_transform=Normalize(d=X.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


labels = [
    "LCEGP",
    "Li",
    "Gao",
    "LCEGP_reuse",
    "ML_IKG",
    "LCEGP_PCS_apx",
    "LCEGP_Matern",
    "LCEGP_Scale",
    "ST_PCS",
    "ST_PCS_apx",
    "ST_IKG",
]
num_labels = len(labels)


def main(
    iterations: int,
    seed: int,
    label: str,
    num_pcs_samples: int = 64,
    num_arms: int = 10,
    num_contexts: int = 10,
    context_dim: int = 1,
    num_full_train: int = 3,
    fit_frequency: int = 1,  # set > 1 if we don't want to fit LCEGP at each iteration.
    fit_tries: int = 1,
    emb_dim: int = 1,
    ground_truth_kwargs: dict = None,
    batch_size: int = 20,
    num_fantasies: int = 16,
    input_dict: dict = None,  # this is for adding more iterations
    randomize_ties: bool = False,
    mode: Optional[str] = None,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
) -> dict:
    r"""This is the updated version for running a single label at a time."""
    assert label in labels, "label not supported!"
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckwargs = {"dtype": dtype, "device": device}
    context_map = torch.rand(num_contexts, context_dim, **ckwargs)
    ground_truth_kwargs = ground_truth_kwargs or dict()

    ground_truth = GroundTruthModel(num_arms, context_map, **ground_truth_kwargs)

    true_means = ground_truth.evaluate_all_true()
    tm_maximizers = true_means.view(num_arms, num_contexts).argmax(dim=0)

    arm_set = torch.arange(0, num_arms, **ckwargs).view(-1, 1)
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
        all_alternatives = X[: num_arms * num_contexts].clone()
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

    start = time()
    existing_iterations = 0
    pcs_estimates = torch.zeros(iterations, **ckwargs)
    correct_selection = torch.zeros(iterations, num_contexts, **ckwargs)
    if input_dict is not None:
        if mode == "-a":
            # adding iterations to existing output
            assert input_dict["label"] == label
            existing_iterations = input_dict["pcs_estimates"].shape[0]
            if existing_iterations >= iterations:
                raise ValueError("Existing output has as many or more iterations!")
            pcs_estimates[:existing_iterations] = input_dict["pcs_estimates"]
            correct_selection[:existing_iterations] = input_dict["correct_selection"]
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
        if "LCEGP" in label:
            if (i - existing_iterations) % fit_frequency != 0:
                # in this case, we will not re-train LCEGP.
                # we will just condition on observations.
                model = old_model.condition_on_observations(
                    X=X[-1].view(1, -1),
                    Y=Y[-1].view(1, 1),
                )
            else:
                model = fit_lcegp(
                    X,
                    Y,
                    emb_dim,
                    fit_tries,
                    old_model=old_model if "reuse" in label else None,
                    adam="Adam" in label,
                    use_matern="Matern" in label,
                    use_outputscale="Scale" in label,
                )
            old_model = model
        elif "ML" in label:
            if (i - existing_iterations) % fit_frequency != 0:
                last_X = X[-1].view(1, -1)
                last_idx = int(last_X[0, 0])
                models = old_model.models
                models[last_idx] = models[last_idx].condition_on_observations(
                    X=last_X[:, 1:], Y=Y[-1].view(-1, 1)
                )
                model = ModelListGP(*models)
            else:
                model = fit_modellist(X, Y, num_arms)
            old_model = model
        elif "ST" in label:
            if (i - existing_iterations) % fit_frequency != 0:
                model = old_model.condition_on_observations(
                    X=X[-1].view(1, -1),
                    Y=Y[-1].view(1, 1),
                )
            else:
                model = fit_singletask(X, Y)
            old_model = model
        elif label in ["Li", "Gao"]:
            model = ContextualIndependentModel(X, Y.squeeze(-1))
        else:
            raise NotImplementedError

        if "LCEGP" in label or "ST" in label:
            # LCEGP or SingleTaskGP
            if "IKG" in label:
                next_point = finite_ikg_maximizer(
                    model, arm_set, context_map, randomize_ties
                )
                next_eval = ground_truth.evaluate(
                    next_point[0, 0].long(), next_point[:, 1:]
                )
            else:
                pcs_vals = torch.zeros(all_alternatives.shape[0], **ckwargs)
                num_batches = math.ceil(all_alternatives.shape[0] / float(batch_size))
                for k in range(num_batches):
                    l_idx = k * batch_size
                    r_idx = min(l_idx + batch_size, all_alternatives.shape[0])
                    pcs_vals[l_idx:r_idx] = estimate_lookahead_generalized_pcs(
                        candidate=all_alternatives[l_idx:r_idx].view(
                            -1, 1, context_dim + 1
                        ),
                        model=model,
                        model_sampler=SobolQMCNormalSampler(num_samples=num_fantasies)
                        if num_fantasies
                        else None,
                        arm_set=arm_set,
                        context_set=context_map,
                        num_samples=64,
                        base_samples=None,
                        func_I=lambda X: (X > 0).to(**ckwargs),
                        rho=lambda X: X.mean(dim=-2),
                        use_approximation="apx" in label,
                    )
                # if the estimate is 1 for multiple points, this would just pick 0,
                # which is not ideal! Added randomization option
                if randomize_ties:
                    max_pcs = pcs_vals.max()
                    max_check = pcs_vals == max_pcs
                    max_count = max_check.sum()
                    max_idcs = torch.arange(
                        0, all_alternatives.shape[0], device=ckwargs["device"]
                    )[max_check]
                    maximizer = max_idcs[
                        torch.randint(max_count, (1,), device=ckwargs["device"])
                    ].squeeze()
                else:
                    maximizer = pcs_vals.argmax()
                next_arm = maximizer // num_contexts
                next_context = maximizer % num_contexts
                next_point = torch.cat(
                    [torch.tensor([next_arm], **ckwargs), context_map[next_context]]
                ).view(1, -1)
        elif "ML" in label:
            # ModelListGP
            if "IKG" in label:
                next_arm, next_context = finite_ikg_maximizer_modellist(
                    model, context_map, randomize_ties
                )
                next_point = torch.cat(
                    [torch.tensor([next_arm], **ckwargs), context_map[next_context]]
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

        if label not in ["LCEGP_IKG", "ST_IKG"]:
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
            rho=lambda X: X.mean(dim=-2),
        )

        # check for correct selection for empirical PCS
        if "LCEGP" in label or "ST" in label:
            post_mean = model.posterior(
                all_alternatives.view(num_arms, num_contexts, context_dim + 1)
            ).mean.squeeze(-1)
        elif "ML" in label:
            post_mean = model.posterior(context_map).mean.t()
        else:
            post_mean = model.means

        maximizers = post_mean.argmax(dim=0)

        correct_selection[i] = tm_maximizers == maximizers

    output_dict = {
        "label": label,
        "X": X,
        "Y": Y,
        "true_means": true_means,
        "pcs_estimates": pcs_estimates,
        "correct_selection": correct_selection,
    }
    return output_dict


if __name__ == "__main__":
    current_dir = path.dirname(path.abspath(__file__))
    exp_dir = path.join(current_dir, sys.argv[1])
    config_path = path.join(exp_dir, "config.json")
    label = sys.argv[2]
    seed = int(sys.argv[3])
    output_path = path.join(exp_dir, f"{str(seed).zfill(4)}_{label}.pt")
    input_dict = None
    mode = None
    if path.exists(output_path):
        if len(sys.argv) > 4 and sys.argv[4] in ["-a", "-f"]:
            mode = sys.argv[4]
            if sys.argv[4] == "-f":
                print("Overwriting the existing output!")
            elif sys.argv[4] == "-a":
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
    output = main(seed=seed, label=label, input_dict=input_dict, mode=mode, **kwargs)
    torch.save(output, output_path)
