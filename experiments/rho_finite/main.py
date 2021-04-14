"""
Comparing the lookahead PCS with MLGP with IKG and Li and Gao.
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
from botorch.models.transforms import Standardize
from botorch.optim.fit import fit_gpytorch_torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions import Branin, Hartmann, Griewank
from botorch.utils import draw_sobol_normal_samples
from botorch.utils.transforms import unnormalize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from contextual_rs.contextual_rs_strategies import (
    li_sampling_strategy,
    gao_sampling_strategy,
    gao_modellist, gao_lcegp,
)
from contextual_rs.finite_ikg import (
    finite_ikg_maximizer_modellist, finite_ikg_maximizer,
)
from contextual_rs.generalized_pcs import (
    estimate_lookahead_generalized_pcs_modellist,
    estimate_current_generalized_pcs,
)
from contextual_rs.models.contextual_independent_model import ContextualIndependentModel
from contextual_rs.models.custom_fit import custom_fit_gpytorch_model
from contextual_rs.models.lce_gp import LCEGP


class GroundTruthModel:
    def __init__(
        self,
        num_arms: int,
        context_map: Tensor,
        num_init_samples: int = None,
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


def fit_lcegp(
    X,
    Y,
    emb_dim,
    fit_tries,
    old_model: LCEGP = None,
    adam: bool = False,
    use_matern: bool = False,
    use_outputscale: bool = False,
    init_strategy: Optional[str] = None,
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
    emb_layer = model.emb_layers[0]
    num_arms = emb_layer.weight.shape[0]
    if init_strategy == "rand":
        new_weight = torch.rand_like(emb_layer.weight)
        emb_layer.weight = torch.nn.Parameter(
            new_weight, requires_grad=True
        )
    elif init_strategy == "gp":
        covar = model.emb_covar_module
        latent_covar = covar(
            torch.linspace(
                0.0,
                1.0,
                num_arms,
            )
        ).add_jitter(1e-4)
        latent_dist = MultivariateNormal(
            torch.zeros(num_arms),
            latent_covar,
        )
        latent_sample = latent_dist.sample().reshape(emb_layer.weight.shape)
        emb_layer.weight = torch.nn.Parameter(
            latent_sample, requires_grad=True
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


labels = [
    "ML_IKG",
    "ML_Gao",
    "ML_Gao_infer_p",
    "LCEGP_Gao_reuse",
    "LCEGP_Gao_reuse_infer_p",
    "Li",
    "Gao",
]
num_labels = len(labels)


def main(
    iterations: int,
    seed: int,
    label: str,
    rho_key: str = "mean",
    num_pcs_samples: int = 64,
    num_arms: int = 10,
    num_contexts: int = 10,
    context_dim: int = 1,
    num_full_train: int = 2,
    fit_frequency: int = 1,  # set > 1 if we don't want to fit the GP at each iteration.
    ground_truth_kwargs: dict = None,
    batch_size: int = 20,
    num_fantasies: int = 16,
    input_dict: dict = None,  # this is for adding more iterations
    randomize_ties: bool = True,
    mode: Optional[str] = None,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
) -> dict:
    r"""This is the updated version for running a single label at a time."""
    assert label in labels, "label not supported!"
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckwargs = {"dtype": dtype, "device": device}
    # fixing context_map for consistency across replications
    old_state = torch.get_rng_state()
    torch.manual_seed(0)
    context_map = torch.rand(num_contexts, context_dim, **ckwargs)
    torch.set_rng_state(old_state)
    ground_truth_kwargs = ground_truth_kwargs or dict()

    def rho(X: Tensor) -> Tensor:
        r"""Operates on -2 dimension. Dimension gets reduced by 1."""
        if rho_key == "mean":
            return X.mean(dim=-2)
        elif rho_key == "worst":
            min_, _ = X.min(dim=-2)
            return min_
        elif "var_" in rho_key:
            cvar = rho_key[0] == "c"
            alpha = float(rho_key[cvar + 4 :])
            alpha_idx = int(alpha * X.shape[-2])
            # This is descending, so we look at the lower tail
            sorted, _ = X.sort(dim=-2, descending=True)
            if cvar:
                return sorted[..., alpha_idx:, :].mean(dim=-2)
            else:
                return sorted[..., alpha_idx, :]
        else:
            raise NotImplementedError  # TODO: maybe mean-var too?

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
        assert torch.allclose(true_means, input_dict["true_means"])
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
        if "ML" in label:
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
        elif "LCEGP" in label:
            if (i - existing_iterations) % fit_frequency != 0:
                model = old_model.condition_on_observations(
                    X=X[-1].view(1, -1), Y=Y[-1].view(1, 1),
                )
            else:
                if "rand" in label:
                    init_strategy = "rand"
                elif "gp" in label:
                    init_strategy = "gp"
                else:
                    init_strategy = None
                model = fit_lcegp(
                    X,
                    Y,
                    emb_dim=1,
                    fit_tries=1,
                    old_model=old_model if "reuse" in label else None,
                    adam="Adam" in label,
                    use_matern="Matern" in label,
                    use_outputscale="Scale" in label,
                    init_strategy=init_strategy,
                )
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
                        randomize_ties,
                        rho=rho if "rho" in label else None,
                    )
                next_point = torch.cat(
                    [torch.tensor([next_arm], **ckwargs), context_map[next_context]]
                ).view(1, -1)
            elif "PCS" in label:
                pcs_vals = torch.zeros(all_alternatives.shape[0], **ckwargs)
                num_batches = math.ceil(all_alternatives.shape[0] / float(batch_size))
                for k in range(num_batches):
                    l_idx = k * batch_size
                    r_idx = min(l_idx + batch_size, all_alternatives.shape[0])
                    sampler = (
                        SobolQMCNormalSampler(num_samples=num_fantasies)
                        if num_fantasies
                        else None
                    )
                    with torch.no_grad():
                        n_f = num_fantasies or 1
                        pcs_vals[
                            l_idx:r_idx
                        ] = estimate_lookahead_generalized_pcs_modellist(
                            candidate=all_alternatives[l_idx:r_idx].view(
                                -1, 1, context_dim + 1
                            ),
                            model=model,
                            model_sampler=sampler,
                            context_set=context_map,
                            num_samples=num_pcs_samples,
                            base_samples=draw_sobol_normal_samples(
                                d=num_contexts * num_arms,
                                n=num_pcs_samples * n_f,
                                **ckwargs,
                            ).view(num_pcs_samples, n_f, 1, num_contexts, num_arms),
                            func_I=lambda X: (X > 0).to(**ckwargs),
                            rho=rho,
                        )
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
        elif "LCEGP" in label:
            if "Gao" in label:
                with torch.no_grad():
                    next_arm, next_context = gao_lcegp(
                        model,
                        arm_set,
                        context_map,
                        randomize_ties,
                        infer_p="infer_p" in label,
                    )
                next_point = torch.cat(
                    [torch.tensor([next_arm], **ckwargs), next_context]
                ).view(1, -1)
            elif "IKG" in label:
                with torch.no_grad():
                    next_point = finite_ikg_maximizer(
                        model, arm_set, context_map, randomize_ties
                    )
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

        if label == "LCEGP_IKG_reuse":
            next_eval = ground_truth.evaluate(
                next_point[0, 0].long(), next_point[:, 1:]
            )
        elif label == "ML_Gao" or label == "ML_Gao_infer_p" or "LCEGP" in label:
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
        elif "LCEGP" in label:
            post_mean = model.posterior(
                all_alternatives.view(num_arms, num_contexts, context_dim + 1)
            ).mean.squeeze(-1)
        else:
            post_mean = model.means

        maximizers = post_mean.argmax(dim=0)

        correct_selection[i] = tm_maximizers == maximizers

    # apply rho to correct selection
    rho_cs = rho(correct_selection.unsqueeze(-1)).squeeze(-1)

    output_dict = {
        "label": label,
        "X": X,
        "Y": Y,
        "true_means": true_means,
        "pcs_estimates": pcs_estimates,
        "correct_selection": correct_selection,
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
    output = main(seed=seed, label=label, input_dict=input_dict, mode=mode, **kwargs)
    torch.save(output, output_path)


if __name__ == "__main__":
    config = sys.argv[1]
    label = sys.argv[2]
    seed = sys.argv[3]
    last_arg = sys.argv[4] if len(sys.argv) > 4 else None
    submitit_main(config, label, seed, last_arg)
