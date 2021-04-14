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
from typing import Union, List, Optional, Tuple, Callable

import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize, Normalize
from botorch.optim.fit import fit_gpytorch_torch
from botorch.optim.parameter_constraints import _arrayify, make_scipy_bounds
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions import Branin, Hartmann, Griewank
from botorch.utils import draw_sobol_samples, draw_sobol_normal_samples
from botorch.utils.transforms import unnormalize
from scipy.optimize import minimize
from torch import Tensor

from gpytorch import ExactMarginalLogLikelihood

from contextual_rs.models.custom_fit import custom_fit_gpytorch_model
from contextual_rs.models.lce_gp import LCEGP
from contextual_rs.generalized_pcs import (
    estimate_lookahead_generalized_pcs,
    estimate_current_generalized_pcs,
)
from contextual_rs.finite_ikg import (
    finite_ikg_eval,
    finite_ikg_eval_modellist,
)


class GroundTruthModel:
    def __init__(
        self,
        num_arms: int,
        dim: int,
        num_init_samples: int = None,
        init_scale: float = 50.0,
        observation_noise: float = 3.0,
        function: Optional[str] = None,
        **ckwargs,
    ):
        r"""
        Generate a GP model for use as the ground truth for function evaluations.
        The dimension is inferred from context_map, as context_map.shape[-1] + 1.
        The underlying GP model is constructed with random data and the
        hyper-parameters are not trained.

        If function is specified, the GP model is replaced with that function,
        where the first dimension is used for the arms, and the remaining dimensions
        are used for contexts. The function should be defined over d_c + 1 dimensions.
        Arm indices are randomly mapped to the underlying space to avoid giving
        SingleTaskGP an advantage.
        In this case, the `num_init_samples` argument is ignored. The `init_scale`
        argument is used to approximately scale the function values to match the scale.
        """
        self.num_arms = num_arms
        self.dim = dim
        if function is None:
            self.arm_map = torch.linspace(0, 1, num_arms, **ckwargs).view(-1, 1)
            num_init_samples = num_init_samples or self.dim * 10
            train_X = torch.rand(num_init_samples, self.dim, **ckwargs)
            train_Y = torch.randn(num_init_samples, 1, **ckwargs) * init_scale
            self.model = SingleTaskGP(
                train_X, train_Y, outcome_transform=Standardize(m=1)
            )
            self.function = None
            self.func_scale = None
        else:
            self.arm_map = torch.rand(num_arms, 1, **ckwargs)
            if function == "branin":
                self.function = Branin(negate=True)
            elif function == "hartmann":
                self.function = Hartmann(dim=3, negate=True)
            elif function == "greiwank":
                self.function = Griewank(dim=2, negate=True)
                self.function.bounds[0, :].fill_(-5)
                self.function.bounds[0, :].fill_(5)
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

    def evaluate_all_true(self, context_map: Tensor) -> Tensor:
        r"""
        Evaluates all arm-context pairs without noise.
        """
        assert context_map.shape[-1] == self.dim - 1
        if context_map.dim() == 2:
            context_map = context_map.repeat(self.num_arms, 1, 1)
        else:
            assert context_map.shape[0] == self.num_arms
        X = torch.cat(
            [
                self.arm_map.view(-1, 1, 1).repeat(1, context_map.shape[-2], 1),
                context_map,
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

    def evaluate_all(self, context_map: Tensor) -> Tensor:
        true_evals = self.evaluate_all_true(context_map)
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


def maximize_contextual_acqf(
    acqf: Callable,
    acqf_kwargs: dict,
    arm_set: Tensor,
    ics_per_arm: int,
    context_dim: int,
    batch_size: int,
    maxfev: int,
    maxiter: int,
    randomize_ties: bool = True,
) -> Tuple[int, Tensor]:
    r"""
    Optimize the acquisition function using Nelder-Mead. Arm indices are
    separated from the solution tensor during optimization.

    Args:
        acqf: Acquisition function to optimize.
        acqf_kwargs: Extra arguments to call acqf with.
        arm_set: Set of arms to optimize with. Independently optimizes for each
            given arm, then returns the overall maximizer.
        ics_per_arm: Number of random initial conditions for each arm.
        context_dim: The dimension of context variable.
        batch_size: Optimization is done in batches. This controls the number of
            solutions evaluated as a batch, e.g., in parallel.
        maxfev: Max number of function evaluations, passed to optimizer.
        maxiter: Max number of iterations, passed to optimizer.
        randomize_ties: If there are multiple maximizers, picks the maximizer randomly.

    Returns:
        The maximizer arm index and the corresponding context.
    """
    num_arms = arm_set.shape[0]
    ics_tensor = torch.cat(
        [
            arm_set.view(-1, 1, 1).repeat(1, ics_per_arm, 1),
            draw_sobol_samples(
                bounds=torch.tensor([[0.0], [1.0]]).repeat(1, context_dim).to(arm_set),
                n=num_arms * ics_per_arm,
                q=1,
            )
            .reshape(num_arms, ics_per_arm, -1)
            .to(arm_set),
        ],
        dim=-1,
    ).view(-1, 1, context_dim + 1)

    def f(x):
        if np.isnan(x).any():
            raise RuntimeError(
                f"{np.isnan(x).sum()} elements of the {x.size} element array "
                f"`x` are NaN."
            )
        X = torch.from_numpy(x).to(ics_tensor).view(shape_X).contiguous()
        X = X.clamp(min=0.0, max=1.0)
        # join the arms
        X = torch.cat([current_arms, X], dim=-1)
        with torch.no_grad():
            loss = -acqf(X, **acqf_kwargs).sum()
        fval = loss.item()
        return fval

    opt_candidates = torch.zeros_like(ics_tensor)
    opt_vals = torch.zeros(ics_tensor.shape[0]).to(ics_tensor)
    num_batches = math.ceil(ics_tensor.shape[0] / float(batch_size))
    for k in range(num_batches):
        l_idx = k * batch_size
        r_idx = min(l_idx + batch_size, ics_tensor.shape[0])
        current_ics_tensor = ics_tensor[l_idx:r_idx]
        current_arms = current_ics_tensor[..., :1]
        current_ic_contexts = current_ics_tensor[..., 1:]
        x0 = _arrayify(current_ic_contexts)
        bounds = make_scipy_bounds(current_ic_contexts, 0.0, 1.0)
        shape_X = current_ic_contexts.shape
        res = minimize(
            f,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxfev": maxfev, "maxiter": maxiter},
        )
        candidates = torch.from_numpy(res.x).to(ics_tensor).view(shape_X).contiguous()
        candidates = candidates.clamp(min=0.0, max=1.0)
        candidates_w_arms = torch.cat([current_arms, candidates], dim=-1)
        opt_candidates[l_idx:r_idx] = candidates_w_arms
        with torch.no_grad():
            opt_vals[l_idx:r_idx] = acqf(candidates_w_arms, **acqf_kwargs)
    if randomize_ties:
        max_val = opt_vals.max()
        max_check = opt_vals == max_val
        max_count = max_check.sum()
        max_idcs = torch.arange(0, ics_tensor.shape[0])[max_check]
        maximizer = max_idcs[torch.randint(max_count, (1,))].squeeze()
    else:
        maximizer = opt_vals.argmax()
    max_candidate = opt_candidates[maximizer]
    return int(max_candidate[..., 0]), max_candidate[..., 1:].view(1, -1)


labels = [
    "LCEGP_PCS",
    "LCEGP_IKG",
    "ML_IKG",
    "ST_PCS",
    "ST_IKG",
]


def main(
    iterations: int,
    seed: int,
    label: str,
    num_pcs_samples: int = 64,
    num_arms: int = 10,
    num_acqf_contexts_base: Optional[int] = None,
    num_eval_contexts: Optional[int] = None,
    context_dim: int = 1,
    num_train_per_arm: Optional[int] = None,
    fit_frequency: int = 1,  # set > 1 if we don't want to fit LCEGP at each iteration.
    fit_tries: int = 1,  # TODO: maybe push this to label
    emb_dim: int = 1,
    ground_truth_kwargs: dict = None,
    batch_size: int = 20,
    num_fantasies: int = 0,  # TODO: label?
    ics_per_arm: Optional[int] = None,
    maxfev: int = 25,
    maxiter: int = 25,
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
    # set default values
    ics_per_arm = ics_per_arm or 10 * context_dim
    num_train_per_arm = num_train_per_arm or 2 * context_dim + 2
    num_acqf_contexts_base = num_acqf_contexts_base or 10 * context_dim
    num_eval_contexts = num_eval_contexts or 40 * context_dim

    # generate the true function
    ground_truth_kwargs = ground_truth_kwargs or dict()
    ground_truth = GroundTruthModel(
        num_arms, context_dim + 1, **ground_truth_kwargs, **ckwargs
    )

    # generate eval contexts and get the true function evaluations
    context_bounds = torch.tensor([[0.0], [1.0]], **ckwargs).repeat(1, context_dim)
    eval_contexts = (
        draw_sobol_samples(
            bounds=context_bounds,
            n=num_eval_contexts,
            q=1,
        )
        .squeeze(-2)
        .to(**ckwargs)
    )
    true_means = ground_truth.evaluate_all_true(eval_contexts)
    tm_maximizers = true_means.view(num_arms, num_eval_contexts).argmax(dim=0)

    # generate the training data
    arm_set = torch.arange(0, num_arms, **ckwargs).view(-1, 1)
    init_contexts = (
        draw_sobol_samples(
            bounds=context_bounds,
            n=num_arms * num_train_per_arm,
            q=1,
        )
        .reshape(num_arms, num_train_per_arm, -1)
        .to(**ckwargs)
    )
    X = torch.cat(
        [
            arm_set.view(-1, 1, 1).expand(-1, num_train_per_arm, -1),
            init_contexts,
        ],
        dim=-1,
    ).view(-1, context_dim + 1)
    num_train_samples = X.shape[0]
    Y = ground_truth.evaluate_all(init_contexts)

    all_eval_alternatives = torch.cat(
        [
            arm_set.view(-1, 1, 1).expand(-1, num_eval_contexts, -1),
            eval_contexts.repeat(num_arms, 1, 1),
        ],
        dim=-1,
    )

    start = time()
    existing_iterations = 0
    pcs_estimates = torch.zeros(iterations, **ckwargs)
    correct_selection = torch.zeros(iterations, num_eval_contexts, **ckwargs)
    # handle various modes
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
        elif mode == "-reeval":
            train_size = X.shape[0]
            reeval_X = input_dict["X"][train_size:]
            reeval_Y = input_dict["Y"][train_size:]
            label += "_reeval"
        else:
            # This should never happen!
            raise RuntimeError("Mode unsupported!")
    old_model = None
    for i in range(existing_iterations, iterations):
        if i % 10 == 0:
            print(
                f"Starting label {label}, seed {seed}, iteration {i}, time: {time()-start}"
            )
        # fitting the model
        if "LCEGP" in label or mode == "-reeval":
            # explicit indices are for -reeval
            if (i - existing_iterations) % fit_frequency != 0:
                # in this case, we will not re-train LCEGP.
                # we will just condition on observations.
                model = old_model.condition_on_observations(
                    X=X[num_train_samples + i - 1].view(1, -1),
                    Y=Y[num_train_samples + i - 1].view(1, 1),
                )
            else:
                model = fit_lcegp(
                    X[: num_train_samples + i],
                    Y[: num_train_samples + i],
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
        else:
            raise NotImplementedError

        num_acqf_contexts = num_acqf_contexts_base + int(math.sqrt(i))
        acqf_contexts = (
            draw_sobol_samples(
                bounds=context_bounds,
                n=num_arms * num_acqf_contexts,
                q=1,
            )
            .reshape(num_arms, num_acqf_contexts, -1)
            .to(**ckwargs)
        )
        if mode == "-reeval":
            next_point = reeval_X[i].view(1, -1)
            next_eval = reeval_Y[i].view(1, 1)
        else:
            # Construct and optimize an acquisition function
            # Should use an SAA approach for consistency across candidates
            # This will happen in two stages:
            #   select the acqf and fill acqf_options
            #   call maximize_contextual_acqf
            if "LCEGP" in label or "ST" in label:
                # LCEGP or SingleTaskGP
                if "IKG" in label:
                    acqf = finite_ikg_eval
                    acqf_kwargs = {
                        "model": model,
                        "arm_set": arm_set,
                        "context_set": acqf_contexts,
                    }
                elif "EI" in label:
                    raise NotImplementedError
                else:
                    acqf = estimate_lookahead_generalized_pcs
                    acqf_kwargs = {
                        "model": model,
                        "model_sampler": SobolQMCNormalSampler(
                            num_samples=num_fantasies
                        )
                        if num_fantasies
                        else None,
                        "arm_set": arm_set,
                        "context_set": acqf_contexts,
                        "num_samples": 64,
                        "base_samples": draw_sobol_normal_samples(
                            d=num_arms * num_acqf_contexts, n=64, **ckwargs
                        ).view(64, 1, 1, -1, 1),
                        "func_I": lambda X: (X > 0).to(**ckwargs),
                        "rho": lambda X: X.mean(dim=-2),
                        "use_approximation": "apx" in label,
                    }
            elif "ML" in label:
                # ModelListGP
                if "IKG" in label:
                    acqf = finite_ikg_eval_modellist
                    acqf_kwargs = {
                        "model": model,
                        "context_set": acqf_contexts,
                    }
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            next_arm, next_context = maximize_contextual_acqf(
                acqf=acqf,
                acqf_kwargs=acqf_kwargs,
                arm_set=arm_set,
                ics_per_arm=ics_per_arm,
                context_dim=context_dim,
                batch_size=batch_size,
                maxfev=maxfev,
                maxiter=maxiter,
                randomize_ties=randomize_ties,
            )
            next_point = torch.cat(
                [torch.tensor(next_arm, **ckwargs).view(1, 1), next_context], dim=-1
            )
            next_eval = ground_truth.evaluate(next_arm, next_context)

        X = torch.cat([X, next_point], dim=0)
        Y = torch.cat([Y, next_eval], dim=0)

        # report current PCS estimate
        pcs_estimates[i] = estimate_current_generalized_pcs(
            model=model,
            arm_set=arm_set,
            context_set=eval_contexts,
            num_samples=num_pcs_samples,
            # base_samples=draw_sobol_normal_samples(
            #     d=num_arms * num_eval_contexts, n=num_pcs_samples, **ckwargs
            # ).view(num_pcs_samples, -1, 1),  # TODO: does not work with MLGP
            base_samples=None,
            func_I=lambda X: (X > 0).to(**ckwargs),
            rho=lambda X: X.mean(dim=-2),
        )

        # check for correct selection for empirical PCS
        if "LCEGP" in label or "ST" in label or mode == "-reeval":
            post_mean = model.posterior(all_eval_alternatives).mean.squeeze(-1)
        elif "ML" in label:
            post_mean = model.posterior(eval_contexts).mean.t()
        else:
            raise NotImplementedError

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
            elif last_arg in ["-reeval", "-reeval-f"]:
                print(f"Re-evaluating the output of {label} using LCEGP.")
                input_dict = torch.load(output_path)
                output_path = path.join(
                    exp_dir, f"{str(seed).zfill(4)}_{label}_reeval.pt"
                )
                if path.exists(output_path) and "-f" not in last_arg:
                    print(
                        "The output file exists for the reeval run!"
                        "Use -reeval-f if to overwrite the existing reeval output!"
                    )
                    quit()
                mode = "-reeval"  # -reeval-f is not handled internally.
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
