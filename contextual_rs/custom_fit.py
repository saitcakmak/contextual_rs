from collections import Callable
from copy import deepcopy
from typing import Any

import torch
from botorch.fit import _set_transformed_inputs
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.optim.utils import sample_all_priors
from contextual_rs.lce_gp import LCEGP
from gpytorch.mlls import MarginalLogLikelihood
from torch import Tensor


def _eval_mll(mll: MarginalLogLikelihood) -> Tensor:
    r"""
    Evaluates the mll on training inputs. The larger the better.
    """
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    return mll(mll.model(*train_inputs), train_targets).sum()


def custom_fit_gpytorch_model(
    mll: MarginalLogLikelihood, optimizer: Callable = fit_gpytorch_scipy, **kwargs: Any
) -> MarginalLogLikelihood:
    r"""
    This is a modified version of BoTorch `fit_gpytorch_model`. `fit_gpytorch_model`
    has some inconsistent behavior in fitting the embedding weights in LCEGP.
    The idea here is to get around this issue by aiming for a global fit.

    Args:
        mll: The marginal log-likelihood of the model. To be maximized.
        optimizer: The optimizer for optimizing the mll starting from an
            initialization of model parameters.
        **kwargs: Optional arguments.

    Returns:
        The optimized mll.
    """
    # TODO: we could add raw samples here as well
    assert isinstance(mll.model, LCEGP), "Only supports LCEGP!"
    num_retries = kwargs.pop("num_retries", 1)
    mll.train()
    original_state_dict = deepcopy(mll.model.state_dict())
    retry = 0
    state_dict_list = list()
    mll_values = torch.zeros(num_retries)
    while retry < num_retries:
        if retry > 0:  # use normal initial conditions on first try
            mll.model.load_state_dict(original_state_dict)
            # randomize the embedding as well
            for emb_layer in mll.model.emb_layers:
                new_weight = torch.randn_like(emb_layer.weight)
                emb_layer.weight = torch.nn.Parameter(new_weight, requires_grad=True)
            sample_all_priors(mll.model)
        mll, _ = optimizer(mll, track_iterations=False, **kwargs)
        mll.eval()
        # record the fitted model and the corresponding mll value
        state_dict_list.append(deepcopy(mll.model.state_dict()))
        mll_values[retry] = _eval_mll(mll)
        retry += 1

    # pick the best among all trained models
    best_idx = mll_values.argmax()
    best_params = state_dict_list[best_idx]
    mll.train()
    mll.model.load_state_dict(best_params)
    _set_transformed_inputs(mll=mll)
    return mll.eval()
