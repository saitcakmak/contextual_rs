import math
from collections import Callable
from copy import deepcopy
from typing import Any

import torch
from botorch.fit import _set_transformed_inputs
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.optim.utils import sample_all_priors
from contextual_rs.models.lce_gp import LCEGP
from gpytorch.mlls import MarginalLogLikelihood


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
    assert isinstance(mll.model, LCEGP), "Only supports LCEGP!"
    num_retries = kwargs.pop("num_retries", 1)
    mll.train()
    original_state_dict = deepcopy(mll.model.state_dict())
    retry = 0
    state_dict_list = list()
    mll_values = torch.zeros(num_retries)
    max_error_tries = kwargs.pop("max_error_tries", 10)
    randn_factor = kwargs.pop("randn_factor", 0.1)
    error_count = 0
    while retry < num_retries:
        if retry > 0:  # use normal initial conditions on first try
            mll.model.load_state_dict(original_state_dict)
            # randomize the embedding as well, reinitializing here.
            # two alternatives for initialization, specified by passing randn_factor
            for i, emb_layer in enumerate(mll.model.emb_layers):
                if randn_factor == 0:
                    new_emb = torch.nn.Embedding(
                        emb_layer.num_embeddings,
                        emb_layer.embedding_dim,
                        max_norm=emb_layer.max_norm,
                    ).to(emb_layer.weight)
                    mll.model.emb_layers[i] = new_emb
                else:
                    new_weight = torch.randn_like(emb_layer.weight) * randn_factor
                    emb_layer.weight = torch.nn.Parameter(
                        new_weight, requires_grad=True
                    )
            sample_all_priors(mll.model)
        mll, info_dict = optimizer(mll, track_iterations=False, **kwargs)
        opt_val = info_dict["fopt"]
        if math.isnan(opt_val):
            if error_count < max_error_tries:
                error_count += 1
                continue
            else:
                state_dict_list.append(None)
                mll_values[retry] = float("-inf")

        # record the fitted model and the corresponding mll value
        state_dict_list.append(deepcopy(mll.model.state_dict()))
        mll_values[retry] = -opt_val  # negate to get mll value
        retry += 1

    # pick the best among all trained models
    best_idx = mll_values.argmax()
    best_params = state_dict_list[best_idx]
    mll.model.load_state_dict(best_params)
    _set_transformed_inputs(mll=mll)
    return mll.eval()
