r"""Defines some commonly used utilities for the experiments,
such as the functions fitting the GP models.
"""

from typing import Optional

from botorch import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.optim.utils import sample_all_priors
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.utils.errors import NotPSDError
from torch import Tensor


def fit_modellist(X: Tensor, Y: Tensor, num_arms: int) -> ModelListGP:
    r"""
    Fit a ModelListGP with a SingleTaskGP model for each arm.

    Args:
        X: A tensor representing all arm-context pairs that have been evaluated.
            First column represents the arm.
        Y: A tensor representing the corresponding evaluations.
        num_arms: An integer denoting the number of arms.

    Returns:
        A fitted ModelListGP.
    """
    mask_list = [X[..., 0] == i for i in range(num_arms)]
    model = ModelListGP(
        *[
            SingleTaskGP(
                X[mask_list[i]][..., 1:],
                Y[mask_list[i]],
                outcome_transform=Standardize(m=1),
                input_transform=Normalize(d=X.shape[-1] - 1),
            )
            for i in range(num_arms)
        ]
    )
    for m in model.models:
        try:
            mll = ExactMarginalLogLikelihood(m.likelihood, m)
            fit_gpytorch_model(mll)
        except NotPSDError:
            sample_all_priors(m)
            mll = ExactMarginalLogLikelihood(m.likelihood, m)
            fit_gpytorch_model(mll)
    return model


def fit_single_gp(X: Tensor, Y: Tensor) -> SingleTaskGP:
    r"""
    Fit a SingleTaskGP on all data.
    """
    model = SingleTaskGP(
        X, Y, outcome_transform=Standardize(m=1), input_transform=Normalize(d=X.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


# # of inputs used to train the model the last time. Used to skip re-fitting if not necessary.
num_last_train_inputs = []


def fit_modellist_with_reuse(X: Tensor, Y: Tensor, num_arms: int, old_model: Optional[ModelListGP] = None) -> ModelListGP:
    r"""
    Fit a ModelListGP with a SingleTaskGP model for each arm.

    If the number of inputs to the sub-model has not changed since the old_model,
    the sub-model is reused rather than being re-fitted from scratch.

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
    skip_count = 0
    for i in range(num_arms):
        num_train = len(Y[mask_list[i]])
        if old_model is not None and len(old_model.models) == len(num_last_train_inputs):
            # If the model has the same inputs, we can reuse it.
            if num_train == num_last_train_inputs[i]:
                models.append(old_model.models[i])
                skip_count += 1
                continue
        # If the model inputs changed, re-fit.
        m = SingleTaskGP(
            X[mask_list[i]][..., 1:],
            Y[mask_list[i]],
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=X.shape[-1] - 1),
        )
        try:
            mll = ExactMarginalLogLikelihood(m.likelihood, m)
            fit_gpytorch_model(mll)
        except NotPSDError:
            sample_all_priors(m)
            mll = ExactMarginalLogLikelihood(m.likelihood, m)
            fit_gpytorch_model(mll)
        models.append(m)
        try:
            num_last_train_inputs[i] = num_train
        except IndexError:
            assert len(num_last_train_inputs) == i
            num_last_train_inputs.append(num_train)
    print(f"Skipped model fitting for {skip_count} out of {num_arms}.")
    return ModelListGP(*models)
