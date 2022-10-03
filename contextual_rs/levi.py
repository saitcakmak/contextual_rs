"""
Implements the LEVI algorithm from Pearce & Branke (2017) EFFICIENT EXPECTED
IMPROVEMENT ESTIMATION FOR CONTINUOUS MULTIPLE RANKING AND SELECTION.

In essence, this is the EI of the given context (ignoring the effects on others)
from sampling at the given alternative. For computing the EI, the current best
value used is the predicted (posterior mean) of the best predicted alternative.

The implementation is loosely based on BoTorch analytic ExpectedImprovement,
with modifications to reduce the number of posterior calls needed.
"""

from typing import Tuple, Optional

import torch
from botorch.models import ModelListGP
from botorch.acquisition import AnalyticAcquisitionFunction
from torch.distributions import Normal
from torch import Tensor


def _compute_all_EI_standard(X: Tensor, model: ModelListGP) -> Tensor:
    r"""Compute the EI for all arms for the given contexts.
    The `best_f` for each context is the maximum of the posterior mean.

    NOTE: This uses the standard noise-free EI formula. This is not the
    same as the EI used in LEVI.

    Args:
        X: A `batch x 1 x d_c`-dim tensor of contexts.
        model: A ModelListGP with a model for each alternative, defined
            over the context space.

    Returns:
        A `batch x num_arms`-dim tensor of EI values.
    """
    posterior = model.posterior(X)
    mean = posterior.mean.squeeze(-2)
    sigma = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-2)
    best_f = mean.max(dim=-1, keepdim=True).values
    u = (mean - best_f) / sigma
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * (updf + u * ucdf)
    return ei


def _compute_all_EI_LEVI(X: Tensor, model: ModelListGP) -> Tensor:
    r"""Compute the EI for all arms for the given contexts.
    The `best_f` for each context is the maximum of the posterior mean.

    NOTE: This uses the EI formula for LEVI, as given in Eq 12 of the paper.

    Args:
        X: A `batch x 1 x d_c`-dim tensor of contexts.
        model: A ModelListGP with a model for each alternative, defined
            over the context space.

    Returns:
        A `batch x num_arms`-dim tensor of EI values.
    """
    posterior = model.posterior(X)
    mean = posterior.mean
    # Get the max mean excluding the alternative corresponding to the given row.
    expanded_mean = mean.expand(-1, mean.shape[-1], -1)
    expanded_mean = expanded_mean - torch.full(mean.shape[-1:],float("inf")).diag()
    # Delta_a as defined after eq 12.
    delta = - (mean.squeeze(-2) - expanded_mean.max(dim=-1).values).abs()

    variance = posterior.variance.clamp_min(1e-9).squeeze(-2)
    variance_w_noise = variance.clone()
    for i in range(variance.shape[-1]):
        raw_noise = model.likelihood.likelihoods[i].noise.item()
        standardize_scale = model.models[i].outcome_transform._stdvs_sq.item()
        variance_w_noise[..., i] = variance_w_noise[..., i] + raw_noise * standardize_scale
    # This is the sigma_tilde as defined after eq 9, evaluated at x_n+1, x_n+1.
    sigma = variance / variance_w_noise.sqrt()
    # Calculate eq 12.
    u = delta / sigma
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = delta * ucdf - sigma * updf
    return ei

def _compute_all_EI(X: Tensor, model: ModelListGP) -> Tensor:
    r"""Compute the EI for all arms for the given contexts.
    The `best_f` for each context is the maximum of the posterior mean.

    NOTE: This uses the EI formula for LEVI, updated from the eq 12 of paper
    to fix the buggy definition.

    Args:
        X: A `batch x 1 x d_c`-dim tensor of contexts.
        model: A ModelListGP with a model for each alternative, defined
            over the context space.

    Returns:
        A `batch x num_arms`-dim tensor of EI values.
    """
    posterior = model.posterior(X)
    mean = posterior.mean
    # Get the max mean excluding the alternative corresponding to the given row.
    expanded_mean = mean.expand(-1, mean.shape[-1], -1)
    expanded_mean = expanded_mean - torch.full(mean.shape[-1:],float("inf")).diag()
    # Delta_a modified from eq 12 to fix the weird absolute value & negative.
    delta = mean.squeeze(-2) - expanded_mean.max(dim=-1).values

    variance = posterior.variance.clamp_min(1e-9).squeeze(-2)
    variance_w_noise = variance.clone()
    for i in range(variance.shape[-1]):
        raw_noise = model.likelihood.likelihoods[i].noise.item()
        standardize_scale = model.models[i].outcome_transform._stdvs_sq.item()
        variance_w_noise[..., i] = variance_w_noise[..., i] + raw_noise * standardize_scale
    # This is the sigma_tilde as defined after eq 9, evaluated at x_n+1, x_n+1.
    sigma = variance / variance_w_noise.sqrt()
    # Calculate eq 12.
    u = delta / sigma
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * (updf + u * ucdf)
    return ei


class PredictiveEI(AnalyticAcquisitionFunction):
    def __init__(self, model: ModelListGP):
        r"""A variant of EI that uses the predicted best alternative as the
        best observed value. This is used in LEVI.

        Args:
            model: A ModelListGP with a model for each alternative, defined
                over the context space.
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)

    def forward(self, X: Tensor) -> Tensor:
        r"""Calculate the max EI over all arms for the given contexts.

        Args:
            X: A `batch x 1 x d_c`-dim tensor of contexts.

        Returns:
            A `batch`-dim tensor of max EI values.
        """
        all_ei = _compute_all_EI(X=X, model=self.model)
        return all_ei.max(dim=-1).values


def discrete_levi(
    model: ModelListGP,
    context_set: Tensor,
    weights: Optional[Tensor] = None,
) -> Tuple[int, Tensor]:
    r"""Get the next alternative / context to evaluate by maximizing LEVI.

    NOTE: This can also be used to get the arm for the given context in the
    continuous context setting.

    Args:
        model: A ModelListGP with a model for each alternative, defined
            over the context space.
        context_set: The set of contexts to consider. `num_contexts x d_c`.
        weights: An optional `num_contexts`-dim tensor of weights. Used to
            weight the EI values before picking the maximizer.

    Returns:
        The index of the next alternative and the next context.
    """
    all_ei = _compute_all_EI(X=context_set.unsqueeze(-2), model=model)
    max_ei, max_idcs = all_ei.max(dim=-1)
    if weights is not None:
        # Applying here since it is the same for all arms.
        max_ei = max_ei * weights
    context_idx = max_ei.argmax()
    next_alternative = max_idcs[context_idx]
    next_context = context_set[context_idx]
    return next_alternative, next_context
