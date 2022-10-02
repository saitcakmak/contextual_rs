"""
Implements the LEVI algorithm from Pearce & Branke (2017) EFFICIENT EXPECTED
IMPROVEMENT ESTIMATION FOR CONTINUOUS MULTIPLE RANKING AND SELECTION.

In essence, this is the EI of the given context (ignoring the effects on others)
from sampling at the given alternative. For computing the EI, the current best
value used is the predicted (posterior mean) of the best predicted alternative.

The implementation is loosely based on BoTorch analytic ExpectedImprovement,
with modifications to reduce the number of posterior calls needed.
"""

from typing import Tuple

import torch
from botorch.models import ModelListGP
from botorch.acquisition import AnalyticAcquisitionFunction
from torch.distributions import Normal
from torch import Tensor

# TODO: how to implement it so that it works for both discrete and continuous context?
# If we go with this, how do we figure out which arm is the best?
# For continuous setup, we can use acqf which get optimized over cont space.
# Then we can pick the best arm from there.
# For discrete, it is probably best to do it all in one.



def _compute_all_EI(X: Tensor, model: ModelListGP) -> Tensor:
    r"""Compute the EI for all arms for the given contexts.
    The `best_f` for each context is the maximum of the posterior mean.

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
) -> Tuple[int, Tensor]:
    r"""Get the next alternative / context to evaluate by maximizing LEVI.

    NOTE: This can also be used to get the arm for the given context in the
    continuous context setting.

    Args:
        model: A ModelListGP with a model for each alternative, defined
            over the context space.
        context_set: The set of contexts to consider. `num_contexts x d_c`.

    Returns:
        The index of the next alternative and the next context.
    """
    all_ei = _compute_all_EI(X=context_set.unsqueeze(-2), model=model)
    max_ei, max_idcs = all_ei.max(dim=-1)
    context_idx = max_ei.argmax()
    next_alternative = max_idcs[context_idx]
    next_context = context_set[context_idx]
    return next_alternative, next_context
