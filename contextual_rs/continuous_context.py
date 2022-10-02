"""
Continuous extension of GP-C-OCBA.
We need to find the minimizer of Zeta for each arm.
We then pick the minimum pair such that it is not the best arm
for that given context.

For each context, just get the min of zeta, excluding the best.
Find the context that minimizes minimum of zeta, then find the
corresponding minimizer arm from there.

"""

import torch
from typing import Tuple
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models import ModelListGP
from botorch.posteriors.posterior import Posterior
from torch import Tensor


def _get_hat_Z(posterior: Posterior) -> Tuple[Tensor, Tensor]:
    r"""Compute the zeta given the posterior.
    The zeta is computed for all arms except the best.
    Also returns the indices for sorting the means.
    """
    # These are both batch x num_arms.
    means = posterior.mean.squeeze(-2)
    variances = posterior.variance.squeeze(-2)
    sorted_means, idcs = means.sort(dim=-1, descending=True)
    sorted_vars = variances.gather(dim=-1, index=idcs)
    num_arms_skip_one = means.shape[-1] -1
    # From here on, we skip the best arm, so batch x num_arms_skip_one.
    added_vars = sorted_vars[:, :1].expand(-1, num_arms_skip_one) + sorted_vars[:, 1:]
    mean_diffs = sorted_means[:, :1].expand(-1, num_arms_skip_one) - sorted_means[:, 1:]
    squared_diffs = mean_diffs.pow(2)
    return squared_diffs / added_vars, idcs


class MinZeta(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: ModelListGP,
    ) -> None:
        r"""The acquisition function for finding the context minimizing the
        zeta in GP-C-OCBA. Intended for use in continuous context spaces.
        This is defined for maximization (to follow BoTorch convention). The
        return value is thus the negative of the minimum of the zeta for the
        given context, with the minimum taken over the arms, excluding the
        predicted best arm.

        Args:
            model: A ModelListGP where each outcome defines one of the arms.
                The model is defined over the context space.
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the negative of the minimizer of zeta for the given contexts.

        Args:
            X: A `batch x 1 x d_c`-dim tensor of contexts.

        Returns:
            A `batch`-dim tensor denoting the negative of the minimizer of zeta
            for each context.
        """
        assert X.shape[-2] == 1 and X.ndim == 3
        posterior = self.model.posterior(X)
        # Compute the zeta, batch x num_arms - 1.
        hat_Z, _ = _get_hat_Z(posterior=posterior)
        # Minimizer of zeta for each context.
        min_hat_Z = hat_Z.min(dim=-1).values
        # Negating for maximization.
        return  -min_hat_Z


def find_next_arm_given_context(
    next_context: Tensor,
    model: ModelListGP,
    kernel_scale: float = 2.0,
    randomize_ties: bool = True,
) -> int:
    r"""Get the next arm to sample given the context.

    Args:
        next_context: A `1 x d_c`-dim tensor denoting the next context.
        model: A ModelListGP where each outcome defines one of the arms.
        kernel_scale: The coefficient used to scale the distances before computing
            the densities. A kernel density estimator is used to approximate the `p`.
        randomize_ties: If there are multiple minimizers of Z, pick the
            returned one randomly.

    Returns:
        The index of the next arm to evaluate.
    """
    posterior = model.posterior(next_context.view(1, 1, -1))
    # This is num_arms - 1 & sorted.
    hat_Z, idcs = _get_hat_Z(posterior)
    flat_Z = hat_Z.view(-1)
    min_Z, minimizer = torch.min(flat_Z, dim=-1)
    if randomize_ties:
        min_check = flat_Z == min_Z
        min_count = min_check.sum()
        if min_count > 1:
            min_idcs = torch.arange(0, flat_Z.shape[0], device=hat_Z.device)[min_check]
            minimizer = min_idcs[
                torch.randint(min_count, (1,), device=hat_Z.device)
            ].squeeze()
    min_sorted_arm = minimizer
    # part 2 b) - train_counts stands in for the p_values
    # we calculate the psi^{(1/2)} here, recorded as y_s_1/2
    train_inputs = model.train_inputs
    weighted_counts = torch.zeros(len(model.models)).to(hat_Z)
    for i, arm_inputs in enumerate(train_inputs):
        arm_inputs = arm_inputs[0]
        # # of obs x 1
        distances = torch.cdist(arm_inputs, next_context)
        densities = torch.exp(-kernel_scale*distances)
        # Then we can sum it over the observations to get the distance weighted
        # observation count for the context.
        weighted_counts[i] = densities.sum(dim=0)
    y_vals = weighted_counts / posterior.variance.squeeze()
    idcs = idcs.view(-1)  # Clear up added dimensions.
    sorted_y_vals = y_vals[idcs]
    y_s_1 = sorted_y_vals[0]
    y_s_2 = sorted_y_vals[1:].sum()
    # check for condition (16) and pick arm accordingly
    # + 1 accounts for the 0th index which disappears when calculating hat Z
    next_sorted_arm = 0 if y_s_1 < y_s_2 else min_sorted_arm + 1
    # map the sorted_arm to original index
    next_arm = idcs[next_sorted_arm]
    return next_arm
