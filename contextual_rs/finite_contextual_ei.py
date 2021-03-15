from typing import Union, Tuple

import torch
from botorch.models import SingleTaskGP
from scipy.stats import norm
from torch import Tensor

from contextual_rs.models.lce_gp import LCEGP


def _f(diff: Tensor, sigma: Tensor) -> Tensor:
    """
    f(diff, sigma) = diff * Phi(diff / sigma) + sigma * phi(diff / sigma)
    """
    assert diff.shape == sigma.shape
    org_device = diff.device
    diff = diff.cpu()
    sigma = sigma.cpu()
    ratio = diff / sigma
    vals = diff * torch.tensor(norm.cdf(ratio)) + sigma * torch.tensor(norm.pdf(ratio))
    return vals.to(org_device)


def contextual_complete_ei(
    model: Union[LCEGP, SingleTaskGP],
    arm_set: Tensor,
    context_set: Tensor,
    randomize_ties: bool = True,
) -> Tuple[int, int]:
    r"""
    Find the maximizer of Contextual Complete Expected Improvement.
    Contextual CEI is a simple extension of CEI to the contextual setting,
    where we calculate the CEI value for each context and for each arm in that
    context, find the maximizer and set that as CCEI for that context.
    The maximizer of CCEI over all contexts, and the arm that maximizes CEI
    for that context are then returned. This much like the MTS algorithm from
    Char et al., except that we use EI rather than TS.

    Args:
        model: An LCEGP instance, for which to find the maximizer of CCEI.
        arm_set: A `num_arms x 1`-dim tensor of arm set.
        context_set: A `num_contexts x d_c`-dim tensor of context set.
        randomize_ties: If True and there are multiple maximizers,
            the result will be randomly selected.

    Returns:
        A tuple of integers denoting the indices of arm-context pair maximizing CCEI.
    """
    assert arm_set.dim() == context_set.dim() == 2
    num_arms = arm_set.shape[0]
    num_contexts = context_set.shape[0]
    # this is num_contexts x num_arms x dim
    arm_context_pairs = torch.cat(
        [
            arm_set.view(1, -1, 1).repeat(num_contexts, 1, 1),
            context_set.view(num_contexts, 1, -1).repeat(1, num_arms, 1),
        ],
        dim=-1,
    )

    with torch.no_grad():
        posterior = model.posterior(arm_context_pairs)
        # means is num_contexts x num_arms
        means = posterior.mean.squeeze(-1)
        # covars is num_contexts x num_arms
        covars = posterior.mvn.covariance_matrix

    # sort the means and covariances
    means, indices = means.sort(dim=-1, descending=True)
    # calculate deltas for future use - deltas are all negative (max - others)
    deltas = - means[..., :1].expand(*means.shape[:-1], num_arms - 1) + means[..., 1:]
    # sort the covariances in the same way
    covars = covars.gather(dim=-2, index=indices.unsqueeze(-1).expand_as(covars))
    # this does the columns, second arm dim
    covars = covars.gather(dim=-1, index=indices.unsqueeze(-2).expand_as(covars))
    # get the first row and expand it to appropriate shape
    c_first = covars[..., :1, :]
    c_first = torch.cat(
        [
            c_first[..., :1].expand(*c_first.shape[:-1], num_arms - 1),
            c_first[..., 1:],
        ],
        dim=-1,
    )
    # writing dimensions explicitly as an implicit shape check
    S_e_upper = c_first.expand(*c_first.shape[:-2], num_arms - 1, 2 * num_arms - 2)
    S_e_low_left = S_e_upper[..., num_arms - 1:].transpose(-1, -2)
    S_e_lower = torch.cat([S_e_low_left, covars[..., 1:, 1:]], dim=-1)
    # A_e S_e product, equivalent to upper - lower
    A_S = S_e_upper - S_e_lower
    # The second matrix product, first half - second half (over last dim)
    S_d = A_S[..., : num_arms - 1] - A_S[..., num_arms - 1:]
    # S_d here is the covariance of the differences.
    # We are only interested in the diagonals for CEI.
    # Diagonals give the variance of difference between the best arm and the one
    # corresponding to the diagonal, with the index taken from indices above.
    S_d_diag = S_d.diagonal(dim1=-2, dim2=-1)

    # now, we're ready to calculate the CEI values
    cei_vals = _f(deltas, S_d_diag)
    # get ccei by maximizing
    ccei_vals, _ = cei_vals.max(dim=-1)

    max_val, max_context_idx = ccei_vals.max(dim=0)
    tmp = cei_vals[max_context_idx] == max_val
    max_arm_idx = torch.arange(0, num_arms - 1)[tmp.cpu()]
    if randomize_ties:
        rand_idx = int(torch.randint(0, len(max_arm_idx), (1,)))
        max_arm_idx = max_arm_idx[rand_idx]
    else:
        max_arm_idx = max_arm_idx[0]
    # the arm index needs to be mapped back to the original index
    # the + 1 is to account for the index of the maximizer
    max_arm_idx = indices[max_context_idx, max_arm_idx + 1]
    return max_arm_idx, max_context_idx
