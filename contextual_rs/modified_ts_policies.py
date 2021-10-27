r"""
These are the modified versions of the TS and TS+ policies of Shen et al. 2021.
The original policy includes `h` and `delta` that the total number of samples
depends on. However, since we use a given max budget, the allocations depend only on
the first stage variance estimates and the total budget. Thus the policy reduces to
a two-stage version of a proportional to variance allocation with the variance
calculated using the formulas given in the paper.
"""
from functools import lru_cache

from torch import Tensor


@lru_cache(maxsize=None)
def x_factor(X: Tensor) -> Tensor:
    r"""
    Calculates (X^T X)^-1 X^T.
    Note: Tensors are cached based on their id's. So, this will be memoized
    only if called using the same exact tensor X.

    Args:
        X: `m x d_c`-dim tensor of design points / contexts.

    Returns:
        An `d_c x m`-dim tensor result of (X^T X)^-1 X^T.
    """
    return X.transpose(-2, -1).matmul(X).inverse().matmul(X.transpose(-2, -1))


def get_beta_hat(X: Tensor, Y_mean: Tensor) -> Tensor:
    r"""
    Get the beta hat following the OLS formula given in the paper.

    WARNING: This assumes that X never changes during a run, so it calculates
    X-dependent values once and stores for later use. If calling this with different
    X, the global variable `x_factor_memoized` needs to be set to None!

    Args:
        X: `m x d_c`-dim tensor of design points / contexts.
        Y_mean: `k x m`-dim tensor average observations.

    Returns:
        `k x d_c`-dim tensor of beta hat values.
    """
    # (X^T X)^-1 X^T - this is d_c x m
    x_factor_ = x_factor(X)
    return Y_mean.matmul(x_factor_.transpose(-2, -1))


def modified_ts(X: Tensor, Y: Tensor, total_budget: int, ratios_only: bool) -> Tensor:
    r"""
    Calculate the variances as in the paper and return the number of
    additional samples to allocate to each arm.

    Note: The returned number of samples need not match the budget exactly.
    The budget is allocated fractionally and then rounded to the nearest integer.

    Args:
        X: `m x d_c`-dim tensor of design points / contexts.
        Y: `n_0 x k x m`-dim tensor of initial observations for `k` arms
            and `m` design points.
        total_budget: Total simulation budget.
        ratios_only: If True, instead of the additional budget, this returns
            the total fraction of samples to allocate to any given arm.

    Returns:
        An `k x m`-dim tensor denoting the number of additional samples to
        allocate to any given arm-context (design point) pair, if `ratios_only=False`.
        Else, returns the total fraction of samples to allocate to the given arm.
    """
    beta_hat = get_beta_hat(X, Y.mean(dim=0))
    # Y - X B: This is n_0 x k x m
    yxb_factor = Y - beta_hat.matmul(X.transpose(-2, -1))
    const = 1.0 / (Y.shape[0] * X.shape[0] - X.shape[-1])
    # These are the S_i^2.
    s_sq = const * yxb_factor.pow(2).sum(dim=-1).sum(dim=0)
    # Repeat to have an estimate for all arm-context pairs.
    s_sq = s_sq.unsqueeze(-1).repeat(1, Y.shape[-1])
    if ratios_only:
        return s_sq / s_sq.sum()
    else:
        budget_allocation = s_sq * total_budget / s_sq.sum()
        additional_budget = (budget_allocation - Y.shape[0]).clamp_min(0)
        return additional_budget.round()


def modified_ts_plus(X: Tensor, Y: Tensor, total_budget: int, ratios_only: bool) -> Tensor:
    r"""
    Calculate the variances as described in the paper for the TS+ procedure
    and return the number of additional samples to allocate to each arm.

        Note: The returned number of samples need not match the budget exactly.
    The budget is allocated fractionally and then rounded to the nearest integer.

    Args:
        X: `m x d_c`-dim tensor of design points / contexts.
        Y: `n_0 x k x m`-dim tensor of initial observations for `k` arms
            and `m` design points.
        total_budget: Total simulation budget.
        ratios_only: If True, instead of the additional budget, this returns
            the total fraction of samples to allocate to any given arm.
    Returns:
        An `k x m`-dim tensor denoting the number of additional samples to
        allocate to any given arm-context (design point) pair, if `ratios_only=False`.
        Else, returns the total fraction of samples to allocate to the given arm.
    """
    y_bar = Y.mean(dim=0)
    s_sq = (Y - y_bar).pow(2).sum(dim=0) / (Y.shape[0] - 1)
    if ratios_only:
        return s_sq / s_sq.sum()
    else:
        budget_allocation = s_sq * total_budget / s_sq.sum()
        additional_budget = (budget_allocation - Y.shape[0]).clamp_min(0)
        return additional_budget.round()
