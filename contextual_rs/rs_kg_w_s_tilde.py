import math
from typing import Tuple, Union

import torch
from contextual_rs.models.unknown_correlation_model import UnknownCorrelationModel

from contextual_rs.models.lce_gp import LCEGP
from scipy.stats import norm, t
from torch import Tensor


def _algorithm_1(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Algorithm 1 from Frazier 2009

    Args:
        a: Input a
        b: Input b, in strictly increasing order

    Returns
        c and A, indices starting with 1 as in original algorithm description!
    """
    # The indices of a and b start with 0, however the rest starts with 1.
    M = a.shape[-1]
    c = torch.empty(M + 1)
    c[0] = -float("inf")
    c[1] = float("inf")
    A = [1]
    for i in range(1, M):
        c[i + 1] = float("inf")
        done = False
        while not done:
            j = A[-1]
            c[j] = (a[j - 1] - a[i]) / (b[i] - b[j - 1])
            if len(A) != 1 and c[j] <= c[A[-2]]:
                A = A[:-1]
            else:
                done = True
        A.append(i + 1)
    return c, torch.tensor(A, dtype=torch.long)


def _f(z: Tensor) -> Tensor:
    """
    The function f defined in the paper as f(z) = phi(z) + z Phi(z) where
    phi and Phi are standard normal PDF and CDF

    Args:
        z: a tensor of input values

    Returns:
        corresponding f(z) values
    """
    z = z.detach()
    return torch.tensor(norm.pdf(z)) + z * torch.tensor(norm.cdf(z))


def find_kg_maximizer_lcegp(model: LCEGP):
    """
    Runs Algorithm 2 as described in the paper.
    This has been modified to work with LCEGP.

    Args:
        model: A fitted LCEGP model with a single categorical input.

    Returns
        The index maximizing KG.
    """
    M = model.category_counts[0]
    all_alternatives = torch.tensor(range(M)).to(model.train_targets).view(-1, 1)
    mu = model.posterior(all_alternatives).mean
    mu = mu.reshape(-1)
    # algorithm loop
    v_star = -float("inf")
    x_star = None
    full_sigma_tilde = model.get_s_tilde(None)
    for i in range(M):
        a = mu.clone()
        b = full_sigma_tilde[i].clone()
        # sort a, b such that b are in non-decreasing order
        # and ties in b are broken so that a_i <= a_i+1 if b_i = b_i+1
        b, index = torch.sort(b)
        a = a[index]
        # handle ties in b, sort a in increasing order if ties found
        if torch.any(b[1:] == b[:-1]):  # pragma: no cover
            for j in range(M):
                a[b == b[j]], _ = torch.sort(a[b == b[j]])
        # remove the redundant entries as described in the algorithm
        remaining = torch.ones(M, dtype=torch.bool)
        remaining[torch.cat([b[1:] == b[:-1], torch.tensor([False])], dim=0)] = 0
        a = a[remaining]
        b = b[remaining]
        # c and A has indices starting at 1!
        c, A = _algorithm_1(a, b)
        b = b[A - 1]
        c = c[A]
        v = torch.log(torch.sum((b[1:] - b[:-1]) * _f(-torch.abs(c[:-1]))))
        if i == 0 or v > v_star:
            v_star = v
            x_star = i
    return x_star


def find_kg_maximizer_ukm(model: UnknownCorrelationModel) -> int:
    """
    Runs Algorithm 2 as described in the paper.
    This is for the multivariate T posterior of UnknownCorrelationModel.

    Args:
        model: A fitted UnknownCorrelationModel.

    Returns
        The index maximizing KG.
    """
    K = model.num_alternatives
    df = model.predictive_df()
    qk = (model.q + 1) / (model.q * df)
    if model.update_method == "KL":
        S0 = math.sqrt(qk) / (
            model.q * (model.b + 1.0 / K) / (model.b + 1.0 / K - K + 1) + 1
        )
    else:
        S0 = 1.0 / math.sqrt((model.q + 1) * model.q * df)
    # algorithm loop
    v_star = -float("inf")
    x_star = None
    for i in range(K):
        a = model.theta.clone()
        b = S0 / model.B[i, i].sqrt() * model.B[:, i]
        # sort a, b such that b are in non-decreasing order
        # and ties in b are broken so that a_i <= a_i+1 if b_i = b_i+1
        b, index = torch.sort(b)
        a = a[index]
        # handle ties in b, sort a in increasing order if ties found
        if torch.any(b[1:] == b[:-1]):  # pragma: no cover
            for j in range(K):
                a[b == b[j]], _ = torch.sort(a[b == b[j]])
        # remove the redundant entries as described in the algorithm
        remaining = torch.ones(K, dtype=torch.bool)
        remaining[torch.cat([b[1:] == b[:-1], torch.tensor([False])], dim=0)] = 0
        a = a[remaining]
        b = b[remaining]
        # c and A has indices starting at 1!
        c, A = _algorithm_1(a, b)
        b = b[A - 1]
        c = c[A[:-1]].abs()
        b_delta = b[1:] - b[:-1]
        expected_values = torch.tensor(t.pdf(c, df)) * (df + c.pow(2)) / (df - 1) - (
            c * (1 - torch.tensor(t.cdf(c, df)))
        )
        v = (b_delta * expected_values).sum()
        if i == 0 or v > v_star:
            v_star = v
            x_star = i
    return x_star


def find_kg_maximizer(model: Union[LCEGP, UnknownCorrelationModel]) -> int:
    if isinstance(model, LCEGP):
        return find_kg_maximizer_lcegp(model)
    else:
        return find_kg_maximizer_ukm(model)
