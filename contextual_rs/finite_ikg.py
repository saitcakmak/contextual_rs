r"""
This defines the integrated KG acquisition function for the
finite arm, finite context setting. This is based on the description
in Pearce and Branke 2018 section 4.1.
This is the same algorithm as the one presented in Frazier et al. 2009,
with some minor differences in the quantities. Frazier aims to find the
maximizer of the KG only, whereas Pearce aims to find the KG value.
It is the algorithm 1 but also includes a bit that is presented in alg 2.
"""
from typing import Tuple, Optional, Callable

import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from scipy.stats import norm
from torch import Tensor

from contextual_rs.models.lce_gp import LCEGP
from contextual_rs.rs_kg_w_s_tilde import _algorithm_1


def _pearce_alg_1(a: Tensor, b: Tensor) -> Tensor:
    r"""
    This calculates the KG value for a given context following the
    description in Algorithm 1 of Pearce and Branke 2018.
    Args:
        a: A 'num_arms'-dim tensor of posterior means
        b: A 'num_arms'-dim tensor of s_tilde values

    Returns:
        The KG value corresponding to given mean and s_tilde vectors.
    """
    # sort a, b such that b are in non-decreasing order
    # and ties in b are broken so that a_i <= a_i+1 if b_i = b_i+1
    # detaching here to avoid errors down the line with scipy
    b, index = torch.sort(b.detach())
    a = a[index].detach()
    M = a.shape[0]
    # handle ties in b, sort a in increasing order if ties found
    if torch.any(b[1:] == b[:-1]):  # pragma: no cover
        for j in range(M):
            a[b == b[j]], _ = torch.sort(a[b == b[j]])
    # remove the redundant entries as described in the algorithm
    remaining = torch.ones(M, dtype=torch.bool)
    remaining[
        torch.cat([b[1:] == b[:-1], torch.tensor([False]).to(a.device)], dim=0)
    ] = 0
    a = a[remaining]
    b = b[remaining]
    # c and A has indices starting at 1!
    c, A = _algorithm_1(a, b)
    a = a[A - 1]
    b = b[A - 1]
    # we want to keep the -inf in c here
    c = c[[0] + A.tolist()]
    # calculate the cdfs and pdfs as vectors
    pdfs = torch.tensor(norm.pdf(c)).to(a)
    cdfs = torch.tensor(norm.cdf(c)).to(a)
    # final value as given in alg 1 of Pearce
    v = torch.sum(a * (cdfs[1:] - cdfs[:-1]) + b * (pdfs[:-1] - pdfs[1:]))
    return v


def _get_s_tilde(model: Model, X, x) -> Tensor:
    if isinstance(model, LCEGP):
        return model.get_s_tilde_general(X, x)
    elif isinstance(model, SingleTaskGP):
        q = X.shape[-2]
        x_cat = torch.cat([x.expand(X.shape[:-2] + x.shape), X], dim=-2)
        full_mvn = model(x_cat)
        full_covar = full_mvn.covariance_matrix
        noise = model.likelihood.noise.squeeze()
        K_x_X = full_covar[..., :-q, -q:]
        K_X_X = full_covar[..., -q:, -q:]
        chol = torch.cholesky(K_X_X + torch.diag(noise.expand(q)).expand_as(K_X_X))
        return K_x_X.matmul(chol.inverse())
    else:
        raise NotImplementedError


def finite_ikg_eval(
    candidates: Tensor,
    model: LCEGP,
    arm_set: Tensor,
    context_set: Tensor,
) -> Tensor:
    r"""
    Calculate the value of IKG for the given candidates, using exact
    computations over finite arm and context sets.

    Args:
        candidates: An `n x 1 x d`-dim tensor of candidates.
        model: An LCEGP instance, for which to find the maximizer of IKG.
        arm_set: A `num_arms x 1`-dim tensor of arm set.
        context_set: A `num_contexts x d_c`-dim tensor of context set.

    Returns:
        An `n`-dim tensor of corresponding IKG values.
    """
    assert arm_set.dim() == 2 and context_set.dim() in [2, 3]
    assert candidates.dim() == 3 and candidates.shape[-2] == 1
    num_arms = arm_set.shape[0]
    num_contexts = context_set.shape[-2]
    if context_set.dim() == 2:
        context_set = context_set.repeat(num_arms, 1, 1)
    arm_context_pairs = torch.cat(
        [
            arm_set.view(-1, 1, 1).repeat(1, num_contexts, 1),
            context_set,
        ],
        dim=-1,
    )

    # means is num_arms x num_contexts
    means = model.posterior(arm_context_pairs).mean.squeeze(-1)

    # this is num_candidates x (num_arms * num_contexts) x 1
    full_sigma_tilde = _get_s_tilde(
        model=model,
        X=candidates,
        x=arm_context_pairs.view(num_arms * num_contexts, -1),
    )

    ikg_vals = torch.zeros(candidates.shape[0]).to(candidates)
    # loop over each candidate and calculate IKG value
    for idx, all_s_tilde in enumerate(full_sigma_tilde):
        ikg_val = torch.tensor(0).to(arm_set)
        shaped_s_tilde = all_s_tilde.reshape(num_arms, num_contexts)
        for c_idx in range(num_contexts):
            ikg_val += _pearce_alg_1(means[:, c_idx], shaped_s_tilde[:, c_idx])
        ikg_vals[idx] = ikg_val
    return ikg_vals


def finite_ikg_maximizer(
    model: LCEGP,
    arm_set: Tensor,
    context_set: Tensor,
    randomize_ties: bool = True,
) -> Tensor:
    r"""
    Find the maximizer of IKG, using exact computations over finite
    arm and context sets.
    This corresponds to the REVI algorithm by Pearce and Branke, where the
    candidate set is also equal to all arm context pairs. If we were to change
    that part and make it a function of a given candidate and optimize that
    somehow, this would be exactly the REVI approximation to IKG.
    Oh, a minor distinction is this ignores the probability distribution over
    contexts, which can be added easily if needed.

    Args:
        model: An LCEGP instance, for which to find the maximizer of IKG.
        arm_set: A `num_arms x 1`-dim tensor of arm set.
        context_set: A `num_contexts x d_c`-dim tensor of context set.
        randomize_ties: If True and there are multiple maximizers,
            the result will be randomly selected.

    Returns:
        A `1 x d`-dim tensor denoting the arm-context pair maximizing IKG.
    """
    assert arm_set.dim() == context_set.dim() in [2, 3]
    num_arms = arm_set.shape[0]
    num_contexts = context_set.shape[-2]
    if context_set.dim() == 2:
        context_set = context_set.repeat(num_arms, 1, 1)
    arm_context_pairs = torch.cat(
        [
            arm_set.view(-1, 1, 1).repeat(1, num_contexts, 1),
            context_set,
        ],
        dim=-1,
    )
    dim = arm_context_pairs.shape[-1]

    # means is num_arms x num_contexts
    means = model.posterior(arm_context_pairs).mean.squeeze(-1)

    # this is (num_arms * num_contexts) x (num_arms * num_contexts) x 1
    full_sigma_tilde = _get_s_tilde(
        model=model,
        X=arm_context_pairs.view(-1, 1, dim),
        x=arm_context_pairs.view(num_arms * num_contexts, -1),
    )

    max_ikg_val = torch.tensor(float("-inf")).to(arm_set)
    max_idx = list()
    # loop over each candidate and calculate IKG value
    for idx, all_s_tilde in enumerate(full_sigma_tilde):
        ikg_val = torch.tensor(0).to(arm_set)
        shaped_s_tilde = all_s_tilde.reshape(num_arms, num_contexts)
        for c_idx in range(num_contexts):
            ikg_val += _pearce_alg_1(means[:, c_idx], shaped_s_tilde[:, c_idx])
        if ikg_val > max_ikg_val:
            max_ikg_val = ikg_val
            max_idx = [idx]
        elif ikg_val == max_ikg_val:
            max_idx.append(idx)
    if randomize_ties:
        rand_idx = int(torch.randint(0, len(max_idx), (1,)))
        max_idx = max_idx[rand_idx]
    else:
        max_idx = max_idx[0]
    return arm_context_pairs.view(-1, dim)[max_idx].view(-1, dim)


def _get_modellist_s_tilde(
    model: ModelListGP, context_set: Tensor, candidates: Optional[Tensor] = None
) -> Tensor:
    r"""
    Calculates the s_tilde for the ModelListGP.

    .. math::
        \tilde{\sigma}(x, X) = K(x, X) / \sqrt{ K(X, X) + \diag(\sigma^2(X)) }

    If candidates is not specified:
        For a single model, we want s_tilde for each context as being the fantasy point,
        e.g, X, and calculate the s_tilde vector for each X for all x=context_set.
        This corresponds to a num_contexts x num_contexts tensor for each arm / model.
        So, this will result in a num_arms x num_contets x num_contexts elements.
    If candidates is specified:
        Candidates becomes X. However, this is not that straightforward. We need to
        separate candidates into candidates for each arm, calculate the s_tilde for
        each arm, and join them together. In this case, the return shape will be
        num_candidates x num_contexts.

    Args:
        model: A ModelListGP with a model corresponding to each arm.
        context_set: A `num_contexts x d_c`-dim tensor of context set.
        candidates: If specified, we process arms one by one, and calculate the
            s_tilde corresponding to each candidate as discussed above.

    Returns:
        If candidates is None:
            A `num_arms x num_contexts x num_contexts`-dim tensor where each
            res[arm, context] with give the s_tilde vector for that arm-context pair.
        Else:
            A `num_candidates x num_contexts`-dim tensor where res[i] gives the
            s_tilde vector for that candidate.
    """
    if candidates is None:
        assert context_set.dim() == 2
        full_post = model.posterior(context_set)
        # covars is num_arms x num_contexts x num_contexts
        covars = full_post.mvn.lazy_covariance_matrix.base_lazy_tensor.evaluate()
        # noises is a num_arms tensor
        noises = torch.stack([l.noise.squeeze() for l in model.likelihood.likelihoods])
        diag_covars = torch.diagonal(covars, dim1=-2, dim2=-1)
        noisy_diag = diag_covars + noises.unsqueeze(-1).expand_as(diag_covars)
        noisy_diag_root = noisy_diag.sqrt()
        expanded = noisy_diag_root.unsqueeze(-1).expand_as(covars)
        return covars / expanded
    else:
        assert candidates.dim() == 3 and candidates.shape[-2] == 1
        q = candidates.shape[-2]
        num_arms = len(model.models)
        num_contexts = context_set.shape[-2]
        if context_set.dim() == 2:
            context_set = context_set.repeat(num_arms, 1, 1)
        s_tilde = torch.zeros(candidates.shape[0], num_contexts).to(candidates)
        for arm_idx, arm_model in enumerate(model.models):
            mask = (candidates[..., 0] == arm_idx).view(-1)
            X = candidates[mask][..., 1:]
            x = context_set[arm_idx]
            x_cat = torch.cat([x.expand(X.shape[:-2] + x.shape), X], dim=-2)
            full_mvn = arm_model(x_cat)
            full_covar = full_mvn.covariance_matrix
            noise = arm_model.likelihood.noise.squeeze()
            K_x_X = full_covar[..., :-1, -1:]
            K_X_X = full_covar[..., -1:, -1:]
            chol = torch.cholesky(K_X_X + torch.diag(noise.expand(q)).expand_as(K_X_X))
            arm_s_tilde = K_x_X.matmul(chol.inverse())
            s_tilde[mask] = arm_s_tilde.squeeze(-1)
        return s_tilde


def finite_ikg_eval_modellist(
    candidates: Tensor,
    model: ModelListGP,
    context_set: Tensor,
) -> Tensor:
    r"""
    Same idea as below. Returns the IKG value instead of the maximizer.
    """
    # allow for 3-dim context_sets
    assert context_set.dim() in [2, 3]
    num_arms = len(model.models)
    num_contexts = context_set.shape[-2]

    # means is num_arms x num_contexts
    if context_set.dim() == 2:
        means = model.posterior(context_set).mean.t()
    else:
        means = torch.zeros(num_arms, num_contexts).to(context_set)
        for arm_idx, arm_model in enumerate(model.models):
            arm_means = arm_model.posterior(context_set[arm_idx]).mean
            means[arm_idx] = arm_means.view(num_contexts)

    # this is (num_arms x num_contexts or num_candidates) x num_contexts
    full_sigma_tilde = _get_modellist_s_tilde(model, context_set, candidates)
    # reshaping to get candidates x num_contexts
    full_sigma_tilde = full_sigma_tilde.view(-1, num_contexts)

    ikg_vals = torch.zeros(candidates.shape[0]).to(context_set)
    for idx, all_s_tilde in enumerate(full_sigma_tilde):
        ikg_val = torch.tensor(0).to(context_set)
        arm_idx = int(candidates[idx][..., 0])
        # expanded s_tilde will be 0 except for the current candidate arm
        expanded_s_tilde = torch.zeros(num_arms, num_contexts).to(context_set)
        expanded_s_tilde[arm_idx] = all_s_tilde
        for c_idx in range(num_contexts):
            ikg_val += _pearce_alg_1(means[:, c_idx], expanded_s_tilde[:, c_idx])
        ikg_vals[idx] = ikg_val
    return ikg_vals


def finite_ikg_maximizer_modellist(
    model: ModelListGP,
    context_set: Tensor,
    weights: Optional[Tensor] = None,
    randomize_ties: bool = True,
    rho: Optional[Callable] = None,
) -> Tuple[int, int]:
    r"""
    Modified to work with MLGP

    Args:
        model: A ModelListGP with a model corresponding to each arm.
        context_set: A `num_contexts x d_c`-dim tensor of context set.
        weights: An optional `num_contexts`-dim tensor to be used when summing
            up the KG values for each context. This is ignored if `rho` is specified.
        randomize_ties: If True and there are multiple maximizers,
            the result will be randomly selected.
        rho: This is an experimental feature. Replaces the integration over
            KG(c) with rho(KG(c)). rho operates on the -2 dimension, so
            kg_vals need to be unsqueezed before calling rho.
            This works pretty bad - not recommended.

    Returns:
        A tuple of arm and context indices corresponding to the maximizer.
    """
    assert context_set.dim() == 2
    num_arms = len(model.models)
    num_contexts = context_set.shape[0]

    # means is num_arms x num_contexts
    means = model.posterior(context_set).mean.t()

    # this is num_arms x num_contexts x num_contexts
    full_sigma_tilde = _get_modellist_s_tilde(model, context_set)
    # reshaping to get candidates x num_contexts
    full_sigma_tilde = full_sigma_tilde.view(-1, num_contexts)

    max_ikg_val = torch.tensor(float("-inf")).to(context_set)
    max_idx = list()
    # loop over each candidate and calculate IKG value
    #   original all_s_tilde is num_arms x num_contexts.
    #   This one is num_contexts. All other arms actually have s_tilde=0
    #   There's a loop over contexts in there. That takes s_tilde for each
    #   arm for a given context. We will pass all zeros and a single non-zero there.
    for idx, all_s_tilde in enumerate(full_sigma_tilde):
        kg_vals = torch.zeros(num_contexts).to(context_set)
        # expanded s_tilde will be 0 except for the current candidate arm
        expanded_s_tilde = torch.zeros(num_arms, num_contexts).to(context_set)
        expanded_s_tilde[idx // num_contexts] = all_s_tilde
        for c_idx in range(num_contexts):
            kg_vals[c_idx] = _pearce_alg_1(means[:, c_idx], expanded_s_tilde[:, c_idx])
        if rho is None:
            if weights is not None:
                kg_vals = kg_vals * weights
            ikg_val = kg_vals.sum()
        else:
            ikg_val = rho(kg_vals.unsqueeze(-1)).squeeze(-1)
        if ikg_val > max_ikg_val:
            max_ikg_val = ikg_val
            max_idx = [idx]
        elif ikg_val == max_ikg_val:
            max_idx.append(idx)
    if randomize_ties:
        rand_idx = int(torch.randint(0, len(max_idx), (1,)))
        max_idx = max_idx[rand_idx]
    else:
        max_idx = max_idx[0]
    return max_idx // num_contexts, max_idx % num_contexts
