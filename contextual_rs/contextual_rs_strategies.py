from typing import Tuple

import torch
from botorch.models import ModelListGP
from torch import Tensor

from contextual_rs.models.contextual_independent_model import ContextualIndependentModel
from contextual_rs.models.lce_gp import LCEGP


def li_sampling_strategy(
    model: ContextualIndependentModel,
) -> Tuple[int, int]:
    r"""
    The sampling strategy introduced in the paper
        [1]: Li et.al. 2020 "Context-dependent ranking and selection
            under a Bayesian framework"
    This is mainly based on eq 4 of the paper.
    Eq 4 is calculated for each arm-context pair, and the maximizer is reported.

    Returns:
        The next arm-context pair to sample.
    """
    means = model.means.clone()
    vars = model.vars.clone()
    num_observations = model.num_observations.clone()
    # we need to expand num_observations for each arm-context pair
    # add add 1 hypothetical observation to the corresponding pair
    expanded_observations = num_observations.repeat(
        model.num_arms, model.num_contexts, 1, 1
    )
    # TODO: how to do this more efficiently?
    for arm in range(model.num_arms):
        for context in range(model.num_contexts):
            expanded_observations[arm, context, arm, context] += 1

    # do the sorting and scaling to get the arguments for eq 4
    scaled_vars = vars / expanded_observations
    sorted_means, idcs = means.sort(dim=0, descending=True)
    sorted_vars = scaled_vars.gather(dim=-2, index=idcs.expand_as(scaled_vars))

    mean_diffs = sorted_means[0].expand(model.num_arms - 1, -1) - sorted_means[1:]
    squared_diff = mean_diffs.pow(2)
    added_vars = (
        sorted_vars[:, :, :1].expand(-1, -1, model.num_arms - 1, -1)
        + sorted_vars[:, :, 1:]
    )
    # this is the term in eq 4 before taking the min
    normalized_term = squared_diff / added_vars
    # take the min over arms, then over contexts to get VFA in eq 4
    tmp, _ = normalized_term.min(dim=-2)
    vfa, _ = tmp.min(dim=-1)
    # the sampling decision is the index maximizing VFA
    maximizer = int(vfa.argmax())
    # maximizer is calculated over flattened tensor, get the arm / context idcs
    max_arm = maximizer // model.num_contexts
    max_context = maximizer % model.num_contexts
    return max_arm, max_context


def gao_sampling_strategy(model: ContextualIndependentModel) -> Tuple[int, int]:
    r"""
    The sampling strategy introduced in the paper
        [2]: Gao et.al. 2019 "Selecting the Optimal System Design under Covariates"
    This follows the algorithm description in Section 3.

    The original algorithm is for minimization! This implementation is for maximization,
    achieved by flipping the index of the best / sort order.

    Returns:
        The next arm-context pair to sample.
    """
    # calculate the hat Z values
    p_st = model.num_observations / model.num_observations.sum()
    sorted_means, idcs = model.means.sort(dim=0, descending=True)
    sorted_vars = model.vars.gather(dim=0, index=idcs)
    sorted_p = p_st.gather(dim=0, index=idcs)
    scaled_vars = sorted_vars / sorted_p
    added_vars = scaled_vars[:1].expand(model.num_arms - 1, -1) + scaled_vars[1:]
    mean_diffs = sorted_means[:1].expand(model.num_arms - 1, -1) - sorted_means[1:]
    squared_diffs = mean_diffs.pow(2)
    hat_Z = squared_diffs / added_vars
    # part 2 a) of the algorithm
    minimizer = torch.argmin(hat_Z)
    min_sorted_arm = minimizer // model.num_contexts
    next_context = minimizer % model.num_contexts
    # step 2 b)
    y_factors = sorted_p[:, next_context].pow(2) / sorted_vars[:, next_context]
    y_s_1 = y_factors[0]
    y_s_2 = y_factors[1:].sum()
    # if the condition (16) holds, pick sorted arm 0, else min_sorted_arm + 1
    # + 1 accounts for the 0th index which disappears when calculating hat Z
    next_sorted_arm = 0 if y_s_1 < y_s_2 else min_sorted_arm + 1
    # need to map the sorted arm to its original index
    next_arm = idcs[next_sorted_arm, next_context]
    return next_arm, next_context


def gao_modellist(
    model: ModelListGP,
    context_set: Tensor,
    randomize_ties: bool = True,
    infer_p: bool = False,
) -> Tuple[int, Tensor]:
    r"""
    Gao sampling strategy adapted to work with ModelListGP by replacing
    vars / p with posterior variance.

    Note: This should not be used in the continuous context setting!

    Args:
        model: A ModelListGP where each model corresponds to a different arm.
        context_set: The set of contexts to consider. `num_contexts x d_c`.
        randomize_ties: If there are multiple maximizers of Z, pick the
            returned one randomly.
        infer_p: This is an experimental feature. If True, the ratio of samples
            allocated to a given alternative `p` is inferred from the ratio of
            the prior and posterior variances. Doesn't work well!

    Returns:
        The maximizer arm and the corresponding context.
    """
    # calculate the hat Z values.
    posterior = model.posterior(context_set)
    # These are num_contexts x num_arms
    means = posterior.mean
    variances = posterior.variance
    sorted_means, idcs = means.sort(dim=-1, descending=True)
    sorted_vars = variances.gather(dim=-1, index=idcs)
    num_arms = len(model.models)
    added_vars = sorted_vars[:, :1].expand(-1, num_arms - 1) + sorted_vars[:, 1:]
    mean_diffs = sorted_means[:, :1].expand(-1, num_arms - 1) - sorted_means[:, 1:]
    squared_diffs = mean_diffs.pow(2)
    hat_Z = squared_diffs / added_vars
    # part 2 a)
    flat_Z = hat_Z.view(-1)
    min_Z, minimizer = torch.min(flat_Z, dim=0)
    if randomize_ties:
        min_check = flat_Z == min_Z
        min_count = min_check.sum()
        if min_count > 1:
            min_idcs = torch.arange(
                0, flat_Z.shape[0], device=hat_Z.device
            )[min_check]
            minimizer = min_idcs[
                torch.randint(min_count, (1,), device=hat_Z.device)
            ].squeeze()
    min_context = minimizer // (num_arms - 1)
    next_context = context_set[min_context]
    min_sorted_arm = minimizer % (num_arms - 1)
    # part 2 b) - train_counts stands in for the p_values
    if infer_p:
        # we just need the ratio for the next context
        prior_variances = [
            m.forward(context_set[min_context].view(1, -1)).variance
            for m in model.models
        ]
        ratio = torch.cat(prior_variances) / variances[min_context]
        y_vals = ratio / variances[min_context]
    else:
        train_inputs = model.train_inputs
        train_counts = torch.zeros(num_arms).to(context_set)
        for i, arm_inputs in enumerate(train_inputs):
            arm_inputs = arm_inputs[0]
            train_counts[i] = (arm_inputs == next_context).all(dim=-1).sum()
        y_vals = train_counts / variances[min_context]
    sorted_y_vals = y_vals[idcs[min_context]]
    y_s_1 = sorted_y_vals[0]
    y_s_2 = sorted_y_vals[1:].sum()
    # check for condition (16) and pick arm accordingly
    # + 1 accounts for the 0th index which disappears when calculating hat Z
    next_sorted_arm = 0 if y_s_1 < y_s_2 else min_sorted_arm + 1
    # map the sorted_arm to original index
    next_arm = idcs[min_context, next_sorted_arm]
    return next_arm, next_context


def gao_lcegp(
    model: LCEGP,
    arm_set: Tensor,
    context_set: Tensor,
    randomize_ties: bool = True,
    infer_p: bool = False,
) -> Tuple[int, Tensor]:
    r"""
    Gao sampling strategy adapted to work with LCEGP by replacing
    vars / p with posterior variance.

    Note: This should not be used in the continuous context setting!

    Args:
        model: An LCEGP modeling the arm-context pairs.
        arm_set: The set of arms to consider, `num_arms x 1`.
        context_set: The set of contexts to consider. `num_contexts x d_c`.
        randomize_ties: If there are multiple maximizers of Z, pick the
            returned one randomly.
        infer_p: This is an experimental feature. If True, the ratio of samples
            allocated to a given alternative `p` is inferred from the ratio of
            the prior and posterior variances.

    Returns:
        The maximizer arm and the corresponding context.
    """
    # calculate the hat Z values.
    num_arms = arm_set.shape[0]
    num_contexts = context_set.shape[0]
    arm_context_pairs = torch.cat(
        [
            arm_set.repeat(num_contexts, 1, 1),
            context_set.view(num_contexts, 1, -1).repeat(1, num_arms, 1)
        ], dim=-1
    )
    posterior = model.posterior(arm_context_pairs)
    # These are num_contexts x num_arms
    means = posterior.mean.squeeze(-1)
    variances = posterior.variance.squeeze(-1)
    sorted_means, idcs = means.sort(dim=-1, descending=True)
    sorted_vars = variances.gather(dim=-1, index=idcs)
    # TODO: could consider using the variance of difference here instead
    added_vars = sorted_vars[:, :1].expand(-1, num_arms - 1) + sorted_vars[:, 1:]
    mean_diffs = sorted_means[:, :1].expand(-1, num_arms - 1) - sorted_means[:, 1:]
    squared_diffs = mean_diffs.pow(2)
    hat_Z = squared_diffs / added_vars
    # part 2 a)
    flat_Z = hat_Z.view(-1)
    min_Z, minimizer = torch.min(flat_Z, dim=0)
    if randomize_ties:
        min_check = flat_Z == min_Z
        min_count = min_check.sum()
        min_idcs = torch.arange(
            0, flat_Z.shape[0], device=hat_Z.device
        )[min_check]
        minimizer = min_idcs[
            torch.randint(min_count, (1,), device=hat_Z.device)
        ].squeeze()
    min_context = minimizer // (num_arms - 1)
    next_context = context_set[min_context]
    min_sorted_arm = minimizer % (num_arms - 1)
    # part 2 b) - train_counts stands in for the p_values
    if infer_p:
        # we just need the ratio for the next context
        prior_variances = model.forward(arm_context_pairs[min_context]).variance.squeeze(-1)
        ratio = prior_variances / variances[min_context]
        y_vals = ratio / variances[min_context]
    else:
        train_inputs = model.train_inputs[0]
        train_counts = torch.zeros(num_arms).to(context_set)
        for arm in arm_set:
            current_pair = torch.cat([arm, next_context], dim=-1)
            train_counts[int(arm)] = (train_inputs == current_pair).all(dim=-1).sum()
        y_vals = train_counts / variances[min_context]
    sorted_y_vals = y_vals[idcs[min_context]]
    y_s_1 = sorted_y_vals[0]
    y_s_2 = sorted_y_vals[1:].sum()
    # check for condition (16) and pick arm accordingly
    # + 1 accounts for the 0th index which disappears when calculating hat Z
    next_sorted_arm = 0 if y_s_1 < y_s_2 else min_sorted_arm + 1
    # map the sorted_arm to original index
    next_arm = idcs[min_context, next_sorted_arm]
    return next_arm, next_context
