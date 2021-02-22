from typing import Tuple

import torch

from contextual_rs.models.contextual_independent_model import ContextualIndependentModel


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
    next_sorted_arm = 0 if y_s_1 < y_s_2 else min_sorted_arm + 1
    # need to map the sorted arm to its original index
    next_arm = idcs[next_sorted_arm, next_context]
    return next_arm, next_context
