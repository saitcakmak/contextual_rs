import warnings
from typing import Optional, Callable

import botorch
import torch
from botorch.models import ModelListGP
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from torch import Tensor


def _min_mahalanobis(delta: Tensor, Sigma: Tensor) -> Tensor:
    r"""
    This is used for the approximation. Calculates the minimum Mahalanobis distance
    given by considering each element of delta individually. If delta is [0.5, 0.3],
    this will calculate the Mahalanobis distance using [0.5, 0] and [0, 0.3]
    independently, and return the minimum as the result for that delta.

    In fact, since we calculate the distances independently, the only place where
    the fact that Sigma is full-rank plays a part is the matrix inverse. Once we have
    the inverse, we can just use the diagonal and use simple element-wise vector
    multiplication.

    Args:
        delta: A `batch_shape x K'`-dim tensor representing the differences in means.
        Sigma: A `batch_shape x K' x K'`-dim tensor representing the covariance matrix
            corresponding to the difference.

    Returns:
        A `batch_shape`-dim tensor with the minimum Mahalanobis distance as
        explained above. To be precise, the Mahalanobis distance calculated here is
        the squared Mahalanobis distance as it avoids the need for sqrt and pow.
    """
    S_inv = Sigma.inverse()
    S_inv_diag = torch.diagonal(S_inv, dim1=-2, dim2=-1)
    distances = delta * S_inv_diag * delta
    min_distances, _ = distances.min(dim=-1)
    return min_distances


def estimate_lookahead_generalized_pcs(
    candidate: Tensor,
    model: Model,
    model_sampler: Optional[MCSampler],
    arm_set: Tensor,
    context_set: Tensor,
    # If not using SAA, number of contexts may need to be passed instead
    num_samples: int,
    base_samples: Optional[Tensor],
    # base_samples could be replaced with an MCSampler
    func_I: Callable[[Tensor], Tensor],
    rho: Callable[[Tensor], Tensor],
    use_approximation: bool = False,
) -> Tensor:
    r"""
    Estimates the lookahead generalized PCS following the description in Algorithm 1.

    Args:
        candidate: The `n x q x (d_x + d_c)`-dim tensor representing the candidate
            point, used for generating the fantasy models.
        model: The current GP model, used for generating the fantasy models.
        model_sampler: An MCSampler object, used for generating the fantasy models.
            If None, the certainty equivalent approximation is used.
        arm_set: An `n_x x d_x`-dim tensor of set of arms under consideration.
        context_set: An `n_c x d_c`-dim tensor of the set of contexts to use. Samples
            will be generated using the `(n_x * n_c) x (d_x + d_c)`-dim tensor resulting
            from concatenation of these two sets.
        num_samples: Number of posterior samples to draw, at each arm-context pair and
            each fantasy model.
        base_samples: A `num_samples x num_fantasies x n x (num_arms * num_contexts) x 1`
            tensor of base samples. If None, it will be randomized internally for each
            draw. The `num_fantasies` and `n` dimensions can be set to `1` to use the
            same set of base samples across those dimensions, via broadcasting.
        func_I: The function used to covert samples of [y(...)] into samples of PCS(c).
            This is an element-wise operation and should not modify the sample shape.
        rho: The functional converting the estimates of PCS(c) into and estimate of
            the generalized PCS. This is an operation over the context dimension and
            should eliminate said dimension, i.e., with input of shape
            `batch_shape x num_contexts x 1`, this should return a `batch_shape x 1`-dim
            tensor output.
        use_approximation: if True, instead of drawing samples from the posterior,
            we use an approximation based on Mahalanobis distance.

    Returns:
        The estimate of the lookahead generalized PCS. Tensor of size `n`.
    """
    # input data verification
    assert arm_set.dim() == 2 and context_set.dim() in [2, 3]
    assert arm_set.shape[-1] + context_set.shape[-1] == candidate.shape[-1]
    assert candidate.dim() == 3

    if isinstance(model, ModelListGP):
        raise NotImplementedError("Use ModelListGP specific version instead!")

    # define for future reference
    full_dim = candidate.shape[-1]
    num_arms = arm_set.shape[0]
    num_contexts = context_set.shape[-2]

    if context_set.dim() == 2:
        context_set = context_set.repeat(num_arms, 1, 1)

    # generate the fantasy model
    if model_sampler is not None:
        fantasy_model = model.fantasize(
            candidate,
            model_sampler,
        )
    else:
        # Using the certainty equivalent approximation here
        # Unsqueeze ensures that we have a num_fantasies dimension of size 1
        Y = model.posterior(candidate).mean.unsqueeze(0)
        fantasy_model = model.condition_on_observations(candidate, Y)

    # generate the tensor of arm-context pairs
    arm_context_pairs = torch.cat(
        [
            arm_set.unsqueeze(-2).expand(-1, num_contexts, -1),
            context_set,
        ],
        dim=-1,
    ).reshape(num_arms * num_contexts, full_dim)

    if use_approximation:
        # this posterior is num_context x fm.batch_shape over num_arms alternatives
        # fm.batch_shape = num_fantasies x num_candidates
        posterior = fantasy_model.posterior(
            arm_context_pairs.reshape(num_arms, num_contexts, -1)
            .transpose(0, 1)
            .view(num_contexts, 1, 1, num_arms, -1)
            .repeat(1, *fantasy_model.batch_shape, 1, 1)
        )
        # num_contexts x fm.batch_shape x num_arms
        means = posterior.mean.squeeze(-1)
        # num_contexts x fm.batch_shape x num_arms x num_arms
        covars = posterior.mvn.covariance_matrix
        means, indices = means.sort(dim=-1, descending=True)
        # calculate deltas for future use
        deltas = means[..., :1].expand(*means.shape[:-1], num_arms - 1) - means[..., 1:]
        # we want to apply the same sort to covars over both arm dimensions
        # this does the rows, so first arm dim
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
        S_e_low_left = S_e_upper[..., num_arms - 1 :].transpose(-1, -2)
        S_e_lower = torch.cat([S_e_low_left, covars[..., 1:, 1:]], dim=-1)
        # A_e S_e product, equivalent to upper - lower
        A_S = S_e_upper - S_e_lower
        # The second matrix product, first half - second half (over last dim)
        S_d = A_S[..., : num_arms - 1] - A_S[..., num_arms - 1 :]
        # We can use this with deltas to get the distance to closes axis
        # min_dist is now simply a num_context x fm.batch_shape tensor,
        # corresponding to pcs_c_est below.
        min_dist = _min_mahalanobis(deltas, S_d)
        # matching the shape
        pcs_c_est = min_dist.permute(1, 2, 0).unsqueeze(-1)
    else:
        # generate the posterior and draw posterior samples
        with botorch.settings.propagate_grads(True):
            posterior = fantasy_model.posterior(arm_context_pairs)
        means = posterior.mean
        # y_samples is `num_samples x num_fantasies x n x (num_arms * num_contexts) x 1`
        y_samples = posterior.rsample(
            sample_shape=torch.Size([num_samples]), base_samples=base_samples
        )

        # separate the arm and context dimensions
        y_samples = y_samples.reshape(*y_samples.shape[:-2], num_arms, num_contexts, 1)
        means = means.reshape(*means.shape[:-2], num_arms, num_contexts, 1)

        # order means across arms and apply the same ordering to y_samples
        means, indices = means.sort(dim=-3, descending=True)
        y_samples = y_samples.gather(
            dim=-3, index=indices.expand(num_samples, *indices.shape)
        )

        # calculate deltas = y_{(1)} - \max_{j!=1} y_{(j)}, actual idx is 0 here
        max_rest, _ = y_samples[..., 1:, :, :].max(dim=-3)
        deltas = y_samples[..., 0, :, :] - max_rest
        # use deltas with func_I to get samples of PCS(c)
        I_vals = func_I(deltas)
        # averaging over samples to get the estimate of PCS(c)
        pcs_c_est = I_vals.mean(dim=0)
    # feeding through rho to get estimates \hat{\rho}[PCS(c); GP_f^i]
    rho_vals = rho(pcs_c_est)
    # average over fantasies to get the final estimate \hat{\rho}[PCS(c)]
    pcs_est = rho_vals.mean(dim=0)
    # return with the last dimension squeezed, i.e., output is an `n`-dim tensor.
    return pcs_est.squeeze(-1)


def estimate_lookahead_generalized_pcs_modellist(
    candidate: Tensor,
    model: ModelListGP,
    model_sampler: Optional[MCSampler],
    context_set: Tensor,
    num_samples: int,
    base_samples: Optional[Tensor],
    func_I: Callable[[Tensor], Tensor],
    rho: Callable[[Tensor], Tensor],
    use_approximation: bool = False,
) -> Tensor:
    r"""
    Modified to work with ModelListGP.
    """
    # input data verification
    assert context_set.dim() == 2
    assert context_set.shape[-1] + 1 == candidate.shape[-1]
    assert candidate.dim() == 3 and candidate.shape[-2] == 1
    assert isinstance(model, ModelListGP)
    if use_approximation:
        raise NotImplementedError
    num_arms = len(model.models)
    num_contexts = context_set.shape[0]
    if base_samples is not None:
        if model_sampler:
            num_fantasies = model_sampler.sample_shape[0]
            if base_samples.shape[1] != num_fantasies:
                warnings.warn(
                    "It is highly recommended to use a different set of base_samples"
                    "for each fantasy model. Repeating the base samples across "
                    "fantasies is very inefficient.",
                    RuntimeWarning,
                )
        else:
            num_fantasies = 1
        base_samples = base_samples.expand(
            num_samples,
            num_fantasies,
            candidate.shape[0],
            num_contexts,
            num_arms,
        )
    # We will handle the fantasies by first sorting the candidates over arms,
    #   calculating the pcs values, then sorting them back.
    #   To calculate the PCS values, we will essentially loop over arms, only
    #   processing the corresponding candidates at any given time.
    sort_idcs = candidate[..., 0].view(-1).argsort()
    sorted_candidates = candidate[sort_idcs]
    pcs_est = torch.zeros(candidate.shape[0]).to(candidate)
    for arm_idx, arm_model in enumerate(model.models):
        mask = sorted_candidates[..., 0].view(-1) == arm_idx
        if mask.sum() == 0:
            continue
        arm_candidates = sorted_candidates[mask][..., 1:]
        remaining_arm_idx = list(range(num_arms))
        remaining_arm_idx.pop(arm_idx)
        # generate the fantasy model
        if model_sampler is not None:
            fantasy_model = arm_model.fantasize(arm_candidates, model_sampler)
        else:
            # certainty equivalent approximation
            Y = arm_model.posterior(arm_candidates).mean.unsqueeze(0)
            fantasy_model = arm_model.condition_on_observations(arm_candidates, Y)

        # generate the posterior and draw posterior samples
        with botorch.settings.propagate_grads(True):
            fm_posterior = fantasy_model.posterior(context_set)
            rem_posterior = model.posterior(
                X=context_set, output_indices=remaining_arm_idx
            )
        fm_means = fm_posterior.mean
        rem_means = rem_posterior.mean.expand(*fm_means.shape[:-2], -1, -1)
        # means is num_fantasies x n x num_contexts x num_arms
        means = torch.cat(
            [rem_means[..., :arm_idx], fm_means, rem_means[..., arm_idx:]], dim=-1
        )
        # pick the base samples corresponding to this arm
        if base_samples is not None:
            arm_base_samples = base_samples[:, :, mask]
            fm_base_samples = arm_base_samples[..., arm_idx : arm_idx + 1]
            rem_base_samples = arm_base_samples[..., remaining_arm_idx]
        else:
            fm_base_samples = None
            rem_base_samples = None
        fm_samples = fm_posterior.rsample(
            sample_shape=torch.Size([num_samples]),
            base_samples=fm_base_samples,
        )
        rem_samples = rem_posterior.rsample(
            sample_shape=torch.Size(fm_samples.shape[:-2]),
            base_samples=rem_base_samples,
        )
        # y_samples is num_samples x num_fantasies x n x num_contexts x num_arms
        y_samples = torch.cat(
            [rem_samples[..., :arm_idx], fm_samples, rem_samples[..., arm_idx:]], dim=-1
        )
        # order means across arms and apply the same ordering to y_samples
        means, indices = means.sort(dim=-1, descending=True)
        y_samples = y_samples.gather(dim=-1, index=indices.expand_as(y_samples))
        # calculate deltas
        max_rest, _ = y_samples[..., 1:].max(dim=-1)
        deltas = y_samples[..., 0] - max_rest
        # get samples of PCS(c)
        I_vals = func_I(deltas)
        # average samples to get the estimate of PCS(c)
        pcs_c_est = I_vals.mean(dim=0)
        # convert to estimates of rho PCS
        rho_vals = rho(pcs_c_est.unsqueeze(-1))
        # average over fantasies to get the final estimate
        pcs_est[mask] = rho_vals.mean(dim=0).squeeze(-1)
    # unsort to get the pcs_est with the original candidate ordering.
    pcs_est = pcs_est.gather(dim=-1, index=sort_idcs.argsort())
    return pcs_est.squeeze(-1)


def estimate_current_generalized_pcs(
    model: Model,
    arm_set: Tensor,
    context_set: Tensor,
    # If not using SAA, number of contexts may need to be passed instead
    num_samples: int,
    base_samples: Optional[Tensor],
    # base_samples could be replaced with an MCSampler
    func_I: Callable[[Tensor], Tensor],
    rho: Callable[[Tensor], Tensor],
    use_approximation: bool = False,
) -> Tensor:
    r"""
    Estimates the generalized PCS implied by the current model.
    This is similar to the description in Algorithm 1, except that it does
    not use any fantasy models.

    Args:
        model: The current GP model, used for generating the fantasy models.
        arm_set: An `n_x x d_x`-dim tensor of set of arms under consideration.
        context_set: An `n_c x d_c`-dim tensor of the set of contexts to use. Samples
            will be generated using the `(n_x * n_c) x (d_x + d_c)`-dim tensor resulting
            from concatenation of these two sets.
        num_samples: Number of posterior samples to draw, at each arm-context pair and
            each fantasy model.
        base_samples: A `num_samples x (num_arms * num_contexts) x 1`
            tensor of base samples. If None, it will be randomized internally for each
            draw.
        func_I: The function used to covert samples of [y(...)] into samples of PCS(c).
            This is an element-wise operation and should not modify the sample shape.
        rho: The functional converting the estimates of PCS(c) into and estimate of
            the generalized PCS. This is an operation over the context dimension and
            should eliminate said dimension, i.e., with input of shape
            `batch_shape x num_contexts x 1`, this should return a `batch_shape x 1`-dim
            tensor output.
        use_approximation: if True, instead of drawing samples from the posterior,
            we use an approximation based on Mahalanobis distance.

    Returns:
        The estimate of the generalized PCS. A scalar tensor.
    """
    # input data verification
    assert arm_set.dim() == 2 and context_set.dim() in [2, 3]
    if context_set.dim() == 3 and isinstance(model, ModelListGP):
        raise NotImplementedError

    # define for future reference
    full_dim = arm_set.shape[-1] + context_set.shape[-1]
    num_arms = arm_set.shape[0]
    num_contexts = context_set.shape[-2]

    if context_set.dim() == 2:
        context_set = context_set.repeat(num_arms, 1, 1)

    # generate the tensor of arm-context pairs
    arm_context_pairs = torch.cat(
        [
            arm_set.unsqueeze(-2).expand(-1, num_contexts, -1),
            context_set,
        ],
        dim=-1,
    ).reshape(num_arms * num_contexts, full_dim)

    if use_approximation:
        """
        How do we account for the correlation between contexts here?
        TLDR: We ignore it, as it is kinda irrelevant.
          The current posterior has the correlation in there. The covariance is
          over the arm-context pairs.
          Does it need to be?
          What are the pros and cons here? The current sampling based method does
          rely on sampling from the arm-context pairs. Is that a good idea or is it
          unnecessary? There's indeed correlation between the pairs. Until we calculate
          rho, we do not have any interaction between contexts. So, we could
          technically ignore that correlation while sampling.
          Following that line of thought, we can ignore it here as well.
          The algorithm description in notes also ignores the correlation between contexts.
        """
        if isinstance(model, ModelListGP):
            # TODO: this would require reshaping the posterior a bit to get right.
            #   Let's make it work with normal models first
            raise NotImplementedError
        # this posterior is num_context batches over num_arms alternatives
        posterior = model.posterior(
            arm_context_pairs.reshape(num_arms, num_contexts, -1).transpose(0, 1)
        )
        # num_contexts x num_arms
        means = posterior.mean.squeeze(-1)
        # num_contexts x num_arms x num_arms
        covars = posterior.mvn.covariance_matrix
        means, indices = means.sort(dim=-1, descending=True)
        # calculate deltas for future use
        deltas = means[..., :1].expand(*means.shape[:-1], num_arms - 1) - means[..., 1:]
        # we want to apply the same sort to covars over both arm dimensions
        # this does the rows, so first arm dim
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
        S_e_low_left = S_e_upper[..., num_arms - 1 :].transpose(-1, -2)
        S_e_lower = torch.cat([S_e_low_left, covars[..., 1:, 1:]], dim=-1)
        # A_e S_e product, equivalent to upper - lower
        A_S = S_e_upper - S_e_lower
        # The second matrix product, first half - second half (over last dim)
        S_d = A_S[..., : num_arms - 1] - A_S[..., num_arms - 1 :]
        # We can use this with deltas to get the distance to closes axis
        # min_dist is now simply a num_context tensor, corresponding to pcs_c_est below.
        min_dist = _min_mahalanobis(deltas, S_d)
        # matching the shape
        pcs_c_est = min_dist.unsqueeze(-1)
    else:
        # generate the posterior and draw posterior samples
        # means is num_arms x num_contexts
        if isinstance(model, ModelListGP):
            # TODO: only works thanks to not-implemented above
            posterior = model.posterior(context_set[0])
            means = posterior.mean.t()
        else:
            posterior = model.posterior(arm_context_pairs)
            means = posterior.mean
        # y_samples is `num_samples x (num_arms * num_contexts) x 1`
        y_samples = posterior.rsample(
            sample_shape=torch.Size([num_samples]), base_samples=base_samples
        )
        if isinstance(model, ModelListGP):
            # swapping num_contexts and num_arms dimensions and matching the shape
            y_samples = y_samples.transpose(-1, -2).unsqueeze(-1)
            means = means.unsqueeze(-1)
        else:
            # separate the arm and context dimensions
            y_samples = y_samples.reshape(
                *y_samples.shape[:-2], num_arms, num_contexts, 1
            )
            means = means.reshape(*means.shape[:-2], num_arms, num_contexts, 1)

        # order means across arms and apply the same ordering to y_samples
        means, indices = means.sort(dim=-3, descending=True)
        y_samples = y_samples.gather(
            dim=-3, index=indices.expand(num_samples, *indices.shape)
        )

        # calculate deltas = y_{(1)} - \max_{j!=1} y_{(j)}, actual idx is 0 here
        max_rest, _ = y_samples[..., 1:, :, :].max(dim=-3)
        deltas = y_samples[..., 0, :, :] - max_rest
        # use deltas with func_I to get samples of PCS(c)
        I_vals = func_I(deltas)
        # averaging over samples to get the estimate of PCS(c)
        pcs_c_est = I_vals.mean(dim=0)
    # feeding through rho to get estimates \hat{\rho}[PCS(c); GP_f^i]
    rho_vals = rho(pcs_c_est)
    # return with the last dimension squeezed, i.e., output is a scalar tensor.
    return rho_vals.squeeze(-1)
