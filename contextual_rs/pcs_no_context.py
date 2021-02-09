from typing import Optional, Callable

import botorch
import torch
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from torch import Tensor


def estimate_lookahead_pcs_no_context(
    candidate: Tensor,
    model: Model,
    model_sampler: MCSampler,
    arm_set: Tensor,
    # If not using SAA, number of contexts may need to be passed instead
    num_samples: int,
    base_samples: Optional[Tensor],
    # base_samples could be replaced with an MCSampler
    func_I: Callable[[Tensor], Tensor],
) -> Tensor:
    r"""
    Estimates the lookahead PCS following the description in Algorithm 1.
    This is simplified to work in the classical R&S setting without contexts.

    Args:
        candidate: The `n x q x d_x`-dim tensor representing the candidate
            point, used for generating the fantasy models.
        model: The current GP model, used for generating the fantasy models.
        model_sampler: An MCSampler object, used for generating the fantasy models.
        arm_set: An `n_x x d_x`-dim tensor of set of arms under consideration.
        num_samples: Number of posterior samples to draw, at each arm-context pair and
            each fantasy model.
        base_samples: A `num_samples x num_fantasies x n x num_arms x 1`
            tensor of base samples. If None, it will be randomized internally for each
            draw. The `num_fantasies` and `n` dimensions can be set to `1` to use the
            same set of base samples across those dimensions, via broadcasting.
        func_I: The function used to covert samples of [y(...)] into samples of PCS(c).
            This is an element-wise operation and should not modify the sample shape.

    Returns:
        The estimate of the lookahead PCS. Tensor of size `n`.
    """
    # input data verification
    assert arm_set.dim() == 2
    assert arm_set.shape[-1] == candidate.shape[-1]
    if candidate.dim() < 3:
        candidate = candidate.unsqueeze(0)
    assert candidate.dim() == 3

    # generate the fantasy model
    fantasy_model = model.fantasize(
        candidate,
        model_sampler,
    )

    # generate the posterior and draw posterior samples
    with botorch.settings.propagate_grads(True):
        posterior = fantasy_model.posterior(arm_set)
    # y_samples is `num_samples x num_fantasies x n x num_arms x 1`
    y_samples = posterior.rsample(
        sample_shape=torch.Size([num_samples]), base_samples=base_samples
    )
    means = posterior.mean

    # order means across arms and apply the same ordering to y_samples
    means, indices = means.sort(dim=-2, descending=True)
    y_samples = y_samples.gather(
        dim=-2, index=indices.expand(num_samples, *indices.shape)
    )

    # calculate deltas = y_{(1)} - \max_{j!=1} y_{(j)}, actual idx is 0 here
    max_rest, _ = y_samples[..., 1:, :].max(dim=-2)
    deltas = y_samples[..., 0, :] - max_rest
    # use deltas with func_I to get samples of PCS
    I_vals = func_I(deltas)
    # averaging over samples to get the estimate of PCS per fantasy
    pcs_fant_est = I_vals.mean(dim=0)
    # average over fantasies to get the final estimate
    pcs_est = pcs_fant_est.mean(dim=0)
    # return with the last dimension squeezed, i.e., output is an `n`-dim tensor.
    return pcs_est.squeeze(-1)


def estimate_current_pcs_no_context(
    model: Model,
    arm_set: Tensor,
    # If not using SAA, number of contexts may need to be passed instead
    num_samples: int,
    base_samples: Optional[Tensor],
    # base_samples could be replaced with an MCSampler
    func_I: Callable[[Tensor], Tensor],
) -> Tensor:
    r"""
    Estimates the PCS implied by the current model.
    This is similar to the description in Algorithm 1, except that it does
    not use any fantasy models.
    This is simplified for the classical R&S setting without contexts.

    Args:
        model: The current GP model, used for generating the fantasy models.
        arm_set: An `n_x x d_x`-dim tensor of set of arms under consideration.
        num_samples: Number of posterior samples to draw, at each arm-context pair and
            each fantasy model.
        base_samples: A `num_samples x num_arms x 1`
            tensor of base samples. If None, it will be randomized internally for each
            draw.
        func_I: The function used to covert samples of [y(...)] into samples of PCS(c).
            This is an element-wise operation and should not modify the sample shape.

    Returns:
        The estimate of the PCS. A scalar tensor.
    """
    # input data verification
    assert arm_set.dim() == 2

    # generate the posterior and draw posterior samples
    posterior = model.posterior(arm_set)
    # y_samples is `num_samples x num_arms x 1`
    y_samples = posterior.rsample(
        sample_shape=torch.Size([num_samples]), base_samples=base_samples
    )
    means = posterior.mean

    # order means across arms and apply the same ordering to y_samples
    means, indices = means.sort(dim=-2, descending=True)
    y_samples = y_samples.gather(
        dim=-2, index=indices.expand(num_samples, *indices.shape)
    )

    # calculate deltas = y_{(1)} - \max_{j!=1} y_{(j)}, actual idx is 0 here
    max_rest, _ = y_samples[..., 1:, :].max(dim=-2)
    deltas = y_samples[..., 0, :] - max_rest
    # use deltas with func_I to get samples of PCS
    I_vals = func_I(deltas)
    # averaging over samples to get the estimate of PCS
    pcs_est = I_vals.mean(dim=0)
    # return with the last dimension squeezed, i.e., output is a scalar tensor.
    return pcs_est.squeeze(-1)
