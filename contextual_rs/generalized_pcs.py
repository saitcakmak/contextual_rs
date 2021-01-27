from typing import Optional, Callable

import botorch
import torch
from botorch.models.gpytorch import MultiTaskGPyTorchModel
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from torch import Tensor


def estimate_generalized_pcs(
    candidate: Tensor,
    model: Model,
    model_sampler: MCSampler,
    arm_set: Tensor,
    context_set: Tensor,
    # If not using SAA, number of contexts may need to be passed instead
    num_samples: int,
    base_samples: Optional[Tensor],
    # base_samples could be replaced with an MCSampler
    func_I: Callable[[Tensor], Tensor],
    rho: Callable[[Tensor], Tensor],
) -> Tensor:
    r"""
    Estimates the generalized PCS following the description in Algorithm 1.

    Args:
        candidate: The `n x q x (d_x + d_c)`-dim tensor representing the candidate
            point, used for generating the fantasy models.
        model: The current GP model, used for generating the fantasy models.
        model_sampler: An MCSampler object, used for generating the fantasy models.
        arm_set: An `n_x x d_x`-dim tensor of set of arms under consideration.
        context_set: An `n_c x d_c`-dim tensor of the set of contexts to use. Samples
            will be generated using the `(n_x * n_c) x (d_x + d_c)`-dim tensor resulting
            from concatenation of these two sets.
        num_samples: Number of posterior samples to draw, at each arm-context pair and
            each fantasy model.
        base_samples: A `num_samples x num_fantasies x n x (num_arms * num_contexts) x 1`
            tensor of base samples. If None, it will be randomized internally for each
            draw. The `num_fantasies` and `n` dimensions can be set to `0` to use the
            same set of base samples across those dimensions, via broadcasting.
        func_I: The function used to covert samples of [y(...)] into samples of PCS(c).
            This is an element-wise operation and should not modify the sample shape.
        rho: The functional converting the estimates of PCS(c) into and estimate of
            the generalized PCS. This is an operation over the context dimension and
            should eliminate said dimension, i.e., with input of shape
            `batch_shape x num_contexts x 1`, this should return a `batch_shape x 1`-dim
            tensor output.

    Returns:
        The estimate of the generalized PCS. Tensor of size `n`.
    """
    # input data verification
    assert arm_set.dim() == 2 and context_set.dim() == 2
    assert arm_set.shape[-1] + context_set.shape[-1] == candidate.shape[-1]
    if candidate.dim() < 3:
        candidate = candidate.unsqueeze(0)
    assert candidate.dim() == 3

    # define for future reference
    full_dim = candidate.shape[-1]
    num_arms = arm_set.shape[0]
    num_contexts = context_set.shape[0]

    # generate the fantasy model
    fantasy_model = model.fantasize(
        candidate,
        model_sampler,
        observation_noise=not isinstance(model, MultiTaskGPyTorchModel),
    )

    # generate the tensor of arm-context pairs
    arm_context_pairs = torch.cat(
        [
            arm_set.unsqueeze(-2).expand(-1, context_set.shape[0], -1),
            context_set.expand(arm_set.shape[0], -1, -1),
        ],
        dim=-1,
    ).reshape(num_arms * num_contexts, full_dim)

    # generate the posterior and draw posterior samples
    with botorch.settings.propagate_grads(True):
        posterior = fantasy_model.posterior(arm_context_pairs)
    # y_samples is `num_samples x num_fantasies x n x (num_arms * num_contexts) x 1`
    y_samples = posterior.rsample(
        sample_shape=torch.Size([num_samples]), base_samples=base_samples
    )
    means = posterior.mean

    # separate the arm and context dimensions
    y_samples = y_samples.reshape(*y_samples.shape[:-2], num_arms, num_contexts, 1)
    means = means.reshape(*means.shape[:-2], num_arms, num_contexts, 1)

    # order means across arms and apply the same ordering to y_samples
    means, indices = means.sort(dim=-3, descending=True)
    y_samples = y_samples.gather(
        dim=-3, index=indices.expand(num_samples, *indices.shape)
    )

    # calculate deltas = y_{(1)} - \max_{j!=1} y_{(j)}, actual idx is 0 here
    max_rest, _ = y_samples[..., 0:, :, :].max(dim=-3)
    deltas = y_samples[..., 0, :, :] - max_rest
    # use deltas with func_I to get samples of PCS(c)
    I_vals = func_I(deltas)
    # averaging over samples to get the estimate of PCS(c)
    pcs_c_est = I_vals.mean(dim=0)
    # feeding through rho to get estimates \hat{\rho}[PCS(c); GP_f^i]
    rho_vals = rho(pcs_c_est)
    # average over fantasies to get the final estimate \hat{\rho}[PCS(c)]
    pcs_est = rho_vals.mean(dim=0)
    # return with the last dimension squeezed, i.e. output is an `n`-dim tensor.
    return pcs_est.squeeze(-1)
