from __future__ import annotations
import torch
from botorch.sampling import MCSampler
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from torch import Tensor

from contextual_rs.rs_base_model import RSBaseModel


class IndependentGP(RSBaseModel):
    r"""
    Implement a GP model defined over a discrete set of inputs, where
    each alternative has an independent normal distribution.
    This model is not Bayesian. It simply stores the mean and standard deviation
    calculated from the input data, and uses these to produce the output.
    Intended for simple comparisons in the classical R&S setting.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
    ) -> None:
        r"""
        A GP model with independent outputs, defined over a finite collection
        of design points, e.g., a finite set of arms in R&S / bandits.

        Args:
            train_X: A `n`-dim tensor of training inputs.
                The inputs are expected to be integer points in
                0, ..., n_alternatives - 1, with each alternative appearing
                in each batch of the training data at least twice.
            train_Y: A `n`-dim tensor of training targets.
        """
        super(IndependentGP, self).__init__(train_X, train_Y)
        self.vars = self.stds.pow(2)

    def posterior(self, X: Tensor) -> MultivariateNormal:
        r"""
        Returns a MultivariateNormal object for sampling from the posterior.

        Args:
            X: A `batch_size x n`-dim tensor of design locations. Integer inputs
                from range 0, ..., n_alternatives - 1.

        Returns:
            A MultivariateNormal object with mean and a diagonal covariance matrix.
        """
        X_l = X.long()
        if torch.any(X != X_l):
            raise ValueError(
                "Inputs must be integers from range 0, ..., n_alternatives - 1!"
            )
        mean = self.means[X_l]
        covar = DiagLazyTensor(self.vars[X_l])
        return MultivariateNormal(mean, covar)

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
    ) -> IndependentGP:
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at `X`
        (2) sample from this posterior (using `sampler`) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model). # TODO: decide on explicit input dimension
            sampler: The sampler used for sampling from the posterior at `X`.

        Returns:
            The constructed fantasy model.
        """
        post_X = self.posterior(X)
        Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
        # TODO: is that shape correct? - should be?
        # TODO: implement condition on observations, or simply make this self-contained.
        #   This requires batch support!
        raise NotImplementedError
        # return self.condition_on_observations(X=X, Y=Y_fantasized, **kwargs)

    # TODO: Thinking of a method corresponding to s_tilde here.
    #   So, s_tilde is technically equivalent to reduction in predictive uncertainty
    #   from the observation, i.e., it is sigma^2_{n+1} - sigma^2_n.
    #   To calculate this difference, we would need to know sigma^2_{n+1}
    #   If the sampling error is known, this can be calculated.
    #   The R code by [2]_ has some code where they do this.
    #   The original KG paper with independent observations does not have any
    #   examples where observation noise is not known. This suggests that it may
    #   not work without knowing the observation noise.
