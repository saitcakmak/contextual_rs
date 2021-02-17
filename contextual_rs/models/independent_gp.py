from __future__ import annotations
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from torch import Tensor

from contextual_rs.models.rs_base_model import RSBaseModel


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

    def add_samples(self, X: Tensor, Y: Tensor) -> None:
        r"""
        Updates the model by adding new samples. It is an efficient alternative
        to re-constructing the model with full data.

        Args:
            X: `n'`-dim tensor of inputs. Expected to be integers
                in 0, ..., n_alternatives - 1.
            Y: `n'`-dim tensor of observations.
        """
        assert X.shape == Y.shape, "X and Y must be of the same shape!"
        assert X.dim() == 1, "X must be a one-dimensional tensor!"
        X_l = X.long()
        if torch.any(X != X_l) or torch.any(X_l > self.num_alternatives):
            raise ValueError(
                "Inputs must be integers from range 0, ..., n_alternatives - 1!"
            )
        # pick the loop based on the size of the input
        if X_l.shape[0] < self.num_alternatives:
            # loop over the inputs, add them one by one
            for x, y in zip(X_l, Y):
                self.alternative_observations[x] = torch.cat(
                    [self.alternative_observations[x], y.view(-1)]
                )
                self.means[x] = self.alternative_observations[x].mean()
                self.stds[x] = self.alternative_observations[x].std()
                self.vars[x] = self.stds[x].pow(2)
        else:
            # loop over alternatives, add if there are observations
            for i in range(self.num_alternatives):
                observations = Y[X_l == i]
                if observations.numel() == 0:
                    continue
                self.alternative_observations[i] = torch.cat(
                    [self.alternative_observations[i], observations]
                )
                self.means[i] = self.alternative_observations[i].mean()
                self.stds[i] = self.alternative_observations[i].std()
                self.vars[i] = self.stds[i].pow(2)
