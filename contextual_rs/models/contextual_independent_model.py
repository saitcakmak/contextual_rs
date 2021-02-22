from __future__ import annotations

from typing import Optional

import torch
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from torch import Tensor
from torch.nn import Module


class ContextualIndependentModel(Module):
    r"""
    Implements a statistical model for the contextual R&S setting,
    where the model is defined over a finite number of arms and contexts
    and each arm-context pair has IID normally distributed observations.
    This is intended for comparison with the following two papers:
        [1]: Li et.al. 2020 "Context-dependent ranking and selection
            under a Bayesian framework"
        [2]: Gao et.al. 2019 "Selecting the Optimal System Design under Covariates"
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        context_map: Optional[Tensor] = None,
    ) -> None:
        r"""
        A model with independent outputs, defined over a finite collection
        of arms and a finite collection of contexts, represented by a 2D tensor
        with categorical (integer) inputs.

        Args:
            train_X: An `n x 2`-dim tensor of training inputs.
                Each 2D input represents the indices of an arm-context pair,
                with the first dimension taking values in 0, ..., n_arms - 1;
                and the second dimension taking values in 0, ..., n_contexts - 1.
                The training data must include at least two observations from
                each arm-context pair!
            train_Y: A `n`-dim tensor of training targets.
            context_map: In both of these papers, despite the focus on
                a finite number of contexts with independent rewards,
                the contexts are introduced as being vectors. If needed,
                we can keep track of the mapping from context index to
                the vector with the `context_map` attribute.
        """
        # verify input shapes
        assert train_X.shape[0] == train_Y.shape[0]
        assert train_Y.dim() == 1
        assert train_X.dim() == 2 and train_X.shape[-1] == 2
        assert torch.all(train_X == train_X.long())

        super().__init__()

        self.train_X = train_X
        self.train_Y = train_Y
        self.context_map = context_map
        ckwargs = {"dtype": train_X.dtype, "device": train_X.device}
        self.arms = train_X[:, 0].unique(sorted=True)
        self.contexts = train_X[:, 1].unique(sorted=True)
        self.num_arms = self.arms.shape[0]
        self.num_contexts = self.contexts.shape[0]

        self.observations = list()
        self.num_observations = torch.zeros(self.num_arms, self.num_contexts, **ckwargs)
        self.means = torch.zeros(self.num_arms, self.num_contexts, **ckwargs)
        self.stds = torch.zeros(self.num_arms, self.num_contexts, **ckwargs)
        # collect the observations and calculate sample statistics.
        for arm in range(self.num_arms):
            self.observations.append(list())
            for context in range(self.num_contexts):
                observations = train_Y[
                    (train_X == torch.tensor([arm, context], **ckwargs)).all(dim=-1)
                ]
                self.observations[arm].append(observations)
                self.num_observations[arm, context] = observations.shape[0]
                self.means[arm, context] = observations.mean()
                std = observations.std()
                if std.isnan():  # pragma: no cover
                    raise RuntimeError(
                        "Expected to see at least two observations from each alternative."
                        f"Got only 1 observation for arm {arm} context {context}."
                    )
                self.stds[arm, context] = std
        self.vars = self.stds.pow(2)

    def posterior(self, X: Tensor) -> GPyTorchPosterior:
        r"""
        Returns a MultivariateNormal object representing the posterior distribution.

        Args:
            X: A `batch_size x n' x 2`-dim tensor of arm-context pairs. Integer inputs
                from range 0, ..., n_arms - 1 and 0, ..., n_contexts - 1.

        Returns:
            A MultivariateNormal object with mean and a diagonal covariance matrix.
        """
        X_l = X.long()
        if torch.any(X != X_l):  # pragma: no cover
            raise ValueError(
                "Inputs `X` must be integers in 0, ..., n_arms - 1 "
                "and 0, ..., n_contexts - 1!"
            )
        mean = self.means[(X_l[..., 0], X_l[..., 1])]
        covar = DiagLazyTensor(self.vars[(X_l[..., 0], X_l[..., 1])])
        return GPyTorchPosterior(MultivariateNormal(mean, covar))

    def add_samples(self, X: Tensor, Y: Tensor) -> None:
        r"""
        Updates the model by adding new samples. It is a slightly more efficient
        alternative to re-constructing the model with full data.

        Args:
            X: `n' x 2`-dim tensor of inputs. Expected to be integers
                in 0, ..., n_arms - 1 and 0, ..., n_contexts - 1.
            Y: `n'`-dim tensor of observations.
        """
        assert X.shape[0] == Y.shape[0], "X and Y must have same number of inputs!"
        assert X.dim() == 2 and X.shape[-1] == 2, "X must be `n' x 2`-dim!"
        X_l = X.long()
        if torch.any(X != X_l):  # pragma: no cover
            raise ValueError(
                "Inputs `X` must be integers in 0, ..., n_arms - 1 "
                "and 0, ..., n_contexts - 1!"
            )
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_Y = torch.cat([self.train_Y, Y], dim=0)
        # pick the loop based on the size of the input
        if X_l.shape[0] < self.num_arms * self.num_contexts:
            # loop over the inputs, add them one by one
            for x, y in zip(X_l, Y):
                self.observations[x[0]][x[1]] = torch.cat(
                    [self.observations[x[0]][x[1]], y.view(-1)]
                )
                self.num_observations[x[0], x[1]] += 1
                self.means[x[0], x[1]] = self.observations[x[0]][x[1]].mean()
                self.stds[x[0], x[1]] = self.observations[x[0]][x[1]].std()
                self.vars[x[0], x[1]] = self.stds[x[0], x[1]].pow(2)
        else:
            # loop over alternatives, add if there are observations
            for arm in range(self.num_arms):
                for context in range(self.num_contexts):
                    observations = Y[
                        (X == torch.tensor([arm, context]).to(self.train_X)).all(dim=-1)
                    ]
                    if observations.numel() == 0:
                        continue
                    self.observations[arm][context] = torch.cat(
                        [self.observations[arm][context], observations]
                    )
                    self.num_observations[arm, context] += observations.shape[0]
                    self.means[arm, context] = self.observations[arm][context].mean()
                    self.stds[arm, context] = self.observations[arm][context].std()
                    self.vars[arm, context] = self.stds[arm, context].pow(2)
