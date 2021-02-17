r"""
A place to store KG acquisition functions that are modified for the R&S setting.
"""
from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from torch.distributions.studentT import StudentT

from contextual_rs.models.lce_gp import LCEGP
from contextual_rs.models.unknown_correlation_model import UnknownCorrelationModel


class DiscreteKG(AcquisitionFunction):
    r"""
    Nested KG with categorical alternatives. Intended for use in R&S setting.
    """

    def __init__(
        self,
        model: LCEGP,
        num_fantasies: Optional[int] = 64,
        sampler: Optional[MCSampler] = None,
        current_value: Optional[Tensor] = None,
    ) -> None:
        if sampler is None:
            # base samples should be fixed for joint optimization over X, X_fantasies
            sampler = SobolQMCNormalSampler(
                num_samples=num_fantasies, resample=False, collapse_batch_dims=True
            )
        elif num_fantasies is not None:
            if sampler.sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        else:
            num_fantasies = sampler.sample_shape[0]
        super().__init__(model=model)
        self.sampler = sampler
        self.num_fantasies = num_fantasies
        self.current_value = current_value

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate DiscreteKG on the candidate set `X`.

        Args:
            X: A `b x q x 1` Tensor with `b` t-batches of `q` design points each.

        Returns:
            A Tensor of shape `b`. For t-batch b, the KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X, sampler=self.sampler, observation_noise=True
        )

        # get the PM and maximize over alternatives
        num_alternatives = self.model.category_counts[0]
        alternatives = torch.arange(num_alternatives).unsqueeze(-1).to(X)
        pm = fantasy_model.posterior(alternatives).mean
        values, _ = pm.max(dim=-2)
        values = values.squeeze(-1)

        if self.current_value is not None:
            values = values - self.current_value

        # return average over the fantasy samples
        return values.mean(dim=0)


class UnknownCorrelationKG(AcquisitionFunction):
    r"""
    This is the KG for use with UnknownCorrelationModel.
    Main difference between this and DiscreteKG is that this uses T
    distribution for fantasies.
    """

    def __init__(
        self,
        model: UnknownCorrelationModel,
        num_fantasies: Optional[int] = 64,
        current_value: Optional[Tensor] = None,
    ) -> None:
        super().__init__(model=model)
        self.num_fantasies = num_fantasies
        self.current_value = current_value
        df = model.predictive_df()
        sampler = StudentT(df)
        self.base_samples = sampler.rsample(torch.Size([num_fantasies]))

    def forward(self, X: Optional[Tensor]) -> Tensor:
        r"""Evaluate UnknownCorrelationKG on the candidate set `X`.

        Args:
            X: An optional scalar Tensor with the alternative to consider.
                If none, KG is computed for all alternatives.

        Returns:
            A Tensor of corresponding shape. KG value is computed by
            maximizing the inner problem for each fantasy and averaging
            over fantasies.
        """
        if X is not None:
            raise NotImplementedError
        # theta^{n+1} = theta^n + s_tilde * T
        # theta is a K-vector
        theta = self.model.theta
        # s_tilde is a K x K tensor with s_tilde[i] corresponding to s_tilde vector for
        # i-th alternative.
        s_tilde = self.model.get_s_tilde(None)
        # expand the base samples to num_fantasies x K x K
        K = theta.shape[0]
        T = self.base_samples.view(-1, 1, 1).expand(-1, K, K)
        # product with s_tilde, again num_fantasies x K x K
        s_tilde_T = s_tilde * T
        # expand theta
        theta = theta.expand(self.num_fantasies, K, -1)
        # add to get theta^{n+1}
        theta_p1 = theta + s_tilde_T
        # max over last dimension, inner maximization
        values, _ = theta_p1.max(dim=-1)
        if self.current_value is not None:
            values = values - self.current_value
        # return average over the fantasy samples
        return values.mean(dim=0)
