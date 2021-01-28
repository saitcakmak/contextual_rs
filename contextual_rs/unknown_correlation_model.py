import math
from typing import Callable, Optional, Union

from scipy.stats import multivariate_t, t
import torch
from torch import Tensor

from contextual_rs.rs_base_model import RSBaseModel


class UnknownCorrelationModel(RSBaseModel):
    r"""
    References:
        [1]: Qu et.al. 2015 "Sequential Selection with Unknown Correlation Structures"
        [2]: Zhang and Song 2017 "Moment-Matching-Based Conjugacy Approximation for
            Bayesian Ranking and Selection"

    This implements the moment-matching approximation presented in [2]_. In
    numerical experiment, this model is shown to outperform the KL minimization
    approach of [1]_. In addition, this does not require a line search to find
    the parameters, which are available in closed form.

    The KL minimization method of [1]_ is also implemented, with the simplification
    that \Delta b = 1/K. Choose between the two approaches by passing
    `update_method` argument. A notable difference between the two approaches
    is that the moment-matching method does a terrible job predicting the
    off-diagonal elements of the covariance matrix. The KL approach is not
    great either, but it at least gets to the same order of magnitude.

    Added the mixed moment-matching and KL method from [2]_ based on
    proposition 3.5 as well.

    This model assumes a normal-Wishart prior

    .. math::
        \mathbf{\mu} \mid \mathbf{R} \sim \mathcal{N}_K
        (\mathbf{\theta}^0, q^0 \mathbf{R}),
        \mathbf{R} \sim \mathcal{W}_K (b^0, \mathbf{B}^0)

    Quoting from [1]_, "If the prior parameters are constructed from
    historical data, the diagonal entries of B^0 will be the sums of squared
    deviations of the first-stage observations from their means. The scalar
    b^0 is analogous to the size of the first-stage sample, so that
    B^0/(b0 − K +1) is precisely the empirical covariance matrix constructed
    from the first-stage data. The parameter q^0 is also analogous to a
    sample size; if first-stage sampling is used, R^{−1}/q^0 will be the
    covariance matrix of the sample mean \mu."
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        update_method: str = "moment-matching",
    ) -> None:
        r"""
        A statistical model based on approximate Bayesian inference and
        assuming unknown correlations between alternatives.
        This model is defined over a finite set of alternatives, which
        are given as integers from 0, ..., n_alternatives - 1.
        The posterior distribution defined by this model is a multivariate-t
        distribution, which is available at Scipy. The posterior samples do not
        satisfy sample path differentiability.

        The priors used here are as follows:
        \theta^0 = 0
        b^0 = num_alternatives + 1
        q^0 = 1
        B^0 = I

        Available full vector observations are processed first, then the remaining
        observations are added one by one.

        # TODO: This does a terrible job at estimating off diagonal covariance.

        Args:
            train_X: A `n`-dim tensor of training inputs.
                The inputs are expected to be integer points in
                0, ..., n_alternatives - 1, with each alternative appearing
                in each batch of the training data at least twice.
            train_Y: A `n`-dim tensor of training targets.
            update_method: The updates used to get the approximate posterior.
                Currently accepts only "moment-matching" and "KL".
        """
        super(UnknownCorrelationModel, self).__init__(train_X, train_Y)
        # record the update method
        if update_method not in ["moment-matching", "KL", "moment-KL"]:
            raise ValueError("Unknown update method!")
        self.update_method = update_method
        # set the prior parameters
        self.theta = torch.zeros(self.num_alternatives).to(train_X)
        self.b = self.num_alternatives + 1.0
        self.q = 1.0
        self.B = torch.eye(self.num_alternatives).to(train_X)
        # collect the full samples, i.e., the `num_alternatives`-dim tensor samples
        full_samples = list()
        num_full_samples = 0
        while True:
            try:
                full_sample = torch.zeros(self.num_alternatives).to(train_X)
                for i in range(self.num_alternatives):
                    full_sample[i] = self.alternative_observations[i][num_full_samples]
                full_samples.append(full_sample.unsqueeze(0))
                num_full_samples += 1
            except IndexError:
                break
        full_samples = torch.cat(full_samples, dim=0)
        # process full_samples first, then the remaining samples
        self.add_full_observations(full_samples)
        # process additional observations as updates
        for i in range(self.num_alternatives):
            Y = self.alternative_observations[i][num_full_samples:]
            X = torch.ones_like(Y) * i
            self.update_parameters(X, Y)

    def add_full_observations(self, Y: Tensor) -> None:
        r"""
        Apply the standard posterior update formulas for normal-Wishart
        for adding a full observation vector.

        Args:
            Y: A `batch_shape x num_alternatives`-dim tensor of full observations.
                Batches are processed by calling the method in a for loop.
        """
        if Y.dim() > 1:
            for y in Y:
                self.add_full_observations(y)
            return None
        if Y.shape != torch.Size([self.num_alternatives]):
            raise ValueError("This method only accepts full observations!")
        # calculate the updates offline, then apply them to self
        q = self.q + 1
        b = self.b + 1
        theta = (self.q * self.theta + Y) / q
        diff = (self.theta - Y).unsqueeze(-1)
        B = self.B + (self.q / q) * diff.matmul(diff.t())
        self.q = q
        self.b = b
        self.theta = theta
        self.B = B

    def update_parameters(self, X: Tensor, Y: Tensor) -> None:
        r"""
        Update the parameters of the normal-Wishart distribution using a
        single observation at a single alternative.

        If non-scalar X, Y are given, the method is called in a for loop,
        processing the samples one at a time.

        Uses the specified update method.

        Args:
            X: A scalar tensor representing the alternative being sampled.
            Y: The corresponding observation, scalar tensor.
        """
        if X.shape != Y.shape:
            raise ValueError("Shapes of `X` and `Y` must match!")
        # handle the no-input and multiple input cases.
        if X.numel() == 0:
            return None
        if X.numel() > 1:
            for x, y in zip(X, Y):
                self.update_parameters(x, y)
            return None
        if X != X.long() or not 0 <= X < self.num_alternatives:
            raise ValueError(
                "Alternatives must be integers in 0, ..., n_alternatives - 1!"
            )
        k = int(X)  # used for cleaner indexing
        K = float(self.num_alternatives)  # for cleaner notation
        int_K = int(K)
        # processing the update in a try / except block and updating
        # the actual (self) parameters once the whole update completes.
        q = self.q + (1 / K)
        b = self.b + (1 / K)
        # We need to index B in non-standard ways for updating theta and B
        # We will also define a number of intermediate values in the process.
        # If any variable is accessed from self, it is time n variable.
        # If it is accessed locally, then it is time n+1 variable.
        if self.update_method == "KL":
            # based on eq 18 and 13 of [1]_
            # this is eq 18
            tmp_factor = (Y - self.theta[k]) / (
                ((self.q * b) / (b - K + 1) + 1) * self.B[k, k]
            )
            theta = self.theta + tmp_factor * self.B[:, k]
            # eq 13
            tmp_factor = self.q * (Y - self.theta[k]).pow(2) / (
                ((self.q * b) / (b - K + 1)) + 1
            ) - self.B[k, k] / self.b
            tmp_col = self.B[:, k].unsqueeze(-1)
            tmp_factor_2 = tmp_col.matmul(tmp_col.t()) / self.B[k, k].pow(2)
            B = (b / self.b) * self.B + (b / (self.b + 1)) * tmp_factor * tmp_factor_2
        elif self.update_method in ["moment-matching", "moment-KL"]:
            # first some common terms between moment-matching and moment-KL
            # eq 20
            theta = self.theta + (self.B[:, k] / self.B[k, k]) * (Y - self.theta[k]) / (
                    self.q + 1
            )
            # Bn_mkmk is B^n -k -k
            mk_mask = torch.ones(int_K, device=X.device, dtype=torch.bool)
            mk_mask[k] = 0
            mk_mk_mask = mk_mask.unsqueeze(-1) * mk_mask
            Bn_mkmk = self.B[mk_mk_mask].reshape(int_K - 1, int_K - 1)
            # tmp_term is B^n -k,k * B^n k, -k  / B^n k, k
            tmp_term = (self.B[mk_mask, k].unsqueeze(-1) * self.B[k, mk_mask]) / self.B[
                k, k
            ]
            # Bn_mkmidk is B^n -k|k
            Bn_mkmidk = Bn_mkmk - tmp_term
            if self.update_method == "moment-matching":
                # This operation is based on Proposition 3.2 of [2]_
                q_tilde = 1 + (self.q * (Y - self.theta[k]).pow(2)) / (
                    (self.q + 1) * self.B[k, k]
                )
                # eq 21, B_mkmk, being B^n+1 -k, -k
                B_mkmk = (
                    (q * (b - K - 1))
                    / (self.b - K)
                    * (
                        Bn_mkmidk / self.q
                        + q_tilde / (self.q + 1) * (Bn_mkmidk / (self.b - K) + tmp_term)
                    )
                )
                # tmp factor is the term common in eq 22 & 23
                tmp_factor = (q * (b - K - 1) * q_tilde) / ((self.q + 1) * (self.b - K))
                # eq 22, B_mkk as B^n+1 -k, k. B_kmk is simply the transpose of this.
                B_mkk = tmp_factor * self.B[mk_mask, k].unsqueeze(-1)
                # eq 23
                B_kk = tmp_factor * self.B[k, k]
            else:
                # eq 27
                tmp_factor = q * (b - K + 1) / ((self.b + 1) * (self.q + 1))
                B_kk = tmp_factor * (
                        self.B[k, k] + (self.q / (self.q + 1)) * (Y - self.theta[k]).pow(2)
                )
                # eq 28
                B_mkk = B_kk * self.B[mk_mask, k] / self.B[k, k]
                # eq 29
                B_mkmk = (b * q / (self.b * self.q)) * Bn_mkmidk + tmp_term
            # putting all together
            B = torch.zeros_like(self.B)
            B[mk_mk_mask] = B_mkmk.view(-1)
            B[mk_mask, k] = B_mkk.view(-1)
            B[k, mk_mask] = B_mkk.view(-1)
            B[k, k] = B_kk
        # update the self params
        self.b = b
        self.q = q
        self.theta = theta
        self.B = B

    def posterior(self, X: Optional[Tensor]) -> Union[t, multivariate_t]:
        r"""
        Returns the posterior distribution of X, which is a t or
        a multivariate-t distribution.

        Args:
            X: A scalar tensor of alternative to get the predictive distribution
                for. If None, a multivariate-t distribution is returned for
                sampling from joint predictive distribution of all alternatives.

        Returns:
            A t or multivariate-t object with mean and covariance frozen to
            the predicted values.
        """
        df = self.b - self.num_alternatives + 1
        if X is None:
            covar = (self.q + 1) * self.B / (self.q * df)
            return multivariate_t(loc=self.theta, shape=covar, df=df)

        X_l = X.long()
        if torch.any(X != X_l):
            raise ValueError(
                "Inputs must be integers from range 0, ..., n_alternatives - 1!"
            )
        if X_l.numel() > 1:
            raise NotImplementedError(
                "This only supports posterior from a single alternative at a time."
            )
        scale = (self.q * df / ((self.q + 1) * self.B[X_l, X_l])).sqrt()
        return t(loc=self.theta[X_l], scale=scale, df=df)

    def get_s_tilde(self, X: Optional[Tensor]) -> Tensor:
        r"""
        Calculate the value of s_tilde for use in KG computations.
        The precise term we calculate here is given in Section 4 of [2]_.
        Additional discussion on s_tilde can be found in Section 3 of [1]_.

        s_tilde can be used with a T random variable (with df = b - K + 1)
        to predict the mu^(n+1) in KG as \theta + s_tilde T.

        This supports both the moment-matching and moment-KL approaches of
        [2]_ and the KL minimization approach of [1]_.

        Args:
            X: A scalar tensor of alternative to get s_tilde for. If None,
                s_tilde is returned for all alternatives.

        Returns:
            The value of s_tilde. If the alternative is specified, this is a
            tensor of shape `num_alternatives`. Otherwise, it is a tensor of
            shape `num_alternatives x num_alternatives`, with s_tilde[i]
            corresponding to s_tilde for i-th alternative.
        """
        if self.update_method in ["moment-matching", "moment-KL"]:
            common_factor = self.q * (self.q + 1) * (self.b - self.num_alternatives + 1)
        elif self.update_method == "KL":
            b_new = self.b + 1.0 / self.num_alternatives
            common_factor = math.sqrt((self.q + 1) / (self.q * (self.b - self.num_alternatives + 1.0))) / (
                (self.q * b_new) / (b_new - self.num_alternatives + 1) + 1
            )
        if X is None:
            expanded_diag = self.B.diag().expand(self.num_alternatives, -1).t()
            if self.update_method in ["moment-matching", "moment-KL"]:
                s_tilde = self.B / (
                        common_factor * expanded_diag
                ).sqrt()
            elif self.update_method == "KL":
                s_tilde = common_factor * self.B / expanded_diag.sqrt()
        else:
            X_l = X.long()
            if torch.any(X != X_l):
                raise ValueError(
                    "Inputs must be integers from range 0, ..., n_alternatives - 1!"
                )
            if X_l.numel() > 1:
                raise NotImplementedError(
                    "This only supports posterior from a single alternative at a time."
                )
            if self.update_method in ["moment-matching", "moment-KL"]:
                s_tilde = self.B[:, X_l] / (common_factor * self.B[X_l, X_l]).sqrt()
            elif self.update_method == "KL":
                s_tilde = common_factor * self.B[:, X_l] / self.B[X_l, X_l].sqrt()
        return s_tilde

