from typing import List, Optional

import torch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior
from torch import Tensor
from torch.nn import ModuleList

MIN_INFERRED_NOISE_LEVEL = 1e-4


class LCEGP(BatchedMultiOutputGPyTorchModel, ExactGP):
    r"""
    A single-output Latent Context Embedding GP model.
    Uses latent context embeddings for categorical variables,
    similar to LCEMGP from BoTorch.

    This implementation is based on LCEMGP and SingleTaskGP.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        categorical_cols: List[int],
        embs_dim_list: Optional[List[int]] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""
        LCEGP with MaternKernel over continuous inputs and RBFKernel over
        embedded inputs.

        Each categorical column is treated independently, using a product kernel.
        This assumes that the training inputs includes all categorical entries at
        least once, i.e., if there are two categorical columns with 3 and 5
        categories respectively, then all 3 categories of the first and all 5
        categories of the second must appear in the training data, requiring at
        least max(category_counts) training inputs.
        It also assumes that the categories are given as indices from 0 to
        n_categories - 1, and will throw an error if this is not the case. Once
        again, all entries from 0 to n_categories - 1 must appear in the training
        input for each categorical column.

        Args:
            train_X: An `n x d`-dim tensor of training features. Categorical
                columns are assumed to take integer values from 0 to n_categories - 1.
            train_Y: An `n x 1`-dim tensor of training outputs.
            categorical_cols: A list of ints denoting the column indices of
                `train_X` to get the categorical variables from.
            embs_dim_list: A list of ints, same size as `categorical_cols`
                giving the embedding dimension for each categorical input.
                If None, defaults to 1D embedding for each dimension.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass. Only applied to continuous columns of the input.
        """
        self._validate_inputs(
            train_X=train_X,
            train_Y=train_Y,
            categorical_cols=categorical_cols,
            embs_dim_list=embs_dim_list,
        )
        dim = train_X.shape[-1]
        categorical_cols = [i % dim for i in categorical_cols]
        continuous_cols = [i for i in range(dim) if i not in categorical_cols]
        self.has_continuous_cols = bool(continuous_cols)
        if input_transform is not None:
            input_transform.to(train_X)
        with torch.no_grad():
            # apply transform only to continuous columns
            transformed_X = train_X.clone()
            if self.has_continuous_cols:
                transformed_X[..., continuous_cols] = self.transform_inputs(
                    X=train_X[..., continuous_cols], input_transform=input_transform
                )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        validate_input_scaling(train_X=transformed_X, train_Y=train_Y)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)

        # default likelihood from SingleTaskGP
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )

        ExactGP.__init__(self, train_X, train_Y, likelihood)
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)

        # construct the covariance module
        # MaternKernel for continuous columns, defaults from SingleTaskGP
        if self.has_continuous_cols:
            self.cont_covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=len(continuous_cols),
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            self._subset_batch_dict = {
                "likelihood.noise_covar.raw_noise": -2,
                "mean_module.constant": -2,
                "cont_covar_module.raw_outputscale": -1,
                "cont_covar_module.base_kernel.raw_lengthscale": -3,
            }
        else:
            self._subset_batch_dict = {
                "likelihood.noise_covar.raw_noise": -2,
                "mean_module.constant": -2,
            }

        # this is based on LCEMGP
        # construct emb_dims based on categorical features
        if embs_dim_list is None:
            #  set embedding_dim = 1 for each categorical variable
            embs_dim_list = [1] * len(categorical_cols)
        n_embs = sum(embs_dim_list)
        self.emb_dims = [
            (self.category_counts[i], embs_dim_list[i])
            for i in range(len(categorical_cols))
        ]
        # contruct embedding layer
        self.emb_layers = ModuleList(
            [
                torch.nn.Embedding(num_embeddings=x, embedding_dim=y, max_norm=1.0)
                for x, y in self.emb_dims
            ]
        )
        # TODO: this is a product kernel. May want to modify this in the future
        self.emb_covar_module = RBFKernel(
            ard_num_dims=n_embs,
            lengthscale_constraint=Interval(
                0.0, 2.0, transform=None, initial_value=1.0
            ),
        )

        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def _validate_inputs(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        categorical_cols: List[int],
        embs_dim_list: Optional[List[int]] = None,
    ) -> None:
        r"""Performs a series of checks on the input arguments."""
        if len(categorical_cols) == 0:
            raise RuntimeError(
                "When not using any categorical inputs, use SingleTaskGP instead!"
            )
        if embs_dim_list is not None and len(embs_dim_list) != len(categorical_cols):
            raise ValueError(
                "`embs_dim_list` and `categorical_dims` must have the same size!"
            )
        # check that categorical entries are from 0 to n_categories - 1
        # record category_counts for use in Embedding
        self.category_counts = list()
        for col in categorical_cols:
            all_categories = train_X[..., col].unique()
            all_cat_long = all_categories.long()
            if torch.any(all_categories != all_cat_long):
                raise ValueError(
                    "Expected categorical inputs to take integer values,"
                    f"found categorical entries {all_categories} in column {col}."
                )
            cat_list = all_cat_long.tolist()
            cat_count = len(cat_list)
            if cat_list != list(range(cat_count)):
                raise ValueError(
                    "Expected to see all categories from 0 to n_categories - 1"
                    "for each categorical column at least once in the training data."
                    f"Received {cat_list} in column {col}"
                )
            self.category_counts.append(cat_count)

    def forward(self, x: Tensor) -> MultivariateNormal:
        r"""
        Computes the prior mean mu(x) and covariance kernel K(x, x) on the inputs.
        Assumes a product covariance structure between the categorical / embedded
        columns and the continuous columns.

        # TODO: for optimization purposes, we may want to consider evaluations in the
            embedded space in the future, i.e., we can create a version of forward
            which accepts `x` that is continuous and is of the embedded dimension.
            This would give us differentiability in the embedded inputs.

        Args:
            x: `batch_shape x q x d`-dim tensor of features.

        Returns:
            The MVN object with mean mu(x) and covariance K(x, x)
        """
        mean = self.mean_module(x)
        # compute covar over continuous columns
        if self.has_continuous_cols:
            x_cont = self.transform_inputs(x[..., self.continuous_cols])
            covar_cont = self.cont_covar_module(x_cont)
        # process categorical columns
        x_cat = x[..., self.categorical_cols]
        x_cat_long = x_cat.long()
        if torch.any(x_cat != x_cat_long):
            raise ValueError(
                "Expected categorical columns to have integer values,"
                f"got {x_cat} for categorical inputs."
            )
        # compute covar over embedded features
        embeddings = [
            emb_layer(x_cat_long[..., i]) for i, emb_layer in enumerate(self.emb_layers)
        ]
        x_emb = torch.cat(embeddings, dim=-1)
        covar_emb = self.emb_covar_module(x_emb)
        # combine covariances together
        if self.has_continuous_cols:
            covar = covar_cont.mul(covar_emb)
        else:
            covar = covar_emb
        return MultivariateNormal(mean, covar)

    def get_s_tilde(self, X: Optional[Tensor]) -> Tensor:
        r"""
        Calculate the value of s_tilde for use in KG computations.
        The calculation is based on the following formula, borrowed from
        Wu & Frazier 2016.

        .. math::
            \tilde{\sigma}(x, X) = K(x, X) / \sqrt{ K(X, X) + \diag(\sigma^2(X)) }

        Note: below, we will use the vector of all alternatives for `x`.
            When X is None, the sqrt becomes the Cholesky decomposition and
            the division becomes matmul with the inverse matrix.

        s_tilde can be used with a Z ~ N(0, 1) random variable
        to predict the mu^(n+1) in KG as \theta + s_tilde Z.

        Args:
            X: A scalar tensor of alternative to get s_tilde for. If None,
                s_tilde is returned for all alternatives.

        Returns:
            The value of s_tilde. If the alternative is specified, this is a
            tensor of shape `num_alternatives`. Otherwise, it is a tensor of
            shape `num_alternatives x num_alternatives`, with s_tilde[i]
            corresponding to s_tilde for i-th alternative.
        """
        if len(self.categorical_cols) != 1 or self.continuous_cols != list():
            raise NotImplementedError(
                "This is defined only for the simple RS setting!"
                "Use `get_s_tilde_general` instead!"
            )
        all_alternatives = (
            torch.tensor(range(self.category_counts[0]))
            .view(-1, 1)
            .to(self.train_targets)
        )
        full_mvn = self(all_alternatives)
        noise = self.likelihood.noise.squeeze()
        if X is None:
            covar = full_mvn.covariance_matrix
            diag_covar = covar.diag()
            noisy_diag = diag_covar + noise
            noisy_diag_root = noisy_diag.sqrt()
            expanded = noisy_diag_root.view(-1, 1).expand_as(covar)
            return covar / expanded
        else:
            X_l = X.long()
            if X.numel() != 1 or X != X_l:
                raise ValueError("X must be a scalar tensor with an integer value!")
            K_x_X = full_mvn.covariance_matrix[:, X_l].view(-1)
            second_term = (full_mvn.covariance_matrix[X_l, X_l] + noise).sqrt()
            return K_x_X / second_term

    def get_s_tilde_general(self, X: Tensor, x: Tensor) -> Tensor:
        r"""
        This returns the s_tilde in the general setting where we have
        more than a single categorical input dimension.

        .. math::
            \tilde{\sigma}(x, X) = K(x, X) \Chol{ K(X, X) + \diag(\sigma^2(X)) }^{-1}

        Args:
            X: `batch_shape x q x d`-dim tensor of candidates.
            x: `n' x d`-dim tensor of points to calculate s_tilde for.

        Returns:
            `batch_shape x n' x q`-dim tensor of s_tilde values.
        """
        q = X.shape[-2]
        x_cat = torch.cat([x.expand(X.shape[:-2] + x.shape), X], dim=-2)
        full_mvn = self(x_cat)
        full_covar = full_mvn.covariance_matrix
        noise = self.likelihood.noise.squeeze()
        K_x_X = full_covar[..., :-q, -q:]
        K_X_X = full_covar[..., -q:, -q:]
        chol = torch.cholesky(K_X_X + torch.diag(noise.expand(q)).expand_as(K_X_X))
        return K_x_X.matmul(chol.inverse())


