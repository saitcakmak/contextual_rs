import torch

from contextual_rs.modified_ts_policies import (
    get_beta_hat,
    modified_ts,
    modified_ts_plus,
)
from test.utils import BotorchTestCase


class TestModifiedTSPolicies(BotorchTestCase):
    def test_get_beta_hat(self):
        # If given a perfectly linear setting, the result should be perfect.
        X = torch.tensor([[1.], [2.]])
        Y = torch.tensor([[1., 2.], [2., 4.]])
        beta = get_beta_hat(X, Y)
        self.assertTrue(torch.allclose(beta, X))

        # 1x + 2y, 3x + 4y
        X = torch.tensor([[1., 2.], [3., 5.]])
        Y = torch.tensor([[5., 13.], [11., 29.]])
        beta = get_beta_hat(X, Y)
        self.assertTrue(
            torch.allclose(beta, torch.tensor([[1., 2.], [3., 4.]]), atol=1e-3)
        )

    def test_modified_ts(self):
        # Try 0 variance.
        X = torch.tensor([[1.], [2.]])
        Y = torch.tensor([[[1., 2.], [2., 4.]]]).repeat(10, 1, 1)
        total_budget = 100
        additional_budget = modified_ts(X, Y, total_budget)
        self.assertTrue(torch.isnan(additional_budget).all())
        # Equal variance, should lead to equal allocation.
        error = torch.randn(10, 1, 1)
        Y = Y + error
        additional_budget = modified_ts(X, Y, total_budget)
        self.assertTrue(
            torch.allclose(additional_budget, torch.full_like(additional_budget, 15.))
        )

    def test_modified_ts_plus(self):
        torch.manual_seed(0)
        # Try 0 variance.
        X = torch.tensor([[1.], [2.]])
        Y = torch.tensor([[[1., 2.], [2., 4.]]]).repeat(10, 1, 1)
        total_budget = 100
        additional_budget = modified_ts_plus(X, Y, total_budget)
        self.assertTrue(torch.isnan(additional_budget).all())
        # Equal variance, should lead to equal allocation.
        error = torch.randn(10, 1, 1)
        Y = Y + error
        additional_budget = modified_ts_plus(X, Y, total_budget)
        self.assertTrue(
            torch.allclose(additional_budget, torch.full_like(additional_budget, 15.))
        )
        # Unequal variance
        Y[:, 1:, 1:] += error
        additional_budget = modified_ts_plus(X, Y, total_budget)
        self.assertTrue(
            torch.allclose(additional_budget, torch.tensor([[4., 4.], [4., 47.]]))
        )




