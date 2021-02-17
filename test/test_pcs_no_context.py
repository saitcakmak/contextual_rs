import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from contextual_rs.custom_fit import custom_fit_gpytorch_model
from contextual_rs.pcs_no_context import (
    estimate_lookahead_pcs_no_context,
    estimate_current_pcs_no_context,
)
from contextual_rs.lce_gp import LCEGP
from test.utils import BotorchTestCase


class TestPCSNoContext(BotorchTestCase):
    def test_estimate_lookahead_pcs_no_context(self):
        dim_x = 1
        num_fantasies = 4
        num_samples = 16
        base_samples = None
        num_arms = 3
        num_train = 5
        train_X = (
            torch.arange(num_arms, dtype=torch.float).repeat(num_train).view(-1, 1)
        )
        # construct and train the model
        model = LCEGP(
            train_X,
            torch.randn(num_train * num_arms, 1),
            categorical_cols=[0],
            embs_dim_list=[2],
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # setup test inputs
        arm_set = torch.tensor(range(num_arms)).unsqueeze(-1)
        q = 2
        candidate = torch.randint(0, num_arms, (q, dim_x), dtype=torch.float)

        func_I = lambda X: (X > 0).to(dtype=torch.float)
        model_sampler = SobolQMCNormalSampler(num_samples=num_fantasies)

        pcs = estimate_lookahead_pcs_no_context(
            candidate=candidate,
            model=model,
            model_sampler=model_sampler,
            arm_set=arm_set,
            num_samples=num_samples,
            base_samples=base_samples,
            func_I=func_I,
        )
        # check output shape
        self.assertEqual(pcs.shape, torch.Size([1]))
        # check that the values are probabilities, i.e., between 0 and 1
        self.assertTrue(torch.equal(pcs, pcs.clamp(min=0, max=1)))

        num_candidates = 4
        candidate = torch.randint(
            0, num_arms, (num_candidates, q, dim_x), dtype=torch.float
        )

        pcs = estimate_lookahead_pcs_no_context(
            candidate=candidate,
            model=model,
            model_sampler=model_sampler,
            arm_set=arm_set,
            num_samples=num_samples,
            base_samples=base_samples,
            func_I=func_I,
        )
        # check output shape
        self.assertEqual(pcs.shape, torch.Size([num_candidates]))
        # check that the values are probabilities, i.e., between 0 and 1
        self.assertTrue(torch.equal(pcs, pcs.clamp(min=0, max=1)))

    def test_estimate_current_pcs_no_context(self):
        def sine_test(X: Tensor) -> Tensor:
            sine = torch.sin(X).sum(dim=-1, keepdim=True)
            noise = torch.randn_like(sine)
            return sine + noise

        torch.manual_seed(0)
        num_fit = 5
        num_arms = 10
        num_train = 5
        num_samples = 100
        train_X = (
            torch.arange(num_arms, dtype=torch.float).repeat(num_train).view(-1, 1)
        )

        # construct and train the model
        model = LCEGP(
            train_X, sine_test(train_X), categorical_cols=[0], embs_dim_list=[2]
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        custom_fit_gpytorch_model(mll, num_retries=num_fit)

        # setup test inputs
        arm_set = torch.tensor(range(num_arms)).unsqueeze(-1)
        base_samples = None

        func_I = lambda X: (X > 0).to(dtype=torch.float)

        pcs = estimate_current_pcs_no_context(
            model=model,
            arm_set=arm_set,
            num_samples=num_samples,
            base_samples=base_samples,
            func_I=func_I,
        )
        # check output shape
        self.assertEqual(pcs.shape, torch.Size())
        # check that the values are probabilities, i.e., between 0 and 1
        self.assertTrue(torch.equal(pcs, pcs.clamp(min=0, max=1)))

        # it should converge to 1 as we add more and more training samples
        num_train = 50
        train_X = (
            torch.arange(num_arms, dtype=torch.float).repeat(num_train).view(-1, 1)
        )

        # construct and train the model
        model = LCEGP(
            train_X, sine_test(train_X), categorical_cols=[0], embs_dim_list=[2]
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        custom_fit_gpytorch_model(mll, num_retries=num_fit)
        pcs2 = estimate_current_pcs_no_context(
            model=model,
            arm_set=arm_set,
            num_samples=num_samples,
            base_samples=base_samples,
            func_I=func_I,
        )
        # check output shape
        self.assertEqual(pcs2.shape, torch.Size())
        # check that the values are probabilities, i.e., between 0 and 1
        self.assertTrue(torch.equal(pcs2, pcs2.clamp(min=0, max=1)))

        # check that the pcs increased
        self.assertGreater(pcs2, pcs)
        print(f"pcs2 {pcs2}, pcs {pcs}")
