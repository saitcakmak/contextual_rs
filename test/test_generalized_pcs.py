import torch
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from contextual_rs.generalized_pcs import (
    estimate_lookahead_generalized_pcs,
    estimate_current_generalized_pcs,
)
from contextual_rs.models.lce_gp import LCEGP
from test.utils import BotorchTestCase


class TestGeneralizedPCS(BotorchTestCase):
    def test_estimate_lookahead_generalized_pcs(self):
        for use_apx in [False, True]:
            # test with SingleTaskGP
            num_train = 10
            dim_x = 2
            dim_c = 1
            dim = dim_x + dim_c
            model = SingleTaskGP(torch.rand(num_train, dim), torch.randn(num_train, 1))
            num_candidates = 2
            candidate = torch.rand(num_candidates, 1, dim)
            num_fantasies = 5
            model_sampler = SobolQMCNormalSampler(num_samples=num_fantasies)
            num_arms = 5
            arm_set = torch.rand(num_arms, dim_x)
            num_contexts = 7
            context_set = torch.rand(num_contexts, dim_c)
            num_samples = 3
            base_samples = torch.randn(
                num_samples, num_fantasies, num_candidates, num_arms * num_contexts, 1
            )
            func_I = lambda X: (X > 0).to(dtype=torch.float)
            rho = lambda X: X.mean(dim=-2)

            pcs = estimate_lookahead_generalized_pcs(
                candidate=candidate,
                model=model,
                model_sampler=model_sampler,
                arm_set=arm_set,
                context_set=context_set,
                num_samples=num_samples,
                base_samples=base_samples,
                func_I=func_I,
                rho=rho,
                use_approximation=use_apx,
            )

            # check output shape
            self.assertEqual(pcs.shape, torch.Size([num_candidates]))
            if not use_apx:
                # check that the values are probabilities, i.e., between 0 and 1
                self.assertTrue(torch.equal(pcs, pcs.clamp(min=0, max=1)))

            # test with LCEMGP
            dim_x = 1
            dim_c = 2
            num_arms = 3
            train_X = torch.cat(
                [
                    torch.randint(0, num_arms, (num_train, dim_x)),
                    torch.rand(num_train, dim_c),
                ],
                dim=-1,
            )
            # ensure each category is in the data
            train_X[:3, 0] = torch.tensor([0.0, 1.0, 2.0])
            # construct and train the model
            model = LCEGP(
                train_X,
                torch.randn(num_train, 1),
                categorical_cols=[0],
                embs_dim_list=[2],
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # setup test inputs
            arm_set = torch.tensor(range(num_arms)).unsqueeze(-1)
            context_set = torch.rand(num_contexts, dim_c)
            base_samples = None
            q = 2
            candidate = torch.cat(
                [
                    torch.randint(0, num_arms, (q, dim_x)),
                    torch.rand(q, dim_c),
                ],
                dim=-1,
            )

            pcs = estimate_lookahead_generalized_pcs(
                candidate=candidate,
                model=model,
                model_sampler=model_sampler,
                arm_set=arm_set,
                context_set=context_set,
                num_samples=num_samples,
                base_samples=base_samples,
                func_I=func_I,
                rho=rho,
                use_approximation=use_apx,
            )
            # check output shape
            self.assertEqual(pcs.shape, torch.Size([1]))
            if not use_apx:
                # check that the values are probabilities, i.e., between 0 and 1
                self.assertTrue(torch.equal(pcs, pcs.clamp(min=0, max=1)))

            num_candidates = 4
            candidate = torch.cat(
                [
                    torch.randint(0, num_arms, (num_candidates, q, dim_x)),
                    torch.rand(num_candidates, q, dim_c),
                ],
                dim=-1,
            )
            pcs = estimate_lookahead_generalized_pcs(
                candidate=candidate,
                model=model,
                model_sampler=model_sampler,
                arm_set=arm_set,
                context_set=context_set,
                num_samples=num_samples,
                base_samples=base_samples,
                func_I=func_I,
                rho=rho,
                use_approximation=use_apx,
            )
            # check output shape
            self.assertEqual(pcs.shape, torch.Size([num_candidates]))
            if not use_apx:
                # check that the values are probabilities, i.e., between 0 and 1
                self.assertTrue(torch.equal(pcs, pcs.clamp(min=0, max=1)))

            # check that the certainty equivalent approximation works fine
            pcs = estimate_lookahead_generalized_pcs(
                candidate=candidate,
                model=model,
                model_sampler=None,
                arm_set=arm_set,
                context_set=context_set,
                num_samples=num_samples,
                base_samples=base_samples,
                func_I=func_I,
                rho=rho,
                use_approximation=use_apx,
            )
            # check output shape
            self.assertEqual(pcs.shape, torch.Size([num_candidates]))
            if not use_apx:
                # check that the values are probabilities, i.e., between 0 and 1
                self.assertTrue(torch.equal(pcs, pcs.clamp(min=0, max=1)))

    def test_estimate_current_generalized_pcs(self):
        def sine_test(X: Tensor) -> Tensor:
            return torch.sin(X * 10.0).sum(dim=-1, keepdim=True)

        for use_apx in [False, True]:
            # running a simple test with LCEGP
            # test with LCEMGP
            dim_x = 1
            dim_c = 2
            num_arms = 3
            num_train = 20
            num_samples = 100
            train_X = torch.cat(
                [
                    torch.randint(0, num_arms, (num_train, dim_x)),
                    torch.rand(num_train, dim_c),
                ],
                dim=-1,
            )
            # ensure each category is in the data
            train_X[:3, 0] = torch.tensor([0.0, 1.0, 2.0])
            # construct and train the model
            model = LCEGP(
                train_X, sine_test(train_X), categorical_cols=[0], embs_dim_list=[2]
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # setup test inputs
            num_contexts = 10
            arm_set = torch.tensor(range(num_arms)).unsqueeze(-1)
            context_set = torch.rand(num_contexts, dim_c)
            base_samples = None

            func_I = lambda X: (X > 0).to(dtype=torch.float)
            rho = lambda X: X.mean(dim=-2)

            pcs = estimate_current_generalized_pcs(
                model=model,
                arm_set=arm_set,
                context_set=context_set,
                num_samples=num_samples,
                base_samples=base_samples,
                func_I=func_I,
                rho=rho,
                use_approximation=use_apx,
            )
            # check output shape
            self.assertEqual(pcs.shape, torch.Size())
            if not use_apx:
                # check that the values are probabilities, i.e., between 0 and 1
                self.assertTrue(torch.equal(pcs, pcs.clamp(min=0, max=1)))

            # it should converge to 1 as we add more and more training samples
            num_train = 200
            train_X = torch.cat(
                [
                    torch.randint(0, num_arms, (num_train, dim_x)),
                    torch.rand(num_train, dim_c),
                ],
                dim=-1,
            )
            # ensure each category is in the data
            train_X[:3, 0] = torch.tensor([0.0, 1.0, 2.0])
            # construct and train the model
            model = LCEGP(
                train_X, sine_test(train_X), categorical_cols=[0], embs_dim_list=[2]
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            pcs2 = estimate_current_generalized_pcs(
                model=model,
                arm_set=arm_set,
                context_set=context_set,
                num_samples=num_samples,
                base_samples=base_samples,
                func_I=func_I,
                rho=rho,
                use_approximation=use_apx,
            )
            # check output shape
            self.assertEqual(pcs2.shape, torch.Size())
            if not use_apx:
                # check that the values are probabilities, i.e., between 0 and 1
                self.assertTrue(torch.equal(pcs2, pcs2.clamp(min=0, max=1)))

            # check that the pcs increased
            self.assertGreater(pcs2, pcs)
