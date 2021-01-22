import torch
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from contextual_rs.generalized_pcs import estimate_generalized_pcs
from contextual_rs.lce_gp import LCEGP
from test.utils import BotorchTestCase


class TestGeneralizedPCS(BotorchTestCase):
    def test_estimate_generalized_pcs(self):
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
        rho = lambda X: X.sum(dim=-2)

        pcs = estimate_generalized_pcs(
            candidate=candidate,
            model=model,
            model_sampler=model_sampler,
            arm_set=arm_set,
            context_set=context_set,
            num_samples=num_samples,
            base_samples=base_samples,
            func_I=func_I,
            rho=rho,
        )

        # check output shape
        self.assertEqual(pcs.shape, torch.Size([num_candidates]))
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
        train_X[:3, 0] = torch.tensor([0., 1., 2.])
        # construct and train the model
        model = LCEGP(
            train_X,
            torch.randn(num_train, 1),
            categorical_cols=[0],
            embs_dim_list=[2]
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

        pcs = estimate_generalized_pcs(
            candidate=candidate,
            model=model,
            model_sampler=model_sampler,
            arm_set=arm_set,
            context_set=context_set,
            num_samples=num_samples,
            base_samples=base_samples,
            func_I=func_I,
            rho=rho,
        )
        # check output shape
        self.assertEqual(pcs.shape, torch.Size([1]))
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
        pcs = estimate_generalized_pcs(
            candidate=candidate,
            model=model,
            model_sampler=model_sampler,
            arm_set=arm_set,
            context_set=context_set,
            num_samples=num_samples,
            base_samples=base_samples,
            func_I=func_I,
            rho=rho,
        )
        # check output shape
        self.assertEqual(pcs.shape, torch.Size([num_candidates]))
        # check that the values are probabilities, i.e., between 0 and 1
        self.assertTrue(torch.equal(pcs, pcs.clamp(min=0, max=1)))
