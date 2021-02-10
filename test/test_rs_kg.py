import torch
from botorch import fit_gpytorch_model
from botorch.models.transforms import Standardize
from botorch.sampling import SobolQMCNormalSampler
from gpytorch import ExactMarginalLogLikelihood

from contextual_rs.unknown_correlation_model import UnknownCorrelationModel
from contextual_rs.lce_gp import LCEGP
from contextual_rs.rs_kg import DiscreteKG, UnknownCorrelationKG
from test.utils import BotorchTestCase


class TestDiscreteKG(BotorchTestCase):
    def setUp(self):
        super(TestDiscreteKG, self).setUp()
        num_alternatives = 5
        num_train = 3
        train_X = (
            torch.arange(num_alternatives, dtype=torch.float)
            .repeat(num_train)
            .unsqueeze(-1)
        )
        train_Y = torch.randn_like(train_X)
        self.model = LCEGP(train_X, train_Y, categorical_cols=[0])

    def test_discrete_kg(self):
        # simple constructor tests
        kg = DiscreteKG(model=self.model)
        self.assertEqual(kg.num_fantasies, 64)
        self.assertIsInstance(kg.sampler, SobolQMCNormalSampler)
        self.assertEqual(kg.sampler.sample_shape, torch.Size([64]))
        self.assertIsNone(kg.current_value)

        # with sampler specified
        kg = DiscreteKG(
            model=self.model, num_fantasies=None, sampler=SobolQMCNormalSampler(4)
        )
        self.assertEqual(kg.num_fantasies, 4)

        with self.assertRaises(ValueError):
            kg = DiscreteKG(
                model=self.model, num_fantasies=16, sampler=SobolQMCNormalSampler(4)
            )

        # simple forward test
        kg = DiscreteKG(model=self.model, current_value=torch.tensor(0.0))
        kg_vals = kg(torch.arange(3, dtype=torch.float).reshape(-1, 1, 1))
        self.assertEqual(kg_vals.shape, torch.Size([3]))

    def test_discrete_kg_logic(self):
        # testing the logic - with these values, alternative 0 should be preferred
        train_X = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        ).view(-1, 1)
        train_Y = torch.tensor(
            [3.5, 4.5, 2.7, 5.0, 1.5, 1.6, 1.4, 1.55, 0.5, 0.6, 0.7, 0.65]
        ).view(-1, 1)
        model = LCEGP(
            train_X, train_Y, categorical_cols=[0], outcome_transform=Standardize(m=1)
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        kg = DiscreteKG(model=model)
        kg_vals = kg(torch.tensor([0.0, 1.0, 2.0]).view(-1, 1, 1))
        self.assertEqual(kg_vals.argmax(), 0)


class TestUnknownCorrelationKG(BotorchTestCase):
    def setUp(self):
        num_alternatives = 5
        num_train = 3
        train_X = torch.arange(num_alternatives, dtype=torch.float).repeat(num_train)
        train_Y = torch.randn_like(train_X)
        self.model = UnknownCorrelationModel(train_X, train_Y)

    def test_unknown_correlation_kg(self):
        kg = UnknownCorrelationKG(model=self.model)
        self.assertEqual(kg.num_fantasies, 64)
        self.assertEqual(kg.base_samples.shape, torch.Size([64]))
        self.assertIsNone(kg.current_value)

        with self.assertRaises(NotImplementedError):
            kg(torch.rand(1))
        kg_vals = kg(None)
        self.assertEqual(kg_vals.shape, torch.Size([self.model.num_alternatives]))

        # with current_value
        kg.current_value = torch.tensor(0.0)
        kg_vals = kg(None)
        self.assertEqual(kg_vals.shape, torch.Size([self.model.num_alternatives]))
