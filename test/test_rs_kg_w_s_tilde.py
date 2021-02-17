"""
This has been tested in use. These are just simple tests to reinforce that.
"""
import torch
from botorch import fit_gpytorch_model
from gpytorch import ExactMarginalLogLikelihood

from contextual_rs.lce_gp import LCEGP
from contextual_rs.rs_kg_w_s_tilde import find_kg_maximizer
from contextual_rs.unknown_correlation_model import UnknownCorrelationModel
from test.utils import BotorchTestCase


class TestRSKGwSTilde(BotorchTestCase):
    def test_rs_kg_w_s_tilde(self):
        num_alternatives = 3
        train_X = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        )
        train_Y = torch.tensor(
            [0.0, 0.1, 0.05, 0.025, 0.5, 0.55, 0.6, 0.52, 2.5, 2.7, 3.0, 3.5]
        )

        # LCEGP test
        model = LCEGP(train_X.view(-1, 1), train_Y.view(-1, 1), [0])
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        maximizer = find_kg_maximizer(model)
        self.assertEqual(maximizer, 2)

        # UnknownCorrelationModel test
        for update_method in ["KL", "moment-matching"]:
            model = UnknownCorrelationModel(train_X, train_Y)
            maximizer = find_kg_maximizer(model)
            self.assertEqual(maximizer, 2)
