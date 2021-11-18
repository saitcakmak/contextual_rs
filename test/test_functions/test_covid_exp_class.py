import torch

from contextual_rs.test_functions.covid_exp_class import CovidSim, CovidEval, CovidSimV2, CovidEvalV2

from test.utils import BotorchTestCase


class TestCovid(BotorchTestCase):
    def test_covid_sim(self):
        sim = CovidSim()
        X = (
            torch.tensor([[0.2, 0.2, 0.0040, 0.0040, 0.0080]])
            .reshape(1, 1, -1)
        )
        self.assertTrue(sim(X).shape == torch.Size([1, 1, 1]))
        X = X.repeat(4, 1, 1)
        self.assertTrue(sim(X).shape == torch.Size([4, 1, 1]))

    def test_covid_eval(self):
        sim = CovidEval()
        X = (
            torch.tensor([[0.3, 0.2, 0.0040, 0.0040, 0.0080]])
            .reshape(1, 1, -1)
            .repeat(4, 1, 1)
        )
        self.assertTrue(sim(X).shape == torch.Size([4, 1, 1]))

    def test_v2(self):
        sim = CovidSimV2()
        X = (
            torch.tensor([[0.2, 0.2, 0.02, 6.0]])
            .reshape(1, 1, -1)
        )
        self.assertTrue(sim(X).shape == torch.Size([1, 1, 1]))
        X = X.repeat(4, 1, 1)
        self.assertTrue(sim(X).shape == torch.Size([4, 1, 1]))
        self.assertTrue(sim.stored_forward(X).shape == torch.Size([4, 1, 1]))
        sim.alpha = 0.5
        self.assertTrue(sim.stored_forward(X).shape == torch.Size([4, 1, 1]))

        sim = CovidEvalV2()
        X = (
            torch.tensor([[0.3, 0.2, 0.02, 6.0]])
            .reshape(1, 1, -1)
            .repeat(4, 1, 1)
        )
        self.assertTrue(sim(X).shape == torch.Size([4, 1, 1]))
        self.assertTrue(sim.stored_forward(X).shape == torch.Size([4, 1, 1]))
