import torch

from contextual_rs.test_functions.esophageal_cancer import EsophagealCancer

from test.utils import BotorchTestCase


class TestEsophagealCancer(BotorchTestCase):
    def test_esophageal_cancer(self):
        torch.manual_seed(1)
        with self.assertRaises(NotImplementedError):
            EsophagealCancer(num_replications=100000, use_cv=True)
        # Comparing against the output of the MATLAB simulator.
        sim = EsophagealCancer(num_replications=100000, use_cv=False)
        X = torch.tensor([[0, 55, 0.08, 0.4, 0.2]])
        self.assertTrue(
            torch.allclose(sim(X), torch.tensor([22.23]), atol=0.05)
        )
        X = torch.tensor([[0, 65, 0.08, 0.6, 0.4]])
        self.assertTrue(
            torch.allclose(sim(X), torch.tensor([15.91]), atol=0.05)
        )
        X = torch.tensor([[1, 65, 0.08, 0.6, 0.4]])
        self.assertTrue(
            torch.allclose(sim(X), torch.tensor([16.76]), atol=0.05)
        )
        X = torch.tensor([[2, 65, 0.08, 0.6, 0.4]])
        self.assertTrue(
            torch.allclose(sim(X), torch.tensor([16.45]), atol=0.05)
        )


