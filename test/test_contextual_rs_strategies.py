from itertools import product

import torch

from contextual_rs.models.contextual_independent_model import ContextualIndependentModel
from contextual_rs.contextual_rs_strategies import (
    li_sampling_strategy,
    gao_sampling_strategy,
)

from test.utils import BotorchTestCase


class TestContextualRSStrategies(BotorchTestCase):
    def test_sampling_strategy(self, sampling_strategy=li_sampling_strategy):
        for dtype, device in product(self.dtype_list, self.device_list):
            torch.manual_seed(5)
            ckwargs = {"dtype": dtype, "device": device}
            # generate a sample model
            n_arms = 3
            n_contexts = 2
            n_observation_per = 4
            # this generates a (n_arms * n_contexts * n_observation_per) x 2
            # tensor of input, which can be reshaped into
            # (n_arms * n_contexts) x n_observations_per x 2 to get the sample statistics
            train_X = (
                torch.cat(
                    [
                        torch.arange(n_arms).view(-1, 1, 1).repeat(1, n_contexts, 1),
                        torch.arange(n_contexts).view(1, -1, 1).repeat(n_arms, 1, 1),
                    ],
                    dim=-1,
                )
                .repeat(1, 1, n_observation_per)
                .reshape(-1, 2)
                .to(**ckwargs)
            )
            train_Y = torch.rand(train_X.shape[0], **ckwargs)
            model = ContextualIndependentModel(train_X, train_Y)

            # test that it works as expected
            next_arm, next_context = sampling_strategy(model)
            self.assertTrue(next_arm < n_arms)
            self.assertTrue(next_context < n_contexts)

            # tests to verify the logic
            # if a arm-context pair has too much uncertainty, it should get selected
            train_Y[-4:] = torch.tensor([0, 1000, -1000, 1], **ckwargs)
            model = ContextualIndependentModel(train_X, train_Y)
            next_arm, next_context = sampling_strategy(model)
            self.assertEqual(next_arm, 2)
            self.assertEqual(next_context, 1)

    def test_gao_sampling_strategy(self):
        self.test_sampling_strategy(gao_sampling_strategy)