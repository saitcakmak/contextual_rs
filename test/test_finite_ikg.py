"""
This is some simple test cases for debugging etc.
More comprehensive test cases are needed.
"""
from itertools import product

import torch
from botorch.models import ModelListGP, SingleTaskGP

from contextual_rs.models.lce_gp import LCEGP
from contextual_rs.finite_ikg import finite_ikg_maximizer, finite_ikg_maximizer_modellist
from test.utils import BotorchTestCase


class TestFiniteIKG(BotorchTestCase):
    def test_finite_ikg(self):
        for dtype, device in product(self.dtype_list, self.device_list):
            ckwargs = {"dtype": dtype, "device": device}
            num_arms = 3
            arm_set = torch.arange(0, num_arms, **ckwargs).view(-1, 1)
            num_contexts = 4
            context_set = torch.rand(num_contexts, 2, **ckwargs)
            dim = 3
            arm_context_pairs = torch.cat(
                [
                    arm_set.view(-1, 1, 1).repeat(1, num_contexts, 1),
                    context_set.repeat(num_arms, 1, 1),
                ],
                dim=-1,
            )
            train_X = arm_context_pairs.view(-1, dim).repeat(3, 1)
            train_Y = torch.randn(train_X.shape[0], 1, **ckwargs)
            model = LCEGP(train_X, train_Y, [0])
            finite_ikg_maximizer(model, arm_set, context_set)

    def test_finite_ikg_modellist(self):
        for dtype, device in product(self.dtype_list, self.device_list):
            ckwargs = {"dtype": dtype, "device": device}
            num_arms = 3
            arm_set = torch.arange(0, num_arms, **ckwargs).view(-1, 1)
            num_contexts = 4
            context_set = torch.rand(num_contexts, 2, **ckwargs)

            model = ModelListGP(
                *[
                    SingleTaskGP(
                        torch.rand(10, 2, **ckwargs),
                        torch.randn(10, 1, **ckwargs)
                    )
                    for _ in range(num_arms)
                ]
            )

            finite_ikg_maximizer_modellist(model, context_set)