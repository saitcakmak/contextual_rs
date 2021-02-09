from itertools import product
from unittest import mock

import torch
from botorch.optim.fit import fit_gpytorch_scipy
from gpytorch import ExactMarginalLogLikelihood

from contextual_rs.lce_gp import LCEGP
from contextual_rs.custom_fit import custom_fit_gpytorch_model, _eval_mll
from test.utils import BotorchTestCase


class TestCustomFit(BotorchTestCase):
    def test_custom_fit(self):
        for dtype, device in product(self.dtype_list, self.device_list):
            ckwargs = {"dtype": dtype, "device": device}

            num_alternatives = 5
            num_train = 3
            train_X = torch.tensor(
                range(num_alternatives), **ckwargs
            ).repeat(num_train).view(-1, 1)
            train_Y = torch.randn_like(train_X)

            # this is identical to a simple fit
            model = LCEGP(train_X, train_Y, categorical_cols=[0])
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            pre_train_mll_value = _eval_mll(mll)
            fitted_mll = custom_fit_gpytorch_model(mll)
            post_train_mll_value = _eval_mll(fitted_mll)
            self.assertGreater(post_train_mll_value, pre_train_mll_value)

            # ensuring that it is called as many times as specified
            mock_optimizer = mock.create_autospec(
                fit_gpytorch_scipy, return_value=(mll, None)
            )
            pre_train_mll_value = _eval_mll(mll)
            num_tries = 5
            fitted_mll = custom_fit_gpytorch_model(
                mll, optimizer=mock_optimizer, num_retries=num_tries
            )
            self.assertEqual(mock_optimizer.call_count, num_tries)
            # fitted mll should be identical to the original mll here
            post_train_mll_value = _eval_mll(fitted_mll)
            self.assertTrue(torch.allclose(pre_train_mll_value, post_train_mll_value))
