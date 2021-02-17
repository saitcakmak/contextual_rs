from itertools import product
from unittest import mock

import torch
from botorch.optim.fit import fit_gpytorch_scipy
from gpytorch import ExactMarginalLogLikelihood

from contextual_rs.lce_gp import LCEGP
from contextual_rs.custom_fit import custom_fit_gpytorch_model
from test.utils import BotorchTestCase


class TestCustomFit(BotorchTestCase):
    def test_custom_fit(self):
        # This doesn't do much, but custom fit has been extensively tested offline.
        for dtype, device in product(self.dtype_list, self.device_list):
            ckwargs = {"dtype": dtype, "device": device}

            num_alternatives = 5
            num_train = 3
            train_X = (
                torch.tensor(range(num_alternatives), **ckwargs)
                .repeat(num_train)
                .view(-1, 1)
            )
            train_Y = torch.randn_like(train_X)

            # just a simple full run
            model = LCEGP(train_X, train_Y, categorical_cols=[0])
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            custom_fit_gpytorch_model(mll, num_retries=3)

            # full run with alternative fit, i.e. passing non-zero randn_factor
            custom_fit_gpytorch_model(mll, num_retries=2, randn_factor=0.5)

            # ensuring that it is called as many times as specified
            mock_optimizer = mock.create_autospec(
                fit_gpytorch_scipy, return_value=(mll, {"fopt": 0.0})
            )
            num_tries = 5
            fitted_mll = custom_fit_gpytorch_model(
                mll, optimizer=mock_optimizer, num_retries=num_tries
            )
            self.assertEqual(mock_optimizer.call_count, num_tries)

            # multi-run with embedding initialization
            mock_optimizer.reset_mock()
            num_tries = 3
            fitted_mll = custom_fit_gpytorch_model(
                mll,
                optimizer=mock_optimizer,
                num_retries=num_tries,
                randn_factor=0,
            )
            self.assertEqual(mock_optimizer.call_count, num_tries)

            # test NaN return values
            mock_optimizer = mock.create_autospec(
                fit_gpytorch_scipy, return_value=(mll, {"fopt": float("nan")})
            )
            num_tries = 3
            # will raise Attribute error since it gets NaN all the time.
            with self.assertRaises(AttributeError):
                fitted_mll = custom_fit_gpytorch_model(
                    mll,
                    optimizer=mock_optimizer,
                    num_retries=num_tries,
                    max_error_tries=5,
                )
            self.assertEqual(mock_optimizer.call_count, num_tries + 5)
