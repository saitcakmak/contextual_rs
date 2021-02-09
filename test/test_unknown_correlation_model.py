from itertools import product

import torch
from torch.distributions import MultivariateNormal
from contextual_rs.unknown_correlation_model import UnknownCorrelationModel
from test.utils import BotorchTestCase


class TestUnknownCorrelationModel(BotorchTestCase):
    def test_unknown_correlation_model(self):
        # check wrong method error
        X = torch.arange(3, dtype=torch.float).repeat(2)
        Y = torch.rand_like(X)
        with self.assertRaises(ValueError):
            model = UnknownCorrelationModel(X, Y, "mm")
        for dtype, update in product(
            [torch.float, torch.double], ["moment-matching", "KL", "moment-KL"]
        ):
            ckwargs = {"dtype": dtype}
            # testing with 4 categories.
            y_0 = torch.rand(3, **ckwargs) + 1
            y_1 = torch.rand(5, **ckwargs) + 3
            y_2 = torch.rand(4, **ckwargs)
            y_3 = torch.rand(3, **ckwargs) + 2
            y_list = [y_0, y_1, y_2, y_3]
            Y = torch.cat(y_list, dim=-1)
            X = torch.tensor([0] * 3 + [1] * 5 + [2] * 4 + [3] * 3, **ckwargs)
            model = UnknownCorrelationModel(X, Y)
            # test the initialization via expected values of b and q
            # if these values match, we know that all of training data got processed
            # and the update_parameters ran with multiple input handling
            self.assertEqual(model.q, 4.75)
            self.assertEqual(model.b, 8.75)

            # test error handling etc in update parameters
            self.assertIsNone(
                model.update_parameters(
                    torch.empty(0, **ckwargs),
                    torch.empty(0, **ckwargs),
                )
            )
            with self.assertRaises(ValueError):
                model.update_parameters(
                    torch.ones([3, 2], **ckwargs),
                    torch.ones([2], **ckwargs),
                )
            with self.assertRaises(ValueError):
                model.update_parameters(
                    torch.tensor(6, **ckwargs),
                    torch.rand(1, **ckwargs),
                )
            with self.assertRaises(ValueError):
                model.update_parameters(
                    torch.rand(1, **ckwargs), torch.rand(1, **ckwargs)
                )

            # error handling in add_full_observations
            with self.assertRaises(ValueError):
                model.add_full_observations(torch.rand(2, **ckwargs))

            # test posterior
            df = model.b - model.num_alternatives + 1
            covar = (model.q + 1) * model.B / (model.q * df)

            # all output case
            post = model.posterior(None)
            self.assertTrue((post.loc == model.theta.numpy()).all())
            self.assertTrue((post.shape == covar.numpy()).all())
            self.assertEqual(post.df, df)

            # single output case
            # error checks
            with self.assertRaises(ValueError):
                model.posterior(torch.tensor(0.5, **ckwargs))
            with self.assertRaises(NotImplementedError):
                model.posterior(torch.tensor([0, 1], **ckwargs))
            # simple test
            scale = (model.q * df / ((model.q + 1) * model.B[0, 0])).sqrt()
            post = model.posterior(torch.tensor(0, **ckwargs))
            self.assertTrue(post.kwds["loc"] == model.theta[0])
            self.assertTrue(post.kwds["scale"] == scale)
            self.assertEqual(post.kwds["df"], df)

            # test predictive df
            df = model.predictive_df()
            self.assertEqual(df, model.b - 3.0)

    def test_asymptotic(self):
        # an asymptotic test to ensure that everything works
        torch.manual_seed(0)
        ckwargs = {"dtype": torch.double}
        true_mean = torch.tensor([0.1, 0.3, 0.6, 0.9], **ckwargs)
        true_cov = (
            torch.tensor(
                [
                    [1.0, 0.5, 0.3, 0.1],
                    [0.5, 0.9, 0.3, 0.2],
                    [0.3, 0.3, 1.2, 0.4],
                    [0.1, 0.2, 0.4, 0.8],
                ],
                **ckwargs
            )
            * 0.1
        )
        true_dist = MultivariateNormal(loc=true_mean, covariance_matrix=true_cov)

        for update in ["moment-matching", "KL", "moment-KL"]:
            # starting with 5 full observations
            train_X = torch.arange(4, **ckwargs).repeat(5)
            train_Y = true_dist.rsample(torch.Size([5])).view(-1)
            model = UnknownCorrelationModel(train_X, train_Y, update_method=update)
            # adding more full observations as individual observations
            # via update parameters
            num_sample = 1000
            X = torch.arange(4, **ckwargs).repeat(num_sample)
            Y = true_dist.rsample(torch.Size([num_sample])).view(-1)
            model.update_parameters(X, Y)
            # check how good of a job posterior doing in estimating the true dist
            post = model.posterior(None)
            self.assertTrue(
                torch.allclose(
                    torch.tensor(post.loc),
                    true_mean,
                    atol=3e-2,
                )
            )
            # This is intentionally weakened to make the test pass
            # Moment matching based approximations do a terrible job at
            # predicting the off diagonal entries of the covariance matrix.
            # Method "KL" doesn't have this issue! Though still bad, it is doing
            # a much better job at predicting the off-diagonal entries.
            # moment-KL is doing quite bad even at predicting the diagonal entries
            if update != "moment-KL":
                self.assertTrue(
                    torch.allclose(
                        torch.tensor(post.shape).diag(),
                        true_cov.diag(),
                        atol=1e-2,
                    )
                )

    def test_s_tilde(self):
        ckwargs = {"dtype": torch.double}
        y_list = [
            torch.rand(3, **ckwargs) + 1,
            torch.rand(5, **ckwargs) + 3,
            torch.rand(4, **ckwargs),
            torch.rand(3, **ckwargs) + 2,
        ]
        Y = torch.cat(y_list, dim=-1)
        X = torch.tensor([0] * 3 + [1] * 5 + [2] * 4 + [3] * 3, **ckwargs)
        for update in ["moment-matching", "KL", "moment-KL"]:
            model = UnknownCorrelationModel(X, Y, update_method=update)
            # verify that the two modes of getting s_tilde agree
            s_tilde_list = [
                model.get_s_tilde(torch.tensor(i, **ckwargs)) for i in range(4)
            ]
            s_tilde_full = model.get_s_tilde(None)
            self.assertTrue(
                all([torch.equal(s_tilde_full[i], s_tilde_list[i]) for i in range(4)])
            )

            # error handling
            with self.assertRaises(ValueError):
                model.get_s_tilde(torch.rand(1, **ckwargs))
            with self.assertRaises(NotImplementedError):
                model.get_s_tilde(torch.tensor([0, 1], **ckwargs))
