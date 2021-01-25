import gpytorch
import torch
from botorch import fit_gpytorch_model
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling import IIDNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from contextual_rs.lce_gp import LCEGP
from test.utils import BotorchTestCase


class TestLCEGP(BotorchTestCase):
    def test_constructor(self):
        for dtype in (torch.float, torch.double):
            ckwargs = {"dtype": dtype}
            # test input checks with categorical_cols
            with self.assertRaises(RuntimeError):
                _ = LCEGP(None, None, [])
            with self.assertRaises(ValueError):
                _ = LCEGP(None, None, [1], [1, 2])

            # test with 1D categorical input
            num_train = 10
            dim_cont = 2

            # test non-integer categorical inputs
            with self.assertRaises(ValueError):
                _ = LCEGP(
                    train_X=torch.rand(num_train, dim_cont, **ckwargs),
                    train_Y=torch.randn(num_train, 1, **ckwargs),
                    categorical_cols=[-1],
                )
            # test missing / improper categorical inputs
            train_X = torch.cat(
                [
                    torch.rand(num_train, dim_cont, **ckwargs),
                    torch.ones(num_train, 1, **ckwargs),
                ],
                dim=-1,
            )
            with self.assertRaises(ValueError):
                _ = LCEGP(
                    train_X=train_X,
                    train_Y=torch.randn(num_train, 1, **ckwargs),
                    categorical_cols=[-1],
                )

            # test a simple full construction with defaults
            train_X = torch.cat(
                [
                    torch.rand(num_train, dim_cont, **ckwargs),
                    torch.arange(0, num_train).unsqueeze(-1).to(**ckwargs),
                ],
                dim=-1,
            )
            train_Y = torch.randn(num_train, 1, **ckwargs)
            model = LCEGP(
                train_X=train_X,
                train_Y=train_Y,
                categorical_cols=[-1],
            )
            # test that everything is set properly
            self.assertEqual(model.categorical_cols, [2])
            self.assertEqual(model.continuous_cols, [0, 1])
            self.assertIsInstance(
                model.likelihood, gpytorch.likelihoods.GaussianLikelihood
            )
            self.assertFalse(hasattr(model, "input_transform"))
            self.assertFalse(hasattr(model, "outcome_transform"))
            self.assertTrue(torch.equal(model.train_inputs[0], train_X))
            self.assertTrue(
                torch.equal(model.train_targets.flatten(), train_Y.flatten())
            )
            self.assertEqual(model.emb_dims, [(10, 1)])
            self.assertEqual(model.category_counts, [10])
            self.assertEqual(model.emb_covar_module.ard_num_dims, 1)

            # test input / outcome transforms and embs_dim_list
            train_X[..., [0, 1]] += 5
            train_Y = train_Y * 10 + 5
            model = LCEGP(
                train_X=train_X,
                train_Y=train_Y,
                categorical_cols=[-1],
                embs_dim_list=[3],
                outcome_transform=Standardize(m=1),
                input_transform=Normalize(d=2),
            )
            self.assertEqual(model.emb_dims, [(10, 3)])
            self.assertEqual(model.emb_covar_module.ard_num_dims, 3)
            self.assertTrue(
                torch.allclose(
                    model.outcome_transform.means,
                    torch.tensor(5.0, **ckwargs),
                    rtol=2.0,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.outcome_transform.stdvs,
                    torch.tensor(10.0, **ckwargs),
                    rtol=4.0,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.input_transform.mins, torch.tensor(5.0, **ckwargs), rtol=2.0
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.input_transform.ranges,
                    torch.tensor(10.0, **ckwargs),
                    rtol=4.0,
                )
            )

            # test with multiple categorical columns
            train_X = torch.cat(
                [
                    torch.rand(num_train, dim_cont, **ckwargs),
                    torch.arange(0, num_train).unsqueeze(-1).to(**ckwargs),
                    torch.tensor([0, 1, 2, 3, 4], **ckwargs).repeat(2).unsqueeze(-1),
                    torch.tensor([0, 1, 2], **ckwargs)
                    .repeat(4)[:num_train]
                    .unsqueeze(-1),
                ],
                dim=-1,
            )
            train_Y = torch.randn(num_train, 1)
            model = LCEGP(
                train_X=train_X,
                train_Y=train_Y,
                categorical_cols=[-3, -2, -1],
                embs_dim_list=[3, 2, 1],
            )
            self.assertEqual(model.emb_dims, [(10, 3), (5, 2), (3, 1)])
            self.assertEqual(model.emb_covar_module.ard_num_dims, 6)
            self.assertEqual(model.categorical_cols, [2, 3, 4])

            # TODO: batch training inputs?

    def _get_sample_model(self, **ckwargs) -> LCEGP:
        num_train = 12
        dim_cont = 3
        train_X = torch.cat(
            [
                torch.rand(num_train, dim_cont, **ckwargs),
                torch.arange(0, num_train).unsqueeze(-1).to(**ckwargs),
                torch.tensor([0, 1, 2, 3, 4, 5], **ckwargs).repeat(2).unsqueeze(-1),
                torch.tensor([0, 1, 2], **ckwargs).repeat(5)[:num_train].unsqueeze(-1),
            ],
            dim=-1,
        )
        train_Y = torch.randn(num_train, 1)
        model = LCEGP(
            train_X=train_X,
            train_Y=train_Y,
            categorical_cols=[-3, -2, -1],
            embs_dim_list=[3, 2, 1],
        )
        return model

    def test_forward(self):
        ckwargs = {"dtype": torch.double}
        model = self._get_sample_model(**ckwargs)
        dim = 6
        num_test = 2
        test_x = torch.rand(num_test, dim, **ckwargs)
        with self.assertRaises(ValueError):
            model.forward(test_x)

        test_x[:, -3:] = 0
        prior = model.forward(test_x)
        self.assertEqual(prior.mean.shape, torch.Size([num_test]))
        self.assertEqual(
            prior.lazy_covariance_matrix.shape, torch.Size([num_test, num_test])
        )

        # batch evaluation
        batch_shape = [5, 3]
        test_x = test_x.expand(*batch_shape, -1, -1)
        prior = model.forward(test_x)
        self.assertEqual(prior.mean.shape, torch.Size([*batch_shape, num_test]))
        self.assertEqual(
            prior.lazy_covariance_matrix.shape,
            torch.Size([*batch_shape, num_test, num_test]),
        )

    def test_posterior(self):
        # TODO: maybe a good idea to add tests verifying output values
        ckwargs = {"dtype": torch.double}
        model = self._get_sample_model(**ckwargs)
        dim = 6
        num_test = 4
        test_x = torch.rand(num_test, dim, **ckwargs)
        test_x[:, -3:] = torch.tensor([0.0, 1.0, 2.0], **ckwargs).repeat(num_test, 1)
        post = model.posterior(test_x)
        self.assertEqual(post.mean.shape, torch.Size([num_test, 1]))
        self.assertEqual(post.variance.shape, torch.Size([num_test, 1]))
        # with batch input
        batch_shape = [3, 6, 2]
        test_x = test_x.expand(*batch_shape, -1, -1)
        post = model.posterior(test_x)
        self.assertEqual(post.mean.shape, torch.Size([*batch_shape, num_test, 1]))
        self.assertEqual(post.variance.shape, torch.Size([*batch_shape, num_test, 1]))

    def test_fantasize(self):
        ckwargs = {"dtype": torch.double}
        model = self._get_sample_model(**ckwargs)
        dim = 6
        q = 2
        fant_x = torch.rand(q, dim, **ckwargs)
        fant_x[:, -3:] = torch.tensor([0.0, 1.0, 0.0], **ckwargs).repeat(q, 1)
        n_f = 3
        fm = model.fantasize(X=fant_x, sampler=IIDNormalSampler(n_f))
        self.assertEqual(fm.train_inputs[0].shape, torch.Size([n_f, 14, dim]))
        self.assertEqual(fm.train_targets.shape, torch.Size([n_f, 14]))
        num_test = 4
        test_x = torch.rand(num_test, dim, **ckwargs)
        test_x[..., -3:] = torch.tensor([2.0, 1.0, 2.0], **ckwargs).repeat(num_test, 1)
        post = fm.posterior(test_x)
        self.assertEqual(post.mean.shape, torch.Size([n_f, num_test, 1]))
        self.assertEqual(post.variance.shape, torch.Size([n_f, num_test, 1]))

        # fantasize on batch candidates
        fant_batch_size = 5
        fant_x = fant_x.repeat(fant_batch_size, 1, 1)
        fant_x[..., :-3] += torch.randn(*fant_x.shape[:-1], 3, **ckwargs) * 0.1
        fm = model.fantasize(X=fant_x, sampler=IIDNormalSampler(n_f))
        fm_batch = [n_f, fant_batch_size]
        self.assertEqual(fm.train_inputs[0].shape, torch.Size([*fm_batch, 14, dim]))
        self.assertEqual(fm.train_targets.shape, torch.Size([*fm_batch, 14]))
        num_test = 4
        test_x = torch.cat(
            [
                torch.rand(*fm_batch, num_test, 3, **ckwargs),
                torch.randint(0, 3, (*fm_batch, num_test, 3), **ckwargs),
            ],
            dim=-1,
        )
        post = fm.posterior(test_x)
        self.assertEqual(post.mean.shape, torch.Size([*fm_batch, num_test, 1]))
        self.assertEqual(post.variance.shape, torch.Size([*fm_batch, num_test, 1]))

    def test_only_categorical_inputs(self):
        # testing the use case with purely categorical inputs
        for dim, dtype in [(1, torch.float), (3, torch.double)]:
            ckwargs = {"dtype": dtype}
            num_train = 20
            train_X = torch.randint(0, 4, size=(num_train, dim), **ckwargs)
            train_X[:4, :] = torch.arange(0, 4).unsqueeze(-1).expand(-1, dim)
            train_Y = torch.randn(num_train, 1, **ckwargs)
            model = LCEGP(
                train_X=train_X,
                train_Y=train_Y,
                categorical_cols=list(range(dim)),
                embs_dim_list=list(range(1, dim + 1)),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # test that the modules are setup correctly
            self.assertFalse(model.has_continuous_cols)
            self.assertEqual(model.emb_dims, [(4, i) for i in range(1, dim + 1)])
            self.assertEqual(model.emb_covar_module.ard_num_dims, sum(range(dim + 1)))

            # continuous inputs should fail
            test_x = torch.rand(3, dim, **ckwargs)
            with self.assertRaises(ValueError):
                model.forward(test_x)

            # batch forward evaluation
            batch_shape = [5, 3]
            num_test = 2
            test_x = torch.randint(0, 4, size=(*batch_shape, num_test, dim), **ckwargs)
            prior = model.forward(test_x)
            self.assertEqual(prior.mean.shape, torch.Size([*batch_shape, num_test]))
            self.assertEqual(
                prior.lazy_covariance_matrix.shape,
                torch.Size([*batch_shape, num_test, num_test]),
            )

            # posterior evaluation
            post = model.posterior(test_x)
            self.assertEqual(post.mean.shape, torch.Size([*batch_shape, num_test, 1]))
            self.assertEqual(post.variance.shape, torch.Size([*batch_shape, num_test, 1]))

            # fantasize
            fant_batch_size = 5
            q = 2
            fant_x = torch.randint(0, 4, size=(fant_batch_size, q, dim), **ckwargs)
            n_f = 7
            fm = model.fantasize(X=fant_x, sampler=IIDNormalSampler(n_f))
            fm_batch = [n_f, fant_batch_size]
            self.assertEqual(fm.train_inputs[0].shape, torch.Size([*fm_batch, 22, dim]))
            self.assertEqual(fm.train_targets.shape, torch.Size([*fm_batch, 22]))
            num_test = 4
            test_x = torch.randint(0, 4, size=(*fm_batch, num_test, dim), **ckwargs)
            post = fm.posterior(test_x)
            self.assertEqual(post.mean.shape, torch.Size([*fm_batch, num_test, 1]))
            self.assertEqual(post.variance.shape, torch.Size([*fm_batch, num_test, 1]))
