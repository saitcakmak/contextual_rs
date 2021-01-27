import torch
from contextual_rs.independent_gp import IndependentGP
from test.utils import BotorchTestCase


class TestIndependentGP(BotorchTestCase):
    def test_independent_gp(self):
        # Note: this also tests RSBaseModel
        for dtype in (torch.float, torch.double):
            ckwargs = {"dtype": dtype}
            # testing with 4 categories.
            y_0 = torch.rand(3, **ckwargs) + 1
            y_1 = torch.rand(5, **ckwargs) + 3
            y_2 = torch.rand(2, **ckwargs)
            y_3 = torch.rand(1, **ckwargs) + 5
            Y = torch.cat([y_0, y_1, y_2, y_3], dim=-1)
            X = torch.rand(3, **ckwargs)
            # test shape mismatch
            with self.assertRaises(ValueError):
                model = IndependentGP(X, Y)
            # test batch error
            Y_ = Y.unsqueeze(-1)
            X = torch.rand_like(Y_)
            with self.assertRaises(NotImplementedError):
                model = IndependentGP(X, Y_)
            # test non-integer error
            X = torch.rand_like(Y)
            with self.assertRaises(ValueError):
                model = IndependentGP(X, Y)
            # test non 0, ..., n_alternatives - 1 inputs
            X = torch.ones_like(Y)
            with self.assertRaises(ValueError):
                model = IndependentGP(X, Y)
            # test expected at least 2 of each error
            X = torch.tensor([0] * 3 + [1] * 5 + [2] * 2 + [3] * 1, **ckwargs)
            with self.assertRaises(ValueError):
                model = IndependentGP(X, Y)
            # proper inputs from here on
            y_3 = torch.rand(3, **ckwargs) + 5
            Y = torch.cat([y_0, y_1, y_2, y_3], dim=-1)
            X = torch.tensor([0] * 3 + [1] * 5 + [2] * 2 + [3] * 3, **ckwargs)
            model = IndependentGP(X, Y)
            self.assertEqual(model.num_alternatives, 4)
            self.assertTrue(torch.equal(model.alternatives, torch.arange(4, **ckwargs)))
            y_list = [y_0, y_1, y_2, y_3]
            for i in range(4):
                self.assertTrue(
                    torch.equal(model.alternative_observations[i], y_list[i])
                )
                self.assertEqual(model.means[i], y_list[i].mean())
                self.assertEqual(model.stds[i], y_list[i].std())

            # testing posterior
            test_X = torch.tensor([5], **ckwargs)
            with self.assertRaises(IndexError):
                model.posterior(test_X)
            test_X += 0.1
            with self.assertRaises(ValueError):
                model.posterior(test_X)
            # single proper input
            test_X = torch.tensor([3], **ckwargs)
            post = model.posterior(test_X)
            self.assertEqual(post.mean, model.means[3])
            self.assertEqual(post.covariance_matrix, model.vars[3])

            # multiple inputs
            test_X = torch.tensor([2, 3], **ckwargs)
            post = model.posterior(test_X)
            self.assertTrue(torch.equal(post.mean, model.means[[2, 3]]))
            self.assertTrue(
                torch.equal(post.covariance_matrix.diag(), model.vars[[2, 3]])
            )

            # batch inputs
            idcs = [[2, 3], [0, 1]]
            test_X = torch.tensor(idcs, **ckwargs)
            post = model.posterior(test_X)
            self.assertTrue(torch.equal(post.mean[0], model.means[idcs[0]]))
            self.assertTrue(
                torch.equal(post.covariance_matrix[0].diag(), model.vars[idcs[0]])
            )
            self.assertTrue(torch.equal(post.mean[1], model.means[idcs[1]]))
            self.assertTrue(
                torch.equal(post.covariance_matrix[1].diag(), model.vars[idcs[1]])
            )

            # fantasize
            with self.assertRaises(NotImplementedError):
                model.fantasize(test_X, lambda X: X)
