import torch
from contextual_rs.models.independent_gp import IndependentGP
from test.utils import BotorchTestCase


class TestIndependentGP(BotorchTestCase):
    def test_independent_gp(self):
        # Note: this also tests RSBaseModel
        for dtype in self.dtype_list:
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
            self.assertEqual(
                post.covariance_matrix, model.vars[3] / model.num_observations[3]
            )

            # multiple inputs
            test_X = torch.tensor([2, 3], **ckwargs)
            post = model.posterior(test_X)
            self.assertTrue(torch.equal(post.mean, model.means[[2, 3]]))
            self.assertTrue(
                torch.equal(
                    post.covariance_matrix.diag(),
                    model.vars[[2, 3]] / model.num_observations[[2, 3]],
                )
            )

            # batch inputs
            idcs = [[2, 3], [0, 1]]
            test_X = torch.tensor(idcs, **ckwargs)
            post = model.posterior(test_X)
            self.assertTrue(torch.equal(post.mean[0], model.means[idcs[0]]))
            self.assertTrue(
                torch.equal(
                    post.covariance_matrix[0].diag(),
                    model.vars[idcs[0]] / model.num_observations[idcs[0]],
                )
            )
            self.assertTrue(torch.equal(post.mean[1], model.means[idcs[1]]))
            self.assertTrue(
                torch.equal(
                    post.covariance_matrix[1].diag(),
                    model.vars[idcs[1]] / model.num_observations[idcs[1]],
                )
            )

    def test_add_new_observations(self):
        for dtype in self.dtype_list:
            ckwargs = {"dtype": dtype}
            # testing with 4 categories.
            y_0 = torch.rand(3, **ckwargs) + 1
            y_1 = torch.rand(5, **ckwargs) + 3
            y_2 = torch.rand(2, **ckwargs)
            y_3 = torch.rand(3, **ckwargs) + 5
            y_list = [y_0, y_1, y_2, y_3]
            Y = torch.cat(y_list, dim=-1)
            X = torch.tensor([0] * 3 + [1] * 5 + [2] * 2 + [3] * 3, **ckwargs)
            model = IndependentGP(X, Y)

            # check the original mean and std
            self.assertTrue(
                torch.equal(model.means, torch.cat([y.mean().view(-1) for y in y_list]))
            )
            self.assertTrue(
                torch.equal(model.stds, torch.cat([y.std().view(-1) for y in y_list]))
            )

            # add new data, single point at first
            model.add_samples(
                torch.tensor([0.0], **ckwargs), torch.tensor([0.5], **ckwargs)
            )
            y_0 = torch.cat([y_0, torch.tensor([0.5], **ckwargs)])
            y_list = [y_0, y_1, y_2, y_3]
            # check the mean and std
            self.assertTrue(
                torch.equal(model.means, torch.cat([y.mean().view(-1) for y in y_list]))
            )
            self.assertTrue(
                torch.equal(model.stds, torch.cat([y.std().view(-1) for y in y_list]))
            )

            # add three points
            y_1_new = torch.rand(1, **ckwargs)
            y_2_new = torch.rand(2, **ckwargs)
            model.add_samples(
                torch.tensor([1, 2, 2], **ckwargs), torch.cat([y_1_new, y_2_new])
            )
            y_1 = torch.cat([y_1, y_1_new])
            y_2 = torch.cat([y_2, y_2_new])
            y_list = [y_0, y_1, y_2, y_3]
            # check the mean and std
            self.assertTrue(
                torch.equal(model.means, torch.cat([y.mean().view(-1) for y in y_list]))
            )
            self.assertTrue(
                torch.equal(model.stds, torch.cat([y.std().view(-1) for y in y_list]))
            )

            # add a large number of new samples
            # this should trigger the other side of the if/else block
            y_1_new = torch.rand(10, **ckwargs)
            y_2_new = torch.rand(20, **ckwargs)
            model.add_samples(
                torch.tensor([1] * 10 + [2] * 20, **ckwargs),
                torch.cat([y_1_new, y_2_new]),
            )
            y_1 = torch.cat([y_1, y_1_new])
            y_2 = torch.cat([y_2, y_2_new])
            y_list = [y_0, y_1, y_2, y_3]
            # check the mean and std
            self.assertTrue(
                torch.equal(model.means, torch.cat([y.mean().view(-1) for y in y_list]))
            )
            self.assertTrue(
                torch.equal(model.stds, torch.cat([y.std().view(-1) for y in y_list]))
            )

            # check that train_X and train_Y were properly updated
            # both should have 47 entries
            self.assertEqual(model.train_X.shape, torch.Size([47]))
            self.assertEqual(model.train_Y.shape, torch.Size([47]))

            # test error checks
            with self.assertRaisesRegex(AssertionError, "X and Y"):
                model.add_samples(torch.rand(2, **ckwargs), torch.rand(1, **ckwargs))
            with self.assertRaisesRegex(AssertionError, "one-dimensional"):
                model.add_samples(
                    torch.ones(5, 2, **ckwargs), torch.ones(5, 2, **ckwargs)
                )
            with self.assertRaisesRegex(ValueError, "must be integers"):
                model.add_samples(torch.rand(3, **ckwargs), torch.rand(3, **ckwargs))
