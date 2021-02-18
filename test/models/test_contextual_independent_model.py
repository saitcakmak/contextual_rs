import torch
from contextual_rs.models.contextual_independent_model import ContextualIndependentModel
from test.utils import BotorchTestCase


class TestContextualIndependentModel(BotorchTestCase):
    def test_contextual_independent_model(self):
        for dtype in self.dtype_list:
            ckwargs = {"dtype": dtype}
            # testing with 3 arms and 2 contexts
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
            context_map = torch.rand(n_arms, n_contexts, **ckwargs)
            model = ContextualIndependentModel(train_X, train_Y, context_map)

            # check arms and contexts identified correctly
            self.assertTrue(torch.equal(model.arms, torch.arange(n_arms, **ckwargs)))
            self.assertTrue(
                torch.equal(model.contexts, torch.arange(n_contexts, **ckwargs))
            )

            # check that the mean and std are set correctly
            self.assertTrue(
                torch.allclose(
                    model.means,
                    train_Y.view(-1, n_observation_per)
                    .mean(dim=-1)
                    .view(n_arms, n_contexts),
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.stds,
                    train_Y.view(-1, n_observation_per)
                    .std(dim=-1)
                    .view(n_arms, n_contexts),
                )
            )

            self.assertTrue(torch.equal(model.context_map, context_map))

            # test posterior
            test_X = train_X.view(-1, n_observation_per, 2)[:, 0]
            post = model.posterior(test_X)
            self.assertTrue(torch.allclose(model.means.view(-1), post.mean))
            self.assertTrue(torch.allclose(model.vars.view(-1), post.variance))
            # with a single input
            test_X = torch.tensor([0, 0], **ckwargs)
            post = model.posterior(test_X)
            self.assertTrue(torch.allclose(model.means[0, 0], post.mean))
            self.assertTrue(torch.allclose(model.vars[0, 0], post.variance))

            # test add samples
            X = train_X
            Y = torch.rand_like(train_Y)
            model.add_samples(X, Y)
            full_Y = torch.cat(
                [train_Y.view(-1, n_observation_per), Y.view(-1, n_observation_per)],
                dim=-1,
            )
            # check that the parameters are updated correctly
            expected_mean = full_Y.mean(dim=-1).view(n_arms, n_contexts)
            expected_std = full_Y.std(dim=-1).view(n_arms, n_contexts)
            self.assertTrue(torch.allclose(model.means, expected_mean))
            self.assertTrue(torch.allclose(model.stds, expected_std))
            self.assertTrue(torch.allclose(model.vars, expected_std.pow(2)))
            self.assertEqual(
                model.train_X.shape,
                torch.Size([n_arms * n_contexts * n_observation_per * 2, 2]),
            )
            self.assertEqual(
                model.train_Y.shape,
                torch.Size([n_arms * n_contexts * n_observation_per * 2]),
            )

            # add only a few samples
            X_ = torch.tensor([[0, 0], [1, 1], [1, 1]], **ckwargs)
            Y_ = torch.rand(3, **ckwargs)
            model.add_samples(X_, Y_)
            new_Y_00 = torch.cat([full_Y[0].view(-1), Y_[0].view(-1)])
            new_Y_11 = torch.cat([full_Y[1 + n_contexts].view(-1), Y_[1:].view(-1)])
            expected_mean[0, 0] = new_Y_00.mean()
            expected_mean[1, 1] = new_Y_11.mean()
            expected_std[0, 0] = new_Y_00.std()
            expected_std[1, 1] = new_Y_11.std()
            # check that the parameters are updated correctly
            self.assertTrue(torch.allclose(model.means, expected_mean))
            self.assertTrue(torch.allclose(model.stds, expected_std))
            self.assertTrue(torch.allclose(model.vars, expected_std.pow(2)))
            self.assertEqual(
                model.train_X.shape,
                torch.Size([n_arms * n_contexts * n_observation_per * 2 + 3, 2]),
            )
            self.assertEqual(
                model.train_Y.shape,
                torch.Size([n_arms * n_contexts * n_observation_per * 2 + 3]),
            )

            # add a lot of samples at a single point
            X_ = torch.tensor([[0, 0]], **ckwargs).repeat(20, 1)
            Y_ = torch.rand(20, **ckwargs)
            model.add_samples(X_, Y_)
            new_Y_00 = torch.cat([new_Y_00, Y_])
            expected_mean[0, 0] = new_Y_00.mean()
            expected_std[0, 0] = new_Y_00.std()
            # check that the parameters are updated correctly
            self.assertTrue(torch.allclose(model.means, expected_mean))
            self.assertTrue(torch.allclose(model.stds, expected_std))
            self.assertTrue(torch.allclose(model.vars, expected_std.pow(2)))
            self.assertEqual(
                model.train_X.shape,
                torch.Size([n_arms * n_contexts * n_observation_per * 2 + 23, 2]),
            )
            self.assertEqual(
                model.train_Y.shape,
                torch.Size([n_arms * n_contexts * n_observation_per * 2 + 23]),
            )
