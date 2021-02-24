import torch
from torch import Tensor
from torch.nn import Module


class RSBaseModel(Module):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
    ) -> None:
        r"""
        Base model for Ranking & Selection type statistical models.

        Contains the common initialization and input checks.

        TODO: Batch inputs are currently not supported. Thinking of a good
            way of handling batches without requiring nested lists / for loops.

        Args:
            train_X: A `n`-dim tensor of training inputs.
                The inputs are expected to be integer points in
                0, ..., n_alternatives - 1, with each alternative appearing
                in each batch of the training data at least twice.
            train_Y: A `n`-dim tensor of training targets.

        # TODO: is an explicit output dimension desirable for any reason?
        """
        if train_X.shape != train_Y.shape:
            raise ValueError("Shapes of `train_X` and `train_Y` do not match!")
        if train_X.dim() > 1:
            raise NotImplementedError("Batch support is not available yet.")
        if torch.any(train_X != train_X.long()):
            raise ValueError(f"Expected integer inputs, got {train_X}!")
        super().__init__()
        self.train_X = train_X
        self.train_Y = train_Y
        device = train_X.device
        dtype = train_X.dtype
        self.alternatives = train_X.unique(sorted=True)
        self.num_alternatives = self.alternatives.shape[-1]
        if torch.any(
            self.alternatives != torch.arange(len(self.alternatives), device=device)
        ):
            raise ValueError(
                "Expected the training input to include all values from"
                f"0, ..., n_alternatives - 1 at least once. Got {self.alternatives}."
            )
        # TODO: may want to improve this later
        #   This current treatment does not work with batches.
        #   I'd prefer not to use nested lists just to handle batches
        #   even this current use of list is sub-optimal
        # `alternative_observations` is a list of observations from the each
        # alternative. For example, `alternative_observations[0]` is a tensor
        # collecting the observations from alternative 0.
        self.alternative_observations = list()
        self.means = torch.zeros(self.num_alternatives, device=device, dtype=dtype)
        self.stds = torch.zeros_like(self.means)
        self.num_observations = torch.zeros_like(self.means)
        for i in range(self.num_alternatives):
            observations = train_Y[train_X == i]
            self.alternative_observations.append(observations)
            self.means[i] = observations.mean()
            std = observations.std()
            if std.isnan():
                raise ValueError(
                    "Expected to see at least two observations from each alternative."
                    f"Got only 1 observation for alternative {i}."
                )
            self.stds[i] = std
            self.num_observations[i] = observations.shape[0]
