from unittest import TestCase
import warnings
import torch


class BotorchTestCase(TestCase):
    r"""Basic test case for Botorch.

    This
        1. sets the default device to be `torch.device("cpu")`
        2. ensures that no warnings are suppressed by default.
            Suppressing repeat warnings now!
    """

    device = torch.device("cpu")
    device_list = ["cpu"] + ["cuda"] if torch.cuda.is_available() else []
    dtype_list = [torch.float, torch.double]

    def setUp(self):
        warnings.resetwarnings()
        warnings.simplefilter("once", append=True)
        pass
