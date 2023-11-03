"""Parent class for datasets."""

import abc

from torch import utils


class Dataset(utils.data.Dataset, abc.ABC):
    """Custom dataset class.

    Args:
        dataset: List of data samples or paths to INRs.
        is_training: Bool, whether this generator is for training or not.
    """

    def __init__(self, dataset, is_training):
        self._is_training = is_training
        self._dataset = dataset

    @abc.abstractmethod
    def __getitem__(self, idx):
        """__getitem__ method of torch.utils.data.Dataset"""

    @abc.abstractmethod
    def __len__(self):
        """__len__ method of torch.utils.data.Dataset"""
