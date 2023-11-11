""""Dataset for MNIST-INRs."""

import glob
import os
from typing import NamedTuple, Tuple, Union

import flags
import torch

from absl import logging
from sklearn.model_selection import train_test_split

from loaders import base_dataset

FLAGS = flags.FLAGS


class Batch(NamedTuple):
    """Copied from DWSNet."""
    weights: Tuple
    biases: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class MnistInrDatasetFactory:
    """Create train-val-test split and return three loaders."""

    def __init__(self, dataset_path, device, val_size=0.2):
        self._device = device
        self._dataset_path = dataset_path
        self._val_size = val_size

    def split(self):

        # Get val set from train set.
        train_val_set = glob.glob(
            os.path.join(self._dataset_path, 'mnist_png_training_*/**/*.pth'))
        train_val_labels = list(
            map(int, [x.split('/')[-3].split('_')[-2] for x in train_val_set]))
        train_set, val_set, train_labels, val_labels = train_test_split(
            train_val_set,
            train_val_labels,
            test_size=self._val_size,
            random_state=FLAGS.random_seed)

        test_set = glob.glob(
            os.path.join(self._dataset_path, 'mnist_png_testing_*/**/*.pth'))
        test_labels = list(
            map(int, [x.split('/')[-3].split('_')[-2] for x in test_set]))

        # TODO(oleg): maybe load it from file just like authors.
        statistics = self._calculate_statistics(train_set, train_labels)

        train_dataset = MnistInrClassificationDataset((train_set, train_labels),
                                                      statistics=statistics,
                                                      device=self._device,
                                                      is_training=True)
        val_dataset = MnistInrClassificationDataset((val_set, val_labels),
                                                    statistics=statistics,
                                                    device=self._device,
                                                    is_training=False)
        test_dataset = MnistInrClassificationDataset((test_set, test_labels),
                                                     statistics=statistics,
                                                     device=self._device,
                                                     is_training=False)

        return train_dataset, val_dataset, test_dataset

    def _calculate_statistics(self, train_set, train_labels):

        train_dataset = MnistInrClassificationDataset((train_set, train_labels),
                                                      statistics=None,
                                                      device='cpu',
                                                      is_training=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=32,
                                                   shuffle=True,
                                                   num_workers=1)

        batch: Batch = next(iter(train_loader))
        weights_mean = [w.mean(0).to(self._device) for w in batch.weights]
        weights_std = [w.std(0).to(self._device) for w in batch.weights]
        biases_mean = [w.mean(0).to(self._device) for w in batch.biases]
        biases_std = [w.std(0).to(self._device) for w in batch.biases]

        return {
            "weights": {
                "mean": weights_mean,
                "std": weights_std
            },
            "biases": {
                "mean": biases_mean,
                "std": biases_std
            },
        }


class MnistInrClassificationDataset(base_dataset.Dataset):
    """Custom dataset for MNIST-INRs data."""

    def __init__(self, dataset, device, is_training, statistics):
        super().__init__(dataset, is_training)
        self._sample_paths, self._labels = dataset
        self._device = device
        self._statistics = statistics

        logging.info(
            f'Dataset size: {len(self._sample_paths)}, device: {device}, training: {is_training}.'
        )

    def __getitem__(self, idx):
        state_dict = torch.load(self._sample_paths[idx],
                                map_location=self._device)

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w])
        weights = tuple([w.unsqueeze(-1) for w in weights])

        biases = tuple([v for w, v in state_dict.items() if "bias" in w])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self._statistics is not None:
            weights, biases = self._normalize_weights_biases(weights, biases)

        label = torch.tensor(self._labels[idx]).to(self._device)
        sample = Batch(weights=weights, biases=biases, label=label)

        return sample

    def _normalize_weights_biases(self, weights, biases):
        wm = self._statistics["weights"]["mean"]
        ws = self._statistics["weights"]["std"]
        bm = self._statistics["biases"]["mean"]
        bs = self._statistics["biases"]["std"]

        weights = tuple((w - m) / s for w, m, s in zip(weights, wm, ws))
        biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))

        return weights, biases

    def __len__(self):
        """__len__ method of torch.utils.data.Dataset"""
        return len(self._sample_paths)
