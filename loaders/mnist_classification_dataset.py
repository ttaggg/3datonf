""""Dataset for MNIST-INRs."""

import glob
import os
from typing import NamedTuple, Tuple, Union

import flags
import torch
import numpy as np

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


def _preprocess_weights_biases(state_dict):

    # NOTE(oleg): currently only NFNet weights' format is used
    # and not DWSNet: need to permute axes here and in inr_to_image.
    # [I, O]
    weights = tuple([v for w, v in state_dict.items() if "weight" in w])
    biases = tuple([v for w, v in state_dict.items() if "bias" in w])

    # [F, I, O]: feature dim add
    weights = tuple([w.unsqueeze(0) for w in weights])
    biases = tuple([b.unsqueeze(0) for b in biases])

    return weights, biases


class MnistInrDatasetFactory:
    """Create train-val-test split and return three loaders."""

    def __init__(self, config, device):
        self._dataset_path = config['dataset_path']
        self._val_size = config.get('val_size', 0.2)
        self._device = device
        self._config = config
        self._normalize = config.get('normalize', False)

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

        statistics = None
        if self._normalize:
            statistics = self._calculate_statistics(train_set, train_labels)

        train_dataset = MnistInrClassificationDataset((train_set, train_labels),
                                                      statistics=statistics,
                                                      is_training=True,
                                                      config=self._config,
                                                      device=self._device)
        val_dataset = MnistInrClassificationDataset((val_set, val_labels),
                                                    statistics=statistics,
                                                    is_training=False,
                                                    config=self._config,
                                                    device=self._device)
        test_dataset = MnistInrClassificationDataset((test_set, test_labels),
                                                     statistics=statistics,
                                                     is_training=False,
                                                     config=self._config,
                                                     device=self._device)

        return train_dataset, val_dataset, test_dataset

    def _calculate_statistics(self, train_set, train_labels):

        train_dataset = MnistInrClassificationDataset((train_set, train_labels),
                                                      statistics=None,
                                                      is_training=True,
                                                      config=self._config,
                                                      device='cpu')

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=1)

        batch = next(iter(train_loader))
        weights_x = [x for x in batch.weights]
        weights_x_2 = [x * x for x in batch.weights]
        biases_x = [x for x in batch.biases]
        biases_x_2 = [x * x for x in batch.biases]

        for _, batch in enumerate(train_loader):
            for j, _ in enumerate(weights_x):
                weights_x[j] += batch.weights[j]
                weights_x_2[j] += batch.weights[j] * batch.weights[j]
            for k, _ in enumerate(biases_x):
                biases_x[k] += batch.biases[k]
                biases_x_2[k] += batch.biases[k] * batch.biases[k]

        num_samples = len(train_loader)
        weights_mean = [(x / num_samples).squeeze(0) for x in weights_x]
        biases_mean = [(x / num_samples).squeeze(0) for x in biases_x]
        weights_std = [
            torch.sqrt((x_2 / num_samples) - (x / num_samples)**2).squeeze(0)
            for x, x_2 in zip(weights_x, weights_x_2)
        ]
        biases_std = [
            torch.sqrt((x_2 / num_samples) - (x / num_samples)**2).squeeze(0)
            for x, x_2 in zip(biases_x, biases_x_2)
        ]

        return {
            "weights": {
                "mean": weights_mean,
                "std": weights_std,
            },
            "biases": {
                "mean": biases_mean,
                "std": biases_std,
            },
        }


class MnistInrClassificationDataset(base_dataset.Dataset):
    """Custom dataset for MNIST-INRs data."""

    def __init__(self, dataset, statistics, is_training, config, device):
        super().__init__(dataset, is_training)
        self._sample_paths, self._labels = dataset
        self._device = device
        self._statistics = statistics
        self._config = config

        logging.info(
            f'Dataset size: {len(self._sample_paths)}, device: {device}, training: {is_training}.'
        )

    def __getitem__(self, idx):
        state_dict = torch.load(self._sample_paths[idx], map_location='cpu')
        weights, biases = _preprocess_weights_biases(state_dict)

        if self._statistics is not None:
            weights, biases = self._normalize_weights_biases(weights, biases)

        label = torch.tensor(self._labels[idx])

        wm = ws = bm = bs = np.array([])
        if self._statistics is not None:
            wm = self._statistics["weights"]["mean"]
            ws = self._statistics["weights"]["std"]
            bm = self._statistics["biases"]["mean"]
            bs = self._statistics["biases"]["std"]

        sample = Batch(weights=weights,
                       biases=biases,
                       label=label,
                       wm=wm,
                       ws=ws,
                       bm=bm,
                       bs=bs)

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
