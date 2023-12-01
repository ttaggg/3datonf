""""Dataset for MNIST-INRs."""

import glob
import os
from typing import NamedTuple, Tuple, Union

# https://github.com/pytorch/pytorch/issues/1355
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import flags
import numpy as np
import torch

from absl import logging
from sklearn.model_selection import train_test_split

from loaders import base_dataset
from networks.inr import InrToImage
from networks.nfn_layers.common import WeightSpaceFeatures

FLAGS = flags.FLAGS


class Batch(NamedTuple):
    """Modiefied from DWSNet."""
    weights: Tuple
    ori_weights: Tuple
    biases: Tuple
    ori_biases: Tuple
    label_image: torch.Tensor
    ori_image: torch.Tensor

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            ori_weights=tuple(w.to(device) for w in self.ori_weights),
            biases=tuple(w.to(device) for w in self.biases),
            ori_biases=tuple(w.to(device) for w in self.ori_biases),
            label_image=self.label_image.to(device),
            ori_image=self.ori_image.to(device))

    def __len__(self):
        return len(self.weights[0])


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
                                                   batch_size=32,
                                                   shuffle=True,
                                                   num_workers=0)

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

    def __init__(self, dataset, statistics, is_training, config, device):
        super().__init__(dataset, is_training)
        self._sample_paths, _ = dataset
        self._device = device
        self._statistics = statistics
        self._inr_to_image = InrToImage((28, 28, 1), device)
        self._transform_type = config['transform_type']
        assert self._transform_type in {
            'rotate', 'dilate'
        }, 'Only rotate and dilate are supported.'

        logging.info(f'Dataset size: {len(self._sample_paths)}, '
                     f'device: {device}, training: {is_training}.')

    @torch.no_grad()
    def __getitem__(self, idx):
        state_dict = torch.load(self._sample_paths[idx],
                                map_location=self._device)

        # NOTE(oleg): currently only NFNet weights' format is used
        # and not DWSNet: need to permute axes here and in inr_to_image.
        # [I, O]
        weights = tuple([v for w, v in state_dict.items() if "weight" in w])
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])

        # [F, I, O]: feature dim add
        weights = tuple([w.unsqueeze(0) for w in weights])
        biases = tuple([b.unsqueeze(0) for b in biases])

        ori_image = self._inr_to_image(weights, biases)
        # squeeze batch dimension, we do not need it here.
        ori_image = ori_image.squeeze(0)

        if self._transform_type == 'dilate':
            label_image = cv2.dilate(ori_image.cpu().detach().numpy(),
                                     np.ones((3, 3), np.uint8),
                                     iterations=1)
        elif self._transform_type == 'rotate':
            label_image = cv2.rotate(
                ori_image.squeeze(0).cpu().detach().numpy(),
                cv2.ROTATE_90_CLOCKWISE)
            label_image = np.expand_dims(label_image, 0)

        label_image = torch.tensor(label_image, device=self._device)

        ori_weights = weights
        ori_biases = biases
        if self._statistics is not None:
            weights, biases = self._normalize_weights_biases(weights, biases)

        # TODO(oleg): instead of giving original weights, pass means
        # and std and re-create them when necessary.
        # Also make metadata instead of huge tuple with everything.
        sample = Batch(weights=weights,
                       ori_weights=ori_weights,
                       biases=biases,
                       ori_biases=ori_biases,
                       label_image=label_image,
                       ori_image=ori_image)

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
