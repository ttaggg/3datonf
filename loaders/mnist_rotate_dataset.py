""""Dataset for MNIST-INRs."""

import glob
import os
import re
from itertools import combinations
from typing import NamedTuple, Tuple, Union

import flags
import numpy as np
import torch

from absl import logging
from PIL import Image
from sklearn.model_selection import train_test_split

from loaders import base_dataset
from networks.inr import InrToImage

FLAGS = flags.FLAGS


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


class Batch(NamedTuple):
    """Modified from DWSNet."""
    weights: Tuple
    ori_weights: Tuple
    biases: Tuple
    ori_biases: Tuple
    weights_out: Tuple
    biases_out: Tuple
    angle: np.ndarray
    image_in: np.ndarray
    image_out: np.ndarray

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
            weights_out=tuple(w.to(device) for w in self.weights_out),
            biases_out=tuple(w.to(device) for w in self.biases_out),
            angle=self.angle.to(device),
            image_in=self.image_in.to(device),
            image_out=self.image_out.to(device))

    def __len__(self):
        return len(self.weights[0])


class MnistInrDatasetFactory:
    """Create train-val-test split and return three loaders."""

    def __init__(self, config, device):
        self._dataset_path = config['dataset_path']
        self._val_size = config.get('val_size', 0.1)
        self._test_size = config.get('val_size', 0.1)
        self._device = device
        self._config = config
        self._normalize = config.get('normalize', False)

    def split(self):

        # Get val set from train set.
        train_val_test_set = []

        # iterate through subdirs, take pairs of samples
        subdirs = glob.glob(os.path.join(self._dataset_path, '**'))
        for subdir in subdirs:
            samples = glob.glob(os.path.join(subdir, '**'))
            # avoid files with invalid name like 1405/model90(1).pt
            models = list(
                filter(lambda x: re.search(r'model[0-9]+.pt$', x), samples))
            models = sorted(models,
                            key=lambda k: int(re.findall('[0-9]+', k)[-1]))

            # make sure sequence is exactly the same as models
            images = list(
                filter(lambda x: re.search(r'image[0-9]+.bmp', x), samples))
            images = sorted(images,
                            key=lambda k: int(re.findall('[0-9]+', k)[-1]))

            angles = [int(re.findall(r'[0-9]+', x)[-1]) for x in models]

            assert len(images) == len(models)

            for l, r in combinations(range(len(models)), 2):
                train_val_test_set.append((models[l], images[l], angles[l],
                                           models[r], images[r], angles[r]))
                train_val_test_set.append((models[r], images[r], angles[r],
                                           models[l], images[l], angles[l]))

        train_val_set, test_set = train_test_split(
            train_val_test_set,
            test_size=self._test_size,
            random_state=FLAGS.random_seed)

        train_set, val_set = train_test_split(train_val_set,
                                              test_size=self._val_size,
                                              random_state=FLAGS.random_seed)

        statistics = None
        if self._normalize:
            statistics = self._calculate_statistics(train_set)

        train_dataset = MnistInrClassificationDataset(train_set,
                                                      statistics=statistics,
                                                      is_training=True,
                                                      config=self._config,
                                                      device=self._device)
        val_dataset = MnistInrClassificationDataset(val_set,
                                                    statistics=statistics,
                                                    is_training=False,
                                                    config=self._config,
                                                    device=self._device)
        test_dataset = MnistInrClassificationDataset(test_set,
                                                     statistics=statistics,
                                                     is_training=False,
                                                     config=self._config,
                                                     device=self._device)

        return train_dataset, val_dataset, test_dataset

    def _calculate_statistics(self, train_set):

        train_dataset = MnistInrClassificationDataset(train_set,
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
        self._samples = dataset
        self._device = device
        self._statistics = statistics

        logging.info(f'Dataset size: {len(self._samples)}, '
                     f'device: {device}, training: {is_training}.')

        self._inr_to_image = InrToImage((28, 28, 1), device)

    @torch.no_grad()
    def __getitem__(self, idx):

        model_in, image_in, angle_in, model_out, image_out, angle_out = self._samples[
            idx]

        state_dict_in = torch.load(model_in, map_location=self._device)
        weights_in, biases_in = _preprocess_weights_biases(state_dict_in)

        state_dict_out = torch.load(model_out, map_location=self._device)
        weights_out, biases_out = _preprocess_weights_biases(state_dict_out)

        image_in = self._inr_to_image(weights_in, biases_in)
        image_in = image_in.squeeze(0)
        image_out = self._inr_to_image(weights_out, biases_out)
        image_out = image_out.squeeze(0)

        ori_weights_in = weights_in
        ori_biases_in = biases_in
        if self._statistics is not None:
            weights_in, biases_in = self._normalize_weights_biases(
                weights_in, biases_in)

        angle = np.array(np.deg2rad(angle_out - angle_in), dtype=np.float32)
        angle = np.expand_dims(angle, 0)
        angle = torch.tensor([np.sin(angle), np.cos(angle)], device=self._device).T

        # TODO(oleg): instead of giving original weights, pass means
        # and std and re-create them when necessary.
        # Also make metadata instead of huge tuple with everything.
        sample = Batch(weights=weights_in,
                       ori_weights=ori_weights_in,
                       biases=biases_in,
                       ori_biases=ori_biases_in,
                       weights_out=weights_out,
                       biases_out=biases_out,
                       angle=angle,
                       image_in=image_in,
                       image_out=image_out)

        return sample

    def _normalize_weights_biases(self, weights, biases):
        wm = self._statistics["weights"]["mean"]
        ws = self._statistics["weights"]["std"]
        bm = self._statistics["biases"]["mean"]
        bs = self._statistics["biases"]["std"]

        # weights = tuple((w - m) / s for w, m, s in zip(weights, wm, ws))
        # biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))
        weights = tuple((w - m) for w, m, s in zip(weights, wm, ws))
        biases = tuple((w - m) for w, m, s in zip(biases, bm, bs))

        return weights, biases

    def __len__(self):
        """__len__ method of torch.utils.data.Dataset"""
        return len(self._samples)
