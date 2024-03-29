""""Dataset for MNIST-INRs."""

import glob
import json
import os
import re
from typing import NamedTuple, Tuple

import flags
# https://github.com/pytorch/pytorch/issues/1355
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import torch

from absl import logging
from PIL import Image
from sklearn.model_selection import train_test_split

from loaders import base_dataset

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
    biases: Tuple
    angle_delta: torch.Tensor
    angle_delta_rad: torch.Tensor
    transformation: torch.Tensor
    image_in: torch.Tensor
    image_out: torch.Tensor
    wm: Tuple
    ws: Tuple
    bm: Tuple
    bs: Tuple

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(weights=tuple(w.to(device) for w in self.weights),
                              biases=tuple(w.to(device) for w in self.biases),
                              angle_delta=self.angle_delta.to(device),
                              angle_delta_rad=self.angle_delta_rad.to(device),
                              transformation=self.transformation.to(device),
                              image_in=self.image_in.to(device),
                              image_out=self.image_out.to(device),
                              wm=tuple(k.to(device) for k in self.wm),
                              ws=tuple(k.to(device) for k in self.ws),
                              bm=tuple(k.to(device) for k in self.bm),
                              bs=tuple(k.to(device) for k in self.bs))

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
        self._enforced_statistics = config.get('enforced_statistics', None)

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

            for l, model in enumerate(models):
                train_val_test_set.append([models[l], images[l], angles[l]])

        train_val_set, test_set = train_test_split(
            train_val_test_set,
            test_size=self._test_size,
            random_state=FLAGS.random_seed)

        train_set, val_set = train_test_split(train_val_set,
                                              test_size=self._val_size,
                                              random_state=FLAGS.random_seed)

        statistics = None
        if self._enforced_statistics is not None:

            with open(self._enforced_statistics) as json_file:
                statistics = json.load(json_file)

            for param, param_value in statistics.items():
                for metric, metric_value in param_value.items():
                    statistics[param][metric] = [
                        torch.tensor(x, dtype=torch.float32)
                        for x in metric_value
                    ]

        if self._normalize and not self._enforced_statistics:
            statistics = self._calculate_statistics(train_set)

        train_dataset = MnistInrDataset(train_set,
                                        statistics=statistics,
                                        is_training=True,
                                        config=self._config,
                                        device=self._device)
        val_dataset = MnistInrDataset(val_set,
                                      statistics=statistics,
                                      is_training=False,
                                      config=self._config,
                                      device=self._device)
        test_dataset = MnistInrDataset(test_set,
                                       statistics=statistics,
                                       is_training=False,
                                       config=self._config,
                                       device=self._device)

        return train_dataset, val_dataset, test_dataset

    def _calculate_statistics(self, train_set):

        train_dataset = MnistInrDataset(train_set,
                                        statistics=None,
                                        is_training=True,
                                        config=self._config,
                                        device='cpu')

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=0)

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

        # Save statistics to load it later without long calculation with
        # "enforced_statistics" in config.

        # import json
        # with open("/Users/oleg/study_repos/3datonf/stat/default_stat.json", "w") as outfile:
        #     json.dump({
        #         "weights": {
        #             "mean": [x.cpu().numpy().tolist() for x in weights_mean],
        #             "std": [x.cpu().numpy().tolist() for x in weights_std],
        #         },
        #         "biases": {
        #             "mean": [x.cpu().numpy().tolist() for x in biases_mean],
        #             "std": [x.cpu().numpy().tolist() for x in biases_std],
        #         },
        #     }, outfile)

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


def _nerf_encode(x, depth=1):

    outputs = []
    for i in range(depth):
        outputs.append((2**i) * torch.pi * torch.sin(x))
        outputs.append((2**i) * torch.pi * torch.cos(x))

    outputs = torch.cat(outputs, dim=-1)

    return outputs


def _getTranslationMatrix2D(dx, dy):
    """
    Returns a numpy affine transformation matrix for a 2D translation of
    (dx, dy)
    """
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


class MnistInrDataset(base_dataset.Dataset):
    """Custom dataset for MNIST-INRs data."""

    def __init__(self, dataset, statistics, is_training, config, device):
        super().__init__(dataset, is_training)
        self._samples = dataset
        self._statistics = statistics
        self._device = device

        logging.info(f'Dataset size: {len(self._samples)}, '
                     f'device: {device}, training: {is_training}.')

    @torch.no_grad()
    def __getitem__(self,
                    idx,
                    angle_input=None,
                    trans_input=None,
                    angle_delta_min=-90,
                    angle_delta_max=90,
                    trans_min=-0.25,
                    trans_max=0.25):

        model_in, image_in, _ = self._samples[idx]

        state_dict_in = torch.load(model_in, map_location='cpu')
        weights_in, biases_in = _preprocess_weights_biases(state_dict_in)

        if angle_input is None:
            angle_delta = np.random.choice(
                list(range(angle_delta_min, angle_delta_max, 5)))
        else:
            angle_delta = angle_input

        if trans_input is None:
            trans = np.random.uniform(low=trans_min, high=trans_max, size=(2,))
        else:
            trans = trans_input

        # Rotation and Translation
        T = _getTranslationMatrix2D(14 * trans[0], 14 * trans[1])
        R = np.vstack(
            [cv2.getRotationMatrix2D((14, 14), angle_delta, 1.0), [0, 0, 1]])
        affine_mat = (np.matrix(T) * np.matrix(R))[0:2, :]

        image_in = np.array(Image.open(image_in).convert('L'),
                            dtype=np.float32) / 255
        image_out = cv2.warpAffine(image_in, affine_mat, (28, 28))
        image_out = torch.tensor(image_out, dtype=torch.float32).unsqueeze(0)

        angle_delta = torch.tensor(angle_delta).unsqueeze(0)
        angle_delta_rad = torch.deg2rad(angle_delta)

        transformation = torch.tensor([angle_delta_rad, trans[0], trans[1]],
                                      dtype=torch.float32).reshape((1, -1))
        transformation = _nerf_encode(transformation, depth=1)

        if self._statistics is not None:
            weights_in, biases_in = self._normalize_weights_biases(
                weights_in, biases_in)

        wm = ws = bm = bs = np.array([])
        if self._statistics is not None:
            wm = self._statistics["weights"]["mean"]
            ws = self._statistics["weights"]["std"]
            bm = self._statistics["biases"]["mean"]
            bs = self._statistics["biases"]["std"]

        sample = Batch(weights=weights_in,
                       biases=biases_in,
                       angle_delta=angle_delta,
                       angle_delta_rad=angle_delta_rad,
                       transformation=transformation,
                       image_in=image_in,
                       image_out=image_out,
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
        return len(self._samples)
