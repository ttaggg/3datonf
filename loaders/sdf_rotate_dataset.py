""""Dataset for SDF-INRs."""

import json
import os
from typing import NamedTuple, Tuple

import flags
import numpy as np
import torch

from absl import logging

import os

from networks.inr3d import InrToShape, Siren
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
    sdf_in: torch.Tensor
    rot_matrix: torch.Tensor
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
                              sdf_in=self.sdf_in.to(device),
                              rot_matrix=self.rot_matrix.to(device),
                              wm=tuple(k.to(device) for k in self.wm),
                              ws=tuple(k.to(device) for k in self.ws),
                              bm=tuple(k.to(device) for k in self.bm),
                              bs=tuple(k.to(device) for k in self.bs))

    def __len__(self):
        return len(self.weights[0])


class SdfDatasetFactory:
    """Create train-val-test split and return three loaders."""

    def __init__(self, config, device):
        self._dataset_path = config['dataset_path']

        all = [
            'Roundbox',
            'Capsule',
            'Cylinder',
            'Dodecahedron',
            'Octabound',
            'Roundbox',
            'Cube',
            'Sphere',
            'Triprismbound',
            'Octahedron',
            'Icosahedron',
            'Torus',
        ]
        self._train_names = all
        self._val_names = ['Hexprism']
        self._test_names = ['Hexprism']
        self._device = device
        self._config = config
        self._normalize = config.get('normalize', False)
        self._enforced_statistics = config.get('enforced_statistics', None)

    def split(self):

        train_set = []
        for name in self._train_names:
            subdir = os.path.join(self._dataset_path, 'sdf_models', name)
            model = os.path.join(subdir, 'model_normal.pt')
            points = os.path.join(subdir, 'points_normal.npz')
            train_set.append((model, points, name))

        val_set = []
        for name in self._val_names:
            subdir = os.path.join(self._dataset_path, 'sdf_models', name)
            model = os.path.join(subdir, 'model_normal.pt')
            points = os.path.join(subdir, 'points_normal.npz')
            val_set.append((model, points, name))

        test_set = []
        for name in self._test_names:
            subdir = os.path.join(self._dataset_path, 'sdf_models', name)
            model = os.path.join(subdir, 'model_normal.pt')
            points = os.path.join(subdir, 'points_normal.npz')
            test_set.append((model, points, name))

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

        train_dataset = SdfDataset(train_set,
                                   statistics=statistics,
                                   is_training=True,
                                   config=self._config,
                                   device=self._device)
        val_dataset = SdfDataset(val_set,
                                 statistics=statistics,
                                 is_training=False,
                                 config=self._config,
                                 device=self._device)
        test_dataset = SdfDataset(test_set,
                                  statistics=statistics,
                                  is_training=False,
                                  config=self._config,
                                  device=self._device)

        return train_dataset, val_dataset, test_dataset

    def _calculate_statistics(self, train_set):

        train_dataset = SdfDataset(train_set,
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


class SdfDataset(base_dataset.Dataset):
    """Custom dataset for SDF-INRs data."""

    def __init__(self, dataset, statistics, is_training, config, device):
        super().__init__(dataset, is_training)
        self._samples = dataset
        self._statistics = statistics
        self._device = device
        self._voxel_dim = 20
        self._inr_to_shape = InrToShape(self._voxel_dim, device)
        self._siren = Siren(in_features=3,
                            out_features=1,
                            hidden_features=128,
                            hidden_layers=2,
                            outermost_linear=True)

        logging.info(f'Dataset size: {len(self._samples)}, '
                     f'device: {device}, training: {is_training}.')

    @torch.no_grad()
    def __getitem__(self,
                    idx,
                    angle_input=None,
                    angle_delta_min=-90,
                    angle_delta_max=90):

        model_in, points_in, name = self._samples[idx]
        # model_in, points, model_out, angle_delta, name =  self._samples[idx]

        state_dict_in = torch.load(model_in, map_location='cpu')
        weights_in, biases_in = _preprocess_weights_biases(state_dict_in)

        sdf_in = self._inr_to_shape(weights_in, biases_in)
        sdf_in = sdf_in.squeeze(0).cpu().numpy()

        if angle_input is None:
            angle_delta = np.random.choice(
                list(range(angle_delta_min, angle_delta_max, 5)))
        else:
            angle_delta = angle_input

        theta = angle_delta_rad = np.deg2rad(angle_delta)
        rot_matrix = torch.tensor(
            [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]],
            dtype=torch.float32)

        angle_delta = torch.tensor(angle_delta).unsqueeze(0)
        angle_delta_rad = torch.tensor(angle_delta_rad)

        transformation = torch.tensor([angle_delta_rad],
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
                       sdf_in=sdf_in,
                       rot_matrix=rot_matrix,
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
