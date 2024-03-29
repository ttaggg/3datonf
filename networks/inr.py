"""Copied from DWSNets and NFN."""

import functorch
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Sine(nn.Module):

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class Siren(nn.Module):

    def __init__(self, dim_in, dim_out, w0=30.0):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(dim_out, dim_in))
        self.bias = nn.Parameter(torch.zeros(dim_out))
        self.activation = Sine(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class INR(nn.Module):
    """from DWSNet, modified"""

    def __init__(
        self,
        in_dim: int = 2,
        n_layers: int = 3,
        up_scale: int = 4,
        out_channels: int = 1,
    ):
        super().__init__()
        hidden_dim = in_dim * up_scale

        self.layers = [Siren(dim_in=in_dim, dim_out=hidden_dim)]
        for _ in range(n_layers - 2):
            self.layers.append(Siren(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_channels))
        self.seq = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


def make_coordinates(shape, bs, coord_range=(-1, 1)):
    x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)


def get_batch_siren(n_layers, in_dim, up_scale, image_size, device):
    siren = INR(n_layers=n_layers, in_dim=in_dim, up_scale=up_scale)
    siren = siren.to(device)
    func_model, _ = functorch.make_functional(siren)
    coords = make_coordinates(image_size, 1)
    coords = coords.to(device)

    def func_inp(p):
        values = func_model(p, coords)[0]
        return torch.permute(values.reshape(*image_size), (2, 1, 0))

    return functorch.vmap(func_inp, (0,))


class InrToImage:

    def __init__(self, image_size, device):
        self._image_size = image_size
        self._device = device
        self._siren_convert = get_batch_siren(n_layers=3,
                                              in_dim=2,
                                              up_scale=16,
                                              image_size=self._image_size,
                                              device=device)

    def __call__(self, weights, biases):
        params = []
        for w, b in zip(weights, biases):

            if w.dim() < 4:
                # [F, I, O] -> [B, F, I, O]
                w = w.unsqueeze(0)
                b = b.unsqueeze(0)

            w = w.squeeze(1)
            b = b.squeeze(1)

            params.append(w)
            params.append(b)

        params = tuple(params)

        image = self._siren_convert(params)
        image = torch.clip(image, -1, 1)
        image = (image + 1) / 2
        return image
