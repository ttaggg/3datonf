import functorch
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SineLayer(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(in_features,
                      hidden_features,
                      is_first=True,
                      omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(hidden_features,
                          hidden_features,
                          is_first=False,
                          omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(hidden_features,
                          out_features,
                          is_first=False,
                          omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


def make_coordinates(voxel_dim, bs, coord_range=(-1.0, 1.0)):
    x_coordinates = np.linspace(coord_range[0], coord_range[1], voxel_dim)
    y_coordinates = np.linspace(coord_range[0], coord_range[1], voxel_dim)
    z_coordinates = np.linspace(coord_range[0], coord_range[1], voxel_dim)
    x_coordinates, y_coordinates, z_coordinates = np.meshgrid(
        x_coordinates, y_coordinates, z_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    z_coordinates = z_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates, z_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)


def get_batch_siren(n_layers, in_dim, shape_size, device):
    siren = Siren(in_features=in_dim,
                  out_features=1,
                  hidden_features=128,
                  hidden_layers=n_layers - 1,
                  outermost_linear=True)
    siren = siren.to(device)

    func_model, _ = functorch.make_functional(siren)
    coords = make_coordinates(shape_size, 1)
    coords = coords.to(device)

    def func_inp(inp):
        p, rot_matrix = inp
        values = func_model(p, coords @ rot_matrix.T)[0]
        return torch.permute(
            values.reshape((shape_size, shape_size, shape_size)), (2, 1, 0))

    return functorch.vmap(func_inp, (0,))


class InrToShape:

    def __init__(self, shape_size, device):
        self._shape_size = shape_size
        self._device = device
        self._siren_convert = get_batch_siren(n_layers=3,
                                              in_dim=2,
                                              shape_size=self._shape_size,
                                              device=device)

    def __call__(self, weights, biases, rot_matrix=None):
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

        params = [p.to(self._device) for p in params]
        params = tuple(params)

        if rot_matrix is None:
            rot_matrix = torch.eye(3).unsqueeze(0)
            rot_matrix = rot_matrix.repeat(weights[0].shape[0], 1, 1)

        rot_matrix = rot_matrix.to(self._device)
        sdf = self._siren_convert((params, rot_matrix))
        return sdf
