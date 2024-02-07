"""MNIST-INRs model."""

import torch

from torch import nn

from models import base_model
from networks.inr3d import InrToShape


class MeanSquareError(nn.Module):

    def __init__(self, config):
        super().__init__()
        self._mse_loss = torch.nn.MSELoss()

    def forward(self, target, prediction):
        loss = self._mse_loss(prediction, target)
        return loss.mean()


class SdfModel(base_model.BaseModel):
    """MNIST model class."""

    def __init__(self, config, network, init_step, device):
        super().__init__(config, network, init_step, device)
        # Losses.
        loss_config = self._config['loss_functions']
        if loss_config['loss_name'] == 'mse':
            self._loss_fn = MeanSquareError(loss_config)
        else:
            raise ValueError('Only mse is supported.')

        self._voxel_dim = 20
        self._inr_to_shape = InrToShape(self._voxel_dim, device)

    def model_outputs(self, inputs):
        outputs = self._network(inputs)
        return outputs

    def compute_loss(self, inputs, outputs):
        """Calculate losses."""
        loss = self._loss_fn(inputs, outputs)
        return loss

    def _unnormalize_weights_biases(self, weights, biases, wm, ws, bm, bs):
        weights = tuple(((w * s) + m) for w, m, s in zip(weights, wm, ws))
        biases = tuple(((w * s) + m) for w, m, s in zip(biases, bm, bs))
        return weights, biases

    def forward(self, inputs, evaluation=False):

        output = self.model_outputs(inputs)
        delta_weights, delta_biases = output.weights, output.biases

        new_weights = [wd + w for wd, w in zip(delta_weights, inputs.weights)]
        new_biases = [bd + b for bd, b in zip(delta_biases, inputs.biases)]

        new_weights, new_biases = self._unnormalize_weights_biases(
            new_weights, new_biases, inputs.wm, inputs.ws, inputs.bm, inputs.bs)

        # We pass rotation matrix and modify input coordinates before applying MLP.
        new_sdf = self._inr_to_shape(new_weights, new_biases, inputs.rot_matrix)
        weights_in, biases_in = self._unnormalize_weights_biases(
            inputs.weights, inputs.biases, inputs.wm, inputs.ws, inputs.bm,
            inputs.bs)

        sdf_in = self._inr_to_shape(weights_in, biases_in)
        sdf_out = self._inr_to_shape(weights_in, biases_in, inputs.rot_matrix)

        loss = self.compute_loss(sdf_in, new_sdf)

        if evaluation:
            return loss, (sdf_in, sdf_out, new_sdf)
        return loss
