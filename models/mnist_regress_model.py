"""MNIST-INRs model."""

import torch

from torch import nn

from models import base_model
from networks.inr import InrToImage


class MeanSquareError(nn.Module):

    def __init__(self, config):
        super().__init__()
        self._mse_loss = torch.nn.MSELoss()

    def forward(self, target, prediction):
        loss = self._mse_loss(prediction, target)
        return loss.mean()


class MnistInrRegressModel(base_model.BaseModel):
    """MNIST model class."""

    def __init__(self, config, network, init_step, device):
        super().__init__(config, network, init_step, device)
        # Losses.
        loss_config = self._config['loss_functions']
        if loss_config['loss_name'] == 'mse':
            self._loss_fn = MeanSquareError(loss_config)
        else:
            raise ValueError('Only mse is supported.')

    def model_outputs(self, inputs):
        outputs = self._network(inputs.weights, inputs.biases,
                                inputs.weights_out, inputs.biases_out,
                                inputs.angle_encoded)
        return outputs

    def compute_loss(self, inputs, outputs):
        """Calculate losses."""
        loss = self._loss_fn(inputs, outputs)
        return loss

    def forward(self, inputs, evaluation=False):

        output = self.model_outputs(inputs)
        loss = self.compute_loss(inputs.angle_delta_rad, output)

        if evaluation:
            return loss, output
        return loss
