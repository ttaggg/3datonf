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


class MnistInrRotateModel(base_model.BaseModel):
    """MNIST model class."""

    def __init__(self, config, network, init_step, device):
        super().__init__(config, network, init_step, device)
        # Losses.
        loss_config = self._config['loss_functions']
        if loss_config['loss_name'] == 'mse':
            self._loss_fn = MeanSquareError(loss_config)
        else:
            raise ValueError('Only mse is supported.')
        self._inr_to_image = InrToImage((28, 28, 1), device)

    def model_outputs(self, inputs):
        outputs = self._network(inputs)
        return outputs

    def compute_loss(self, inputs, outputs):
        """Calculate losses."""
        loss = self._loss_fn(inputs, outputs)
        return loss

    def forward(self, inputs, evaluation=False):

        output = self.model_outputs(inputs)
        delta_weights, delta_biases = output.weights, output.biases
        new_weights = [
            wd + w for wd, w in zip(delta_weights, inputs.ori_weights)
        ]
        new_biases = [bd + b for bd, b in zip(delta_biases, inputs.ori_biases)]

        new_image = self._inr_to_image(new_weights, new_biases)
        # new_image = torch.clip(new_image, -1, 1)
        # new_image =  (new_image + 1) / 2

        loss = self.compute_loss(inputs.image_out, new_image)

        if evaluation:
            return loss, new_image
        return loss
