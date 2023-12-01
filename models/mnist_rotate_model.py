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
        # loss = 0
        # for tr, pr in zip(target, prediction):
            # loss += self._mse_loss(pr, tr)
        loss = self._mse_loss(target, prediction) 
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

        image_out = self._inr_to_image(new_weights, new_biases)
        # w_loss = self.compute_loss(inputs.weights_out, new_weights)
        # b_loss = self.compute_loss(inputs.biases_out, new_biases)
        # loss = w_loss + b_loss
        loss = self.compute_loss(inputs.image_in, image_out)

        if evaluation:
            return loss, (new_weights, new_biases)
        return loss