"""MNIST-INRs model."""

import torch

from torch import nn

from models import base_model


class CrossEntropyLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self._bce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, target, prediction):
        loss = self._bce_loss(prediction, target)
        return loss.mean()


class MnistInrClassificationModel(base_model.BaseModel):
    """MNIST model class."""

    def __init__(self, config, network, init_step, device):
        super().__init__(config, network, init_step, device)
        # Losses.
        loss_config = self._config['loss_functions']
        if loss_config['loss_name'] == 'cross_entropy':
            self._loss_fn = CrossEntropyLoss(loss_config)
        else:
            raise ValueError('Only cross_entropy is supported.')

        self._network_name = config['network']['network_name']
        self._device = device

    def model_outputs(self, inputs):
        outputs = self._network(inputs)
        return outputs

    def compute_loss(self, inputs, outputs):
        """Calculate losses."""
        loss = self._loss_fn(inputs.label, outputs)
        return loss
