"""Base model class."""

import abc
import torch


class BaseModel(torch.nn.Module, abc.ABC):
    """Base model class."""

    def __init__(self, config, network, init_step):
        super().__init__()
        self._config = config
        self._init_step = init_step
        self._network = network

    @abc.abstractmethod
    def model_outputs(self, inputs):
        """Model outputs."""

    @abc.abstractmethod
    def compute_loss(self, inputs, outputs, evaluation):
        """Compute and return losses."""

    def forward(self, inputs, evaluation=False):
        outputs = self.model_outputs(inputs)
        loss = self.compute_loss(inputs, outputs)
        if evaluation:
            return loss, outputs
        return loss

    @property
    def init_step(self):
        return self._init_step

    @property
    def network(self):
        return self._network
