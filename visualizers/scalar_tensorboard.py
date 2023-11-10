"""Visuzalize scalar data in Tensorflow."""

import os

from torch.utils import tensorboard


class ScalarTensorboardVisualizer:

    def __init__(self, output_dir):
        self._scalar_writer = tensorboard.SummaryWriter(
            os.path.join(output_dir, 'scalars'))

    def log(self, metrics_dict, step):
        for tag, value in metrics_dict.items():
            self._scalar_writer.add_scalar(tag, value, step)
