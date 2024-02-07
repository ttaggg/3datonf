"""MNIST trainer."""

import collections

import numpy as np
import torch

from absl import logging

from networks.inr3d import InrToShape
from trainers import base_trainer
from utils import trainer_utils


class SdfTrainer(base_trainer.Trainer):
    """Trainer for MNIST-INRs."""

    def __init__(self, config, model, output_dir, device, visualizers):
        super().__init__(config, model, output_dir, device, visualizers)
        self._voxel_dim = 20
        self._inr_to_shape = InrToShape(self._voxel_dim, device)
        self._config = config

    def _compute_gradients(self, inputs):
        """Calculate gradients."""
        loss = self._model(inputs).mean()
        loss.backward()

    @torch.no_grad()
    def evaluate(self, prefix=''):
        logging.info(f'Evaluation for step {self._current_step}.')

        self._model.eval()

        metrics_dict = collections.defaultdict(list)

        for i, inputs in enumerate(self._test_data):

            inputs = self._move_to_device(inputs)
            losses, (shape_in, shape_out,
                     new_shape) = self._model(inputs, evaluation=True)

            # Log losses and metrics every step.
            metrics_dict[f'loss'].append(losses.detach().cpu().numpy())
            metrics_dict[f'mse'].append((shape_out - new_shape).square().mean())

        current_lr = trainer_utils.get_learning_rate(self._lr_scheduler,
                                                     self._config)
        metrics_dict['lr'] = [current_lr]

        mean_mse = torch.stack(
            metrics_dict[f'mse']).mean().detach().cpu().numpy()
        metrics_dict[f'mse'] = [mean_mse]
        metrics_dict[f'psnr'].append(-10 * np.log10(mean_mse))

        for metric_name, values in metrics_dict.items():
            scalar = torch.tensor(np.stack(values)).mean()
            metrics_dict[metric_name] = scalar
            logging.info(f'{metric_name}: {scalar:.3f}')

        if 'scalars' in self._vis:
            self._vis['scalars'].log(metrics_dict, prefix, self._current_step)

        self._model.train()
        return metrics_dict
