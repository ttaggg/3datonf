"""MNIST trainer."""

import collections

import numpy as np
import torch

from absl import logging

from trainers import base_trainer
from utils import trainer_utils


class MnistTrainer(base_trainer.Trainer):
    """Trainer for MNIST-INRs."""

    def __init__(self, config, model, output_dir, device, visualizers):
        super().__init__(config, model, output_dir, device, visualizers)

    def _compute_gradients(self, inputs):
        """Calculate gradients."""
        loss = self._model(inputs).mean()
        loss.backward()

    @torch.no_grad()
    def evaluate(self, prefix=''):
        logging.info(f'Evaluation for step {self._current_step}.')

        self._model.eval()

        metrics_dict = collections.defaultdict(list)

        ground_truth = []
        predictions = []
        for _, inputs in enumerate(self._test_data):

            inputs = self._move_to_device(inputs)
            losses, outputs = self._model(inputs, evaluation=True)
            # Save more stuff for dataset wide calculation.
            ground_truth.extend(inputs.label.cpu().numpy())
            predictions.extend(outputs.argmax(1).cpu().numpy())

            # Log losses every step.
            metrics_dict[f'{prefix}loss'].append(losses.detach().cpu())

        accuracy = np.array(ground_truth) == np.array(predictions)
        metrics_dict[f'{prefix}accuracy'] = accuracy.astype(np.float32)

        current_lr = trainer_utils.get_learning_rate(self._lr_scheduler)
        metrics_dict['lr'] = current_lr

        for metric_name, values in metrics_dict.items():
            scalar = torch.tensor(np.stack(values)).mean()
            metrics_dict[metric_name] = scalar
            logging.info(f'{metric_name}: {scalar:.3f}')

        if 'scalars' in self._vis:
            self._vis['scalars'].log(metrics_dict, self._current_step)

        self._model.train()
        return metrics_dict
