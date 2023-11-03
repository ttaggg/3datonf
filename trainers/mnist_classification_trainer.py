"""MNIST trainer."""

import collections

import numpy as np
import torch

from trainers import base_trainer
from utils import trainer_utils


class MnistTrainer(base_trainer.Trainer):
    """Trainer for MNIST-INRs."""

    def __init__(self, config, model, output_dir, device):
        super().__init__(config, model, output_dir, device)

    def _compute_gradients(self, inputs):
        """Calculate gradients."""
        loss = self._model(inputs).mean()
        loss.backward()

    @torch.no_grad()
    def evaluate(self, prefix=''):
        self._model.eval()

        log_scalars = collections.defaultdict(list)

        current_lr = trainer_utils.get_learning_rate(self._lr_scheduler)
        log_scalars['lr'].append(current_lr)

        ground_truth = []
        predictions = []
        for _, inputs in enumerate(self._test_data):

            inputs = self._move_to_device(inputs)
            losses, outputs = self._model(inputs, evaluation=True)

            # Log losses and metrics.
            log_scalars[f'{prefix}loss'].append(losses.detach().cpu())

            # Log more stuff when metrics are ready.
            ground_truth.extend(inputs.label.cpu().numpy())
            predictions.extend(outputs.argmax(1).cpu().numpy())

        log_scalars[f'{prefix}accuracy'] = (
            np.array(ground_truth) == np.array(predictions)).astype(np.float32)

        self._log_scalars_tensorboard(log_scalars, self._current_step)

        self._model.train()
        return log_scalars
