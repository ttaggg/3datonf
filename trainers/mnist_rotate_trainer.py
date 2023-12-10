"""MNIST trainer."""

import collections

import numpy as np
import torch

from absl import logging

from networks.inr import InrToImage
from trainers import base_trainer
from utils import trainer_utils


class MnistTrainer(base_trainer.Trainer):
    """Trainer for MNIST-INRs."""

    def __init__(self, config, model, output_dir, device, visualizers):
        super().__init__(config, model, output_dir, device, visualizers)
        self._inr_to_image = InrToImage((28, 28, 1), device)

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
            losses, new_image = self._model(inputs, evaluation=True)

            # Log losses every step.
            metrics_dict[f'loss'].append(losses.detach().cpu())

            if i < self._vis_n_batches and self._vis['images']:

                meta = np.rad2deg(inputs.angle.cpu().numpy()).flatten()
                meta = np.rint(meta)
                log_images = torch.cat(
                    [inputs.image_in, inputs.image_out, new_image], dim=-1)
                self._vis['images'].log(log_images, prefix, self._current_step,
                                        meta)

        current_lr = trainer_utils.get_learning_rate(self._lr_scheduler)
        metrics_dict['lr'] = current_lr

        for metric_name, values in metrics_dict.items():
            scalar = torch.tensor(np.stack(values)).mean()
            metrics_dict[metric_name] = scalar
            logging.info(f'{metric_name}: {scalar:.3f}')

        if 'scalars' in self._vis:
            self._vis['scalars'].log(metrics_dict, prefix, self._current_step)

        self._model.train()
        return metrics_dict
