"""Parent trainer class."""

import abc
import os

import numpy as np
import torch
from absl import logging
from torch.utils import tensorboard

from utils import trainer_utils as tu


class Trainer(abc.ABC):

    def __init__(self, config, model, output_dir, device):
        # General.
        self._config = config
        self._model = model
        self._output_dir = output_dir
        self._scalar_writer = tensorboard.SummaryWriter(
            os.path.join(output_dir, 'scalars'))
        # Training.
        self._current_step = None
        self._optimizer = None
        self._lr_scheduler = None
        self._device = device
        # Evaluation.
        self._test_data = None
        self._evaluate_freq = None

    def set_for_training(self, init_step):
        """Set attributes necessary for training."""
        self._current_step = init_step

        gpus_available = torch.cuda.device_count()
        logging.info(f'Running on {gpus_available} GPU(s).')
        self._model = torch.nn.DataParallel(self._model,
                                            device_ids=list(
                                                range(gpus_available)))
        self._model.to(self._device)
        self._model.train()

        self._optimizer = tu.get_optimizer(
            model_params=self._model.parameters(),
            config=self._config['optimizer'])
        self._lr_scheduler = tu.get_lr_scheduler(
            optimizer=self._optimizer, config=self._config['lr_scheduler'])

    def set_for_evaluation(self, init_step, test_data):
        """Set attributes necessary for evaluation."""
        self._current_step = init_step
        self._model.to(self._device)
        self._test_data = test_data

    def _move_to_device(self, inputs):
        if isinstance(inputs, dict):
            for key, value in inputs.items():
                inputs[key] = value.to(self._device)
        else:
            inputs.to(self._device)
        return inputs

    def _on_batch_start(self):
        self._optimizer.zero_grad()

    def train_step(self, inputs):
        """Make one training step."""
        self._on_batch_start()
        inputs = self._move_to_device(inputs)
        self._compute_gradients(inputs)
        self._on_batch_end()

    def _on_batch_end(self):
        self._optimizer.step()
        self._current_step += 1

        if (self._current_step == 1):
            # dummy evaluation before first epoch
            self.evaluate()

    def on_epoch_end(self):
        """Actions to do in the end of each epoch."""
        self._save(self._current_step)
        outputs = self.evaluate()

        if self._config['lr_scheduler']['name'] == 'reduce_on_plateau':
            # NOTE(oleg): StepLR and ReduceLROnPlateau have different API :(
            target_metric = self._config['lr_scheduler']['target_metric']
            self._lr_scheduler.step(np.mean(outputs[target_metric]))
        else:
            self._lr_scheduler.step()

    def _save(self, step):
        """Save checkpoint."""
        ckpt_dir = os.path.join(self._output_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_path = os.path.join(ckpt_dir, f'state_dict_{step}.pt')
        torch.save(self._model.module.network.state_dict(), checkpoint_path)
        logging.info(f'Model was saved to {checkpoint_path}.')

    def _log_scalars_tensorboard(self, log_scalars, step):
        logging.info(f'Evaluation for step {step}.')
        for tag, values in log_scalars.items():
            scalar = torch.tensor(np.stack(values)).mean()
            logging.info(f'{tag}: {scalar:.3f}')
            self._scalar_writer.add_scalar(tag, scalar, step)

    def evaluate(self, prefix=''):
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_gradients(self, inputs):
        """Calculate gradients.

        outputs = models_outputs_fn(inputs)
        loss = loss_fn(inputs, outputs)
        loss.backward()
        """
