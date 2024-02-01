"""Trainer utils."""

import json

import numpy as np
import torch

from absl import logging


def get_optimizer(model_params, config):
    """Choose optimizer.

    Args:
        model_params: Trainable params.
        optimizer_config: Dictionary.
    Returns:
        optim.Adam or RMSprop.
    Raises:
        ValueError if optimizer name is not one of
            adam or rmsprop.
    """
    logging.info(f'Optimizer: {json.dumps(config, indent=2)}')

    optimizer = config.get('name', 'adam')
    learning_rate = config.get('learning_rate', 1e-3)
    optimizer_params = config.get('params', {})

    logging.info(f'Optimizer: {optimizer}.')
    logging.info(f'Learning rate: {learning_rate}.')
    if optimizer == 'adam':
        return torch.optim.Adam(model_params,
                                lr=learning_rate,
                                **optimizer_params)
    if optimizer == 'adamw':
        return torch.optim.AdamW(model_params,
                                 lr=learning_rate,
                                 **optimizer_params)

    if optimizer == 'rmsprop':
        return torch.optim.RMSprop(model_params,
                                   lr=learning_rate,
                                   **optimizer_params)
    if optimizer == 'sgd':
        return torch.optim.SGD(model_params,
                               lr=learning_rate,
                               **optimizer_params)
    raise ValueError('Only "adam", "sgd" and "rmsprop" are supported,'
                     f'given: {optimizer}.')


def get_lr_scheduler(optimizer, config):
    logging.info(f'LR scheduler: {json.dumps(config, indent=2)}')

    lr_scheduler = config.get('name', 'step_lr')
    params = config.get('params', {})
    if lr_scheduler == 'step_lr':
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif lr_scheduler == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)


def get_learning_rate(lr_scheduler, config=None):

    if lr_scheduler is None:
        logging.info('LR scheduler was not set.')
        if config is not None:
            return config['optimizer']['learning_rate']
        return np.nan

    if lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
        if hasattr(lr_scheduler, '_last_lr'):
            return lr_scheduler._last_lr
        if config is not None:
            return config['optimizer']['learning_rate']
        logging.info(
            'LR scheduler did not make any steps, LR is not updated yet,'
            ' cannot log LR yet.')
        return np.nan

    return lr_scheduler.get_last_lr()
