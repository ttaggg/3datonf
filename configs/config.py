"""Baseline config."""

# pylint: disable=invalid-name
# yapf: disable

import os

_DATA_PATH = '../datasets'
_BATCH_SIZE = 512

model = {
    'model_name': 'mnist_classification_model',
    'network': {
        'network_name': 'dwsnet_classification',
        'network_params': {
            'weight_shapes': tuple([(2, 32, ), (32, 32, ), (32, 1, )]),
            'bias_shapes': tuple([(32,), (32,), (1, )]),
            'input_features': 1,
            'hidden_dim': 32,
            'n_hidden': 4,
            'bn': True,
        }
    },
    'batch_size': _BATCH_SIZE,
    'loss_functions': {
        'loss_name': 'cross_entropy',
    },
}

training = {
    # General.
    'trainer_name': 'mnist_classification_trainer',
    'batch_size': _BATCH_SIZE,
    'num_epochs': 50,
    # Optimizer.
    'visualizers': {
        'scalars': 'scalar_visualizer'
    },
    'optimizer': {
        'name': 'adamw',
        'learning_rate': 1e-3,
        'amsgrad': True,
        'weight_decay': 5e-4
    },
    # LR-scheduler.
    'lr_scheduler': {
        'name': 'step_lr',
        'params': {
            'step_size': 20,
            'gamma': 0.3,
        }
    }
}


data = {
    'task': 'mnist_classification',
    'dataset_path':  os.path.join(_DATA_PATH, 'mnist-inrs'),
    'num_workers': 8,
    'augmentation': {
    }
}