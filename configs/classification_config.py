"""Baseline config."""

# pylint: disable=invalid-name
# yapf: disable

import os

_DATA_PATH = '../../datasets'
_BATCH_SIZE = 256

model = {
    'model_name': 'mnist_classification_model',
    'network': {
        'network_name': 'nfn_classification',
        'network_params': {
            'weight_shapes': tuple([(32, 2,), (32, 32,), (1, 32,)]),
            'bias_shapes': tuple([(32,), (32,), (1, )]),
            'inp_enc_cls': 'gaussian',
            'pos_enc_cls': None,
            'hidden_chan': 128,
            'hidden_layers': 3,
            'mode': 'HNP',
            'out_scale': 0.01,
            'lnorm': False,
            'dropout': 0,
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
    'vis_n_batches': 1,
    # Optimizer.
    'visualizers': {
        'scalars': 'scalar_visualizer',
    },
    'optimizer': {
        'name': 'adamw',
        'learning_rate': 5e-4,
        'amsgrad': True,
        'weight_decay': 5e-4
    },
    # LR-scheduler.
    'lr_scheduler': {
        'name': 'reduce_on_plateau',
        'params': {
            'factor': 0.5,
            'patience': 3,
            'threshold': 0.001,
            'min_lr': 1e-6,
        }
    }
}

data = {
    'task': 'mnist_classification',
    'dataset_path':  os.path.join(_DATA_PATH, 'mnist-inr-medium'),
    'normalize': True,
    'num_workers': 4,
}