"""Baseline config."""

# pylint: disable=invalid-name
# yapf: disable

import os


_DATA_PATH = '../../datasets'
_BATCH_SIZE = 512

model = {
    'model_name': 'mnist_rotate_model',
    'network': {
        'network_name': 'transfer_rotate_net',
        'network_params': {
            'weight_shapes': tuple([(32, 2,), (32, 32,), (1, 32,)]),
            'bias_shapes': tuple([(32,), (32,), (1, )]),
            'pos_enc_cls': None,
            'hidden_chan': 128,
            'hidden_layers': 3,
            'mode': 'HNP_angle',
            'out_scale': 0.01,
            'lnorm': False,
            'dropout': 0,
        }
    },
    'batch_size': _BATCH_SIZE,
    'loss_functions': {
        'loss_name': 'mse',
    },
}

training = {
    # General.
    'trainer_name': 'mnist_rotate_trainer',
    'batch_size': _BATCH_SIZE,
    'num_epochs': 50,
    'vis_n_batches': 1,
    # Optimizer.
    'visualizers': {
        'scalars': 'scalar_visualizer',
        'images': 'image_visualizer'
    },
    'optimizer': {
        'name': 'adamw',
        'learning_rate': 1e-3,
        # 'amsgrad': True,
        'weight_decay': 5e-4
    },
    # LR-scheduler.
    'lr_scheduler': {
        'name': 'step_lr',
        'params': {
            'step_size': 10,
            'gamma': 0.3,
        }
    }
}


data = {
    'task': 'mnist_rotate',
    'dataset_path':  os.path.join(_DATA_PATH, 'mnist-inrs-rotate-full'),
    'normalize': False,
    'num_workers': 0,
}