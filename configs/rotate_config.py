"""Baseline config."""

# pylint: disable=invalid-name
# yapf: disable

import os

_DATA_PATH = '../../datasets'
_BATCH_SIZE = 810

model = {
    'model_name': 'mnist_rotate_model',
    'network': {
        'network_name': 'transfer_rotate_net',
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
            'input_dim': 6,
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
    'num_epochs': 500,
    'vis_n_batches': 1,
    # Optimizer.
    'visualizers': {
        'scalars': 'scalar_visualizer',
        'images': 'image_visualizer'
    },
    'optimizer': {
        'name': 'adamw',
        'learning_rate': 1e-3,
        'amsgrad': True,
        'weight_decay': 5e-4
    },
    # LR-scheduler.
    # 'lr_scheduler': {
    #     'name': 'reduce_on_plateau',
    #     'params': {
    #         'factor': 0.5,
    #         'patience': 5,
    #         'threshold': 0.001,
    #         'min_lr': 1e-6,
    #     }
    # }
    'lr_scheduler': {
        'name': 'step_lr',
        'params': {
            'step_size': 50,
            'gamma': 0.2,
        }
    }
}


data = {
    'task': 'mnist_rotate',
    'dataset_path':  os.path.join(_DATA_PATH, 'less_converged'),
    'normalize': True,
    'enforced_statistics': './stat/default_stat.json',
    'num_workers': 4,
}
