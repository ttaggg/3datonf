"""Pipeline utils."""

import os
from sys import platform
from absl import logging

if platform == 'darwin':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    logging.warning(
        'KMP_DUPLICATE_LIB_OK is set to True on macOS, '
        'Intel says that it can silently produce incorrect results.')

import collections
import random
import re
import runpy
import shutil
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.models import regnet_y_800mf

from loaders import mnist_classification_dataset, mnist_stylize_dataset, mnist_rotate_dataset
from models import mnist_classification_model, mnist_stylize_model, mnist_rotate_model
from networks.dwsnets_networks import DWSModelForClassification, DWSModel
from networks.nfn_networks import TransferNet
from trainers import mnist_classification_trainer, mnist_stylize_trainer, mnist_rotate_trainer
from visualizers import scalar_visualizer, image_visualizer


@dataclass
class Config:
    model: collections.defaultdict = None
    training: collections.defaultdict = None
    data: collections.defaultdict = None


def initialize(output_dir, seed, use_gpus):
    """Initilialize output directory, random seeds and logging levels."""

    # Set torch.
    torch.set_num_threads(1)

    # Set visible GPUs.
    os.environ['CUDA_VISIBLE_DEVICES'] = use_gpus

    # Initialize output directory.
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Set logging levels.
    logging.get_absl_handler().use_absl_log_file('log_file', log_dir)
    logging.set_stderrthreshold(logging.INFO)

    # Fix seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info(f'Random seed is {seed}.')


def set_device(force_device):

    if force_device in {'cuda', 'mps', 'cpu'}:
        return torch.device(force_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dict_to_defaultdict(config):
    """Recursively convert all dicts to defaultdicts."""
    if not isinstance(config, collections.defaultdict):
        if isinstance(config, dict):
            new_config = collections.defaultdict(collections.defaultdict)
            new_config.update({
                k: _dict_to_defaultdict(v) for k, v in config.items()
            })
            return new_config
    return config


def get_config(config_file, output_dir):
    """Parse config.

    Args:
        config_file: path to config file,
            consists of config dictionaries.
        output_dir: path to output directory to
            copy config and make it easier to reproduce results
    Returns:
        config: dataclass with config dictionaries.
    Raises:
        FileNotFoundError if config_file was not found.
    """
    if os.path.exists(config_file):
        shutil.copyfile(config_file,
                        os.path.join(output_dir, os.path.basename(config_file)))
        config = runpy.run_path(config_file)
        # TODO(oleg): make config immutable.
        dd_config = _dict_to_defaultdict(config)
        return Config(model=dd_config['model'],
                      training=dd_config['training'],
                      data=dd_config['data'])
    raise FileNotFoundError(f'File {config_file} was not found.')


def create_trainer(config, model, visualizers, output_dir, device):
    """Get appropriate trainer.

    Args:
        config: Dict, config.
        model: nn.Module.
        output_dir: String, path to output directory.
    Returns:
        trainer: instance of trainer
    Raises:
        ValueError if trainer name is unknown.
    """
    trainer_name = config['trainer_name']
    if trainer_name == 'mnist_classification_trainer':
        return mnist_classification_trainer.MnistTrainer(
            config, model, output_dir, device, visualizers)
    elif trainer_name == 'mnist_stylize_trainer':
        return mnist_stylize_trainer.MnistTrainer(config, model, output_dir,
                                                  device, visualizers)
    elif trainer_name == 'mnist_rotate_trainer':
        return mnist_rotate_trainer.MnistTrainer(config, model, output_dir,
                                                 device, visualizers)
    raise ValueError(f'Unknown trainer was given: {trainer_name}.')


def create_visualizers(config, output_dir):
    """Get appropriate visuzalizers.

    Args:
        config: Dict, config.
        output_dir: String, path to output directory.
    Returns:
        visualizers: dict of different visualizers
    Raises:
        ValueError if visualizer name is unknown.
    """

    vis_dict = collections.defaultdict(lambda: None)
    visualizer_names = config.get('visualizers', {})
    for vis_type, vis_name in visualizer_names.items():
        if vis_name == 'scalar_visualizer':
            vis_dict[vis_type] = scalar_visualizer.ScalarTensorboardVisualizer(
                output_dir)
        elif vis_name == 'image_visualizer':
            vis_dict[vis_type] = image_visualizer.ImageTensorboardVisualizer(
                output_dir)
        else:
            raise ValueError(f'Unknown visualizer was given: {vis_name}.')
    return vis_dict


def create_loader(data_config, model_config, device):
    """Get appropriate loader.

    Args:
        data_config: Dict, data config.
        model_config: Dict, model config.
        device: string
    Returns: loader: instances of train, val, test loaders
    Raises:
        ValueError if task_name is unknown.
    """
    task_name = data_config.pop('task')
    batch_size = model_config['batch_size']

    kwargs = {}
    num_workers = data_config.pop('num_workers', 0)
    if num_workers > 0:
        kwargs = {'pin_memory': False, 'multiprocessing_context': 'fork'}

    if task_name == 'mnist_classification':

        factory = mnist_classification_dataset.MnistInrDatasetFactory(
            data_config, device)
        train_loader, val_loader, test_loader = factory.split()

    elif task_name == 'mnist_stylize':

        factory = mnist_stylize_dataset.MnistInrDatasetFactory(
            data_config, device)
        train_loader, val_loader, test_loader = factory.split()

    elif task_name == 'mnist_rotate':

        factory = mnist_rotate_dataset.MnistInrDatasetFactory(
            data_config, device)
        train_loader, val_loader, test_loader = factory.split()

    else:
        raise ValueError(f'Unknown task was given: {task_name}.')

    train_loader = torch.utils.data.DataLoader(train_loader,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(val_loader,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             **kwargs)
    return train_loader, val_loader, test_loader


def _load_state_dict(state_dict_path):
    """Load state_dict from path.

    Args:
        state_dict_path: Path to state_dict.
    Returns:
        state_dict: state_dict for the network.
        step: Integer, current step.
    Raises:
        FileNotFoundError if state_dict cannot be found or not pt.
    """

    if not os.path.exists(state_dict_path) or not state_dict_path.endswith(
            'pt'):
        raise ValueError(
            'Please give path to valid checkpoint, or '
            f'directory with checkpoints, given: {state_dict_path}.')

    # TODO(oleg): there is a bug in regex.
    step = int(re.match(r'(.*)state_dict_(.*).pt', state_dict_path).group(2))
    state_dict = torch.load(state_dict_path, map_location='cpu')

    return state_dict, step


def create_network(network_configs, state_dict_path):

    networks_dict = {
        'regnet_y_800mf': regnet_y_800mf,
        'dwsnet_classification': DWSModelForClassification,
        'dwsnet': DWSModel,
        'transfer_net': TransferNet,
    }

    network_name = network_configs['network_name']
    network_params = network_configs['network_params']

    if network_name not in networks_dict:
        raise ValueError(f'Unknown network name is given: {network_name}.')

    network = networks_dict[network_name](**network_params).float()
    init_step = 0

    # NOTE(oleg): this logic is quite bad and should be changed.
    # Ways to load stuff:
    # 1. setting state_dict in "network" part of the config: this will load
    # state dict and is made for finetuning of the models trained with this pipeline.
    # 2. setting weights flag during prediction: in this case we just load everything
    # for prediction.
    # Priority is 2 > 1, i.e. if we set "weights" flag it will override
    # whatever state_dict is in the config.

    if state_dict_path is None:
        state_dict_path = network_configs.get('state_dict', None)

    if state_dict_path is not None:
        network_state_dict, init_step = _load_state_dict(state_dict_path)
        network.load_state_dict(network_state_dict, strict=True)
        logging.info(f'Weights for {network_name} are loaded from '
                     f'state_dict {state_dict_path}.')

    return init_step, network


def create_model(model_configs, device, weights=None):
    """Choose the model.

    Args:
        model_configs: Dictionary, model config, must include networks' config.
    Returns:
        model: nn.Module object.
    """

    init_step, network = create_network(model_configs['network'], weights)

    if model_configs['model_name'] == 'mnist_classification_model':
        model = mnist_classification_model.MnistInrClassificationModel(
            config=model_configs,
            network=network,
            init_step=init_step,
            device=device)
    elif model_configs['model_name'] == 'mnist_stylize_model':
        model = mnist_stylize_model.MnistInrStylizeModel(config=model_configs,
                                                         network=network,
                                                         init_step=init_step,
                                                         device=device)
    elif model_configs['model_name'] == 'mnist_rotate_model':
        model = mnist_rotate_model.MnistInrRotateModel(config=model_configs,
                                                       network=network,
                                                       init_step=init_step,
                                                       device=device)
    else:
        raise ValueError(f'Unknown model name: {model_configs["model_name"]}.')

    return model
