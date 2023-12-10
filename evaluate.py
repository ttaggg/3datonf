"""Evaluation file.

python evaluate.py \
    --config=./configs/tsplice_config.py \
    --output_dir=<output directory path> \
    --weights=<state_dict.pt path>
"""

from absl import app

import flags
import numpy as np
import torch

from utils import pipeline_utils as pu
from visualizers import image_visualizer
from loaders.mnist_stylize_dataset import Batch

FLAGS = flags.FLAGS

# TODO(oleg): clean everything here


def main(_):

    # NOTE(oleg): for model evaluation we need the config that was used
    # during the training, but the good news is config is always copied to the
    # respective output directory during training (together with performance
    # files, logs, tensorboard scalars and so on).

    # Set config, output directory, logging levels and random seeds.
    pu.initialize(output_dir=FLAGS.output_dir,
                  seed=FLAGS.random_seed,
                  use_gpus=FLAGS.use_gpus)
    config = pu.get_config(FLAGS.config, FLAGS.output_dir)
    device = pu.set_device(FLAGS.device)

    # Create dataset for test.
    _, _, test_data = pu.create_loader(config.data, config.model, device)

    def generator():
        for i in range(16):
            angles = [15.0]
            yield torch.utils.data._utils.collate.default_collate(
                [test_data.__getitem__(i, a) for a in angles])

    class IterDataset(torch.utils.data.IterableDataset):

        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return self.generator()

    test_loader = IterDataset(generator)

    # Visualizer
    vis = image_visualizer.ImageTensorboardVisualizer(FLAGS.output_dir)

    # Load or create model.
    model = pu.create_model(config.model, device, FLAGS.weights)
    model.eval()
    model.to(device)

    for i, sample in enumerate(test_loader):
        images = [sample.image_in]
        for j in range(6):

            _, params, new_image = model(sample, evaluation=True)
            new_weights, new_biases = params

            new_weights = [x.squeeze(dim=0) for x in new_weights]
            new_biases = [x.squeeze(dim=0) for x in new_biases]

            norm_new_weights, norm_new_bises = test_data._normalize_weights_biases(
                new_weights, new_biases)

            sample = torch.utils.data._utils.collate.default_collate([
                Batch(weights=norm_new_weights,
                      ori_weights=new_weights,
                      biases=norm_new_bises,
                      ori_biases=new_biases,
                      image_out=new_image,
                      image_in=new_image,
                      angle=15.)
            ])

            images.append(new_image)

        log_images = torch.cat(images, dim=-1)

        vis.log(log_images, prefix=f'test_{i}', step=0)


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'weights'])
    app.run(main)
