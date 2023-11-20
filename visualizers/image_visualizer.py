"""Visuzalize scalar data in Tensorflow."""

import os

from torch.utils import tensorboard


class ImageTensorboardVisualizer:

    def __init__(self, output_dir):
        self._image_writer = tensorboard.SummaryWriter(
            os.path.join(output_dir, 'images'))

    def log(self, images, prefix, step):
        for i, image in enumerate(images):
            self._image_writer.add_image(f'{prefix}image_{i}', image, step)
