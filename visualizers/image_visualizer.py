"""Visuzalize scalar data in Tensorflow."""

import os

from torch.utils import tensorboard


class ImageTensorboardVisualizer:

    def __init__(self, output_dir):
        self._image_writer = tensorboard.SummaryWriter(
            os.path.join(output_dir, 'images'))

    def log(self, images, prefix, step, meta=None):

        for i, image in enumerate(images):
            suffix = '' if meta is None else meta[i]
            self._image_writer.add_image(f'{prefix}image_{i}_{suffix}', image, step)
