"""Evaluation file.

python evaluate.py \
    --config=./configs/tsplice_config.py \
    --output_dir=<output directory path> \
    --weights=<state_dict.pt path>
"""

from absl import app
import os

import flags
import numpy as np
import torch
from skimage import metrics
from PIL import Image

from scipy import ndimage

from utils import pipeline_utils as pu
from visualizers import image_visualizer

FLAGS = flags.FLAGS


def main(_):
    # Set config, output directory, logging levels and random seeds.
    pu.initialize(output_dir=FLAGS.output_dir,
                  seed=FLAGS.random_seed,
                  use_gpus=FLAGS.use_gpus)
    config = pu.get_config(FLAGS.config, FLAGS.output_dir)
    device = pu.set_device(FLAGS.device)

    config.data['enforced_statistics'] = './stat/default_stat.json'

    # Create dataset for test.
    _, _, test_loader = pu.create_loader(config.data, config.model, device)

    # Load or create model.
    model = pu.create_model(config.model, device, FLAGS.weights)
    model.to(device)
    model.eval()

    def _psnr(target, prediction):
        target = target.squeeze(0).squeeze(0).detach().cpu().numpy()
        prediction = prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
        return metrics.peak_signal_noise_ratio(target,
                                               prediction,
                                               data_range=1.)

    def _ssim(target, prediction):
        target = target.squeeze(0).squeeze(0).detach().cpu().numpy()
        prediction = prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
        return metrics.structural_similarity(target, prediction, data_range=1.)

    vis = image_visualizer.ImageTensorboardVisualizer(FLAGS.output_dir)

    # PSNR / SSIM calculation: whole set, random angles and translations
    metrics_dict = {'psnr': [], 'ssim': []}
    for i, sample in enumerate(test_loader):
        sample = torch.utils.data._utils.collate.default_collate([sample])
        _, (images_in, images_out, new_images) = model(sample, evaluation=True)
        metrics_dict['psnr'].append(_psnr(images_out, new_images))
        metrics_dict['ssim'].append(_ssim(images_out, new_images))

        if i < 10:
            log_image = torch.cat([images_in, images_out, new_images], dim=-1)
            vis.log(log_image, prefix=f'test_{i}', step=0)

    print('Metrics: ')
    for k, v in metrics_dict.items():
        print(k, np.mean(v))

    # Nice pics
    angles = np.linspace(-90, 90, 18)
    trans = np.concatenate([
        np.linspace(0.0, 0.4, 6),
        np.linspace(0.4, -0.4, 6),
        np.linspace(-0.4, 0.4, 6)
    ])
    for i, sample in enumerate(test_loader):

        images_in = []
        images_out = []
        new_images = []
        for angle, tr in zip(angles, trans):
            sample = test_loader.__getitem__(i,
                                             angle_input=angle,
                                             trans_input=[0., tr])
            sample = torch.utils.data._utils.collate.default_collate([sample])
            _, (image_in, image_out, new_image) = model(sample, evaluation=True)

            image_in = sample.image_in.unsqueeze(0)

            images_in.append(image_in)
            images_out.append(image_out)
            new_images.append(new_image)

        images_in = torch.cat(images_in, -1)
        images_out = torch.cat(images_out, -1)
        new_images = torch.cat(new_images, -1)
        log_image = torch.cat([images_in, images_out, new_images], dim=-2)

        log_image = log_image.squeeze(0).moveaxis(
            0, -1).squeeze(-1).detach().cpu().numpy()
        log_image = (log_image * 255).astype(np.uint8)
        log_image = ndimage.zoom(log_image, 5.0)

        im = Image.fromarray(log_image)
        im.save(f".samples/sample{i}.jpeg")


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'weights'])
    app.run(main)
