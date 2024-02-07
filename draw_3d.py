"""Evaluation file.

python evaluate.py \
    --config=./configs/tsplice_config.py \
    --output_dir=<output directory path> \
    --weights=<state_dict.pt path>
"""

from absl import app

import flags
import torch
from skimage import measure

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec

from utils import pipeline_utils as pu

FLAGS = flags.FLAGS


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
    _, _, test_loader = pu.create_loader(config.data, config.model, device)

    # Load or create model.
    model = pu.create_model(config.model, device, FLAGS.weights)
    model.to(device)
    model.eval()

    for _, sample in enumerate(test_loader):
        sample = torch.utils.data._utils.collate.default_collate([sample])
        _, (sdf_in, sdf_out, new_sdf) = model(sample, evaluation=True)

        sdf_in = sdf_in.detach().squeeze(0).cpu().numpy()
        verts, faces, normals, values = measure.marching_cubes(sdf_in)

        gs = gridspec.GridSpec(1, 3)

        fig = plt.figure(figsize=plt.figaspect(0.33))
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        mesh = Poly3DCollection(verts[faces],
                                linewidths=1,
                                shade=True,
                                facecolors=[0.5, 0.5, 1])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.title.set_text('from the input weights')
        ax.set_xlim(0., 21)
        ax.set_ylim(0., 21)
        ax.set_zlim(0., 21)

        new_sdf = new_sdf.detach().squeeze(0).cpu().numpy()
        verts, faces, normals, values = measure.marching_cubes(new_sdf)

        ax = fig.add_subplot(gs[0, 2], projection='3d')
        mesh = Poly3DCollection(verts[faces],
                                linewidths=1,
                                shade=True,
                                facecolors=[0.5, 0.5, 1])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.title.set_text('from the output weights')
        ax.set_xlim(0., 21)
        ax.set_ylim(0., 21)
        ax.set_zlim(0., 21)

        sdf_out = sdf_out.detach().squeeze(0).cpu().numpy()
        verts, faces, normals, values = measure.marching_cubes(sdf_out)
        ax = fig.add_subplot(gs[0, 1], projection='3d')
        mesh = Poly3DCollection(verts[faces],
                                linewidths=1,
                                shade=True,
                                facecolors=[0.5, 0.5, 1])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.title.set_text('ground truth')
        ax.set_xlim(0., 21)
        ax.set_ylim(0., 21)
        ax.set_zlim(0., 21)
        plt.show()


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'weights'])
    app.run(main)
