"""All flags that are used in the pipeline."""

import os

from absl import flags

flags.DEFINE_string('output_dir', os.getcwd(), 'Path to output directory.')
flags.DEFINE_string('config', None, 'Path to train settings.')
flags.DEFINE_string('weights', None, 'Which checkpoint to use for prediction.')
flags.DEFINE_integer('random_seed', 42, 'Random seed value.')
flags.DEFINE_string('use_gpus', '0', 'List of GPUs to see.')

FLAGS = flags.FLAGS


def mark_flags_as_required(*args, **kwargs):
    """"Override absl.flags.mark_flags_as_required """
    flags.mark_flags_as_required(*args, **kwargs)
