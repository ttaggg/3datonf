"""Evaluation file.

python evaluate.py \
    --config=./configs/tsplice_config.py \
    --output_dir=<output directory path> \
    --weights=<state_dict.pt path>
"""

from absl import app
from absl import logging

import flags
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

    # Create dataset for test.
    batch_size = config.training['batch_size']
    _, _, test_data = pu.create_loader(config.data, batch_size)

    # Load or create model.
    model = pu.create_model(config.model, FLAGS.weights)
    model.eval()

    # Create trainer and initialize everything.
    trainer = pu.create_trainer(config.training, model, FLAGS.output_dir)
    trainer.set_for_evaluation(model.init_step, test_data)

    # Run evaluation on the whole test set.
    logging.info(f'Evaluation is starting.')
    trainer.evaluate(prefix='test_')
    logging.info(f'Evaluation is done.')


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'weights'])
    app.run(main)
