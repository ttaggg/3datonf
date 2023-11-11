"""Train file.

python train.py \
    --config=./configs/tsplice_config.py \
    --output_dir=<output directory path>
"""

import flags
import torch

from absl import app
from absl import logging

from utils import pipeline_utils as pu

FLAGS = flags.FLAGS


def main(_):
    """Run pipeline."""

    # Set config, output directory, logging levels and random seeds.
    pu.initialize(output_dir=FLAGS.output_dir,
                  seed=FLAGS.random_seed,
                  use_gpus=FLAGS.use_gpus)
    config = pu.get_config(FLAGS.config, FLAGS.output_dir)
    device = pu.set_device(FLAGS.device)

    # Create datasets for train and val.
    batch_size = config.training['batch_size']
    train_data, val_data, _ = pu.create_loader(config.data, batch_size, device)

    # Load or create model.
    model = pu.create_model(config.model)

    # Create visualizers
    visualizers = pu.create_visualizers(config.training, FLAGS.output_dir)

    # Create trainer and initialize everything.
    trainer = pu.create_trainer(config.training, model, visualizers,
                                FLAGS.output_dir, device)
    trainer.set_for_training(init_step=model.init_step)
    trainer.set_for_evaluation(model.init_step, val_data)

    # Start training.
    for epoch in range(1, config.training['num_epochs'] + 1):
        logging.info(f'Epoch {epoch} is starting.')

        for step, data_batch in enumerate(train_data):
            trainer.train_step(data_batch)
            if step % 100 == 0:
                logging.info(f'{step} / {len(train_data)} steps are done.')

        trainer.on_epoch_end()
        logging.info(f'Epoch {epoch} is done.')


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    app.run(main)
