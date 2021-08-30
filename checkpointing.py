"""Functions to write and restore TensorFlow checkpoints."""
import logging
import pathlib
from typing import Optional, Tuple

import tensorflow as tf


def init_checkpoint_and_manager(
        checkpoint_path: pathlib.Path,
        **kwargs) -> Tuple[tf.train.Checkpoint, tf.train.CheckpointManager]:
    """Initialize checkpoint and checkpoint manager.

    Args:
        checkpoint_path: directory to save checkpoints into.
        **kwargs: additional keyword arguments to be included into the
            checkpoints.

    Returns:
        tf.train.Checkpoint
        tf.train.CheckpointManager
    """
    check_pt = tf.train.Checkpoint(step=tf.Variable(1), **kwargs)
    check_pt_manager = tf.train.CheckpointManager(check_pt,
                                                  checkpoint_path,
                                                  max_to_keep=10)
    return check_pt, check_pt_manager


def write_checkpoint_if_necesssary(
        check_pt: tf.train.Checkpoint,
        check_pt_manager: tf.train.CheckpointManager,
        check_pt_every_n_epochs: Optional[int]):
    """Write checkpoint if necessary and update checkpoint step.

    Args:
        check_pt: checkpoint to be updated / written
        check_pt_manager: checkpoint manager object
        check_pt_every_n_epochs: every `check_pt_every_n_epochs` epochs a
            checkpoint is written. If None no checkpoints are written.
    """
    check_pt.step.assign_add(1)
    if check_pt_every_n_epochs and int(
            check_pt.step) % check_pt_every_n_epochs == 0:
        save_path = check_pt_manager.save()
        logging.info("Saved checkpoint for step %s: %s", int(check_pt.step),
                     save_path)


def restore_checkpoint_if_exists(check_pt: tf.train.Checkpoint,
                                 check_pt_manager: tf.train.CheckpointManager):
    """Restore from checkpoints if prior checkpoints exist.

    Args:
        check_pt: checkpoint to be updated / written
        check_pt_manager: checkpoint manager object
    """
    check_pt.restore(check_pt_manager.latest_checkpoint)
    if check_pt_manager.latest_checkpoint:
        logging.info("Restored from %s", check_pt_manager.latest_checkpoint)
    else:
        logging.info("Initializing from scratch.")
