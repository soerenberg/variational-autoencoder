"""Export training images closest to the example images.

This script loads the example images included in checkpoint data and finds
the instances in the training set which are closest to those with respect to
the Euclidean distance in latent space, i.e. after applying the encoder.
"""
import argparse
import functools
import pathlib
from typing import List, Tuple

import tensorflow as tf
import tqdm

import checkpointing
import matplotlib
import matplotlib.pyplot as plt  # noqa: F401
import train_vae


def _encode_preserve_data_and_label(
        encoder: tf.keras.Model, tensor: tf.Tensor,
        label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Encode a single data point, pass through the data pt and label."""
    return tf.split(encoder(tensor[tf.newaxis, ...]),
                    num_or_size_splits=2,
                    axis=1)[0][0], tensor, label


def _euclidean_dist(
        batch: tf.Tensor, latent_tensor: tf.Tensor, data_pt: tf.Tensor,
        label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute Euclidean distance between a tensor and a batch of tensors."""
    squared_diff = tf.square(latent_tensor[tf.newaxis, ...] - batch)
    return tf.reduce_sum(squared_diff, axis=1), data_pt, label


def _get_closest_latent_points(
    latent_dataset: tf.data.Dataset
) -> List[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """Compute closest data points from a transformed data set."""
    current_closest_images: List[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]] = []
    for dists, img, lbl in tqdm.tqdm(latent_dataset):
        if not current_closest_images:
            current_closest_images = [(d.numpy(), img, lbl) for d in dists]
            continue

        for i, dist in enumerate(dists):
            if current_closest_images[i][0] > dist:
                current_closest_images[i] = (dist.numpy(), img, lbl)
    return current_closest_images


def _export_images(closest_images: List[Tuple[tf.Tensor, tf.Tensor,
                                              tf.Tensor]],
                   image_dir: pathlib.Path):
    """Export images to a directory."""
    for i, (_, image, label) in enumerate(closest_images):
        # for monotone images replicate the sole channel
        if image.shape[-1] == 1:
            image = tf.tile(image, (1, 1, 3))
        matplotlib.image.imsave(
            image_dir / f"train_image_closest_to_{i}_label_{label}.png",
            image.numpy())


def parse_cmd_line_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        action="store",
        dest="model_dir",
        required=True,
        help="Root directory for saving checkpoints and events.")
    parser.add_argument("--dataset",
                        "-d",
                        action="store",
                        dest="dataset",
                        required=True,
                        help="Name of training dataset, e.g. 'cifar10'.")
    parser.add_argument("--latent_dim",
                        action="store",
                        dest="latent_dim",
                        type=int,
                        default=2,
                        help="Latent dimension of the VAE. Defaults to 2.")
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Verbosity level. Repeat to increase. '-vvv' equals logging info."
    )
    return parser.parse_args()


def run() -> None:
    """Execute training step(s) for the VAE model."""
    parsed_args = parse_cmd_line_args()
    # set_up_logging(parsed_args.verbose)

    latent_dim = parsed_args.latent_dim
    model_dir = parsed_args.model_dir

    model_dir = pathlib.Path(model_dir)

    checkpoint_dir = model_dir / "checkpoints"
    reader = tf.train.load_checkpoint(checkpoint_dir)

    fixed_latents = reader.get_tensor(
        "fixed_latents/.ATTRIBUTES/VARIABLE_VALUE")

    train_dataset, _, input_shape = train_vae.fetch_datasets(
        parsed_args.dataset)

    model = train_vae.build_model(latent_dim, input_shape)

    check_pt, check_pt_manager = checkpointing.init_checkpoint_and_manager(
        checkpoint_path=checkpoint_dir, model=model)
    check_pt.restore(check_pt_manager.latest_checkpoint).expect_partial()

    if not check_pt_manager.latest_checkpoint:
        print(f"No (latest) checkpoint data found in {checkpoint_dir}. Abort.")
        return

    latent_dataset = train_dataset.map(
        functools.partial(_encode_preserve_data_and_label, model.encoder)).map(
            functools.partial(_euclidean_dist, fixed_latents))

    _export_images(_get_closest_latent_points(latent_dataset),
                   model_dir / "images")


if __name__ == "__main__":
    run()
