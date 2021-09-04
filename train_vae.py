"""Execute one or more training steps from the command line."""
import argparse
import logging
import pathlib
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import checkpointing
import autoencoder
import vae_loss


def build_model(latent_dim: int) -> autoencoder.VariationalAutoEncoder:
    return autoencoder.VariationalAutoEncoder(
        input_shape=(28, 28, 1),
        encoder_configs=[
            autoencoder.EncoderConfig(32, 3, 1),
            autoencoder.EncoderConfig(64, 3, 2),
            autoencoder.EncoderConfig(64, 3, 2),
            autoencoder.EncoderConfig(64, 3, 1)
        ],
        decoder_configs=[
            autoencoder.DecoderConfig(64, 3, 1),
            autoencoder.DecoderConfig(64, 3, 2),
            autoencoder.DecoderConfig(32, 3, 2),
            autoencoder.DecoderConfig(1, 3, 1)
        ],
        latent_dim=latent_dim)


def fetch_datasets():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    train_images = train_images[..., np.newaxis] / 255.
    test_images = test_images[..., np.newaxis] / 255.

    train_images = train_images.astype("float32")
    test_images = test_images.astype("float32")

    batch_size = 32

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images,
         train_labels)).shuffle(len(train_images)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels)).shuffle(len(test_images)).batch(batch_size)

    return train_dataset, test_dataset


def write_events(writer, scalars, images, step):
    with writer.as_default():
        for name, value in scalars.items():
            tf.summary.scalar(name, value, step=step)
        for name, values in images.items():
            tf.summary.image(name,
                             values,
                             step=step,
                             max_outputs=values.shape[0])


def export_images(images: tf.Tensor, image_dir: pathlib.Path,
                  global_step: tf.Tensor) -> None:
    """Export batch of images to separate png files.

    Args:
        images: tensorial image batch. Must have shape
            (num batches, width, height, channels).
        image_dir: directory to export images into.
        global_step: global step count to be included into the image file
            names.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        img_count = str(i).zfill(int(np.ceil(np.log10(len(images)))))
        step_count = str(global_step.numpy()).zfill(8)
        out_file = image_dir / f"image_{img_count}_step_{step_count}.png"

        # for monotone images replicate the sole channel
        if image.shape[-1] == 1:
            image = tf.tile(image, (1, 1, 3))
        matplotlib.image.imsave(out_file, image.numpy())


def train_model(model,
                train_dataset,
                test_dataset,
                num_epochs,
                learning_rate,
                latent_dim,
                model_dir,
                check_pt_every_n_epochs=None):
    model_dir = pathlib.Path(model_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    global_step = tf.Variable(0,
                              name="global_step",
                              trainable=False,
                              dtype=tf.int64)

    fixed_latents = tf.Variable(tf.random.normal(shape=(10, latent_dim)),
                                name="fixed_latents",
                                trainable=False)

    check_pt, check_pt_manager = checkpointing.init_checkpoint_and_manager(
        checkpoint_path=model_dir / "checkpoints",
        optimizer=optimizer,
        model=model,
        iterator=iter(train_dataset),
        fixed_latents=fixed_latents,
        global_step=global_step)

    checkpointing.restore_checkpoint_if_exists(check_pt, check_pt_manager)

    train_elbo = tf.keras.metrics.Mean("train_elbo")
    test_elbo = tf.keras.metrics.Mean("test_elbo")

    writer = tf.summary.create_file_writer(str(model_dir / "events"))

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        for train_x, _ in train_dataset:
            model.train_step(train_x, optimizer, train_elbo)

        checkpointing.write_checkpoint_if_necesssary(check_pt,
                                                     check_pt_manager,
                                                     check_pt_every_n_epochs)

        for test_x, _ in test_dataset:
            test_elbo(-tf.reduce_mean(vae_loss.compute_loss(model, test_x)))

        elapsed_time = time.time() - start_time

        example_images = tf.nn.sigmoid(model.decoder(fixed_latents))
        global_step.assign_add(1)
        write_events(writer,
                     scalars=dict(train_elbo=train_elbo.result(),
                                  test_elbo=test_elbo.result(),
                                  learning_rate=learning_rate),
                     images=dict(example_images=example_images),
                     step=global_step)
        export_images(example_images, model_dir / "images", global_step)

        print(f"Epoch: {epoch}, mean test set ELBO {test_elbo.result()}, "
              f"time elapsed: {elapsed_time}")
    return model


def set_up_logging(rank: int) -> None:
    """Get Logger and set verbosity level.

    Args:
        rank: level to set, must be in {0, 1, ..., 5}
    """
    logger = logging.getLogger()
    level = 50 - 10 * rank
    logger.setLevel(level=level)


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
    parser.add_argument("--latent_dim",
                        action="store",
                        dest="latent_dim",
                        type=int,
                        default=2,
                        help="Latent dimension of the VAE. Defaults to 2.")
    parser.add_argument("--num_epochs",
                        action="store",
                        dest="num_epochs",
                        type=int,
                        default=1,
                        help="Num epochs to train. Defaults to 1.")
    parser.add_argument("--learning_rate",
                        action="store",
                        dest="learning_rate",
                        type=float,
                        default=.0005,
                        help="Learning rate. Defaults to 0.0005.")
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
    train_dataset, test_dataset = fetch_datasets()
    model = build_model(latent_dim=parsed_args.latent_dim)

    train_model(model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                num_epochs=parsed_args.num_epochs,
                learning_rate=parsed_args.learning_rate,
                latent_dim=parsed_args.latent_dim,
                model_dir=pathlib.Path(parsed_args.model_dir),
                check_pt_every_n_epochs=1)


if __name__ == "__main__":
    run()
