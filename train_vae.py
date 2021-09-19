"""Execute one or more training steps from the command line."""
import argparse
import functools
import logging
import pathlib
from typing import Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

import checkpointing
import autoencoder
import vae_loss


def build_model(
        latent_dim: int,
        input_shape: Tuple[int, int,
                           int]) -> autoencoder.VariationalAutoEncoder:
    encoder = autoencoder.Encoder(input_shape=input_shape,
                                  latent_dim=latent_dim,
                                  config=[
                                      autoencoder.EncoderConfig(32, 3, 1),
                                      autoencoder.EncoderConfig(64, 3, 2),
                                      autoencoder.EncoderConfig(64, 3, 2),
                                      autoencoder.EncoderConfig(64, 3, 1)
                                  ])
    decoder = autoencoder.Decoder(input_shape=encoder.shape_before_flattening,
                                  latent_dim=latent_dim,
                                  config=[
                                      autoencoder.DecoderConfig(64, 3, 1),
                                      autoencoder.DecoderConfig(64, 3, 2),
                                      autoencoder.DecoderConfig(32, 3, 2),
                                      autoencoder.DecoderConfig(
                                          input_shape[-1], 3, 1)
                                  ])

    return autoencoder.VariationalAutoEncoder(encoder=encoder, decoder=decoder)


def preprocessing(images: np.ndarray) -> np.ndarray:
    """Simple preprocessing for input data."""
    if images.ndim == 3:
        images = images[..., np.newaxis]
    if images.ndim != 4:
        raise ValueError(f"Input data has invalid shape {images.shape}.")

    # Scale into range [0, 1]
    min_value, max_value = images.min(), images.max()
    images = (images + min_value) / (max_value + min_value)

    return images.astype("float32")


def fetch_datasets(
        dataset: str
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.TensorShape]:
    """Load dataset and apply rudimentary preprocessing.

    Args:
        dataset: name of dataset to be loaded.

    Returns:
        training dataset
        test dataset
        input shape of a single image
    """
    if dataset in ["cifar10", "cifar100", "fashion_mnist", "mnist"]:
        raw_dataset = getattr(tf.keras.datasets, dataset)
    else:
        raise ValueError(f"Dataset '{dataset}' unknown.")

    (train_images, train_labels), (test_images,
                                   test_labels) = raw_dataset.load_data()

    train_images = preprocessing(train_images)
    test_images = preprocessing(test_images)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).shuffle(len(train_images))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels)).shuffle(len(test_images))

    return train_dataset, test_dataset, train_images.shape[1:]


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
            (num batches, width, height, channels). Number of batches must be
            a square.
        image_dir: directory to export images into.
        global_step: global step count to be included into the image file
            names.
    """
    num_images = images.shape[0]
    edge_len = np.sqrt(num_images).astype(int)
    if edge_len**2 != num_images:
        raise ValueError(
            f"Number of example images must be a square, found {num_images}.")

    image_list = []
    image_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        img_count = str(i).zfill(int(np.ceil(np.log10(len(images)))))
        step_count = str(global_step.numpy()).zfill(8)
        out_file = image_dir / f"image_{img_count}_step_{step_count}.png"

        # for monotone images replicate the sole channel
        if image.shape[-1] == 1:
            image = tf.tile(image, (1, 1, 3))
        matplotlib.image.imsave(out_file, image.numpy())
        image_list.append(image)

    # Stitch images together into a square
    image_table = np.concatenate([
        np.concatenate(image_list[i * edge_len:(i + 1) * edge_len], axis=0)
        for i in range(edge_len)
    ], axis=1)  # yapf:disable
    matplotlib.image.imsave(image_dir / f"grid_step_{step_count}.png",
                            image_table)


def get_cmap(num_labels: int) -> str:
    """Returns suitable cmap for given number of distinct labels."""
    if num_labels <= 10:
        return "tab10"
    if num_labels <= 20:
        return "tab20"
    return "inferno"


def export_planar_encoding_plot(encoder: autoencoder.Encoder,
                                dataset: tf.data.Dataset,
                                image_dir: pathlib.Path,
                                global_step: tf.Tensor,
                                kind: str = "",
                                max_pts: Optional[int] = None):
    """Export a figure of the encoding of a dataset on the latent space.

    If the latent space has a higher dimension than 2, it will be projected
    into planar space using PCA first.

    Args:
        encoder: encoder to encode the dataset
        dataset: dataset to encode
        image_dir: directory to export image into
        global_step: global step to be included into the file name
        kind: will be included into the filename, e.g. 'train' or 'test'
        max_pts: the first `max_pts` of dataset will be included in the plot,
            or if `None` all points will be included. Defaults to None.
    """
    def encode(encoder: tf.keras.Model, tensor: tf.Tensor,
               label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode a single data point."""
        return tf.split(encoder(tensor[tf.newaxis, ...]),
                        num_or_size_splits=2,
                        axis=1)[0][0], label

    max_pts = max_pts or len(dataset)
    dataset = dataset.map(functools.partial(
        encode, encoder)).take(max_pts).batch(max_pts)
    images, labels = next(iter(dataset))

    images = images.numpy()
    labels = labels.numpy()

    pca_note = ""
    if images.shape[1] > 2:
        pca_note = f", projected dim. {images.shape[1]} to dim. 2 with PCA"
        images = PCA(n_components=2).fit_transform(images)

    fig, ax = plt.subplots(figsize=(10, 10))  # pylint: disable=invalid-name

    scatter = ax.scatter(images[:, 0],
                         images[:, 1],
                         c=labels,
                         label=labels,
                         cmap=get_cmap(num_labels=len(np.unique(labels))),
                         s=16,
                         alpha=1.)

    # Since points are standard normally distributed we can use limits safely.
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)

    ax.legend(*scatter.legend_elements(), loc="upper right", title="Labels")
    ax.set_title(
        f"Encoding on planar latent space ({kind}) - {images.shape[0]} pts"
        f"{pca_note}.")

    step_count = str(global_step.numpy()).zfill(8)
    fig.savefig(image_dir / f"planar_encoding_{kind}_step_{step_count}.png", )


def get_indices_of_closest_vectors(elements: tf.Tensor,
                                   tensors: tf.Tensor) -> tf.Tensor:
    """Compute indices of closest points for a batch of tensors in a set.

    For each tensor `t` out of a given batch of tensors `elements`, where the
    first dimension references the different vectors, determines the index of
    of the closest vector in `tensors`, where again the first dimensions
    references the different instances.
    Here, 'closest' mean closest in terms of the Euclidean distance.

    Args:
        elements: (num_examples, dims)
        datasets: (size of set, dims)

    Returns:
        tf.Tensor of shape (num_examples,) and dtype tf.int64
    """
    diffs = elements[:, tf.newaxis, ...] - tensors[tf.newaxis, :, ...]
    squared_distances = tf.reduce_sum(tf.square(diffs), axis=2)
    return tf.argmin(squared_distances, axis=1)


def train_model(model,
                train_dataset,
                test_dataset,
                num_epochs,
                batch_size,
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

    fixed_latents = tf.Variable(tf.random.normal(shape=(16, latent_dim)),
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
        print(f"\nepoch {epoch}/{num_epochs}")
        progbar = tf.keras.utils.Progbar(train_dataset.cardinality().numpy(),
                                         stateful_metrics=[train_elbo])

        for i, (train_x, _) in enumerate(train_dataset.batch(batch_size)):
            model.train_step(train_x, optimizer, train_elbo)
            progbar.update((i + 1) * batch_size,
                           values=[("train_elbo", train_elbo.result())])

        checkpointing.write_checkpoint_if_necesssary(check_pt,
                                                     check_pt_manager,
                                                     check_pt_every_n_epochs)

        logging.info("Evaluate test set ELBO")
        for test_x, _ in test_dataset.batch(batch_size):
            test_elbo(-tf.reduce_mean(vae_loss.compute_loss(model, test_x)))

        example_images = tf.nn.sigmoid(model.decoder(fixed_latents))
        global_step.assign_add(1)
        write_events(writer,
                     scalars=dict(train_elbo=train_elbo.result(),
                                  test_elbo=test_elbo.result(),
                                  learning_rate=learning_rate,
                                  batch_size=batch_size),
                     images=dict(example_images=example_images),
                     step=global_step)
        export_images(example_images, model_dir / "images", global_step)

        export_planar_encoding_plot(encoder=model.encoder,
                                    dataset=train_dataset,
                                    image_dir=model_dir / "images",
                                    global_step=global_step,
                                    kind="train",
                                    max_pts=10000)
        export_planar_encoding_plot(encoder=model.encoder,
                                    dataset=test_dataset,
                                    image_dir=model_dir / "images",
                                    global_step=global_step,
                                    kind="test",
                                    max_pts=10000)
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
    parser.add_argument("--num_epochs",
                        action="store",
                        dest="num_epochs",
                        type=int,
                        default=1,
                        help="Num epochs to train. Defaults to 1.")
    parser.add_argument("--batch_size",
                        action="store",
                        dest="batch_size",
                        type=int,
                        default=32,
                        help="Batch size. Defaults to 32.")
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
    set_up_logging(parsed_args.verbose)
    train_dataset, test_dataset, input_shape = fetch_datasets(
        parsed_args.dataset)
    model = build_model(latent_dim=parsed_args.latent_dim,
                        input_shape=input_shape)

    train_model(model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                num_epochs=parsed_args.num_epochs,
                batch_size=parsed_args.batch_size,
                learning_rate=parsed_args.learning_rate,
                latent_dim=parsed_args.latent_dim,
                model_dir=pathlib.Path(parsed_args.model_dir),
                check_pt_every_n_epochs=1)


if __name__ == "__main__":
    run()
