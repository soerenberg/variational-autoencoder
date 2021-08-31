import pathlib
import time
from typing import NamedTuple, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

import checkpointing
import vae_loss


class EncoderConfig(NamedTuple):
    """NamedTuple for configuring a Conv2D layer in an encoder.

    Fields:
        filter: num filters in layer
        kernel_size: kernel size in layer
        stride: stride in layer
    """
    filter: int
    kernel_size: int
    stride: int


class DecoderConfig(NamedTuple):
    """NamedTuple for configuring a Conv2DTranspose layer in an encoder.

    Fields:
        t_filter: num filters in layer
        t_kernel_size: kernel size in layer
        t_stride: stride in layer
    """
    t_filter: int
    t_kernel_size: int
    t_stride: int


class VariationalAutoEncoder(keras.Model):
    """Variational Auto Encoder model."""
    def __init__(self, input_shape, encoder_configs, decoder_configs,
                 latent_dim, learning_rate, **kwargs):
        super().__init__(**kwargs)

        self._input_shape = input_shape
        self._encoder_configs = tuple(encoder_configs)
        self._decoder_configs = tuple(decoder_configs)
        self._latent_dim = latent_dim

        self._learning_rate = learning_rate

        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    @classmethod
    def from_latent_dim(cls, latent_dim):
        """Create a VAE with predefined architecture with desired latent dim.
        """
        return cls(input_shape=(28, 28, 1),
                   encoder_configs=[
                       EncoderConfig(32, 3, 1),
                       EncoderConfig(64, 3, 2),
                       EncoderConfig(64, 3, 2),
                       EncoderConfig(64, 3, 1)
                   ],
                   decoder_configs=[
                       DecoderConfig(64, 3, 1),
                       DecoderConfig(64, 3, 2),
                       DecoderConfig(32, 3, 2),
                       DecoderConfig(1, 3, 1)
                   ],
                   latent_dim=latent_dim,
                   learning_rate=0.0005)

    def _build_encoder(self):
        inputs = tf.keras.layers.Input(self._input_shape, name="encoder_input")

        tensor = inputs
        for i, conf in enumerate(self._encoder_configs):
            tensor = tf.keras.layers.Conv2D(filters=conf.filter,
                                            kernel_size=conf.kernel_size,
                                            strides=conf.stride,
                                            padding="same",
                                            name=f"encoder_conv_{i}")(tensor)
            tensor = tf.keras.layers.BatchNormalization()(tensor)
            tensor = tf.keras.layers.LeakyReLU()(tensor)
            tensor = tf.keras.layers.Dropout(rate=.25)(tensor)

        tensor = tf.keras.layers.Flatten()(tensor)

        output = tf.keras.layers.Dense(self._latent_dim +
                                       self._latent_dim)(tensor)

        return keras.models.Model(inputs, output, name="encoder")

    def _build_decoder(self):
        inputs = tf.keras.layers.Input(shape=(self._latent_dim, ),
                                       name="decoder_input")

        tensor = tf.keras.layers.Dense(7 * 7 * 64)(inputs)
        tensor = tf.keras.layers.Reshape((7, 7, 64))(tensor)

        for i, conf in enumerate(self._decoder_configs):
            tensor = tf.keras.layers.Conv2DTranspose(
                filters=conf.t_filter,
                kernel_size=conf.t_kernel_size,
                strides=conf.t_stride,
                padding="same",
                name=f"decoder_conv_t_{i}")(tensor)

            if i < len(self._decoder_configs) - 1:
                tensor = tf.keras.layers.BatchNormalization()(tensor)
                tensor = tf.keras.layers.LeakyReLU()(tensor)
                tensor = tf.keras.layers.Dropout(rate=.25)(tensor)

        # Note that we did not add any activation on the end, the decoder
        # therefore returns values on the logit scale.

        decoder_output = tensor

        return tf.keras.models.Model(inputs, decoder_output)

    def encode(self, tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mean, log_var = tf.split(self._encoder(tensor),
                                 num_or_size_splits=2,
                                 axis=1)
        return mean, log_var

    def decode(self, z: tf.Tensor, apply_sigmoid=False) -> tf.Tensor:
        logits = self._decoder(z)

        if apply_sigmoid:
            return tf.sigmoid(logits)

        return logits

    @staticmethod
    def sample_from_latent_conditional(mean: tf.Tensor,
                                       log_var: tf.Tensor) -> tf.Tensor:
        """Sample from the from latent space.

        This method returns samples from the distribution
            p(z | X), parametrized by `mean` and `log_var`.

        Args:
            mean: mean of the normal distribution to be sampled from.
            log_var: log variange of the normal distribution to be sampled
                from. Must have the same shape as `mean`.

        Returns:
            tf.Tensor: tensor of same shape as `mean` and `log_var` containing
                samples of the specified normal distribution.
        """
        std_normal_samples = tf.random.normal(shape=mean.shape)
        return mean + std_normal_samples * tf.exp(.5 * log_var)


@tf.function
def train_step(model, tensor_batch, optimizer, train_elbo):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(vae_loss.compute_loss(model, tensor_batch))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_elbo(-loss)


def train_model(train_dataset,
                test_dataset,
                num_epochs,
                latent_dim,
                model_dir,
                check_pt_every_n_epochs=None):
    model = VariationalAutoEncoder.from_latent_dim(latent_dim=latent_dim)
    model_dir = pathlib.Path(model_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate=.0005)
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

        for train_x in train_dataset:
            train_step(model, train_x, optimizer, train_elbo)

        checkpointing.write_checkpoint_if_necesssary(check_pt,
                                                     check_pt_manager,
                                                     check_pt_every_n_epochs)

        for test_x in test_dataset:
            test_elbo(-tf.reduce_mean(vae_loss.compute_loss(model, test_x)))

        elapsed_time = time.time() - start_time

        with writer.as_default():
            tf.summary.scalar("train_elbo",
                              train_elbo.result(),
                              step=global_step)
            tf.summary.scalar("test_elbo",
                              test_elbo.result(),
                              step=global_step)

            imgs = tf.nn.sigmoid(model._decoder(fixed_latents))
            tf.summary.image("example_images",
                             imgs,
                             step=global_step,
                             max_outputs=imgs.shape[0])

        print(f"Epoch: {epoch}, mean test set ELBO {test_elbo.result()}, "
              f"time elapsed: {elapsed_time}")
        global_step.assign_add(1)
    return model


def get_datasets():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    train_images = train_images[..., np.newaxis] / 255.
    test_images = test_images[..., np.newaxis] / 255.

    train_images = train_images.astype("float32")
    test_images = test_images.astype("float32")

    batch_size = 32

    # TODO test sizen
    train_size = 100
    test_size = 10

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        train_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(
        test_size).batch(batch_size)

    return train_dataset, test_dataset
