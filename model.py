import time
from typing import NamedTuple, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


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
                 latent_dim, learning_rate, r_loss_factor, **kwargs):
        super().__init__(**kwargs)

        self._input_shape = input_shape
        self._encoder_configs = tuple(encoder_configs)
        self._decoder_configs = tuple(decoder_configs)
        self._latent_dim = latent_dim

        self._encoder = None
        self._decoder = None
        self._model = None

        self._learning_rate = learning_rate

        self._mu = None
        self._log_var = None
        self._r_loss_factor = r_loss_factor
        self._z = None

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
                   learning_rate=0.0005,
                   r_loss_factor=1000)

    def _build_encoder(self):
        inputs = tf.keras.layers.Input(self._input_shape, name="encoder_input")

        tensor = tf.keras.layers.Rescaling(scale=1. / 255.)(inputs)
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

        self._mu = tf.keras.layers.Dense(self._latent_dim, name="mu")(tensor)
        self._log_var = tf.keras.layers.Dense(self._latent_dim,
                                              name="log_var")(tensor)

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
            else:
                tensor = tf.keras.layers.Activation("sigmoid")(tensor)

        decoder_output = tf.keras.layers.Rescaling(scale=255.)(tensor)

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


def log_normal_pdf(sample, mean, logvar, raxis=1):
    # TODO
    log2pi = tf.math.log(2. * np.pi)  # TODO needed?
    return tf.reduce_sum(
        -.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model: VariationalAutoEncoder,
                 tensor_batch: tf.Tensor) -> tf.Tensor:
    """Compute VAE loss for a single batch.

    Args:
        model: VAE model
        tensor_batch: batch of training data points, e.g. for 3 channel
            pictures of size 28 x 28 this tensor will have the shape
                (batch_size, 28, 28, 3).

    Returns:
        tf.Tensor: a scalar tensor equal to the loss of the given batch
    """

    # mean, log_var have shape (batch_size, latent_dim)
    mean, log_var = model.encode(tensor_batch)

    # add random noise
    eps = tf.random.normal(shape=mean.shape)
    latent_z = mean + eps * tf.exp(.5 * log_var)

    x_logit = model.decode(latent_z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit,
                                                        labels=tensor_batch)

    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(latent_z, 0., 0.)
    logqz_x = log_normal_pdf(latent_z, mean, log_var)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, tensor_batch, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, tensor_batch)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train_model(train_dataset, test_dataset):
    model = VariationalAutoEncoder.from_latent_dim(latent_dim=2)
    print(model._encoder.summary())

    optimizer = tf.keras.optimizers.Adam(learning_rate=.0005)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        metric = VAE_loss()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)

        for test_x in test_dataset:
            metric.update_state(model, test_x)

        elapsed_time = time.time() - start_time

        print(
            f"Epoch: {epoch}, test set ELBO {metric.result()}, time elapsed: "
            "{elapsed_time}")


class VAE_loss(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elbo = self.add_weight("elbo", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, model, test_x) -> None:
        self.count.assign_add(tf.ones([], tf.float32))

        loss_value = compute_loss(model, test_x)
        self.elbo.assign_add(-loss_value)

    def result(self) -> tf.Tensor:
        return self.elbo / self.count


def get_datasets():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    batch_size = 32

    # TODO test sizen
    train_size = 100
    test_size = 10

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        train_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(
        test_size).batch(batch_size)

    return train_dataset, test_dataset
