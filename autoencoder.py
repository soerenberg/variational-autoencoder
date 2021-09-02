from typing import NamedTuple, Tuple

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
                 latent_dim, **kwargs):
        super().__init__(**kwargs)

        self._input_shape = input_shape
        self._encoder_configs = tuple(encoder_configs)
        self._decoder_configs = tuple(decoder_configs)
        self._latent_dim = latent_dim

        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    @property
    def encoder(self) -> tf.keras.models.Model:
        """Accessor method"""
        return self._encoder

    @property
    def decoder(self) -> tf.keras.models.Model:
        """Accessor method"""
        return self._decoder

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
                   latent_dim=latent_dim)

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
    def train_step(self, tensor_batch: tf.Tensor,
                   optimizer: tf.keras.optimizers.Optimizer,
                   train_elbo: tf.keras.metrics.Metric):
        """Perform a training step / epoch.

        Args:
            tensor_batch: Batched tensor for training.
            optimizer: optimizer to apply gradients to.
            train_elbo: metric to collect the mean elbo of the batch.
        """
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(vae_loss.compute_loss(self, tensor_batch))
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        train_elbo(-loss)
