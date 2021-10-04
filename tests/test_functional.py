"""Functional tests."""
import numpy as np
import pytest
import tensorflow as tf

import autoencoder
import train_vae


@pytest.mark.functional
@pytest.mark.parametrize("input_shape,train_size,test_size,latent_dim",
                         [((28, 28, 1), 11, 13, 2), ((28, 28, 3), 12, 10, 3),
                          ((16, 16, 1), 7, 2, 3), ((32, 32, 3), 2, 7, 2)])
def test_model_noise_run(input_shape, train_size, test_size, latent_dim,
                         tmp_path):
    """Simple functional test, building model on noise data."""
    np.random.seed(1)
    train_images = np.random.uniform(low=0,
                                     high=255.,
                                     size=(train_size, ) +
                                     input_shape).astype("float32")
    train_labels = np.random.choice(np.arange(10), size=train_size)

    test_images = np.random.uniform(low=0,
                                    high=255.,
                                    size=(test_size, ) +
                                    input_shape).astype("float32")
    test_labels = np.random.choice(np.arange(10), size=test_size)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).shuffle(len(train_images))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels)).shuffle(len(test_images))

    vautoencoder = autoencoder.VariationalAutoEncoder.from_latent_dim(
        latent_dim=latent_dim, input_shape=input_shape)

    learning_rate = .0005
    num_epochs = 3
    batch_size = 2

    train_vae.train_model(vautoencoder,
                          train_dataset,
                          test_dataset,
                          num_epochs,
                          batch_size,
                          learning_rate,
                          latent_dim,
                          tmp_path,
                          check_pt_every_n_epochs=1)
