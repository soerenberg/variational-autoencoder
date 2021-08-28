"""Tests for model.py module."""
import numpy as np
import pytest
import tensorflow as tf

import model


@pytest.mark.functional
def test_sth():
    noise_train = np.random.uniform(low=0, high=255.,
                                    size=(10, 28, 28, 1)).astype("float32")

    noise_test = np.random.uniform(low=0, high=255.,
                                   size=(10, 28, 28, 1)).astype("float32")

    batch_size = 2

    train_dataset = tf.data.Dataset.from_tensor_slices(noise_train).shuffle(
        len(noise_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(noise_test).shuffle(
        len(noise_test)).batch(batch_size)

    vautoencoder = model.VariationalAutoEncoder.from_latent_dim(latent_dim=2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=.0005)

    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        metric = model.VAE_loss()
        for train_x in train_dataset:
            model.train_step(vautoencoder, train_x, optimizer)

        for test_x in test_dataset:
            metric.update_state(vautoencoder, test_x)
