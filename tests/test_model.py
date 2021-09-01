"""Tests for model.py module."""
import pathlib

import numpy as np
import pytest
import tensorflow as tf

import checkpointing
import model
import vae_loss


@pytest.mark.functional
def test_model_noise_run(tmp_path):
    """Simple functional test, building model on noise data."""
    noise_train = np.random.uniform(low=0, high=255.,
                                    size=(10, 28, 28, 1)).astype("float32")

    noise_test = np.random.uniform(low=0, high=255.,
                                   size=(10, 28, 28, 1)).astype("float32")

    batch_size = 2

    train_dataset = tf.data.Dataset.from_tensor_slices(noise_train).shuffle(
        len(noise_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(noise_test).shuffle(
        len(noise_test)).batch(batch_size)

    latent_dim = 2
    vautoencoder = model.VariationalAutoEncoder.from_latent_dim(
        latent_dim=latent_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=.0005)
    global_step = tf.Variable(0,
                              name="global_step",
                              trainable=False,
                              dtype=tf.int64)

    fixed_latents = tf.Variable(tf.random.normal(shape=(10, latent_dim)),
                                name="fixed_latents",
                                trainable=False)

    check_pt, check_pt_manager = checkpointing.init_checkpoint_and_manager(
        checkpoint_path=tmp_path,
        optimizer=optimizer,
        model=vautoencoder,
        iterator=iter(train_dataset),
        fixed_latents=fixed_latents,
        global_step=global_step)

    checkpointing.restore_checkpoint_if_exists(check_pt, check_pt_manager)

    train_elbo = tf.keras.metrics.Mean("train_elbo")
    test_elbo = tf.keras.metrics.Mean("test_elbo")

    num_epochs = 3
    for _ in range(1, num_epochs + 1):
        for train_x in train_dataset:
            vautoencoder.train_step(train_x, optimizer, train_elbo)

        checkpointing.write_checkpoint_if_necesssary(check_pt,
                                                     check_pt_manager,
                                                     check_pt_every_n_epochs=1)

        for test_x in test_dataset:
            test_elbo(
                -tf.reduce_mean(vae_loss.compute_loss(vautoencoder, test_x)))


class TestVariationalAutoEncoder:
    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    @pytest.mark.parametrize("mean,log_var,std_samples,expected", [
        (np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)),
        (np.zeros(1), np.zeros(1), np.ones(1), np.ones(1)),
        (np.ones(1), np.zeros(1), np.ones(1), 2 * np.ones(1)),
        (np.ones(1), np.zeros(1), -np.ones(1), np.zeros(1)),
        (np.zeros((2, 3)), np.zeros((2, 3)), np.ones((2, 3)), np.ones((2, 3))),
        (np.array([[0, 1], [2, 3]]), np.array(
            [[2, 0], [4, 6]]), np.array([[1, -1], [2, -2]]),
         np.array([[np.exp(1), 0], [2 + 2 * np.exp(2), 3 + -2 * np.exp(3)]]))
    ])
    def test_sample_from_latent_conditional(self, mean, log_var, std_samples,
                                            dtype, mocker, expected):
        """Test correctness of sample_from_latent_conditional method."""
        mocker.patch.object(tf.random, "normal", return_value=std_samples)

        result = model.VariationalAutoEncoder.sample_from_latent_conditional(
            tf.constant(mean, dtype), tf.constant(log_var, dtype))

        np.testing.assert_array_almost_equal(result.numpy(), expected)
