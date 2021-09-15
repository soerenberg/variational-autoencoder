"""Tests for autoencoder.py module."""
import numpy as np
import pytest
import tensorflow as tf

import autoencoder


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

        result = (
            autoencoder.VariationalAutoEncoder.sample_from_latent_conditional(
                tf.constant(mean, dtype), tf.constant(log_var, dtype)))

        np.testing.assert_array_almost_equal(result.numpy(), expected)
