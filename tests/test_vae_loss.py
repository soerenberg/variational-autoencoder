"""Tests for vae_loss.py module."""
import numpy as np
import pytest
import tensorflow as tf

import vae_loss


@pytest.mark.parametrize("dtype,decimal", [(tf.float32, 6), (tf.float64, 7)])
@pytest.mark.parametrize("mu,log_var,expected", [
    (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1)),
    (np.zeros((3, 4)), np.zeros((3, 4)), np.zeros(3)),
    (np.ones((3, 4)), np.zeros((3, 4)), np.ones(3) * 2),
    (np.ones((3, 4)), np.ones((3, 4)), np.ones(3) * 2 * (np.exp(1) - 1)),
    (np.tile(np.array([0, 0, 1, 1]),
             reps=[4, 1]).T, np.tile(np.array([0, 1, 0, 1]), reps=[4, 1]).T,
     np.array([0, 2 * (np.exp(1) - 2), 2, 2 * (np.exp(1) - 1)])),
])
def test_kl_divergence_normal_std_normal(mu, log_var, dtype, decimal,
                                         expected):
    result = vae_loss.kl_divergence_normal_std_normal(
        tf.constant(mu, dtype), tf.constant(log_var, dtype))

    np.testing.assert_almost_equal(result.numpy(), expected, decimal=decimal)
