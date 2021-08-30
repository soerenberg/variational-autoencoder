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


@pytest.fixture(name="mocked_model")
def fixture_mocked_model(mocker):
    return mocker.Mock()


class TestComputeLoss:
    # @pytest.mark.xfail
    @pytest.mark.parametrize("batch_size,latent_dim,width,height,channels",
                             [(1, 1, 1, 1, 1), (32, 2, 28, 28, 3),
                              (32, 2, 28, 28, 1), (3, 3, 30, 7, 4)])
    def test_calls_secondary_functions(self, batch_size, latent_dim, width,
                                       height, channels, mocker, mocked_model):
        """Simple test to check that compute_loss calls the correct secondary
        or helper functions.
        Note that we intentionally allow testing for more than one atomic thing
        at once in this test to avoid repeating a lot of code or increasing the
        complexity of this test and its setup.
        """
        mean = tf.zeros((batch_size, latent_dim), tf.float32)
        log_var = tf.ones((batch_size, latent_dim), tf.float32)

        mocked_model.encode = mocker.Mock(return_value=(mean, log_var))

        x_logit = tf.ones((batch_size, width, height, channels))
        mocked_model.decode = mocker.Mock(return_value=x_logit)

        latent_z = tf.ones((batch_size, width, height, channels))
        mocked_model.sample_from_latent_conditional = mocker.Mock(
            return_value=latent_z)

        kl_divergence = tf.ones((batch_size, ))
        mocker.patch.object(vae_loss,
                            "kl_divergence_normal_std_normal",
                            return_value=kl_divergence)

        tensor_batch = tf.zeros((batch_size, width, height, channels))
        vae_loss.compute_loss(mocked_model, tensor_batch)

        mocked_model.encode.assert_called_once_with(tensor_batch)
        mocked_model.decode.assert_called_once_with(latent_z)
        mocked_model.sample_from_latent_conditional.assert_called_once_with(
            mean, log_var)
        vae_loss.kl_divergence_normal_std_normal.assert_called_once_with(
            mean, log_var)  # pylint: disable=no-member
