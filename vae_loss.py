"""Compute losses for VariationalAutoencoders."""
import tensorflow as tf


def kl_divergence_normal_std_normal(mu: tf.Tensor,
                                    log_var: tf.Tensor) -> tf.Tensor:
    """KL divergence of multivariate Gaussian with standard gaussian.

    For a given distribution N(mu, sigma^2) this function computes
        KL( N(mu, sigma^2) || N(0, I) ).

    Args:
        mu: mean/location of the given Gaussion. Must have shape
            (batch_size, d), where `d` is the dimension of the distributions.
        log_var: logarithm of the variance, i.e. log(sigma^2) in the above
            example. Must have the same shape as `mu`.

    Returns:
        tf.Tensor: Result, a tensor of shape (d,), cf. above.
    """
    return .5 * tf.reduce_sum(tf.exp(log_var) + tf.square(mu) - 1 - log_var,
                              axis=1)
