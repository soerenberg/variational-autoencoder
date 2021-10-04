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


def compute_loss(model, tensor_batch: tf.Tensor) -> tf.Tensor:
    """Compute VAE loss for a single batch.

    Args:
        model: VAE model
        tensor_batch: batch of training data points, e.g. for 3 channel
            pictures of size (height, width) this tensor will have the shape
                (batch_size, height, width, 3).

    Returns:
        tf.Tensor: a scalar tensor equal to the loss of the given batch
    """

    # mean, log_var have shape (batch_size, latent_dim)
    mean, log_var = model.encode(tensor_batch)

    # add random noise
    latent_z = model.sample_from_latent_conditional(mean, log_var)

    # Note: x_logit has shape (batch, height, width, x)
    x_logit = model.decode(latent_z)  # Note: the decoder returns logits.

    # Note: cross_ent has shape (batch, height, width, x)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit,
                                                        labels=tensor_batch)
    # Note: summed_cross_ent has shape (batch,)
    summed_cross_ent = tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    kl_divergence = kl_divergence_normal_std_normal(mean, log_var)

    return kl_divergence + summed_cross_ent
