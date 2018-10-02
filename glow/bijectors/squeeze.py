"""Squeeze bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.distributions as tfd

__all__ = [
    "Squeeze",
]


tfb = tfd.bijectors


class Squeeze(tfb.Bijector):
    """Squeeze Bijector

    TODO: Should probably be renamed since this does not only squeeze,
    but also transposes some axis in between.
    """

    def __init__(self,
                 factor=2,
                 forward_min_event_ndims=3,
                 inverse_min_event_ndims=3,
                 *args,
                 **kwargs):
        """Instantiates the `Squeeze` reshaping bijector."""
        self._factor = factor

        # Note that the super is not for Squeeze but Reshape
        super(Squeeze, self).__init__(
            forward_min_event_ndims=forward_min_event_ndims,
            inverse_min_event_ndims=inverse_min_event_ndims,
            *args,
            **kwargs)

    @property
    def factor(self):
        return self._factor

    def _forward(self, x):
        factor = self._factor
        (H, W, C) = x.shape[-3:]
        intermediate_event_shape = (H//factor, factor, W//factor, factor, C)
        output_event_shape = (H//factor, W//factor, C*factor**2)

        sample_batch_shape = tf.shape(x)[:-3]
        intermediate_shape = tf.concat([
            sample_batch_shape, intermediate_event_shape], axis=0)
        output_shape = tf.concat([
            sample_batch_shape, output_event_shape], axis=0)

        y = tf.reshape(x, intermediate_shape)
        y = tf.transpose(y, [0, 1, 3, 5, 2, 4])
        y = tf.reshape(y, output_shape)

        return y

    def _inverse(self, y):
        factor = self._factor
        (H, W, C) = y.shape[-3:]
        intermediate_event_shape = (H, W, C//factor**2, factor, factor)
        output_event_shape = (H*factor, W*factor, C//factor**2)

        sample_batch_shape = tf.shape(y)[:-3]
        intermediate_shape = tf.concat([
            sample_batch_shape, intermediate_event_shape], axis=0)
        output_shape = tf.concat([
            sample_batch_shape, output_event_shape], axis=0)

        x = tf.reshape(y, intermediate_shape)
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, output_shape)

        return x

    def _forward_log_det_jacobian(self, x, *args, **kwargs):
        return tf.constant(0.0, dtype=x.dtype)

    def _inverse_log_det_jacobian(self, y, *args, **kwargs):
        return tf.constant(0.0, dtype=y.dtype)
