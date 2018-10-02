"""Parallel bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import tensorflow as tf
import tensorflow.contrib.distributions as tfd


__all__ = [
    "Parallel",
]


tfb = tfd.bijectors


def _use_static_shape(input_tensor, ndims):
    return input_tensor.shape.is_fully_defined() and isinstance(ndims, int)


class Parallel(tfb.Bijector):
    """Bijector which applies a set of bijectors in parallel.

    Example Use:

    ```python
    parallel = Parallel([Exp(), Identity()], name="parallel_of_exp_and_identity")
    ```

    Results in:

    * Forward:

     ```python
     axis = 1
     Parallel([Exp(), Identity()], axis=axis).forward(x)
     = tf.concatenate([
           exp.forward(tf.split(x, axis=axis)),
           tf.split(x, axis=axis)
       ], axis=axis)
     = tf.concatenate([
           tf.exp(tf.split(x, axis=axis)),
           tf.split(x, axis=axis)
       ], axis=axis)
     ```

    * Inverse:

     ```python
     axis = 1
     Parallel([Exp(), Identity()], axis=axis).forward(y)
     = tf.concatenate([
           exp.inverse(tf.split(y, axis=axis)),
           tf.split(y, axis=axis)
       ], axis=axis)
     = tf.concatenate([
           tf.log(tf.split(y, axis=axis)),
           tf.split(y, axis=axis)
       ], axis=axis)
     ```

    """

    def __init__(self,
                 bijectors=None,
                 split_axis=None,
                 split_proportions=None,
                 validate_args=False,
                 name=None):
        """Instantiates `Parallel` bijector.

        Args:
            bijectors: Python `list` of bijector instances. An empty list makes this
                bijector equivalent to the `Identity` bijector.
            split_axis: Python `int`. Axis to make the split on.
            split_proportions: Python `list` of `int`s. The relative sizes of
                the splits to use for each bijector.
            validate_args: Python `bool` indicating whether arguments should be
                checked for correctness.
            name: Python `str`, name given to ops managed by this object.
                Default: E.g.,
                `Parallel([Exp(), Softplus()]).name == "parallel_of_exp_and_softplus"`.

        Raises:
            ValueError: if bijectors have different dtypes.
        """
        bijectors = bijectors or ()
        self._bijectors = bijectors

        if split_axis is None:
            assert not bijectors, (
                "split_axis has to be defined if parallel has bijectors.")

        self._split_axis = split_axis

        self._split_proportions = split_proportions or (1, ) * len(bijectors)

        self._bijectors, self._split_proportions = zip(*[
            (bijector, split_proportion)
            for bijector, split_proportion in
            zip(self._bijectors, self._split_proportions)
            if split_proportion > 0
        ])

        assert len(self._split_proportions) == len(self._bijectors), (
            "Split proportion must be defined for every bijector, got {}"
            "".format(self._split_proportions))
        assert all(
            isinstance(x, int) for x in self._split_proportions
        ), "Every element in split_proportions must be an integer."

        for bijector in bijectors:
            if not bijector._is_injective:  # pylint: disable=protected-access
                raise NotImplementedError(
                    "Invert is not implemented for non-injective bijector ({})"
                    "".format(bijector.name))

        dtype = list(set([b.dtype for b in bijectors]))
        if len(dtype) > 2:
            raise ValueError("incompatible dtypes: %s" % dtype)
        elif len(dtype) == 2:
            dtype = dtype[1] if dtype[0] is None else dtype[0]
        elif len(dtype) == 1:
            dtype = dtype[0]
        else:
            dtype = None

        forward_min_event_ndims = max(
            bijector.forward_min_event_ndims for bijector in bijectors)
        inverse_min_event_ndims = min(
            bijector.forward_min_event_ndims for bijector in bijectors)

        super(Parallel, self).__init__(
            graph_parents=list(itertools.chain.from_iterable(
                b.graph_parents for b in bijectors)),
            forward_min_event_ndims=forward_min_event_ndims,
            inverse_min_event_ndims=inverse_min_event_ndims,
            is_constant_jacobian=all(
                b.is_constant_jacobian for b in bijectors),
            validate_args=validate_args,
            dtype=dtype,
            name=name or (
                "identity" if not bijectors else
                "_and_".join(["parallel"] + [b.name for b in bijectors])))

    @property
    def bijectors(self):
        return self._bijectors

    def _forward(self, x, **kwargs):
        proportions = self._split_proportions
        axis = self._split_axis
        bijectors = self._bijectors

        num_splits = sum(proportions)
        split_x = tf.split(x, num_splits, axis=axis)

        ys = []
        for i, (bijector, split_size) in enumerate(
                zip(bijectors, proportions)):
            i_start = sum(proportions[:i])
            i_end = i_start + split_size
            y = bijector.forward(
                tf.concat(split_x[i_start:i_end], axis=axis),
                **kwargs.get(bijector.name, {}))
            ys.append(y)

        full_y = tf.concat(ys, axis=axis)

        return full_y

    def _inverse(self, y, **kwargs):
        proportions = self._split_proportions
        axis = self._split_axis
        bijectors = self._bijectors

        num_splits = sum(proportions)
        split_y = tf.split(y, num_splits, axis=axis)

        xs = []
        for i, (bijector, split_size) in enumerate(
                zip(bijectors, proportions)):
            i_start = sum(proportions[:i])
            i_end = i_start + split_size
            x = bijector.inverse(
                tf.concat(split_y[i_start:i_end], axis=axis),
                **kwargs.get(bijector.name, {}))
            xs.append(x)

        full_x = tf.concat(xs, axis=axis)

        return full_x

    def _forward_log_det_jacobian(self, x, **kwargs):
        x = ops.convert_to_tensor(x, name="x")

        fldj = math_ops.cast(0., dtype=x.dtype.base_dtype)

        if not self.bijectors:
            return fldj

        event_ndims = self._maybe_get_static_event_ndims(
            self.forward_min_event_ndims)

        proportions = self._split_proportions
        axis = self._split_axis
        bijectors = self._bijectors

        num_splits = sum(proportions)
        split_x = tf.split(x, num_splits, axis=axis)

        fldjs = []
        for i, (bijector, split_size) in enumerate(
                zip(bijectors, proportions)):
            i_start = sum(proportions[:i])
            i_end = i_start + split_size
            fldj = bijector.forward_log_det_jacobian(
                tf.concat(split_x[i_start:i_end], axis=axis),
                event_ndims=event_ndims,
                **kwargs.get(bijector.name, {}))
            fldjs.append(fldj)

        full_fldj = tf.concat(fldjs, axis=axis)

        return full_fldj

    def _inverse_log_det_jacobian(self, y, **kwargs):
        y = ops.convert_to_tensor(y, name="y")
        ildj = math_ops.cast(0., dtype=y.dtype.base_dtype)

        if not self.bijectors:
            return ildj

        event_ndims = self._maybe_get_static_event_ndims(
            self.inverse_min_event_ndims)

        proportions = self._split_proportions
        axis = self._split_axis
        bijectors = self._bijectors

        num_splits = sum(proportions)
        split_y = tf.split(y, num_splits, axis=axis)

        ildjs = []
        for i, (bijector, split_size) in enumerate(
                zip(bijectors, proportions)):
            i_start = sum(proportions[:i])
            i_end = i_start + split_size
            ildj = bijector.inverse_log_det_jacobian(
                tf.concat(split_y[i_start:i_end], axis=axis),
                event_ndims=event_ndims,
                **kwargs.get(bijector.name, {}))
            ildjs.append(ildj)

        full_ildj = tf.concat(ildjs, axis=axis)

        return full_ildj
