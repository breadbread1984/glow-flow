#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

class Identity(tfp.bijectors.Bijector):
	def __init__(self,validate_args = False, name = "identity"):
		super(Identity,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
	def _forward(self,x):
		return x;
	def _inverse(self,y):
		return y;
	def _inverse_log_det_jacobian(self,y):
		ildj = tf.constant([0], dtype = y.dtype);
		ildj = tf.tile(ildj,tf.shape(y)[:-3]);
		tf.debugging.assert_equal(tf.debugging.is_nan(ildj),tf.tile([False],tf.shape(y)[:-3]));
		return ildj;
