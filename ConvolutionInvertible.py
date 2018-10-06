#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

class ConvolutionInvertible(tfp.bijectors.Bijector):
	def __init__(self,validate_args = False, name = "convolution_invertible"):
		super(ConvolutionInvertible,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
		self.built = False;
	def build(self,x):
		shape = x.get_shape();
		#shared weight between forward and inverse conv operators
		with tf.variable_scope(self._name):
			self.w = tf.get_variable("w", shape = [1,1,shape[-1],shape[-1]], dtype = tf.float32,initializer = tf.initializers.orthogonal());
		self.built = True;
	def _forward(self,x):
		if self.built == False: build(x);
		y = tf.nn.conv2d(x,filter = self.w,padding = 'same');
		return y;
	def _inverse(self,y):
		if self.built == False: build(y);
		x = tf.nn.conv2d(y,filter = tf.matrix_inverse(self.w),padding = 'same');
		return x;
	def _inverse_log_det_jacobian(self,y):
		if self.built == False: build(y);
		#tensorflow has no LU decomposition implement, so get determinant directly
		detJ = tf.matrix_determinant(tf.matrix_inverse(self.w));
		return tf.log(tf.abs(detJ)); #equals sum_i log(|S_i|) where S is a diagonal matrix derived from inv(W)
	def _forward_log_det_jacobian(self,x):
		if self.built == False: build(x);
		return -self._inverse_log_det_jacobian(x);