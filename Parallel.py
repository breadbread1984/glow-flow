#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

class Parallel(tfp.bijectors.Bijector):
	def __init__(self, bijectors, weights = None, axis = -1, validate_args = False, name = 'parallel'):
		#weights must be an array composed of integers
		assert len(bijectors) == len(weights);
		assert weights is not None;
		forward_min_event_ndims = min(bijector.forward_min_event_ndims for bijector in bijectors);
		super(Parallel,self).__init__(forward_min_event_ndims = forward_min_event_ndims, validate_args = validate_args, name = name);
		self.bijectors = bijectors;
		self.axis = axis;
		self.weights = weights;
	def _forward(self,x):
		splits = tf.split(x,sum(self.weights), axis = self.axis);
		y = [];
		for i,(bijector,weight) in enumerate(zip(self.bijectors,self.weights)):
			i_start = sum(self.weights[:i]); #start index of splits
			i_end = i_start + weight;	#end index of splits
			if i_end - i_start >= 2:
				#if the tensor is composed by multiple slices
				y.append(bijector.forward(tf.concat(splits[i_start:i_end], axis = self.axis)));
			elif i_end - i_start == 1:
				#if the tensor is just one slice
				y.append(bijector.forward(splits[i_start]));
			#else no action
		return tf.concat(y,axis = self.axis);
	def _inverse(self,y):
		splits = tf.split(y,sum(self.weights), axis = self.axis);
		x = [];
		for i,(bijector,weight) in enumerate(zip(self.bijectors,self.weights)):
			i_start = sum(self.weights[:i]);
			i_end = i_start + weight;
			x.append(bijector.inverse(tf.concat(splits[i_start:i_end], axis = self.axis)));
		return tf.concat(x,axis = self.axis);
	def _inverse_log_det_jacobian(self,y):
		splits = tf.split(y,sum(self.weights), axis = self.axis);
		ildjs = [];
		for i,(bijector,weight) in enumerate(zip(self.bijectors,self.weights)):
			i_start = sum(self.weights[:i]);
			i_end = i_start + weight;
			ildjs.append(bijector.inverse_log_det_jacobian(tf.concat(splits[i_start:i_end], axis = self.axis)));
		return tf.add_n(ildjs);
