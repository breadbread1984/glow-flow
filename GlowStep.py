#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
import ConvolutionInvertible;

class GlowStep(tfp.bijectors.Bijector):
	def __init__(self, depth = 2, validate_args = False, name = 'GlowStep'):
		super(GlowStep,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
		self.depth = depth;
		self.built = False;
	def build(self,x):
		shape = x.get_shape();
		# setup network structure
		layers = [];
		for i in range(self.depth):
			layers.append(tfp.bijectors.BatchNormalization(batchnorm_layer = tf.layers.BatchNormalization()));
			layers.append(ConvolutionInvertible.ConvolutionInvertible(name = self._name + "/conv_inv_{}".format(i)));
			layers.append(tfp.bijectors.Reshape(event_shape_in = list(shape[1:]), event_shape_out = [np.prod(shape[1:])]));
			layers.append(tfp.bijectors.RealNVP(num_masked = np.prod(shape[1:]) // 2, shift_and_log_scale_fn = tfp.bijectors.real_nvp_default_template(hidden_layers = [512,512])));
			layers.append(tfp.bijectors.Reshape(event_shape_out = list(shape[1:]), event_shape_in = [np.prod(shape[1:])]));
		# Note that tfb.Chain takes a list of bijectors in the *reverse* order
		self.flow = tfp.bijectors.Chain(list(reversed(layers)));
		self.built = True;
	def _forward(self,x):
		if self.built == False: self.build(x);
		return self.flow.forward(x);
	def _inverse(self,y):
		if self.built == False: self.build(y);
		return self.flow.inverse(y);
	def _inverse_log_det_jacobian(self,y):
		if self.built == False: self.build(y);
		return self.flow.inverse_log_det_jacobian(y, event_ndims = 3);
	def _forward_log_det_jacobian(self,x):
		if self.built == False: self.build(x);
		return self.flow.forward_log_det_jacobian(x, event_ndims = 3);
