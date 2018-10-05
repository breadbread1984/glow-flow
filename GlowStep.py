#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
import ConvolutionInvertible;

class GlowStep(tfp.bijectors.Bijector):
	def __init__(self, shape, depth = 2, validate_args = False, name = 'GlowStep'):
		super(GlowStep,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
		# setup network structure
		layers = [];
		for i in range(depth):
			layers.append(tfp.bijectors.BatchNormalization(batchnorm_layer = tf.layers.BatchNormalization()));
			layers.append(ConvolutionInvertible.ConvolutionInvertible(shape.as_list()[-1],name = self._name + "/conv_inv_{}".format(i)));
			layers.append(tfp.bijectors.Reshape(event_shape_out = [-1,np.prod(shape.as_list()[1:])]));
			layers.append(tfp.bijectors.RealNVP(num_masked = np.prod(shape.as_list()[1:]) // 2, shift_and_log_scale_fn = tfp.bijectors.real_nvp_default_template(hidden_layers = [512,512])));
			layers.append(tfp.bijectors.Reshape(event_shape_out = [-1] + shape.as_list()[1:]));
		# Note that tfb.Chain takes a list of bijectors in the *reverse* order
		self.flow = tfp.bijectors.Chain(list(reversed(layers)));
	def _forward(self,x):
		return self.flow.forward(x);
	def _inverse(self,y):
		return self.flow.inverse(y);
	def _inverse_log_det_jacobian(self,y):
		return self.flow.inverse_log_det_jacobian(y);
