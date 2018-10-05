#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
import Parallel,Squeeze,GlowStep;

class Glow(tfp.bijectors.Bijector):
	def __init__(self, shape, levels = 2, depth = 2, validate_args = False, name = 'Glow'):
		super(Glow,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
		# setup network structure
		layers = [];
		for i in range(levels):
			layers.append(tfp.bijectors.Invert(Squeeze(factor = 2**(i+1))));
			layers.append(Parallel(
				bijectors = [GlowStep(shape,depth = depth),tfp.bijectors.Identity()], 
				weights = [1, 2**i-1],
				axis = -1
			));
			layers.append(Squeeze(factor = 2**(i+1)));
		# Note that tfb.Chain takes a list of bijectors in the *reverse* order
		self.flow = tfp.bijectors.Chain(list(reversed(layers)));
	def _forward(self, x):
		return self.flow.forward(x);
	def _inverse(self, y):
		return self.flow.inverse(y);
	def _inverse_log_det_jacobian(self, y):
		return self.flow.inverse_log_det_jacobian(y);
