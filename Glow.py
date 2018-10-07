#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
import Parallel,Squeeze,GlowStep;

class Glow(tfp.bijectors.Bijector):
	def __init__(self, levels = 2, depth = 2, validate_args = False, name = 'Glow'):
		super(Glow,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
		# setup network structure
		layers = [];
		for i in range(levels):
			layers.append(Squeeze.Squeeze(factor = 2)); #h,w,c->h/2,w/2,c*4
			layers.append(Parallel.Parallel(
				bijectors = [GlowStep.GlowStep(depth = depth,name = "glow_step_{}".format(i)),tfp.bijectors.Identity()], 
				weights = [1, 2**i-1],
				axis = -1
			));
		# Note that tfb.Chain takes a list of bijectors in the *reverse* order
		self.flow = tfp.bijectors.Chain(list(reversed(layers)));
	def _forward(self, x):
		return self.flow.forward(x);
	def _inverse(self, y):
		return self.flow.inverse(y);
	def _inverse_log_det_jacobian(self, y):
		return self.flow.inverse_log_det_jacobian(y);
