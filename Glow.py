#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
import Parallel,Squeeze,GlowStep;

class Glow(tfp.bijectors.Bijector):
	def __init__(self, levels = 2, depth = 2, validate_args = False, name = 'Glow'):
		super(Glow,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
		self.levels = levels;
		self.depth = depth;
		self.built = False;
	def build(self, x):
		shape = x.get_shape();
		# setup network structure
		layers = [];
		for i in range(self.levels):
			layers.append(Squeeze.Squeeze(factor = 2**i)); #h,w,c->h/2^i,w/2^i,c*4^i
			layers.append(Parallel.Parallel(
				bijectors = [GlowStep.GlowStep(depth = self.depth,name = "glow_step_{}".format(i)),tfp.bijectors.Identity()], 
				weights = [1, 2**i-1],
				axis = -1
			));
			layers.append(tfp.bijectors.Invert(Squeeze.Squeeze(factor = 2**i)));
		# Note that tfb.Chain takes a list of bijectors in the *reverse* order
		self.flow = tfp.bijectors.Chain(list(reversed(layers)));
		self.built = True;
	def _forward(self, x):
		if self.built == False: self.build(x);
		#from image->code
		return self.flow.forward(x);
	def _inverse(self, y):
		if self.built == False: self.build(y);
		#from code->image
		return self.flow.inverse(y);
	def _inverse_log_det_jacobian(self, y):
		if self.built == False: self.build(y);
		return self.flow.inverse_log_det_jacobian(y,event_ndims = 3);
	def _forward_log_det_jacobian(self, x):
		if self.built == False: self.build(x);
		return self.flow.forward_log_det_jacobian(x,event_ndims = 3);
