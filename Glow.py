#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
from Parallel import Parallel;
from Squeeze import Squeeze;
from Identity import Identity;
from GlowStep import GlowStep;

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
            layers.append(Squeeze(factor = 2**i, name = self._name + "/space2batch_{}".format(i))); #h,w,c->h/2^i,w/2^i,c*4^i
            layers.append(Parallel(
                bijectors = [GlowStep(depth = self.depth,name = self._name + "/glow_step_{}".format(i)),Identity()], 
                weights = [1, 2**i-1],
                axis = -1
            ));
            layers.append(tfp.bijectors.Invert(Squeeze(factor = 2**i, name = self._name + "/batch2space_{}".format(i))));
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
        ildj = self.flow.inverse_log_det_jacobian(y,event_ndims = 3);
        tf.debugging.assert_equal(tf.debugging.is_nan(ildj),tf.tile([False],tf.shape(y)[:-3]));
        return ildj;
