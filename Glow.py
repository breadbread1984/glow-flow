#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
from Squeeze import Squeeze;
from GlowStep import GlowStep;
from Split import Split;

class Glow(tfp.bijectors.Bijector):

    #forward: dimension (b,h,w,c)->(b,h/2^levels,w/2^levels,c*2^(levels-1))
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
            layers.append(Squeeze(factor = 2, name = self._name + "/space2batch_{}".format(i))); #c'=c*4
            layers.append(GlowStep(depth = self.depth,name = self._name + "/glow_step_{}".format(i)));
            if i < self.levels - 1:
                layers.append(Split()); #c''=c'/2=2*c
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
