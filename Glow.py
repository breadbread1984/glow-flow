#!/usr/bin/python3

import numpy as np;
import tensorflow_probability as tfp;
from Squeeze import Squeeze;
from GlowStep import GlowStep;
from Split import Split;

class Glow(tfp.bijectors.Bijector):

    #forward: dimension (b,h,w,c)->(b,h/2^levels,w/2^levels,c*2^(levels+1))
    def __init__(self, input_shape, levels = 2, depth = 2, validate_args = False, name = 'Glow'):
        super(Glow,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
        # setup network structure
        input_shape = np.array(input_shape);
        layers = [];
        for i in range(levels):
            layers.append(Squeeze(factor = 2, name = self._name + "/space2batch_{}".format(i))); #c'=c*4
            input_shape[-1] = input_shape[-1] * (4 if i == 0 else 2);
            input_shape[-3:-1] = input_shape[-3:-1] // 2;
            layers.append(GlowStep(input_shape, depth = depth,name = self._name + "/glowstep_{}".format(i)));
            if i < levels - 1:
                layers.append(Split(name = self._name + "/split_{}".format(i))); #c''=c'/2=2*c
        # Note that tfb.Chain takes a list of bijectors in the *reverse* order
        self.flow = tfp.bijectors.Chain(list(reversed(layers)));

    def _forward(self, x):
        #from image->code
        return self.flow.forward(x);

    def _inverse(self, y):
        #from code->image
        return self.flow.inverse(y);

    def _inverse_log_det_jacobian(self, y):
        ildj = self.flow.inverse_log_det_jacobian(y,event_ndims = 3);
        return ildj;
