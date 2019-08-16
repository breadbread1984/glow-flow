#!/usr/bin/python3

import numpy as np;
import tensorflow_probability as tfp;
from ConvolutionInvertible import ConvolutionInvertible;
from ActNorm import ActNorm;
from AffineCoupling import AffineCoupling;

class GlowStep(tfp.bijectors.Bijector):

    def __init__(self, input_shape, depth = 2, validate_args = False, name = 'GlowStep'):
        super(GlowStep,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
        input_shape = np.array(input_shape);
        layers = [];
        for i in range(depth):
            layers.append(ActNorm(name = self._name + "/actnorm_{}".format(i)));
            layers.append(ConvolutionInvertible(input_shape = input_shape, name = self._name + "/conv_inv_{}".format(i)));
            layers.append(AffineCoupling(input_shape = input_shape, name = self._name + "/affinecoupling_{}".format(i)));
        # Note that tfb.Chain takes a list of bijectors in the *reverse* order
        self.flow = tfp.bijectors.Chain(list(reversed(layers)));

    def _forward(self,x):
        return self.flow.forward(x);

    def _inverse(self,y):
        return self.flow.inverse(y);

    def _inverse_log_det_jacobian(self,y):
        ildj = self.flow.inverse_log_det_jacobian(y, event_ndims = 3);
        return ildj;
