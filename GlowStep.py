#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
from ConvolutionInvertible import ConvolutionInvertible;
from ActNorm import ActNorm;
from AffineCoupling import AffineCoupling;

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
            layers.append(ActNorm(name = self._name + "/actnorm_{}".format(i)));
            layers.append(ConvolutionInvertible(name = self._name + "/conv_inv_{}".format(i)));
            layers.append(AffineCoupling(name = self._name + "/affinecoupling_{}".format(i)));
        # Note that tfb.Chain takes a list of bijectors in the *reverse* order
        self.flow = tfp.bijectors.Chain(list(reversed(layers)));
        self.built = True;

    def _forward(self,x):
        if self.built == False: self.build(x);
        return self.flow.forward(x);

    def _inverse(self,y):
        if self.built == False: self.build(y);
        return self.flow.inverse(y);

    def _forward_log_det_jacobian(self, x):
        if self.built == False: self.build(x);
        fldj = self.flow.forward_log_det_jacobian(x, event_ndims = 3);
        return fldj;

    def _inverse_log_det_jacobian(self,y):
        if self.built == False: self.build(y);
        ildj = self.flow.inverse_log_det_jacobian(y, event_ndims = 3);
        return ildj;
