#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
from Squeeze import Squeeze;
from GlowStep import GlowStep;
from Split import Split;

class Glow(tfp.bijectors.Bijector):

    #forward: dimension (b,h,w,c)->(b,h/2^levels,w/2^levels,c*2^(levels+1))
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
            layers.append(GlowStep(depth = self.depth,name = self._name + "/glowstep_{}".format(i)));
            if i < self.levels - 1:
                layers.append(Split(name = self._name + "/split_{}".format(i))); #c''=c'/2=2*c
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

    def _forward_log_det_jacobian(self, x):
        if self.built == False: self.build(x);
        fldj = self.flow.forward_log_det_jacobian(x,event_ndims = 3);
        return fldj;

    def _inverse_log_det_jacobian(self, y):
        if self.built == False: self.build(y);
        ildj = self.flow.inverse_log_det_jacobian(y,event_ndims = 3);
        return ildj;

class GlowModel(tf.keras.Model):

    def __init__(self, levels = 2, shape = (227,227,3)):
        assert type(levels) is int and levels > 0;
        assert type(shape) is tuple and len(shape) == 3;
        super(GlowModel, self).__init__();
        code_dims = (shape[0] // 2**levels, shape[1] // 2**levels, shape[2] * 2**(levels + 1));
        # 1-D vector code distribution
        self.base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc = tf.zeros([np.prod(code_dims)], dtype = tf.float32),
            scale_diag = tf.ones([np.prod(code_dims)], dtype = tf.float32)
        );
        self.transformed_dist = tfp.distributions.TransformedDistribution(
            distribution = self.base_distribution,
            bijector = tfp.bijectors.Chain([
                tfp.bijectors.Invert(Glow(levels = levels)),
                tfp.bijectors.Reshape(event_shape_out = list(code_dims), event_shape_in = [np.prod(code_dims)])
            ]),
            name = "transformed_dist"
        );

    def call(self, input = None, training = False):
        if training:
            assert issubclass(type(input),tf.Tensor);
            result = tf.keras.layers.Lambda(lambda x: self.transformed_dist.log_prob(x))(input);
        else:
            assert type(input) is int and input >= 1;
            result = tf.keras.layers.Lambda(lambda x: self.transformed_dist.sample(x))(input);
        return result;
