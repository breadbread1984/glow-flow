#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

class ActNorm(tfp.bijectors.Bijector):

    def __init__(self, validate_args = False, name = 'actnorm'):
        super(ActNorm,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
        self.initialized = False;

    def getStatics(self, input):
        assert issubclass(type(input),tf.Tensor);
        #expectation
        mean = tf.math.reduce_mean(input, axis = [0,1,2]);
        #variance
        variance = tf.math.reduce_mean(tf.math.square(input - mean), axis = [0,1,2]);
        stdvar = tf.math.sqrt(variance);
        return mean,stdvar;

    def _forward(self, x):
        #initialize with the first batch
        if self.initialized == False:
            mean,stdvar = self.getStatics(x);
            self.loc = -mean;
            self.scale = tf.math.reciprocal(stdvar + 1e-6);
            print("loc",self.loc);
            print("scale",self.scale);
            self.initialized = True;
        return (x + self.loc) * self.scale;

    def _inverse(self, y):
        #initialize with the first batch
        if self.initialized == False:
            mean,stdvar = self.getStatics(y);
            self.loc = mean;
            self.scale = stdvar;
            print("loc",self.loc);
            print("scale",self.scale);
            self.initialized = True;
        return y / self.scale - self.loc;

    def _inverse_log_det_jacobian(self, y):
        #df^{-1}(y) / dy = 1 / stdvar
        #ildj = log(abs(diag(1/stdvar))), where diag(1/stdvar) is a (h*w) x (h*w) matrix
        shape = tf.shape(y);
        ildj = -tf.math.reduce_prod(tf.cast(shape[1:2], dtype = tf.float32)) \
               * tf.math.reduce_sum(tf.math.log(tf.abs(self.scale)));
        ildj = tf.tile([ildj],[tf.shape(y)[0]]);
        return ildj;
