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
        #get mean
        mean = 0;
        for (batch,image) in enumerate(input):
            #get expectation over spatial, but not channel
            mean_batch = tf.math.reduce_mean(image, axis = [0,1]);
            mean = mean_batch / (batch + 1) + batch * mean / (batch + 1);
        #get variance
        variance = 0;
        for (batch,image) in enumerate(input):
            sq_batch = tf.math.square(image - mean);
            variance_batch = tf.math.reduce_mean(sq_batch, axis = [0,1]);
            variance = variance_batch / (batch + 1) + batch * variance / (batch + 1);
        stdvar = tf.math.sqrt(variance);
        return mean,stdvar;
    def initForward(self, x):
        mean,stdvar = self.getStatics(x);
        self.loc = tf.Variable(-mean, name = 'mean');
        self.scale = tf.Variable(1. / (stdvar + 1e-6), name = 'stdvar');
        self.initialized = True;
    def initInverse(self, y):
        mean,stdvar = self.getStatics(y);
        self.loc = tf.Variable(mean, name = 'mean');
        self.scale = tf.Variable(stdvar, name = 'stdvar');
        self.initialized = True;
    def _forward(self, x):
        #initialize with the first batch
        if self.initialized == False:
            self.initForward(x);
        return (x + self.loc) * self.scale;
    def _inverse(self, y):
        #initialize with the first batch
        if self.initialized == False:
            self.initInverse(y);
        return y / self.scale - self.loc;
    def _inverse_log_det_jacobian(self, y):
        #df^{-1}(y) / dy = 1 / stdvar
        #ildj = log(abs(diag(1/stdvar))), where diag(1/stdvar) is a (h*w) x (h*w) matrix
        shape = y.get_shape();
        ildj = - np.prod(shape[1:2]) * tf.math.log(tf.abs(self.scale));
        return ildj;
