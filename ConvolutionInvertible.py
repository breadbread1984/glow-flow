#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

class ConvolutionInvertible(tfp.bijectors.Bijector):

    def __init__(self,validate_args = False, name = "convolution_invertible"):
        super(ConvolutionInvertible,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
        self.initialized = False;

    def build(self,x):
        shape = x.get_shape();
        #shared weight between forward and inverse conv operators
        #shape=(height,width,channel_in,channel_out)
        self.w = tf.Variable(np.random.normal(size = (1, 1, shape[-1], shape[-1])), dtype = tf.float32);
        self.initialized = True;

    def _forward(self,x):
        if self.initialized == False: self.build(x);
        y = tf.nn.conv2d(x,filters = self.w,strides=(1,1,1,1),padding = 'SAME');
        return y;

    def _inverse(self,y):
        if self.initialized == False: self.build(y);
        x = tf.nn.conv2d(y,filters = tf.matrix_inverse(self.w),strides=(1,1,1,1),padding = 'SAME');
        return x;

    def _inverse_log_det_jacobian(self,y):
        if self.initialized == False: self.build(y);
        #slogdet is the LU decomposition implement of log(det|dy/dx|)
        ildj = tf.reshape(-tf.linalg.slogdet(self.w).log_abs_determinant,[1]);
        ildj = tf.tile(ildj,[tf.shape(y)[0]]);
        return ildj;
