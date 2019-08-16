#!/usr/bin/python3

from math import pi;
import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

class Split(tfp.bijectors.Bijector):

    def __init__(self, input_shape, validate_args = False, name = 'split'):
        super(Split,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
        input_shape = (input_shape[-3], input_shape[-2], input_shape[-1] // 2);
        inputs = tf.keras.Input(shape = input_shape);
        results = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(3, 3), padding='same')(inputs);
        self.conv = tf.keras.Model(inputs = inputs, outputs = results);

    def _forward(self, x):
        #xb is thought to be a part of the encoding which follows a normal distribution
        xa, xb = tf.split(x, 2, axis = -1);
        return xa;

    def _inverse(self, ya):
        #yb is a sampled part of the encoding which follows a normal distribution
        theta = self.conv(ya);
        mean, logs = tf.split(theta, 2, axis = -1);
        yb = tf.random.normal(mean.get_shape()) * tf.math.exp(logs) + mean;
        y = tf.concat([ya,yb], axis = -1);
        return y;

    #NOTE: log det|dy / dx| != -log det|dx / dy| for this bijector
    def _forward_log_det_jacobian(self, x):
        #log det|dy/dx| = log N(xb; mean,s^2)
        xa, xb = tf.split(x, 2, axis = -1);
        theta = self.conv(xa);
        mean, logs = tf.split(theta, 2, axis = -1);
        fldj = -0.5 * (logs * 2 + tf.math.square(xb - mean) / tf.math.exp(logs * 2) + tf.math.log(2 * pi));
        fldj = tf.math.reduce_sum(fldj, axis = [1,2,3]);
        return fldj;

    def _inverse_log_det_jacobian(self, y):
        #log det|dx/dy| = log 1 = 0
        ildj = tf.zeros([tf.shape(y)[0]], dtype = y.dtype);
        return ildj;
