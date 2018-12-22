#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

class AffineCoupling(tfp.bijectors.Bijector):

    def __init__(self, hidden_filters = 512, validate_args = False, name = 'affinecoupling'):
        super(AffineCoupling,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
        self.hidden_filters = hidden_filters;
        self.initialized = False;
        
    def build(self, x):
        input = tf.keras.Input(shape = x.get_shape()[-3:]);
        output = tf.keras.layers.Conv2D(filters = self.hidden_filters, kernel_size = (3,3), padding = 'same')(input);
        output = tf.keras.layers.BatchNormalization()(output);
        output = tf.keras.layers.ReLU()(output);
        output = tf.keras.layers.Conv2D(filters = self.hidden_filters, kernel_size = (1,1), padding = 'same')(output);
        output = tf.keras.layers.BatchNormalization()(output);
        output = tf.keras.layers.ReLU()(output);
        output = tf.keras.layers.Conv2D(filters = int(x.get_shape()[-1] * 2), kernel_size = (3,3), padding = 'same')(output);
        shift,log_scale = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, axis = -1))(output)
        scale = tf.keras.layers.Lambda(lambda x: tf.math.exp(x))(log_scale);
        self.nn = tf.keras.Model(input,[shift,scale]);
        self.initialized = True;

    def _forward(self, x):
        xa,xb = tf.split(x, 2, axis = -1);
        if self.initialized == False:
            self.build(xa);
        ya = xa;
        outputs = self.nn(xa);
        scales = outputs[0];
        shifts = outputs[1];
        yb = scales * xb + shifts;
        y = tf.concat([ya,yb], axis = -1);
        return y;

    def _inverse(self, y):
        ya,yb = tf.split(y, 2, axis = -1);
        if self.initialized == False:
            self.build(ya);
        xa = ya;
        outputs = self.nn(ya);
        scales = outputs[0];
        shifts = outputs[1];
        xb = (scales - shifts) / scales;
        x = tf.concat([xa,xb], axis = -1);
        return x;

    def _inverse_log_det_jacobian(self, y):
        ya,yb = tf.split(y, 2, axis = -1);
        if self.initialized == False:
            self.build(ya);
        outputs = self.nn(ya);
        scales = outputs[0];
        ildj = -tf.math.reduce_sum(tf.math.log(tf.abs(scales)));
        ildj = tf.tile([ildj],[tf.shape(y)[0]]);
        return ildj;
