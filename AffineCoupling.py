#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

class AffineCoupling(tfp.bijectors.Bijector):

    def __init__(self, input_shape, hidden_filters = 512, validate_args = False, name = 'affinecoupling'):
        super(AffineCoupling,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
        input_shape = np.array(input_shape);
        input_shape[-1] = input_shape[-1] // 2;
        # build network
        inputs = tf.keras.Input(shape = input_shape[-3:].tolist());
        output = tf.keras.layers.Conv2D(filters = hidden_filters, kernel_size = (3,3), padding = 'same')(inputs);
        output = tf.keras.layers.BatchNormalization()(output);
        output = tf.keras.layers.ReLU()(output);
        output = tf.keras.layers.Conv2D(filters = hidden_filters, kernel_size = (1,1), padding = 'same')(output);
        output = tf.keras.layers.BatchNormalization()(output);
        output = tf.keras.layers.ReLU()(output);
        output = tf.keras.layers.Conv2D(filters = int(input_shape[-1] * 2), kernel_size = (3,3), padding = 'same')(output);
        shift,log_scale = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, axis = -1))(output)
        scale = tf.keras.layers.Lambda(lambda x: tf.math.exp(x))(log_scale);
        self.nn = tf.keras.Model(inputs = inputs, outputs = (shift, scale));

    def _forward(self, x):
        xa, xb = tf.split(x, 2, axis = -1);
        ya = xa;
        scales, shifts = self.nn(xa);
        yb = scales * xb + shifts;
        y = tf.concat([ya, yb], axis = -1);
        return y;

    def _inverse(self, y):
        ya, yb = tf.split(y, 2, axis = -1);
        xa = ya;
        scales, shifts = self.nn(ya);
        xb = (scales - shifts) / scales;
        x = tf.concat([xa, xb], axis = -1);
        return x;

    def _inverse_log_det_jacobian(self, y):
        ya,yb = tf.split(y, 2, axis = -1);
        scales, _ = self.nn(ya);
        ildj = -tf.math.reduce_sum(tf.math.log(tf.abs(scales)));
        ildj = tf.tile([ildj],[tf.shape(y)[0]]);
        return ildj;
