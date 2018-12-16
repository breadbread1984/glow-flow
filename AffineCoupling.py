#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

class AffineCoupling(tfp.bijectors.Bijector):

    def __init__(self, filters = [256,256,256], num_blocks = 1, validate_args = False, name = 'affinecoupling'):
        assert type(filters) is list and len(filters) == 3;
        super(AffineCoupling,self).__init__(forward_min_event_ndims = 3, validate_args = validate_args, name = name);
        self.filters = filters;
        self.num_blocks = num_blocks;
        self.initialized = False;

    def build(self, a_shape):
        assert type(a_shape) is tuple and len(a_shape) == 3;
        #a_shape equals b_shape
        b_channels = a_shape[-1];
        #bottleneck layers
        input = tf.keras.Input(shape = a_shape);
        isFirst = True;
        for i in range(self.num_blocks):
            output = self.convBlock(input, filters = self.filters, skip_reduce = isFirst);
            input = output;
            isFirst = False;
        #output
        output = tf.keras.layers.Conv2D(filters = b_channels * 2, kernel_size = (1,1), strides = (1,1), padding = 'same')(output);
        shift,log_scale = tf.keras.layers.Lambda(lambda x: tf.split(x, 2))(output);
        scale = tf.keras.layers.Lambda(lambda x: tf.math.exp(x))(log_scale);
        #create model object
        self.resnet = tf.keras.Model(inputs = input, outputs = [scale,shift])();
        self.initialized = True;

    def convBlock(self, x, filters, skip_reduce = False, strides = (1,1)):
        assert type(filters) is list and len(filters) == 3;
        assert type(skip_reduce) is bool;
        assert type(strides) is tuple and len(strides) == 2;
        if skip_reduce:
            skip_conv = tf.keras.layers.Conv2D(filters = filters[2], kernel_size = (1,1), strides = strides, padding = 'same')(x);
            skip = tf.keras.layers.BatchNormalization()(skip_conv);
        else:
            skip = x;
        conv1 = tf.keras.layers.Conv2D(filters = filters[0], kernel_size = (1,1), strides = strides, padding = 'same')(x);
        norm1 = tf.keras.layers.BatchNormalization()(conv1);
        relu1 = tf.keras.layers.ReLU()(norm1);
        conv2 = tf.keras.layers.Conv2D(filters = filters[1], kernel_size = (3,3), strides = (1,1), padding = 'same')(relu1);
        norm2 = tf.keras.layers.BatchNormalization()(conv2);
        relu2 = tf.keras.layers.ReLU()(norm2);
        conv3 = tf.keras.layers.Conv2D(filters = filters[2], kernel_size = (1,1), strides = (1,1), padding = 'same')(relu2);
        norm3 = tf.keras.layers.BatchNormalization()(conv3);
        sum = tf.keras.layers.Add()([skip, norm3]);
        relu3 = tf.keras.layers.ReLU()(sum);
        return relu3;

    def _forward(self, x):
        xa,xb = tf.split(x, 2);
        if self.initialized == False:
            self.build(tuple([int(dim) for dim in tf.shape(xa)[1:]]));
        ya = xa;
        outputs = self.resnet(xa);
        scales = outputs[0];
        shifts = outputs[1];
        yb = scales * xb + shifts;
        y = tf.concat([ya,yb]);

    def _inverse(self, y):
        ya,yb = tf.split(y, 2);
        if self.initialized == False:
            self.build(tuple([int(dim) for dim in tf.shape(ya)[1:]]));
        xa = ya;
        outputs = self.resnet(ya);
        scales = outputs[0];
        shifts = outputs[1];
        xb = (scales - shifts) / scales;
        x = tf.concat([xa,xb]);

    def _inverse_log_det_jacobian(self, y):
        ya,yb = tf.split(y, 2);
        outputs = self.resnet(ya);
        scales = outputs[0];
        ildj = -tf.math.reduce_sum(tf.math.log(tf.abs(scales)));
        return ildj;
