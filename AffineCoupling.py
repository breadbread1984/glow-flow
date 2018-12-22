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
        
    def build(self, x):
        self.resnet = ResNet(x.shape, self.filters, self.num_blocks);
        self.initialized = True;

    def _forward(self, x):
        if self.initialized == False:
            self.build(x);
        xa,xb = tf.split(x, 2, axis = -1);
        ya = xa;
        outputs = self.resnet(xa);
        scales = outputs[0];
        shifts = outputs[1];
        yb = scales * xb + shifts;
        y = tf.concat([ya,yb]);

    def _inverse(self, y):
        if self.initialized == False:
            self.build(y);
        ya,yb = tf.split(y, 2, axis = -1);
        xa = ya;
        outputs = self.resnet(ya);
        scales = outputs[0];
        shifts = outputs[1];
        xb = (scales - shifts) / scales;
        x = tf.concat([xa,xb]);

    def _inverse_log_det_jacobian(self, y):
        ya,yb = tf.split(y, 2, axis = -1);
        outputs = self.resnet(ya);
        scales = outputs[0];
        ildj = -tf.math.reduce_sum(tf.math.log(tf.abs(scales)));
        return ildj;

class ResNet(tf.keras.Model):

    def __init__(self, shape, filters, num_blocks):
        assert type(filters) is list and len(filters) == 3;
        assert type(num_blocks) is int;
        super(ResNet,self).__init__();
        #setup network structure once
        data = tf.keras.Input(shape = shape[-3:]);
        input = data;
        isFirst = True;
        for i in range(num_blocks):
            output = self.convBlock(input, filters, skip_reduce = isFirst);
            input = output;
            ifFirst = False;
        output = tf.keras.layers.Conv2D(filters = int(input.shape[-1]) * 2, kernel_size = (1,1), strides = (1,1), padding = 'same')(output);
        shift,log_scale = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, axis = -1))(output);
        scale = tf.keras.layers.Lambda(lambda x: tf.math.exp(x))(log_scale);
        #save it as a tf.keras.Model
        self.resnet = tf.keras.Model(data, [scale,shift]);

    def convBlock(self, x, filters, skip_reduce = False, strides = (1,1)):
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

    def call(self, input):
        return self.resnet(input);
