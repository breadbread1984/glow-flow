#!/usr/bin/python
# -*- coding: utf-8 -*-

import os;
import numpy as np;
import tensorflow as tf;
import pickle;

def write_tfrecord(data,label,filename):
        assert data.shape[1:] == (28,28);
        if True == os.path.exists(filename):
                os.remove(filename);
        label = np.array(label,dtype = np.int64);
        if True == os.path.exists(filename):
                os.remove(filename);
        writer = tf.io.TFRecordWriter(filename);
        for i in range(data.shape[0]):
                trainsample = tf.train.Example(features = tf.train.Features(
                        feature = {
                                'data':tf.train.Feature(bytes_list = tf.train.BytesList(value = [data[i,...].tobytes()])),
                                'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [label[i]]))
                        }
                ));
                writer.write(trainsample.SerializeToString());
        writer.close();

def create_mnist():
        (train_x,train_y),(test_x,test_y) = tf.keras.datasets.mnist.load_data();
        if False == os.path.exists('dataset'):
                os.mkdir('dataset');
        write_tfrecord(train_x,train_y,os.path.join('dataset','trainset.tfrecord'));
        write_tfrecord(test_x,test_y,os.path.join('dataset','testset.tfrecord'));

if __name__ == "__main__":
        create_mnist();

