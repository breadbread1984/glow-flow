#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
from Glow import Glow;

batch_size = 200;

def parse_function(serialized_example):
    feature = tf.io.parse_single_example(
        serialized_example,
        features = {
            'data':tf.io.FixedLenFeature((),dtype = tf.string,default_value = ''),
            'label':tf.io.FixedLenFeature((),dtype = tf.int64,default_value = 0)
        }
    );
    data = tf.io.decode_raw(feature['data'],out_type = tf.uint8);
    data = tf.reshape(data,[28,28,1]);
    data = tf.pad(data, paddings = [[2,2],[2,2],[0,0]], mode = 'CONSTANT');
    data = tf.cast(data,dtype = tf.float32);
    label = tf.cast(feature['label'],dtype = tf.int32);
    return data,label;

def main():
    #prepare dataset
    trainset = tf.data.TFRecordDataset(os.path.join('dataset','trainset.tfrecord')).map(parse_function).shuffle(100).batch(100);
    testset = tf.data.TFRecordDataset(os.path.join('dataset','testset.tfrecord')).map(parse_function).batch(100);
    #create model
    glow = tfp.distributions.TransformedDistribution(
        distribution = tfp.distributions.Normal(loc = 0., scale = 1.),
        bijector = tfp.bijectors.Invert(Glow(levels = 2)),
        name = "transformed_dist",
        event_shape = (32,32,1)
    );
    #lr = tf.compat.v1.train.cosine_decay(1e-3, global_step = optimizer.iterations, decay_steps = 1000);
    optimizer = tf.keras.optimizers.Adam(1e-3);
    #check point
    if False == os.path.exists('checkpoints'): os.mkdir('checkpoints');
    checkpoint = tf.train.Checkpoint(model = glow, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
    #create log
    log = tf.summary.create_file_writer('checkpoints');
    #train model
    print("training");
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
    while True:
        for (images,_) in trainset:
            with tf.GradientTape() as tape:
                loss = -tf.math.reduce_mean(glow.log_prob(images));
            avg_loss.update_state(loss);
            #write log
            if tf.equal(optimizer.iterations % 10, 0):
                with log.as_default():
                    tf.summary.scalar('loss',avg_loss.result(), step = optimizer.iterations);
                print('Step #%d Loss: %.6f' % (optimizer.iterations,avg_loss.result()));
                avg_loss.reset_states();
            grads = tape.gradient(loss, glow.trainable_variables);
            optimizer.apply_gradients(zip(grads, glow.trainable_variables));
        #save model once every epoch
        checkpoint.save(os.path.join('checkpoints','ckpt'));
        if loss < 0.01: break;
    #save model parameters (without network structure)
    if False == os.path.exists('model'): os.mkdir('model');
    model.save_weights('./model/glow_model');
    #TODO: test model

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
