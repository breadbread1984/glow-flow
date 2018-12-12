#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
import Glow;

batch_size = 200;
levels = 2;

class GlowModel(tf.keras.Model):
    def __init__(self, shape):
        super(GlowModel, self).__init__();
        # 1-D vector code distribution
        self.base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc = tf.zeros([np.prod(shape[-3:])]),
            scale_diag = tf.ones([np.prod(shape[-3:])])
        );
        self.transformed_dist = tfp.distributions.TransformedDistribution(
            distribution = self.base_distribution,
            bijector = tfp.bijectors.Chain([
                tfp.bijectors.Invert(Glow.Glow(levels = levels)),
                tfp.bijectors.Reshape(event_shape_out = list(shape[1:]), event_shape_in = [np.prod(shape[1:])])
            ]),
            name = "transformed_dist"
        );

    def call(self, input = None, training = False):
        if training:
            assert issubclass(type(input),tf.Tensor);
            result = tf.keras.layers.Lambda(lambda x: self.transformed_dist.log_prob(x))(input);
        else:
            assert type(input) is int and input >= 1;
            result = tf.keras.layers.Lambda(lambda x: self.transformed_dist.sample(x))(input);
        return result;

def parse_function(serialized_example):
    feature = tf.parse_single_example(
        serialized_example,
        features = {
            'data':tf.FixedLenFeature((),dtype = tf.string,default_value = ''),
            'label':tf.FixedLenFeature((),dtype = tf.int64,default_value = 0)
        }
    );
    data = tf.decode_raw(feature['data'],out_type = tf.uint8);
    data = tf.reshape(data,[28,28,1]);
    data = tf.pad(data, paddings = [[2,2],[2,2],[0,0]], mode = 'CONSTANT');
    data = tf.cast(data,dtype = tf.float32);
    label = tf.cast(feature['label'],dtype = tf.int32);
    return data,label;

def main(unused_argv):
    tf.enable_eager_execution();
    #prepare dataset
    trainset = tf.data.TFRecordDataset(os.path.join('dataset','trainset.tfrecord')).map(parse_function).shuffle(100).batch(100);
    testset = tf.data.TFRecordDataset(os.path.join('dataset','testset.tfrecord')).map(parse_function).batch(100);
    #create model
    model = GlowModel(shape = (32,32,1));
    lr = tf.train.cosine_decay(1e-3, global_step = tf.train.get_or_create_global_step(), decay_steps = 1000);
    optimizer = tf.train.AdamOptimizer(lr);
    #check point
    if False == os.path.exists('checkpoints'): os.mkdir('checkpoints');
    checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer, optimizer_step = tf.train.get_or_create_global_step());
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
    #create log
    log = tf.contrib.summary.create_file_writer('checkpoints');
    log.set_as_default();
    #train model
    print("training");
    while True:
        for (batch,(images,_)) in enumerate(trainset):
            with tf.GradientTape() as tape:
                loss = -tf.math.reduce_mean(model(images, training = True),name = 'loss');
            #write log
            with tf.contrib.summary.record_summaries_every_n_global_steps(2, global_step = tf.train.get_global_step()):
                tf.contrib.summary.scalar('loss',loss);
            grads = tape.gradient(loss, model.variables);
            optimizer.apply_gradients(zip(grads,model.variables),global_step = tf.train.get_global_step());
            if batch % 100 == 0: print('Step #%d\tLoss: %.6f' % (batch,loss));
        #save model once every epoch
        checkpoint.save(os.path.join('checkpoints','ckpt'));
        if loss < 0.01: break;
    #save model parameters (without network structure)
    if False == os.path.exists('model'): os.mkdir('model');
    model.save_weights('./model/glow_model');
    #TODO: test model

if __name__ == "__main__":
    tf.app.run();
