#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;
import Glow;

batch_size = 100;
levels = 2;

def main(unused_argv):
    generator = tf.estimator.Estimator(model_fn=model_fn, model_dir="generator_model");
    tf.logging.set_verbosity(tf.logging.DEBUG);
    logging_hook = tf.train.LoggingTensorHook(tensors={"loss": "loss"}, every_n_iter=1);
    generator.train(input_fn=train_input_fn, steps=200000, hooks=[logging_hook]);
    eval_results = generator.evaluate(input_fn=eval_input_fn, steps=1);
    print(eval_results);


def parse_function(serialized_example):
    feature = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature((), dtype=tf.string, default_value=''),
            'label': tf.FixedLenFeature((), dtype=tf.int64, default_value=0)
        }
    );
    data = tf.decode_raw(feature['data'], out_type=tf.uint8);
    data = tf.reshape(data, [28, 28, 1]);
    data = tf.pad(data, paddings=[[2, 2], [2, 2], [0, 0]], mode='CONSTANT');
    data = tf.cast(data, dtype=tf.float32);
    label = tf.cast(feature['label'], dtype=tf.int32);
    return data, label;


def train_input_fn():
    dataset = tf.data.TFRecordDataset(['dataset/trainset.tfrecord']);
    dataset = dataset.map(parse_function);
    dataset = dataset.shuffle(buffer_size=batch_size);
    dataset = dataset.batch(batch_size);
    dataset = dataset.repeat(None);
    iterator = dataset.make_one_shot_iterator();
    features, labels = iterator.get_next();
    return features, labels;


def eval_input_fn():
    dataset = tf.data.TFRecordDataset(['dataset/testset.tfrecord']);
    dataset = dataset.map(parse_function);
    dataset = dataset.batch(batch_size);
    dataset = dataset.repeat(None);
    iterator = dataset.make_one_shot_iterator();
    features, labels = iterator.get_next();
    return features, labels;


def model_fn(features, labels, mode):
    shape = tf.shape(features);
    base_distribution = tfp.distributions.Normal(loc=0., scale=1.);
    transformed_dist = tfp.distributions.TransformedDistribution(
        distribution = base_distribution,
        bijector = tfp.bijectors.Invert(Glow(levels=levels)),
        name = "transformed_dist",
        event_shape = shape[-3:]
    );
    # predict mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        samples = transformed_dist.sample(batch_size);
        return tf.estimator.EstimatorSpec(mode=mode, predictions=samples);
    # train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = -tf.reduce_mean(transformed_dist.log_prob(features));
        # learning rate
        lr = tf.train.cosine_decay(1e-2, global_step=tf.train.get_or_create_global_step(), decay_steps=1000);
        optimizer = tf.train.AdamOptimizer(lr);
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step());
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op);
    # eval mode
    if mode == tf.estimator.ModeKeys.EVAL:
        loss = -tf.reduce_mean(transformed_dist.log_prob(features));
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={"loss": loss});

    raise Exception('Unknown mode of estimator!');


if __name__ == "__main__":
    tf.app.run();