#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
import Glow;

batch_size = 256;
levels = 2;

def main(unused_argv):
	generator = tf.estimator.Estimator(model_fn = model_fn, model_dir = "generator_model");
	tf.logging.set_verbosity(tf.logging.DEBUG);
	logging_hook = tf.train.LoggingTensorHook(tensors = {"loss":"loss"}, every_n_iter = 1);
	generator.train(input_fn = train_input_fn, steps = 200000, hooks = [logging_hook]);
	eval_results = generator.evaluate(input_fn = eval_input_fn, steps = 1);
	print(eval_results);

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
	data = tf.cast(data,dtype = tf.float32);
	label = tf.cast(feature['label'],dtype = tf.int32);
	return data,label;
	
def train_input_fn():
	dataset = tf.data.TFRecordDataset(['dataset/trainset.tfrecord']);
	dataset = dataset.map(parse_function);
	dataset = dataset.shuffle(buffer_size = batch_size);
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
	shape = features.get_shape();
	# 1-D vector code distribution
	base_distribution = tfp.distributions.MultivariateNormalDiag(
		loc = tf.zeros([batch_size,np.prod(features.shape[-3:])]),
		scale_diag = tf.ones([batch_size,np.prod(features.shape[-3:])])
	);
	# normalizing flow
	# The TransformedDistribution defines forward direction from code to image
	# I define Glow's forward direction from image to code
	# Therefore, I give the Glow bijector in the inverted direction
	transformed_dist = tfp.distributions.TransformedDistribution(
		distribution = base_distribution,
		bijector = tfp.bijectors.Chain([
			tfp.bijectors.Invert(Glow.Glow(levels = levels)),
			tfp.bijectors.Reshape(event_shape_out = list(shape[1:]), event_shape_in = [np.prod(shape[1:])])
		]),
		name = "transformed_dist"
	);
	# predict mode
	if mode == tf.estimator.ModeKeys.PREDICT:
		samples = transformed_dist.sample(batch_size);
		return tf.estimator.EstimatorSpec(mode = mode, predictions = samples);
	# train mode
	if mode == tf.estimator.ModeKeys.TRAIN:
		loss = -tf.reduce_mean(transformed_dist.log_prob(features));
		#learning rate
		lr = tf.train.cosine_decay(1e-4, global_step = tf.train.get_or_create_global_step(), decay_steps = 1000);
		optimizer = tf.train.AdamOptimizer(learning_rate);
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step());
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op);
	# eval mode
	if mode == tf.estimator.ModeKeys.EVAL:
		loss = -tf.reduce_mean(transformed_dist.log_prob(features));
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = {"loss": loss});
	
	raise Exception('Unknown mode of estimator!');

if __name__ == "__main__":
	tf.app.run();
