from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import signal
from datetime import datetime

# Dependency imports
import tensorflow as tf
import tensorflow.contrib.distributions as tfd

from scripts.datasets import get_input_fns
from glow.bijectors import GlowFlow

AVAILABLE_DATASETS = ('mnist', )

TMP_DIR = os.getenv("TEST_TMP_DIR", "/tmp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',
                        type=str,
                        default='.',
                        help="Base directory for the logs.")

    parser.add_argument('--dataset',
                        type=str,
                        choices=AVAILABLE_DATASETS,
                        default='mnist')

    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(TMP_DIR, "data/"),
        help="Base directory for the data.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(
            TMP_DIR,
            "data/models/",
            datetime.now().strftime('%Y%m%dT%H%M%S')),
        help="Directory to save the model.")

    parser.add_argument("--visualize_every",
                        type=int,
                        default=500,
                        help="Frequency at which to save visualizations.")

    parser.add_argument("--n_train",
                        type=int,
                        default=50000,
                        help="Training epoch size.")

    parser.add_argument("--n_test",
                        type=int,
                        default=-1,
                        help="Validation epoch size.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="Batch size.")

    parser.add_argument("--max_steps",
                        type=int,
                        default=int(1e6),
                        help="Total number of training steps to run.")

    parser.add_argument(
        "--activation",
        type=lambda activation: getattr(tf.nn, activation),
        default="leaky_relu",
        help="Activation function for all hidden layers.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_levels", type=int, default=3,
                        help="Number of Glow levels in the flow.")
    parser.add_argument("--level_depth", type=int, default=3,
                        help="Number of flow steps in each Glow level.")

    args = parser.parse_args()

    return args


def bits_per_dim(negative_log_likelihood, image_shape):
    image_size = tf.reduce_prod(image_shape)
    return ((negative_log_likelihood + tf.log(256.0) * image_size)
            / (image_size * tf.log(2.0)))


def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width, height, depth = shape[-3], shape[-2], shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images


def image_tile_summary(name, tensor, rows=8, cols=8):
    tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)


def model_fn(features, labels, mode, params, config):
    """Build the glow flow model function for use in an estimator.

    Arguments:
        features: The input features for the estimator.
        labels: The labels, unused here.
        mode: Signifies whether it is train or test or predict.
        params: Some hyperparameters as a dictionary.
        config: The RunConfig, unused here.
        Returns:
        EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """
    base_distribution = tfd.MultivariateNormalDiag(
        loc=tf.zeros(features.shape[-3:]),
        scale_diag=tf.ones(features.shape[-3:]))

    glow_flow = GlowFlow(
        num_levels=params['num_levels'],
        level_depth=params['level_depth'])

    transformed_glow_flow = tfd.TransformedDistribution(
        distribution=base_distribution,
        bijector=glow_flow,
        name="transformed_glow_flow")

    image_tile_summary("input", tf.to_float(features), rows=1, cols=16)

    z = glow_flow.inverse(features)
    prior_log_probs = base_distribution.log_prob(z)
    prior_log_likelihood = -tf.reduce_mean(prior_log_probs)
    log_det_jacobians = glow_flow.inverse_log_det_jacobians(features)
    log_probs = log_det_jacobians + log_det_jacobians

    # Sanity check, remove when tested
    assert tf.equal(log_probs, transformed_glow_flow.log_prob(features))

    negative_log_likelihood = -tf.reduce_mean(log_probs)
    bpd = bits_per_dim(negative_log_likelihood, features.shape[-3:])

    loss = negative_log_likelihood

    tf.summary.scalar(
        "negative_log_likelihood",
        tf.reshape(negative_log_likelihood, []))
    tf.summary.scalar("bit_per_dim", tf.reshape(bpd, []))

    #  TODO: prior likelihood and log det jacobians?
    # tf.summary.scalar("prior_ll", tf.reshape(tf.reduce_mean(prior_ll), []))

    z_l2 = tf.norm(z, axis=1)
    z_l2_mean, z_l2_var = tf.nn.moments(z_l2)
    log_det_jacobians_mean, log_det_jacobians_var = tf.nn.moments(
        log_det_jacobians)
    prior_log_likelihood_mean, prior_log_likelihood_var = tf.nn.moments(
        prior_log_likelihood)

    tf.summary.scalar("log_det_jacobians_mean",
                      tf.reshape(log_det_jacobians_mean, []))
    tf.summary.scalar("log_det_jacobians_var",
                      tf.reshape(log_det_jacobians_var, []))

    tf.summary.scalar("prior_log_likelihood_mean",
                      tf.reshape(prior_log_likelihood_mean, []))
    tf.summary.scalar("prior_log_likelihood_var",
                      tf.reshape(prior_log_likelihood_var, []))

    tf.summary.scalar("l2_z_mean", tf.reshape(z_l2_mean, []))
    tf.summary.scalar("z_l2_var", tf.reshape(z_l2_var, []))

    # Generate samples for visualization.
    random_image = transformed_glow_flow.sample(16)
    image_tile_summary(
        "random/sample", tf.to_float(random_image), rows=4, cols=4)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(
        params['learning_rate'], global_step, params['max_steps'])
    tf.summary.scalar("learning_rate", learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    capped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, clip_norm=params['clip_gradient'])
    capped_gradients_and_variables = zip(capped_gradients, variables)
    train_op = optimizer.apply_gradients(
        capped_gradients_and_variables, global_step=global_step)

    gradient_norm = tf.check_numerics(
        gradient_norm, "Gradient norm contains NaNs or Infs.")
    tf.summary.scalar("gradient_norm", tf.reshape(gradient_norm, []))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "log_probs": tf.metrics.mean(log_probs),
        })


def train_model(args):
    # TODO: make these tensorflow datasets or something more efficient
    train_input_fn, eval_input_fn = get_input_fns(
        args.dataset, args.data_dir, args.batch_size)

    estimator = tf.estimator.Estimator(
        model_fn,
        params=vars(args),
        config=tf.estimator.RunConfig(
            model_dir=args.model_dir,
            save_checkpoints_steps=args.visualize_every))

    for _ in range(args.max_steps // args.visualize_every):
        estimator.train(train_input_fn, steps=args.visualize_every)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    args = parse_args()

    train_model(args)
