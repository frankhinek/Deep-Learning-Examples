'''
A simple example showing how to restore a trained TensorFlow model and its
weights from disk.  Given the default hyperparameters, the restored model will
have already been trained for 100 epochs.

The accompanying tf-save-model.py script must be run first.
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math

# The file path to save the data
save_file = './train_model.ckpt'

# Parameters
learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('.', one_hot=True)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]), name='weights_0')
bias = tf.Variable(tf.random_normal([n_classes]), name='bias_0')

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,\
    labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Class used to save and/or restore Tensor model and variables
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as session:
    saver.restore(session, save_file)

    test_accuracy = session.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
