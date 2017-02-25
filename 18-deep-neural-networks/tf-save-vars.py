'''
A simple example showing how to save two TensorFlow variables to disk.
The accompanying tf-restore-vars.py script demonstrates how to restore these
variables.

The following files should be written to disk assuming you are running
TensorFlow 0.12 or later:
checkpoint
model.ckpt.data-00000-of-00001
model.ckpt.index
model.ckpt.meta
'''

import tensorflow as tf

# The file path to save the data
save_file = "./model.ckpt"

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize all the Variables
    sess.run(tf.global_variables_initializer())

    # Show the values of weights and bias
    print('Weights:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))

    # Save the model
    saver.save(sess, save_file)
