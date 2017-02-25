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

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)

    # Show the values of weights and bias
    print('Weights:')
    print(session.run(weights))
    print('Bias:')
    print(session.run(bias))

    # Save the model
    saver.save(session, save_file)
