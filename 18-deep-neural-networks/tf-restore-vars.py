'''
A simple example showing how to restore two TensorFlow variables from disk.
The accompanying tf-save-vars.py script demonstrates how to save these
variables.
'''

import tensorflow as tf

# The file path to save the data
save_file = "./model.ckpt"

# Two Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as session:
    # Load the weights and bias
    saver.restore(session, save_file)

    # Show the values of weights and bias
    print('Weight:')
    print(session.run(weights))
    print('Bias:')
    print(session.run(bias))
