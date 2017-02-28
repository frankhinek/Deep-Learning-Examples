'''
A simple Convolutional Neural Network (CNN) example using TensorFlow library.

- Input tensor is of shape [1, 4, 4, 1] (batch, in_height, in_width, in_channels)
- Filter/kernel tensor is of shape [2, 2, 1, 3] (filter_height, filter_width,
in_channels, out_channels])
- Output bias is of shape [3] (out_channels)
'''

import tensorflow as tf
import numpy as np

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)

def conv2d(input):
    # Filter (weights and bias)
    # The shape of the filter weight is (height, width, input_depth, output_depth)
    # The shape of the filter bias is (output_depth)
    F_W = tf.Variable(tf.truncated_normal([2, 2, 1, 3]))
    F_b = tf.Variable(tf.truncated_normal([3]))

    # The stride for each dimension (batch_size, height, width, depth)
    strides = [1, 2, 2, 1]

    # The padding is 'VALID'
    padding = 'VALID'

    return tf.nn.conv2d(input, F_W, strides, padding) + F_b

out = conv2d(X)

# Initializing the variables
init = tf. global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)

    # Print session results
    print("Output shape: " + str(out.shape))
    print("\nConvolution result:")
    print(session.run(out))
