'''
A simple max pooling example using TensorFlow library tf.nn.max_pool.

tf.nn.max_pool performs max pooling on the input in the form (value, ksize,
strides, padding).

- value is of shape [batch, height, width, channels] and type tf.float32
- ksize is the filter/window size for each dimension of the input tensor
- strides of the sliding window for each dimension of the input tensor
- padding is the padding algorithm (either 'VALID' or 'SAME')

Reference: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
'''
import tensorflow as tf
import numpy as np

# `tf.nn.max_pool` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)

def maxpool(input):
    # The ksize (filter size) for each dimension (batch_size, height, width, depth)
    ksize = [1, 2, 2, 1]

    # The stride for each dimension (batch_size, height, width, depth)
    strides = [1, 2, 2, 1]

    # The padding, 'VALID'
    padding = 'VALID'

    return tf.nn.max_pool(input, ksize, strides, padding)

out = maxpool(X)

# Initializing the variables
init = tf. global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)

    # Print session results
    print("Output shape: " + str(out.shape))
    print("\nMax Pool result:")
    print(session.run(out))
