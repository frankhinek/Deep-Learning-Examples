'''
This quiz starts with the code from Quiz 1 (Implementing a ReLU activation
using TensorFlow) and challenges you to add a dropout layer.
'''

import tensorflow as tf

# Values for input-to-hidden and hidden-to-output weights
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Store layer weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input layer features
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])

# Probability of keeping neurons.  A keep_prob of 0.5 will keep half the units
# and dropout half, randomly.
keep_prob = tf.placeholder(tf.float32)

# Construct model
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
output = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)

    # Print session results
    print(session.run(output, feed_dict = { keep_prob: 0.5 } ))
