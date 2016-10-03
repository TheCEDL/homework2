from gym.spaces import Box, Discrete
import numpy as np
from scipy.signal import lfilter

def linear(inputs, output_size,
			weights_initializer=initializers.xavier_initializer(),
			biases_initializer=tf.zeros_initializer, synthetic=False,
			activation_fn=None, batch_norm=True, name='linear'):
	
	var = {}
	shape = inputs.get_shape().as_list()
	with tf.variable_scope(name):
		var['w'] = tf.get_variable('w', [shape[1], output_size], tf.float32,
						initializer=weights_initializer)
		var['b'] = tf.get_variable('b', [output_size],
						initializer=biases_initializer)
		out = tf.nn.bias_add(tf.matmul(inputs, var['w']), var['b'])

		if batch_norm:
			out = tf.contrib.layers.batch_norm(out)
		if activation_fn is not None:
			out = activation_fn(out)
		if synthetic:
			with tf.variable_scope('synthetic_grad'):
				out_shape = out.get_shape()
				h1, var['l1_w'], var['l1_b'] = linear(out, 4000, weights_initializer=tf.zeros_initializer,
									biases_initializer=tf.zeros_initializer, activation_fn=tf.nn.relu, batch_norm=True, name='l1')
				synthetic_grad, var['l2_w'], var['l2_b'] = linear(h1, out_shape[1], weights_initializer=tf.zeros_initializer,
									biases_initializer=tf.zeros_initializer, activation_fn=tf.nn.relu, batch_norm=True, name='l2')
			return out, var['w'], var['b'], synthetic_grad
		else:
			return out, var['w'], var['b']

def flatten_space(space):
	if isinstance(space, Box):
		return np.prod(space.shape)
	elif isinstance(space, Discrete):
		return space.n
	else:
		raise ValueError("Env must be either Box or Discrete.")

"""
Problem 3:

1. Read the example provided in HW2_Policy_Graident.ipynb
2. Uncomment below function and implement it.

Sample solution is about 1~7 lines.
"""

def discount_cumsum(x, discount_rate):

	
    # YOUR CODE HERE >>>>>>
    # return ???
    # <<<<<<<<
