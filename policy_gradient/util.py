from gym.spaces import Box, Discrete
import numpy as np
from scipy.signal import lfilter
import math
import pdb

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
	discounted_r = np.zeros(len(x))
	num_r = len(x)
	for i in range(num_r):
		discounted_r[i] = x[i]*math.pow(discount_rate,i)
	discounted_r = np.cumsum(discounted_r[::-1])
	return discounted_r[::-1]
	# <<<<<<<<
