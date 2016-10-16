from gym.spaces import Box, Discrete
import numpy as np
from scipy.signal import lfilter

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
    dis_rw = [x[-1]]
    for i in range(1, len(x)):
        # add the current time step discounted reward with the previous one
        dis_rw.insert(0, x[-i] * (discount_rate**i) + dis_rw[0])

    return dis_rw

    # return ???
    # <<<<<<<<
