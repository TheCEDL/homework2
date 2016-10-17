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

# def discount_cumsum(x, discount_rate):
    # YOUR CODE HERE >>>>>>
def discount_cumsum(x, discount_rate):
    len_x = len(x)
    discount_x = np.zeros(len_x)
    for i in range(len_x):
        discount_x[i] = x[i] * discount_rate**i
    discount_x =np.cumsum(discount_x[::-1])
    return discount_x
    # return ???
    # <<<<<<<<
