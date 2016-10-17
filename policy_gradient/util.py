from gym.spaces import Box, Discrete
import numpy as np
from scipy.signal import lfilter
import math

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
    R = [x[-1]]
    for V in x[:-1][::-1]:
        R.append(V + discount_rate*R[-1])
    
    return np.array(R[::-1])
    # <<<<<<<<
