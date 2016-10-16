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
    # The sequence of x should be [t, t+1, t+2], so the accumulated discounted reward should be
    #  [t + (rate^1)*(t+1) + (rate^2)*(t+2), (t+1) + (rate^1)*(t+2), t+2]
    # 
    # For example: x = [1, 2, 3], rate = 0.9:
    #  [1 + (0.9^1)*2 + (0.9^2)*3, 2 + (0.9^1)*3, 3] = [5.23, 4.7, 1]
    dis_rw = [x[-1]]
    for i in range(1, len(x)):
        dis_rw.insert(0, dis_rw[-i] * discount_rate + x[-i-1])

    return dis_rw

    # return ???
    # <<<<<<<<
