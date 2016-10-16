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
    sum = 0
    acc_sum = [0 for _ in range(len(x))]
    for i in range(len(x)-1,-1,-1):
        exp = len(x)-i-1
        sum += x[i]*(discount_rate**exp)
        acc_sum[i]=sum

    return np.asarray(acc_sum)
    # <<<<<<<<q