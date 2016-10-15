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

def discount_cumsum(x, discount_rate):
    discount_array = np.power(discount_rate, np.arange(len(x)+1))
    return [sum(x[i:] * discount_array[:-i-1]) for i in range(len(x))]
