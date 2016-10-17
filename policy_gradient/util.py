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
    # MY CODE HERE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # x = [1 ,1 ,1, 1] , discount_rate = 0.5
    # rewards = x = [1,1,1,1]    
    # iteration #1: rewards = [1,1,1+0.5,1] = [1,1,1.5,1]
    # iteration #2: rewards = [1,1+0.5*1.5,1.5,1] = [1,1.75,1.5,1]
    # iteration #3: rewards = [1+0.5*1.75,1.75,1.5,1] = [1.875,1.75,1.5,1]
    # return rewards
    
    rewards = np.copy(x)
    for j in range(len(x)-2,-1,-1): # j is the index of rewards array
        rewards[j] = rewards[j]+discount_rate*rewards[j+1]
    return rewards
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<