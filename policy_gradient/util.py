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
    a=np.zeros(x.shape)+x
    tail=x.shape[0]-1

    for i in range(tail):
        temp=1
        for j in range(i+1,x.shape[0]):
            temp=temp*discount_rate
            a[i]=a[i]+x[j]*temp
    return a


