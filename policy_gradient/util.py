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
    a=np.zeros(x.shape)
    sz=x.shape[0]

    for i in range(sz-2,-1,-1):
        temp=1                  
        for j in range(sz-i-1):            
            temp=temp*discount_rate                   
            a[i]=a[i]+x[i]*temp   
            
    a=a+x
    return a
