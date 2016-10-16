from gym.spaces import Box, Discrete
import numpy as np
#from scipy.signal import lfilter

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
    # return ???
    # <<<<<<<<
def discount_cumsum(x,discount_rate):
	cumsum=np.zeros(np.array(x).shape[0])
	cumsum[np.array(x).shape[0]-1]=x[np.array(x).shape[0]-1]
	rate=1
	for i in range(np.array(x).shape[0]-2,-1,-1):
		rate=rate*discount_rate
		cumsum[i]=x[i]*rate+cumsum[i+1]
        return cumsum


if __name__ == '__main__':
	print discount_cumsum(np.array([1, 1, 1, 1]), 0.99)
