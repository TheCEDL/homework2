import matplotlib.pyplot as plt
import numpy as np
import pickle
import pdb

with open('wbl.pkl', 'rb') as f:
    avg_r_wbl, std_r_wbl, max_r_wbl, min_r_wbl = pickle.load(f)
with open('wobl.pkl', 'rb') as f:
    avg_r_wobl, std_r_wobl, max_r_wobl, min_r_wobl = pickle.load(f)


std_r_wbl = np.asarray(std_r_wbl)
std_r_wobl = np.asarray(std_r_wobl)

x = np.arange(1, std_r_wbl.shape[0]+1, dtype=np.float64)
plt.plot(x, std_r_wbl)
x = np.arange(1, std_r_wobl.shape[0]+1, dtype=np.float64)
plt.plot(x, std_r_wobl)

plt.legend(['std with baseline', 'std without baseline'], loc='upper left')

plt.xlabel('Iterations')
plt.ylabel('Standard Deviation (square root of Variances)')
plt.title('Comparison of std with/without baseline')
plt.grid(True)
plt.savefig('std_comparison.png')
plt.show()
