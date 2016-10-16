import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('loss.pkl', 'rb') as f:
    avg_r, max_r, min_r = pickle.load(f)

avg_r = np.asarray(avg_r)
max_r = np.asarray(max_r)
min_r = np.asarray(min_r)

x = np.arange(1.0, avg_r.shape[0]+1, 1.0)
plt.plot(x, avg_r)
plt.plot(x, max_r)
plt.plot(x, min_r)

plt.legend(['average', 'max', 'min'], loc='upper left')

plt.xlabel('Iterations')
plt.ylabel('Return values')
plt.title('Return without baseline')
plt.grid(True)
plt.savefig('return_wo_baseline.png')
plt.show()
