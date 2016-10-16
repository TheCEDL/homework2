import numpy as np
import matplotlib.pyplot as plt
import seaborn

f=np.load("./substract.npz")["data"]
return_lst=np.array([])
adv_lst=np.array([])

for i in range(len(f)):
    return_lst = np.concatenate([return_lst,f[i]["returns"]])
    print f[i]["returns"].shape
    adv_lst = np.concatenate([adv_lst,f[i]["advantages_wo_norm"]])

var = np.var(adv_lst)
plt.plot(adv_lst)
plt.show()
print var

#587.128956331
#116.95866528

#587.128956331
#542.499453896
