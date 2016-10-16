import numpy as np
import matplotlib.pyplot as plt
import seaborn

type = "without_baseline"
result_list = []
max_len = 0
for i in range(10):
    f = np.load('./%s_%d.npz'%(type,i+1))
    result_list.append(f['average_return_list'])
    if f['average_return_list'].shape[0]>max_len:
        max_len = f['average_return_list'].shape[0]

for i in range(10):
    x_axis = np.arange(1, len(result_list[i])+1)
    plt.plot(x_axis,result_list[i],label='iteration %d'%(i+1))
    plt.legend(bbox_to_anchor=(0.1, 1), loc=1, borderaxespad=0.)

plt.xlabel('Iteration')
plt.ylabel('Average_return')
plt.title('10 different training cycle')
plt.show()

# non_zero = np.zeros((max_len),dtype=np.float32)
# sum_lst = np.zeros((max_len),dtype = np.float32)
# std_lst = np.zeros((max_len),dtype=np.float32)
# print non_zero.shape
# for j in range(max_len):
#     element = []
#     for i in range(10):
#
#         try:
#             if result_list[i][j]>0:
#                 non_zero[j]+=1
#                 element.append(result_list[i][j])
#             sum_lst[j] += result_list[i][j]
#         except:
#             pass
#     element = np.asarray(element)
#     v = (np.var(element))**0.5
#     std_lst[j]=v
# print non_zero
# print std_lst
# mean_lst = sum_lst/non_zero
# x_axis=np.arange(1,max_len+1)
# #plt.plot(x_axis,mean_lst,'b')
# plt.errorbar(x_axis,mean_lst,yerr = [std_lst,std_lst])
# plt.xlabel('Iteration')
# plt.ylabel('Average_return')
# plt.title('Average progress plot with std')
# plt.show()

