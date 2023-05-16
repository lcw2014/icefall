import matplotlib.pyplot as plt
import os.path as PATH
import sys
import numpy as np
import torch
from collections import OrderedDict


# file_list = [
#     # 'results_baseline.txt', # baseline
#     # 'results_scheme0.txt', # 0
#     # 'result_surplus_random_init_only.txt', # 1-1
#     # 'result_surplus_copy_last_only.txt', # 1-2
#     # 'result_last_layer.txt', # 2
#     # 'result_adapter.txt', # 3-1
#     # 'result_adapter_16.txt', # 3-2
#     # 'result_whole-layer_40epochs.txt',
#     # 'result_whole-layer_45epochs.txt',
#     # 'result_average_whole-layers_train-extra.txt',
#     # 'result_average_whole-layers.txt',
#     'results_per_spkid.txt',
#     'results_per_spkid_baseline.txt',
#     'results_per_spkid_fed_35_avg.txt',
#     'results_per_spkid_fed_35_avg_test.txt',
#     'results_per_spkid_fed_40_avg.txt',
# ]

# file_list1 = []
# file_list2 = []

# for i in range(5,31):
#     file_list1.append(f'plm_average_topk_userlibri_{i}.txt')
#     file_list2.append(f'plm_average_topk{i}.txt')
alphas = np.arange(0, 1.1, 0.1)
alphas = np.array([round(a,1) for a in alphas])

dir_path = 'pruned_transducer_stateless5'
results = []
for alpha in alphas:
    data_path = PATH.join(dir_path,f'results_per_book_convex_{alpha}.txt')
    with open(data_path,'r') as f:
        temp = np.array([float(t.strip().split('\t')[1]) for t in f.readlines()])
        results.append(np.mean(temp))
plt.figure()
plt.plot(alphas,results)
plt.ylabel('WER')
plt.xlabel('alpha')
plt.savefig('pruned_transducer_stateless5/average_FED_baseline.png')


# # FED LM
# alpha=['1e-3', '1e-3', '1e-3', '1e-3', '1e-4', '1e-4', '1e-4', '1e-4', '1e-2', '1e-2', '1e-2',]
# beta=['1e-2', '1e-3', '1e-4', '1e-5', '1e-3', '1e-4', '1e-5', '1e-2', '1e-3','1e-4', '1e-5',]
# # alpha = ['1e-4']
# # beta = ['1e-1']
# dir_path = 'pruned_transducer_stateless5'
# results = OrderedDict()


# data_path1 = PATH.join(f'{dir_path}', 'results_per_book_fed_45_1e-4_1e-1.txt')
# with open(data_path1) as f:
#     baseline = [float(t.strip().split('\t')[1]) for t in  f.readlines()]
#     baseline = [np.average(np.array(baseline)) for _ in range(6)]
    

# for epoch in [35,40,45]:
#     for a, b in zip(alpha,beta):
#         file1 = f'results_per_book_fed_{epoch}_{a}_{b}.txt'
#         file2 = f'results_per_book_fed_{epoch}_{a}_{b}_avg.txt'
#         data_path1 = PATH.join(dir_path, file1)
#         data_path2 = PATH.join(dir_path, file2)
#         with open(data_path1) as f:
#             data1 = [float(t.strip().split('\t')[1]) for t in  f.readlines()]
#         with open(data_path2) as f:
#             data2 = [float(t.strip().split('\t')[1]) for t in  f.readlines()]
#         if epoch == 35:
#             results.setdefault(f'{a}_{b}',[np.average(np.array(data1)),np.average(np.array(data2)),])
#         else:
#             results[f'{a}_{b}'].extend([np.average(np.array(data1)),np.average(np.array(data2)),])
######################################
# # without FED
# for epoch in [35,40,45]:
#     for a, b in zip(alpha,beta):
#         file1 = f'results_per_book_{epoch}.txt'
#         file2 = f'results_per_book_{epoch}_avg.txt'
#         data_path1 = PATH.join(dir_path, file1)
#         data_path2 = PATH.join(dir_path, file2)
#         with open(data_path1) as f:
#             data1 = [float(t.strip().split('\t')[1]) for t in  f.readlines()]
#         with open(data_path2) as f:
#             data2 = [float(t.strip().split('\t')[1]) for t in  f.readlines()]
#         if epoch == 35:
#             results.setdefault(f'{a}_{b}',[np.average(np.array(data1)),np.average(np.array(data2)),])
#         else:
#             results[f'{a}_{b}'].extend([np.average(np.array(data1)),np.average(np.array(data2)),])
# result1 = OrderedDict()
# result2 = OrderedDict(
# for i, file in enumerate(zip(file_list1,file_list2)):
#     file1, file2 = file[0], file[1]
#     data_path1 = PATH.join(dir_path, file1)
#     data_path2 = PATH.join(dir_path, file2)
#     with open(data_path1) as f1, open(data_path2) as f2:
#         data1 = [float(t.strip().split('\t')[1]) for t in  f1.readlines()]
#         data2 = [float(t.strip().split('\t')[1]) for t in  f2.readlines()]

#     result1[file1.split('.')[0]] = np.average(np.array(data1))
#     result2[file2.split('.')[0]] = np.average(np.array(data2))
######################################

# plt.figure()
# x = np.arange(5,31)
# plt.plot(x,result1.values(),label='average and train')
# plt.plot(x,result2.values(),label='average')
# plt.ylabel('WER')
# plt.xlabel('The number of averaged models')
# plt.legend()
# plt.savefig(PATH.join(dir_path, 'Averaged PLM results.png'))

# print(results.keys())

# ax1 = plt.figure(1,dpi=100)
# keys = results.keys()
# values = results.values()

# plt.boxplot(values)
# plt.xticks(np.arange(1,len(file_list)+1), [
#     # 'baseline',
#     # '0',
#     # '1-1',
#     # '1-2',
#     # '2',
#     '3-1',
#     '3-2'])
# plt.ylabel('WER')
# plt.xlabel('LM Personalization Methods')
# # plt.ylim(0,12)
# plt.savefig(PATH.join(dir_path,'plm_results.png'))

# print([np.average(v) for v in values])
# exit()
# with open('pruned_transducer_stateless5/result_per_1k.txt') as f:
#     result_dict = OrderedDict()
#     for l in f.readlines():
#         value = l.strip().split('\t')[1]
#         key = l.split('\t')[0].split('_')[1]
#         result_dict.setdefault(key, [])
#         result_dict[key].append(float(value))

# # FED graph
# x1 = [35,'35_avg',40,'40_avg',45,'45_avg']
# ax2 = plt.figure(2)
# for k in results.keys():
#     print(results[k])
#     plt.plot(x1, results[k], label=k)
# plt.plot(x1,baseline,label='baseline')
# plt.xlabel('epoch')
# # plt.xticks(np.arange(0,20001,5000))
# # plt.xlim(0,20000)
# plt.yticks()
# plt.ylim(4.0,5.0)
# plt.ylabel('WER')
# plt.grid()
# ax2.legend(title='alpha_beta',bbox_to_anchor=(1.1,0.7))
# plt.savefig(PATH.join(dir_path,'fed_plm_results.png'),dpi=150,bbox_inches='tight')
