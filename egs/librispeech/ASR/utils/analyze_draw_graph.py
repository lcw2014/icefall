import matplotlib.pyplot as plt
import os.path as PATH
import sys
import numpy as np
from collections import OrderedDict


file_list = [
    # 'results_baseline.txt', # baseline
    # 'results_scheme0.txt', # 0
    # 'result_surplus_random_init_only.txt', # 1-1
    # 'result_surplus_copy_last_only.txt', # 1-2
    # 'result_last_layer.txt', # 2
    # 'result_adapter.txt', # 3-1
    # 'result_adapter_16.txt', # 3-2
    'result_whole-layer_40epochs.txt',
    'result_whole-layer_45epochs.txt',
    'result_average_whole-layers_train-extra.txt',
    'result_average_whole-layers.txt',
]

dir_path = 'pruned_transducer_stateless5'
results = OrderedDict()
for i, file in enumerate(file_list):
    data_path = PATH.join(dir_path, file)
    with open(data_path) as f:
        data = [float(t.strip().split('\t')[1]) for t in  f.readlines()]
    results[file.split('.')[0]] = np.array(data)

print(results.keys())

ax1 = plt.figure(1,dpi=100)
keys = results.keys()
values = results.values()

plt.boxplot(values)
plt.xticks(np.arange(1,len(file_list)+1), [
    # 'baseline',
    # '0',
    # '1-1',
    '1-2',
    '2',
    '3-1',
    '3-2'])
plt.ylabel('WER')
plt.xlabel('LM Personalization Methods')
# plt.ylim(0,12)
plt.savefig(PATH.join(dir_path,'plm_results.png'))

print([np.average(v) for v in values])
exit()
with open('pruned_transducer_stateless5/result_per_1k.txt') as f:
    result_dict = OrderedDict()
    for l in f.readlines():
        value = l.strip().split('\t')[1]
        key = l.split('\t')[0].split('_')[1]
        result_dict.setdefault(key, [])
        result_dict[key].append(float(value))


ax2 = plt.figure(2)
for k in result_dict.keys():
    print(result_dict[k])
    plt.plot(np.arange(1, len(result_dict[k]) + 1) * 1000, result_dict[k], label=k)

plt.xlabel('The Number of Data')
plt.xticks(np.arange(0,20001,5000))
plt.xlim(0,20000)
# plt.yticks([k[0] for k in result_dict.values()])
plt.ylabel('WER')
plt.grid()
ax2.legend(bbox_to_anchor=(1.08,0.75),title='Book ID')
plt.savefig(PATH.join(dir_path,'plm_reults_per_1k.png'),dpi=150,bbox_inches='tight')
