import matplotlib.pyplot as plt
import json
from argparser import get_args
import math
import numpy as np

# BASELINE = 'SC1'
# METHODS = ['SC0', 'SC1', 'SC4', 'BRUCK', 'RD', '2DMESH', '3DMESH']
# BASELINE = 'NCCL'
# METHODS = ['Ring light-bar', 'NCCL', 'Pair light-bar', 'BRUCK', 'RD', '2DMESH', '3DMESH']
LIBs = ['NCCL', 'MSCCL']
COLORS = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']

y = [
    [0.03, 0.06, 0.11, 0.21, 0.35, 0.41, 0.41, 0.90, 1.00, 0.88, 1.02, 1.30, 1.45, 1.51, 1.52, 1.74, 1.73, 1.75, 1.74, 1.74, 1.68],
    [0.02, 0.05, 0.09, 0.18, 0.32, 0.53, 0.71, 0.84, 0.82, 0.86, 1.06, 1.37, 1.81, 2.20, 1.67, 1.73, 1.73, 1.72, 1.76, 1.67, 1.63],
]

def main():
    args = get_args()
    # file_name = f'results/{args.gpus}cu_all2all.json'
    # with open(file_name, encoding='utf-8') as f:
    #     results = json.load(f)
    # print(f'results: {results}')
    # x = np.log2(np.array(results[BASELINE]['SIZE']))
    x = np.arange(10, 10 + len(y[0]))
    
    # BASELINE_TIME = np.array(results[BASELINE]['time'])
    for i, lib_name in enumerate(LIBs):
        plt.plot(x, y[i], 'o-', color = COLORS[i], label=lib_name)#o-:圆形
    
    plt.title(f"All2All in Different DL CCL (4 A100 40G)")
    plt.xlabel("message_size_log (Byte)")#横坐标名字
    plt.ylabel("Bus Bandwidth (GB/s)")#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.savefig(f'results/All2All in Different DL CCL.png')
    plt.show()


if __name__ == '__main__':
    main()