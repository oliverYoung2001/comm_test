import matplotlib.pyplot as plt
import json
from argparser import get_args
import math
import numpy as np

BASELINE = 'SC1'
METHODS = ['SC0', 'SC1', 'SC4', 'BRUCK', 'RD', '2DMESH', '3DMESH']
COLORS = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']

def main():
    args = get_args()
    file_name = f'results/{args.gpus}cu_all2all.json'
    with open(file_name, encoding='utf-8') as f:
        results = json.load(f)
    # print(f'results: {results}')
    x = np.log2(np.array(results[BASELINE]['SIZE']))
    BASELINE_TIME = np.array(results[BASELINE]['time'])
    for i, method in enumerate(METHODS):
        plt.plot(x, np.array(results[method]['time']) / BASELINE_TIME,'o-',color = COLORS[i],label=method)#o-:圆形
    
    plt.xlabel("message_size_log (Byte)")#横坐标名字
    plt.ylabel("speedup")#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.savefig(f'results/{args.gpus}cu_all2all.png')
    plt.show()


if __name__ == '__main__':
    main()