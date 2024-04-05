import numpy as np
import matplotlib.pyplot as plt
import json 
import math
from common import *


def upper_bound(arr, val):
    for i, v in enumerate(arr):
        if v > val:
            return i
    return len(arr)
    
def draw_flash_attn_flops(FILE_NAMEs, DTYPE):
    # flash_attn: [] -> [mbs * S, Nh, E]
    # args: (S, mbs, Nh, E)
    # for mbs in [1]:
    #     for Nh in [8, 16, 32]:
    #         for E in [64, 128]:
    #             for S in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
    B = np.array([1])
    Nh = np.array([8, 16, 32])
    E = np.array([64, 128])
    S = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152])
    S_str = ['256', '512', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1M', '2M']
    MAX_MUL = 1 * 8 * 64 * 2097152  # 1G 
    xt = np.arange(0, 300, 301 / len(S))
    Times_fwd = np.zeros((len(Nh), len(E), len(S)))
    Times_bwd = np.zeros((len(Nh), len(E), len(S)))
    FLOPs_fwd = np.zeros((len(Nh), len(E), len(S)))
    FLOPs_bwd = np.zeros((len(Nh), len(E), len(S)))
    
    for FILE_NAME in FILE_NAMEs:
        with open(FILE_NAME, 'r') as f:
            COMP_TABLE = parse_prof_data(json.load(f))
        
        print(f'COMP_TABLE: {COMP_TABLE}')
        for shapes, times in COMP_TABLE['flash_attn'].items():
            s, b, nh, e = shapes
            assert(b in B)
            assert(s in S)
            assert(e in E)
            assert(len(times) == 2)
            # b_ = B.tolist().index(b)
            s_ = S.tolist().index(s)
            nh_ = Nh.tolist().index(nh)
            e_ = E.tolist().index(e)
            Times_fwd[nh_][e_][s_] += times[0]
            Times_bwd[nh_][e_][s_] += times[1]

    for nh_, nh in enumerate(Nh):
        for e_, e in enumerate(E):
            for s_, s in enumerate(S):
                if 1 * nh * e * s <= MAX_MUL:
                    Times_fwd[nh_][e_][s_] /= len(FILE_NAMEs)
                    Times_bwd[nh_][e_][s_] /= len(FILE_NAMEs)
                    FLOPs_fwd[nh_][e_][s_] = 4.0 * s * s * e * nh / pow(1000, 4) / Times_fwd[nh_][e_][s_] * pow(1000, 2) / 2   # Tflops, causal=True
                    FLOPs_bwd[nh_][e_][s_] = 4.0 * s * s * e * nh / pow(1000, 4) / Times_bwd[nh_][e_][s_] * pow(1000, 2) / 2 * 2.5   # Tflops, causal=True
    print(f'FLOPs_fwd: {FLOPs_fwd}')
    print(f'FLOPs_bwd: {FLOPs_bwd}')
    # plot fwd:
    for e_, e in enumerate(E):
        for nh_, nh in enumerate(Nh):
            ub = min(len(S), upper_bound(S, MAX_MUL / (1 * nh * e)))
            print(f'nh: {nh}, e: {e}, ub_S: {S[ub - 1]}')
            plt.plot(xt[: ub], FLOPs_fwd[nh_][e_][: ub], 'o-', label=f'falsh_attn(S,mbs=1,Nh={nh},E={e})')
        # plt.xlim(-10,300,20)   #设定x轴显示范围
        # plt.xticks(xt, S_str)  #修改x轴刻度，并将刻度旋转30度

        # TITLE = f'flash_attn_fwd_flops_nh={nh}'
        # plt.title(TITLE)
        # plt.xlabel("Seq_len")#横坐标名字
        # plt.ylabel("FLOPs (Tflops)")#纵坐标名字
        # plt.legend(loc = "best")#图例
        # plt.savefig(f"./results/{TITLE}.png")
        # plt.show()
        # plt.clf()
    plt.xlim(-10,300,20)   #设定x轴显示范围
    plt.xticks(xt, S_str)  #修改x轴刻度，并将刻度旋转30度

    TITLE = f'flash_attn_fwd_flops_{DTYPE}'
    plt.title(TITLE)
    plt.xlabel("Seq_len")#横坐标名字
    plt.ylabel("FLOPs (Tflops)")#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.savefig(f"./results/{TITLE}.png")
    plt.show()
    plt.clf()
    
    # plot bwd:
    for e_, e in enumerate(E):
        for nh_, nh in enumerate(Nh):
            ub = min(len(S), upper_bound(S, MAX_MUL / (1 * nh * e)))
            plt.plot(xt[: ub], FLOPs_bwd[nh_][e_][: ub], 'o-', label=f'falsh_attn(S,mbs=1,Nh={nh},E={e})')
        # plt.xlim(-10,300,20)   #设定x轴显示范围
        # plt.xticks(xt, S_str)  #修改x轴刻度，并将刻度旋转30度

        # TITLE = f'flash_attn_bwd_flops_nh={nh}'
        # plt.title(TITLE)
        # plt.xlabel("Seq_len")#横坐标名字
        # plt.ylabel("FLOPs (Tflops)")#纵坐标名字
        # plt.legend(loc = "best")#图例
        # plt.savefig(f"./results/{TITLE}.png")
        # plt.show()
        # plt.clf()
    plt.xlim(-10,300,20)   #设定x轴显示范围
    plt.xticks(xt, S_str)  #修改x轴刻度，并将刻度旋转30度

    TITLE = f'flash_attn_bwd_flops_{DTYPE}'
    plt.title(TITLE)
    plt.xlabel("Seq_len")#横坐标名字
    plt.ylabel("FLOPs (Tflops)")#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.savefig(f"./results/{TITLE}.png")
    plt.show()
    plt.clf()

def draw_coll_comm(GPU_NUMs, HOST):
    for GPU_NUM in GPU_NUMs:
        with open(f'./prof_data/coll_comm_bench_{GPU_NUM}_{HOST}.json') as f:
            BW_TABLE = json.load(f)
        assert 'meta' in BW_TABLE.keys()
        sizes = BW_TABLE['meta']
        size_strs = [convert_size(s, infix='', suffix='') for s in sizes]
        xt_len = 300
        xt = np.arange(0, xt_len, (xt_len + 1) / len(sizes))
    
        del BW_TABLE['meta']
        for coll_name, bws in BW_TABLE.items():
            for k, v in bws.items():
                assert(len(v) == len(sizes))
            plt.plot(xt, bws['tput'], 'o-', label=f'{coll_name}')

        plt.xlim(-10,xt_len,20)   #设定x轴显示范围
        plt.xticks(xt, size_strs, size=8)  #修改x轴刻度，并将刻度旋转30度

        TITLE = f'coll_comm_tput_ws{GPU_NUM}'
        plt.title(TITLE)
        plt.xlabel("Message Size Per GPU (B)")#横坐标名字
        plt.ylabel("Throughput Per GPU (GB/s)")#纵坐标名字
        plt.legend(loc = "best")#图例
        plt.savefig(f"./prof_data/{TITLE}.png")
        plt.show()
        plt.clf()
    
def main():
    # coll_comm_bench_${GPU_NUM}_${HOST}
    # FILE_NAMEs = [f'./prof_data/coll_comm_bench.json' for i in range(NUM_FILE)]
    GPU_NUMs = [2, 4, 8]
    HOST = 'g4004'
    draw_coll_comm(GPU_NUMs, HOST)
    
    # matmul
    # NUM_FILE = 1
    # DTYPE = 'bf16'
    # FILE_NAMEs = [f'./prof_data/time_matmul_tp=1_{DTYPE}_{i}.json' for i in range(NUM_FILE)]
    # draw_matmul_flops(FILE_NAMEs, DTYPE)
    
    
    
if __name__ == '__main__':
    main()