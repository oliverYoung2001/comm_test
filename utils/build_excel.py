import pandas as pd
from pandas import DataFrame
import os
import numpy as np
from argparser import get_args
import json
pd.set_option('mode.chained_assignment', None)

GPU_NUM = 8

def make_excel_test():
    GPUIDs = [0, 1]
    BD = 0
    file_path = r'./results/net_test_nico3_8_ref.xlsx'
    if not os.path.exists(file_path):
        df = DataFrame(
            np.zeros((GPU_NUM, GPU_NUM)),
            # index=[i + 1 for i in range(GPU_NUM)],
            columns=[i for i in range(GPU_NUM)],
        )
        # print(f'df: {df}')
        # print(f'df.head(): {df.head()}')
        # print(f'df.columns: {df.columns}')
        # print(f'df.index: {df.index}')
        df.to_excel(file_path, sheet_name='Sheet1')
    df = pd.read_excel(file_path, index_col=0)
    df[GPUIDs[1]][GPUIDs[0]] = BD
    # df.ix[GPUIDs[0] + 1, GPUIDs[1] + 1] = BD
    print(df)
    DataFrame(df).to_excel(file_path, sheet_name='Sheet1')
    
def test():
    data = { 'name': ['zs', 'ls', 'ww'], 'age': [11, 12, 13], 'gender': ['man', 'man', 'woman']}
    df = DataFrame(data)
    print(df)
    df.to_excel('./results/new.xlsx')

def write_new_excel(args):
    # GPU_NUM = 7
    with open('results/' + args.input_file_name + '.json', 'r', encoding='utf-8') as f:
        p2p = json.load(f)
    
    
    # output_files = ['./results/binet_test_cuda_7.xlsx']#, './results/net_test_cuda_7.xlsx']
    output_file = 'results/' + args.input_file_name + '.xlsx'
    if not os.path.exists(output_file):
        df_si = DataFrame(
            np.array(p2p['P2P_SI']),
            # index=[i + 1 for i in range(GPU_NUM)],
            columns=[i for i in range(len(p2p['P2P_SI']))],
        )
        # df_si = df_si.style.set_caption('SI')
        df_bi = DataFrame(
            np.array(p2p['P2P_BI']),
            # index=[i + 1 for i in range(GPU_NUM)],
            columns=[i for i in range(len(p2p['P2P_BI']))],
        )
        # df_bi = df_bi.style.set_caption('BI')
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_si.to_excel(writer, sheet_name='Sheet1', startrow=0)
            df_bi.to_excel(writer, sheet_name='Sheet1', startrow=df_si.shape[0] + 1)
    
    df_si = pd.read_excel(output_file, sheet_name='Sheet1', index_col=0, nrows=len(p2p['P2P_SI']))
    df_bi = pd.read_excel(output_file, sheet_name='Sheet1', index_col=0, nrows=len(p2p['P2P_BI']), header=len(p2p['P2P_SI']) + 1)
    assert(df_si.shape == df_bi.shape)
    assert(df_si.shape == (len(p2p['P2P_SI']), len(p2p['P2P_SI'][0])))
    assert(df_bi.shape == (len(p2p['P2P_BI']), len(p2p['P2P_BI'][0])))
    for src in range(df_si.shape[0]):
        for dst in range(df_si.shape[1]):
            if p2p['P2P_SI'][src][dst] > 0:
                # df_si.loc[src].loc[dst] = p2p['P2P_SI'][src][dst]
                df_si[src][dst] = p2p['P2P_SI'][src][dst]
            if p2p['P2P_BI'][src][dst] > 0:
                # df_bi.loc[src].loc[dst] = p2p['P2P_BI'][src][dst]
                df_bi[src][dst] = p2p['P2P_BI'][src][dst]
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_si.to_excel(writer, sheet_name='Sheet1', startrow=0)
        df_bi.to_excel(writer, sheet_name='Sheet1', startrow=df_si.shape[0] + 1)
            
    # print(f'df_si: {df_si}')
    # print(f'df_bi: {df_bi}')
    
    # df_si = DataFrame(
    #     np.array(p2p['P2P_SI']),
    #     # index=[i + 1 for i in range(GPU_NUM)],
    #     columns=[i for i in range(len(p2p['P2P_SI']))],
    # )
    # # df_si = df_si.style.set_caption('SI')
    # df_bi = DataFrame(
    #     np.array(p2p['P2P_BI']),
    #     # index=[i + 1 for i in range(GPU_NUM)],
    #     columns=[i for i in range(len(p2p['P2P_BI']))],
    # )
    # # df_bi = df_bi.style.set_caption('BI')
    # with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    #     df_si.to_excel(writer, sheet_name='Sheet1', startrow=0)
    #     df_bi.to_excel(writer, sheet_name='Sheet1', startrow=df_si.shape[0] + 1)
            
    
def main():
    args = get_args()
    # make_excel_test()
    # test()
    write_new_excel(args)
    
    
    
if __name__ == '__main__':
    main()