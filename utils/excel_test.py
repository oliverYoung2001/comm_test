import pandas as pd
from pandas import DataFrame
import os
import numpy as np

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

def write_new_excel():
    GPU_NUM = 7
    input_file = "./results/tmp.log"
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        output_files = ['./results/binet_test_cuda_7.xlsx']#, './results/net_test_cuda_7.xlsx']
        for i, file_path in enumerate(output_files):
            result_table = np.array([
                list(map((lambda x: float(x)), lines[l].strip('\n\r ').split(' ')))
                for l in range(i * GPU_NUM, (i + 1) * GPU_NUM)
            ])
            # result_table = [
            #     lines[l].strip('\n\r ').split(' ')
            #     for l in range(i * GPU_NUM, (i + 1) * GPU_NUM)
            # ]
            print(f'result_table: {result_table}')
            df = DataFrame(
                result_table,
                # index=[i + 1 for i in range(GPU_NUM)],
                columns=[i for i in range(GPU_NUM)],
            )
            # print(f'df: {df}')
            # print(f'df.head(): {df.head()}')
            # print(f'df.columns: {df.columns}')
            # print(f'df.index: {df.index}')
            df.to_excel(file_path, sheet_name='Sheet1')
            
    
def main():
    # make_excel_test()
    # test()
    write_new_excel()
    
    
    
if __name__ == '__main__':
    main()