import pandas as pd
from pandas import DataFrame
import os
import numpy as np

GPU_NUM = 8

def make_excel_test():
    GPUIDs = [0, 1]
    BD = 20.2
    file_path = r'./results/net_test.xlsx'
    if not os.path.exists(file_path):
        df = DataFrame(
            np.zeros((GPU_NUM, GPU_NUM)),
            # index=[i + 1 for i in range(GPU_NUM)],
            columns=[i + 1 for i in range(GPU_NUM)],
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
    
def main():
    make_excel_test()
    # test()
    
    
    
if __name__ == '__main__':
    main()