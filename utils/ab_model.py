import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def main():
    NUM = 23
    SIZE = np.zeros((NUM))
    for i in range(NUM):
        SIZE[i] = 1 << i
    # BW = np.array([
    #     0.043660,
    #     0.084615,
    #     0.134467,
    #     0.268721,
    #     1.101678,
    #     2.109389,
    #     3.887590,
    #     6.644005,
    #     10.232428,
    #     14.076982,
    #     17.385683,
    #     19.669329,
    #     21.030006,
    #     21.641304,
    #     22.092847,
    #     22.377636,
        
    # ])
    T = np.array([
        0.043686,
        0.045083,
        0.056738,
        0.056783,
        0.027701,
        0.028935,
        0.031400,
        0.036746,
        0.047719,
        0.069373,
        0.112341,
        0.198596,
        0.371493,
        0.721999,
        1.414485,
        2.792967,
        5.559907,
        11.091417,
        22.157078,
        44.283284,
        88.538600,
        177.046527,
        354.052678,
    ])
    N = np.zeros((NUM))
    N[0] = 1024 / pow(1024, 3)     # 1KB = 1 / pow(1024, 2) GB
    for i in range(1, NUM):
        N[i] = N[i - 1] * 2
    
    TIMES = 2000
    
    N_ = sm.add_constant(N) # 若模型中有截距，必须有这一步
    model = sm.OLS(T, N_).fit() # 构建最小二乘模型并拟合
    print(model.summary()) # 输出回归结果

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    predicts = model.predict() # 模型的预测值
    plt.scatter(N, T, label='实际值') # 散点图
    plt.plot(N, predicts, color = 'red', label='预测值')
    plt.legend() # 显示图例，即每条线对应 label 中的内容
    plt.show() # 显示图形
    
if __name__ == '__main__':
    main()