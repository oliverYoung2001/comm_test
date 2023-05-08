import numpy as np
import math

def main():
    n = 1000
    f = np.zeros((n), dtype=np.float32)
    f[0] = 1
    f[1] = 0
    for i in range(2, n):
        for k in range(2, i + 1):
            f[i] += f[i - k]
        f[i] /= i
    for i in range(0, 10):
        print(f'{i}: {f[i]}, {f[i] * math.factorial(i)}')
        # print(f'{i}: {f[i]}')
        
    # ss = np.zeros((n), dtype=np.float32)
    # ss[0] = 1 / 2
    # ss[1] = 1 / 3
    # for i in range(2, n):
    #     ss[i] = (ss[i - 2] + (i + 1) * ss[i - 1]) / (i + 2)
    # for i in range(0, n):
    #     # print(f'{i}: {f[i]}, {f[i] * math.factorial(i)}')
    #     print(f'{i}: {ss[i]}')
    
    
if __name__ == '__main__':
    main()