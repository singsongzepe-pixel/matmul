
from typing import List
import time
import random

def matmul(a: List[List[int]], b: List[List[int]], n: int):
    c = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]

    return c


def main():
    random.seed(42)
    
    # 4096 is too hard to calculate, so we change it to be a 1024x1024 matrix
    # a is 1024x1024 matrix
    a = [[random.random() for _ in range(1024)] for _ in range(1024)]
    # b is 1024x1024 matrix
    b = [[random.random() for _ in range(1024)] for _ in range(1024)]

    start = time.time()
    c = matmul(a, b, len(a))
    end = time.time()
    print(f"Matrix Multiplication of 1024x1024 matrix Time: {end - start}")

if __name__ == '__main__':
    main()


# we record the time cost of the matrix multiplication there
# test result:
# 0: 212.67234325408936
