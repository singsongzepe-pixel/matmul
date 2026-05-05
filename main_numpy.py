import time
import numpy as np


def main():
    a = np.random.rand(4096, 4096)
    b = np.random.rand(4096, 4096)

    # record start time
    start = time.time()
    c = a @ b.T
    end = time.time()

    # print in millisecond
    print(f"Matrix Multiplication of 4096x4096 matrix Time: {(end - start) * 1000:.6f} milliseconds")

if __name__ == '__main__':
    main()
