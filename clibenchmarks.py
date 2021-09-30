import argparse
import os

from scipy.io import mmread

from benchmarks.spmv_benchmark import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", metavar='filename',
                        help="filename of matrix data")
    parser.add_argument("-t", "--times", type=int, metavar='int',
                        help="The number of times each test is run")
    args = parser.parse_args()
    matrix_data = mmread(args.file).tocsr()
    t = args.times

    spmv_scipy_benchmark(matrix_data, t)

    slice_heights = [2, 4, 8, 16]
    os.environ['PYOPENCL_CTX'] = '2'
    spmv_cpu_csr_benchmark(matrix_data, t)
    spmv_cpu_exp_sell_benchmark(matrix_data, t)
    for s in slice_heights:
        spmv_cpu_sell_benchmark(matrix_data, s, t)

    slice_heights = [32, 64, 128, 256, 512]
    os.environ['PYOPENCL_CTX'] = '0'
    spmv_gpu_csr_benchmark(matrix_data, t)
    spmv_pycuda_csr_benchmark(matrix_data, t)
    for s in slice_heights:
        spmv_gpu_sell_benchmark(matrix_data, s, t)
        spmv_pycuda_sell_benchmark(matrix_data, s, t)


if __name__ == "__main__":
    main()
