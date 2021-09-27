import argparse
import os

from scipy.io import mmread
from numba import set_num_threads

from benchmarks.linearoperator_benchmark import solver_cpu_benchmark
from benchmarks.spmv_benchmark import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", type=int, metavar='int',
                        help="number of threads (for Numba)")
    parser.add_argument("-f", "--file", metavar='filename',
                        help="filename of matrix data")
    parser.add_argument("-s", "--slice", type=int, metavar='int',
                        help="slice height of the Sliced ELLPACK format")
    args = parser.parse_args()
    matrix_data = mmread(args.file).tocsr()
    datasets = ['data/cant.mtx', 'data/consph.mtx',
                'data/cop20k_A.mtx', 'data/pdb1HYS.mtx', 'data/rma10.mtx']


if __name__ == "__main__":
    set_num_threads(8)
    datasets = ['data/cant.mtx']
    t = 200

    # slice_heights = [2, 4, 8, 16]
    # os.environ['PYOPENCL_CTX'] = '2'
    # for d in datasets:
    #     print("matrix name:", d)
    #     matrix_data = mmread(d).tocsr()
    #     spmv_cpu_csr_benchmark(matrix_data, t)
    #     for s in slice_heights:
    #         spmv_cpu_sell_benchmark(matrix_data, s, t)

    slice_heights = [128]
    os.environ['PYOPENCL_CTX'] = '0'
    for d in datasets:
        print("matrix name:", d)
        matrix_data = mmread(d).tocsr()
        spmv_gpu_csr_benchmark(matrix_data, t)
        for s in slice_heights:
            spmv_gpu_sell_benchmark(matrix_data, s, t)

    # matrix_data = mmread('data/cant.mtx').tocsr()
    # solver_cpu_benchmark(matrix_data)
