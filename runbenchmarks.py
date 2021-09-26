import argparse

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
    opencl_test(matrix_data, args.slice)


def opencl_test(matrix_data, slice_height):
    datasets = ['data/consph.mtx', 'data/cant.mtx', 'data/rma10.mtx']
    for d in datasets:
        matrix_data = mmread(d).tocsr()


if __name__ == "__main__":
    set_num_threads(8)

    # matrix_data = mmread('data/cant.mtx').tocsr()
    # os.environ['PYOPENCL_CTX'] = '2'
    # spmv_cpu_benchmark(matrix_data, 4, 2, 100)
    # os.environ['PYOPENCL_CTX'] = '0'
    # spmv_gpu_benchmark(matrix_data, 128, 2, 100)

    matrix_data = mmread('data/cant.mtx').tocsr()
    solver_cpu_benchmark(matrix_data)
