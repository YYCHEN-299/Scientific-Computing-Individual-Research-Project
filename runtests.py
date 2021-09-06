import argparse
import os

from numba import set_num_threads
from scipy.io import mmread

from benchmarks.spmv_performance_benchmarks import numba_performance_benchmark
from benchmarks.spmv_performance_benchmarks import opencl_performance_benchmark


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
    numba_test(args.threads, matrix_data, args.slice)
    print("========================================")
    opencl_test(matrix_data, args.slice)


def numba_test(threads, matrix_data, slice_height):
    set_num_threads(threads)
    numba_performance_benchmark(matrix_data, slice_height, 100)


def opencl_test(matrix_data, slice_height):
    # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    opencl_performance_benchmark(matrix_data, slice_height, 100)


if __name__ == "__main__":
    os.environ['PYOPENCL_CTX'] = '0'
    datasets = ['data/consph.mtx', 'data/cant.mtx',
                'data/mac_econ_fwd500.mtx', 'data/mc2depi.mtx',
                'data/pdb1HYS.mtx', 'data/pwtk.mtx', 'data/rail4284.mtx',
                'data/rma10.mtx', 'data/scircuit.mtx', 'data/shipsec1.mtx',
                'data/webbase-1M.mtx']
    for str in datasets:
        matrix_data = mmread(str).tocsr()
        opencl_test(matrix_data, 64)
