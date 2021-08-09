import sys

from scipy.io import mmread
from numba import set_num_threads
from argparse import ArgumentParser, REMAINDER

from benchmarks.test_spmv_speed import random_data_test, data_set_test, avx_test


def main():
    # TODO command line test code
    # set_num_threads(8)
    # row, column, nnz, slice height, loop number
    random_data_test(100, 200, 0, 64, 100)

    # matrix_data = mmread('data/consph.mtx').tocsr()
    # data_set_test(matrix_data, 5, 100)
    # avx_test()

    return 1


if __name__ == "__main__":
    main()
