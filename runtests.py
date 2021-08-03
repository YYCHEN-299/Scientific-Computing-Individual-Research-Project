import sys

from scipy.io import mmread
from numba import set_num_threads
from argparse import ArgumentParser, REMAINDER

from benchmarks.test_spmv_speed import speed_test, performance_test


def main():
    # TODO command line test code
    # set_num_threads(8)
    # speed_test(1000, 2000, 10, 5, 100)

    matrix_data = mmread('data/consph.mtx').tocsr()
    performance_test(matrix_data, 5, 100)

    return 1


if __name__ == "__main__":
    main()
