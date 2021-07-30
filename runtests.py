import sys

from scipy.io import mmread
from numba import set_num_threads
from argparse import ArgumentParser, REMAINDER

from benchmarks.test_spmv_speed import speed_test, performance_test


def main(argv):
    set_num_threads(8)
    # speed_test(10000, 500, 10, 5, 100)

    matrix_data = mmread('data/lp_nug30.mtx').tocsr()
    performance_test(matrix_data, 4, 500)

    return 1


if __name__ == "__main__":
    main(1)
