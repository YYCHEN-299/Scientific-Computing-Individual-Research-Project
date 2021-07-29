import sys

from numba import set_num_threads
from argparse import ArgumentParser, REMAINDER

from tests.test_spmv_speed import speed_test


def main(argv):
    set_num_threads(4)
    speed_test(1000, 1000, 0, 4)
    return 1


if __name__ == "__main__":
    main(1)
