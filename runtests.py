import argparse

from numba import set_num_threads
from scipy.io import mmread

from benchmarks.spmv_performance_benchmarks import data_set_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", type=int, metavar='int',
                        help="number of threads")
    parser.add_argument("-f", "--file", metavar='filename',
                        help="filename of matrix data")
    parser.add_argument("-s", "--slice", type=int, metavar='int',
                        help="slice height of the Sliced ELLPACK format")
    args = parser.parse_args()

    set_num_threads(args.threads)
    matrix_data = mmread(args.file).tocsr()
    data_set_test(matrix_data, args.slice, 100)


if __name__ == "__main__":
    main()
