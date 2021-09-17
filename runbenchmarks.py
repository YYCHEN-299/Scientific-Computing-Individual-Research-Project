import argparse
import os
import numpy as np
import time

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.sparse import csc_matrix, csr_matrix
from numba import set_num_threads, threading_layer
from scipy.io import mmread

from benchmarks.spmv_performance_benchmarks import numba_performance_benchmark
from benchmarks.spmv_performance_benchmarks import opencl_performance_benchmark
from benchmarks.spmv_performance_benchmarks import cuda_performance_benchmark
from benchmarks.spmv_performance_benchmarks import class_performance_benchmark
from benchmarks.spmv_performance_benchmarks import test_tool
from FasterSpMV.spmv import SpMVOperator
from FasterSpMV.matrix_tools import *


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
    numba_performance_benchmark(matrix_data, slice_height, 1000)


def opencl_test(matrix_data, slice_height):
    os.environ['PYOPENCL_CTX'] = '0'
    # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    datasets = ['data/consph.mtx', 'data/cant.mtx',
                'data/mac_econ_fwd500.mtx', 'data/mc2depi.mtx',
                'data/pdb1HYS.mtx', 'data/pwtk.mtx', 'data/rail4284.mtx',
                'data/rma10.mtx', 'data/scircuit.mtx', 'data/shipsec1.mtx',
                'data/webbase-1M.mtx']
    for str in datasets:
        matrix_data = mmread(str).tocsr()
        opencl_performance_benchmark(matrix_data, slice_height, 100)


def class_test():
    n_row = 2000
    n_col = 2000
    sp_matrix, nnz_count, row_max_nnz = random_spmatrix(n_row, n_col, 0)
    csr_rowptr, csr_colidx, csr_val = spmatrix_to_csr(sp_matrix)
    slice_height = 4

    csrspmvop = SpMVOperator('numba', 'csr', n_row, n_col,
                             csr_rowptr, csr_colidx, csr_val, slice_height)
    sellspmvop = SpMVOperator('numba', 'sell', n_row, n_col,
                              csr_rowptr, csr_colidx, csr_val, slice_height)
    sellspmvop1 = SpMVOperator('numba', 'sell', n_row, n_col,
                               csr_rowptr, csr_colidx, csr_val, 2)
    sellspmvop2 = SpMVOperator('numba', 'sell', n_row, n_col,
                               csr_rowptr, csr_colidx, csr_val, 8)
    A = LinearOperator((n_row, n_col), matvec=csrspmvop.run)
    K = LinearOperator((n_row, n_col), matvec=sellspmvop.run)
    K1 = LinearOperator((n_row, n_col), matvec=sellspmvop1.run)
    K2 = LinearOperator((n_row, n_col), matvec=sellspmvop2.run)
    b = np.ones(n_col, dtype=np.float32)
    zz = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    Z = zz.tocsc().astype(np.float32)

    t_start = time.perf_counter()
    x, exitCode = gmres(Z, b)
    t_end = time.perf_counter()
    print('normal e code:', exitCode, ', t:', (t_end - t_start))

    t_start = time.perf_counter()
    x, exitCode = gmres(A, b)
    t_end = time.perf_counter()
    print('csr e code:', exitCode, ', t:', (t_end - t_start))

    t_start = time.perf_counter()
    x, exitCode = gmres(K, b)
    t_end = time.perf_counter()
    print('sell 4 e code:', exitCode, ', t:', (t_end - t_start))


if __name__ == "__main__":
    # matrix_data = mmread('data/cant.mtx').tocsr()
    # cuda_performance_benchmark(matrix_data, 64, 100)
    # matrix_data = 0
    # opencl_test(matrix_data, 32)
    # os.environ['PYOPENCL_CTX'] = '2'
    # matrix_data = mmread('data/consph.mtx').tocsr()
    # class_performance_benchmark(matrix_data, 32, 100)
    # cuda_performance_benchmark(matrix_data, 32, 100)
    # os.environ['PYOPENCL_CTX'] = '2'
    # class_test()
    # matrix_data = mmread('data/consph.mtx').tocsr()
    # numba_test(8, matrix_data, 4)
    test_tool()
