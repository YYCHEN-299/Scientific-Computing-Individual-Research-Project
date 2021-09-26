import argparse
import os

from scipy.sparse.linalg import LinearOperator, gmres
from scipy.io import mmread
from numba import set_num_threads

from benchmarks.linearoperator_benchmark import solver_cpu_benchmark
from benchmarks.spmv_benchmark import *
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
    opencl_test(matrix_data, args.slice)


def opencl_test(matrix_data, slice_height):
    datasets = ['data/consph.mtx', 'data/cant.mtx',
                'data/mac_econ_fwd500.mtx', 'data/mc2depi.mtx',
                'data/pdb1HYS.mtx', 'data/pwtk.mtx', 'data/rail4284.mtx',
                'data/rma10.mtx', 'data/scircuit.mtx', 'data/shipsec1.mtx',
                'data/webbase-1M.mtx']
    for str in datasets:
        matrix_data = mmread(str).tocsr()


def test_op_class():
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
    # os.environ['PYOPENCL_CTX'] = '2'
    # spmv_cpu_benchmark(matrix_data, 4, 2, 100)
    # os.environ['PYOPENCL_CTX'] = '0'
    # spmv_gpu_benchmark(matrix_data, 128, 2, 100)

    matrix_data = mmread('data/cant.mtx').tocsr()
    solver_cpu_benchmark(matrix_data)

    # os.environ['PYOPENCL_CTX'] = '2'
    # sp_matrix = mmread('data/cant.mtx').tocsr()
    # n_row, n_col = sp_matrix.shape
    # n_row = int(n_row)
    # n_col = int(n_col)
    # print("Sparse matrix row:", n_row, ", column:", n_col)
    #
    # # get CSR data
    # csr_rowptr = sp_matrix.indptr.astype(np.int32)
    # csr_colidx = sp_matrix.indices.astype(np.int32)
    # csr_val = sp_matrix.data.astype(np.float32)
    # sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    # v = np.ones(n_col, dtype=np.float32)
    # y_exact = sp_A.dot(v)  # SciPy SpMV
    #
    # csr_spmvop = SpMVOperator('opencl', 'sell', 'CPU', n_row, n_col,
    #                           csr_rowptr, csr_colidx, csr_val, 4)
    #
    # z = csr_spmvop.run_matvec(v)
    # print(z)
    # print(y_exact)
    #
    # csr_spmvop = SpMVOperator('opencl', 'csr', 'CPU', n_row, n_col,
    #                           csr_rowptr, csr_colidx, csr_val, 4)
    # csr_A = LinearOperator((n_row, n_col), matvec=csr_spmvop.run_matvec)
    #
    # start = time.perf_counter()
    # x, exitCode = gmres(csr_A, y_exact)  # exit code 0 = converge
    # end = time.perf_counter()
    # err = np.linalg.norm(
    #     x - v, np.inf) / np.linalg.norm(v, np.inf)
    # print("OpenCL CPU CSR run time:", (end - start))
    # print("OpenCL CPU CSR exit code:", exitCode)
    # print("OpenCL CPU CSR result error:", err)
    # print("")
