import os
import time

import numpy as np
from scipy.sparse import csr_matrix

from scipy.sparse.linalg import LinearOperator, gmres

from FasterSpMV.matrix_tools import *
from FasterSpMV.spmv import SpMVOperator


def solver_cpu_benchmark(sp_matrix):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # get CSR data
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # set x and get y
    x_exact = np.ones(n_col, dtype=np.float32)
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y = sp_A.dot(x_exact)

    # === Numba CSR test ===
    csr_spmvop = SpMVOperator('numba', 'csr', 'CPU', n_row, n_col,
                              csr_rowptr, csr_colidx, csr_val, 2)
    csr_A = LinearOperator((n_row, n_col), matvec=csr_spmvop.run_matvec)

    start = time.perf_counter()
    x, exitCode = gmres(csr_A, y)  # exit code 0 = converge
    end = time.perf_counter()
    err = np.linalg.norm(
        x - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)
    print("Numba CSR run time:", (end - start))
    print("Numba CSR exit code:", exitCode)
    print("Numba CSR result error:", err)
    print("")

    # === Numba Sliced ELLPACK test ===
    sell_spmvop = SpMVOperator('numba', 'sell', 'CPU', n_row, n_col,
                               csr_rowptr, csr_colidx, csr_val, 2)
    sell_A = LinearOperator((n_row, n_col), matvec=sell_spmvop.run_matvec)

    start = time.perf_counter()
    x, exitCode = gmres(sell_A, y)  # exit code 0 = converge
    end = time.perf_counter()
    err = np.linalg.norm(
        x - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)
    print("Numba SELL run time:", (end - start))
    print("Numba SELL exit code:", exitCode)
    print("Numba SELL result error:", err)
    print("")

    # === OpenCL CPU CSR test ===
    os.environ['PYOPENCL_CTX'] = '2'

    csr_spmvop = SpMVOperator('opencl', 'csr', 'CPU', n_row, n_col,
                              csr_rowptr, csr_colidx, csr_val, 2)
    csr_A = LinearOperator((n_row, n_col), matvec=csr_spmvop.run_matvec)

    start = time.perf_counter()
    x, exitCode = gmres(csr_A, y)  # exit code 0 = converge
    end = time.perf_counter()
    err = np.linalg.norm(
        x - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)
    print("OpenCL CPU CSR run time:", (end - start))
    print("OpenCL CPU CSR exit code:", exitCode)
    print("OpenCL CPU CSR result error:", err)
    print("")

    # === OpenCL CPU Sliced ELLPACK test ===
    sell_spmvop = SpMVOperator('opencl', 'sell', 'CPU', n_row, n_col,
                               csr_rowptr, csr_colidx, csr_val, 2)
    sell_A = LinearOperator((n_row, n_col), matvec=sell_spmvop.run_matvec)

    start = time.perf_counter()
    x, exitCode = gmres(sell_A, y)  # exit code 0 = converge
    end = time.perf_counter()
    err = np.linalg.norm(
        x - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)
    print("OpenCL CPU SELL run time:", (end - start))
    print("OpenCL CPU SELL exit code:", exitCode)
    print("OpenCL CPU SELL result error:", err)
    print("")


def solver_scipy_benchmark(sp_matrix):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # get CSR data
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # set x and get y
    x_exact = np.ones(n_col, dtype=np.float32)
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y = sp_A.dot(x_exact)

    # === SciPy test ===
    start = time.perf_counter()
    x, exitCode = gmres(sp_A, y)  # exit code 0 = converge
    end = time.perf_counter()
    err = np.linalg.norm(
        x - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)
    print("SciPy run time:", (end - start))
    print("SciPy exit code:", exitCode)
    print("SciPy result error:", err)
    print("")


def solver_gpu_benchmark(sp_matrix):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # get CSR data
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # set x and get y
    x_exact = np.ones(n_col, dtype=np.float32)
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y = sp_A.dot(x_exact)

    # # === OpenCL GPU CSR test ===
    # os.environ['PYOPENCL_CTX'] = '0'
    #
    # csr_spmvop = SpMVOperator('opencl', 'csr', 'GPU', n_row, n_col,
    #                           csr_rowptr, csr_colidx, csr_val, 128)
    # csr_A = LinearOperator((n_row, n_col), matvec=csr_spmvop.run_matvec)
    #
    # start = time.perf_counter()
    # x, exitCode = gmres(csr_A, y)  # exit code 0 = converge
    # end = time.perf_counter()
    # err = np.linalg.norm(
    #     x - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)
    # print("OpenCL GPU CSR run time:", (end - start))
    # print("OpenCL GPU CSR exit code:", exitCode)
    # print("OpenCL GPU CSR result error:", err)
    # print("")
    #
    # # === OpenCL GPU Sliced ELLPACK test ===
    # sell_spmvop = SpMVOperator('opencl', 'sell', 'GPU', n_row, n_col,
    #                            csr_rowptr, csr_colidx, csr_val, 128)
    # sell_A = LinearOperator((n_row, n_col), matvec=sell_spmvop.run_matvec)
    #
    # start = time.perf_counter()
    # x, exitCode = gmres(sell_A, y)  # exit code 0 = converge
    # end = time.perf_counter()
    # err = np.linalg.norm(
    #     x - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)
    # print("OpenCL GPU SELL run time:", (end - start))
    # print("OpenCL GPU SELL exit code:", exitCode)
    # print("OpenCL GPU SELL result error:", err)
    # print("")

    # === CUDA CSR test ===
    csr_spmvop = SpMVOperator('cuda', 'csr', 'GPU', n_row, n_col,
                              csr_rowptr, csr_colidx, csr_val, 128)
    csr_A = LinearOperator((n_row, n_col), matvec=csr_spmvop.run_matvec)

    start = time.perf_counter()
    x, exitCode = gmres(csr_A, y)  # exit code 0 = converge
    end = time.perf_counter()
    err = np.linalg.norm(
        x - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)
    print("CUDA CSR run time:", (end - start))
    print("CUDA CSR exit code:", exitCode)
    print("CUDA CSR result error:", err)
    print("")

    # === CUDA Sliced ELLPACK test ===
    sell_spmvop = SpMVOperator('cuda', 'sell', 'GPU', n_row, n_col,
                               csr_rowptr, csr_colidx, csr_val, 128)
    sell_A = LinearOperator((n_row, n_col), matvec=sell_spmvop.run_matvec)

    start = time.perf_counter()
    x, exitCode = gmres(sell_A, y)  # exit code 0 = converge
    end = time.perf_counter()
    err = np.linalg.norm(
        x - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)
    print("CUDA SELL run time:", (end - start))
    print("CUDA SELL exit code:", exitCode)
    print("CUDA SELL result error:", err)
