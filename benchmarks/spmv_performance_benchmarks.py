import math
import time

import pyopencl as cl
import numpy as np
import numba
from numba import cuda
from scipy.sparse import csr_matrix

from code.debug_tools import find_instr
from code.matrix_tools import csr_to_sellpack, spmatrix_to_csr, random_spmatrix
from code.numba_spmv import numba_csr_spmv, numba_sliced_ellpack_spmv
from code.opencl_spmv import BaseSELLSpMV, SELLSpMV, CSRSpMV
from code.cuda_spmv import cuda_csr_spmv
from code.cuda_spmv import cuda_sliced_ellpack_spmv


def numba_random_data_test(n_row, n_col, per_nnz, slice_height, t):
    """
    Test Numba SpMV performance
    """

    # generate a sparse matrix fill with random value
    sp_matrix, nnz_count, row_max_nnz = random_spmatrix(n_row, n_col, per_nnz)
    nnz_per = (nnz_count / (n_row * n_col)) * 100
    avg_nnz = nnz_count / n_row
    print(str(nnz_count) +
          " non-zero elements in this sparse matrix (" + str(nnz_per) + "%).")
    print("Row average non-zero elements: " + str(avg_nnz) +
          ", row max non-zero elements: " + str(row_max_nnz))

    # convert sparse matrix to CSR format
    csr_rowptr, csr_colidx, csr_val = spmatrix_to_csr(sp_matrix)

    # convert CSR to Sliced ELLPACK format
    ell_colidx, ell_sliceptr, ell_val = csr_to_sellpack(
        csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # generate data
    slice_count = math.ceil(n_row / slice_height)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # performance test
    csr_y = numba_csr_spmv(n_row, csr_rowptr, csr_colidx, csr_val, x)
    # start test
    csr_start = time.perf_counter()
    for i in range(t):
        numba_csr_spmv(n_row, csr_rowptr, csr_colidx, csr_val, x)
    csr_end = time.perf_counter()
    print("CSR format runtime: ", (csr_end - csr_start) / t)

    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    # performance test
    ell_y = numba_sliced_ellpack_spmv(slice_count, ell_sliceptr,
                                      ell_colidx, ell_val, x, slice_height)
    # start test
    ell_start = time.perf_counter()
    for i in range(t):
        numba_sliced_ellpack_spmv(slice_count, ell_sliceptr,
                                  ell_colidx, ell_val, x, slice_height)
    ell_end = time.perf_counter()
    print("Sliced ELLPACK format runtime:", (ell_end - ell_start) / t)

    ell_rel_error = np.linalg.norm(
        ell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(ell_rel_error, 5)}.")

    # print(sliced_ellpack_spmv.parallel_diagnostics(level=4))
    # print(sliced_ellpack_spmv.inspect_asm()
    #       [list(sliced_ellpack_spmv.inspect_asm().keys())[0]])
    find_instr(numba_sliced_ellpack_spmv, keyword='add')


def numba_performance_benchmark(sp_matrix, slice_height, t):
    """
    Test Numba SpMV performance
    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    print("Sparse matrix row: ", n_row, ", column: ", n_col)

    # convert sparse matrix to CSR format
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    ell_colidx, ell_sliceptr, ell_val = csr_to_sellpack(
        csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # generate data
    slice_count = math.ceil(n_row / slice_height)

    # CSR test
    csr_y = numba_csr_spmv(n_row, csr_rowptr, csr_colidx, csr_val, x)
    # start test
    csr_perf_start = time.perf_counter()
    for i in range(t):
        numba_csr_spmv(n_row, csr_rowptr, csr_colidx, csr_val, x)
    csr_perf_end = time.perf_counter()
    print("CSR perf count: ", (csr_perf_end - csr_perf_start) / t)

    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    # Sliced ELLPACK test
    sell_y = numba_sliced_ellpack_spmv(slice_count, ell_sliceptr,
                                       ell_colidx, ell_val, x, slice_height)
    # start test
    ell_perf_start = time.perf_counter()
    for i in range(t):
        numba_sliced_ellpack_spmv(slice_count, ell_sliceptr,
                                  ell_colidx, ell_val, x, slice_height)
    ell_perf_end = time.perf_counter()
    print("SELL perf count: ", (ell_perf_end - ell_perf_start) / t)

    # check error
    ell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(ell_rel_error, 5)}.")

    # print(sliced_ellpack_spmv.parallel_diagnostics(level=4))
    # print(sliced_ellpack_spmv.inspect_asm()
    #       [list(sliced_ellpack_spmv.inspect_asm().keys())[0]])
    find_instr(numba_sliced_ellpack_spmv, keyword='mul')


def opencl_performance_benchmark(sp_matrix, slice_height, t):
    """
    Test OpenCL SpMV performance
    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    print("Sparse matrix row: ", n_row, ", column: ", n_col)

    # convert sparse matrix to CSR format
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    ell_colidx, ell_sliceptr, ell_val = csr_to_sellpack(
        csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr),
                      shape=(n_row, n_col), dtype=np.float32)
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # generate data
    slice_count = math.ceil(n_row / slice_height)

    # CSR test
    csr_spmv = CSRSpMV(n_row, np.array(csr_rowptr), np.array(csr_colidx),
                       np.array(csr_val, dtype=np.float32), x)
    csr_spmv.run()
    csr_y = csr_spmv.get_result()

    csr_time = csr_spmv.run(t)
    print("CSR perf time: ", csr_time / t)

    # Base Sliced ELLPACK test
    base_sell_spmv = BaseSELLSpMV(n_row, slice_count, ell_sliceptr,
                                  ell_colidx, ell_val, x, slice_height)
    base_sell_spmv.run()
    bsell_y = base_sell_spmv.get_result()

    sell_time = base_sell_spmv.run(t)
    print("Base SELL perf time: ", sell_time / t)

    # Sliced ELLPACK test
    sell_spmv = SELLSpMV(n_row, slice_count, ell_sliceptr,
                         ell_colidx, ell_val, x, slice_height)
    sell_spmv.run()
    sell_y = sell_spmv.get_result()
    # print(csr_y)
    # print(sell_y[:n_row])
    # print(bsell_y[:n_row])
    # print(y_exact)

    sell_time = sell_spmv.run(t)
    print("SELL perf time: ", sell_time / t)

    # check error
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    bsell_rel_error = np.linalg.norm(
        bsell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(bsell_rel_error, 5)}.")

    sell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(sell_rel_error, 5)}.")


def cuda_performance_benchmark(sp_matrix, slice_height, t):
    """
    Test OpenCL SpMV performance
    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    print("Sparse matrix row: ", n_row, ", column: ", n_col)

    # convert sparse matrix to CSR format
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    ell_colidx, ell_sliceptr, ell_val = csr_to_sellpack(
        csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr),
                      shape=(n_row, n_col), dtype=np.float32)
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # generate data
    slice_count = math.ceil(n_row / slice_height)

    # CSR test
    nblocks = (n_row,)  # global blocks
    nthreads = (1,)  # threads per block, better be a multiple of 32
    csr_y = np.zeros(n_row, dtype=np.float32)

    # CUDA buffer
    bf_csr_rowptr = cuda.to_device(csr_rowptr)
    bf_csr_colidx = cuda.to_device(csr_colidx)
    bf_csr_val = cuda.to_device(csr_val)
    bf_x = cuda.to_device(x)
    bf_csr_y = cuda.to_device(csr_y)

    cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
                                     bf_csr_val, bf_x, bf_csr_y)
    csr_perf_start = time.perf_counter()
    for i in range(t):
        cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
                                         bf_csr_val, bf_x, bf_csr_y)
    csr_perf_end = time.perf_counter()
    print("CSR perf count: ", (csr_perf_end - csr_perf_start) / t)
    output_csr_y = bf_csr_y.copy_to_host()

    # Sliced ELLPACK test
    nblocks = (slice_count,)  # global blocks
    nthreads = (slice_height,)  # threads per block, better be a multiple of 32
    sell_y = np.zeros(slice_height * slice_count, dtype=np.float32)

    # CUDA buffer
    bf_sell_y = cuda.to_device(sell_y)
    bf_ell_sliceptr = cuda.to_device(ell_sliceptr)
    bf_ell_colidx = cuda.to_device(ell_colidx)
    bf_ell_val = cuda.to_device(ell_val)
    bf_x = cuda.to_device(x)

    # first run
    cuda_sliced_ellpack_spmv[nblocks, nthreads](bf_ell_sliceptr,
                                                bf_ell_colidx,
                                                bf_ell_val, bf_x,
                                                slice_height,
                                                bf_sell_y)

    # calculate running time
    ell_perf_start = time.perf_counter()
    for i in range(t):
        cuda_sliced_ellpack_spmv[nblocks, nthreads](bf_ell_sliceptr,
                                                    bf_ell_colidx,
                                                    bf_ell_val, bf_x,
                                                    slice_height,
                                                    bf_sell_y)
    ell_perf_end = time.perf_counter()
    print("SELL perf count: ", (ell_perf_end - ell_perf_start) / t)
    output_sell_y = bf_sell_y.copy_to_host()

    print(output_csr_y)
    print(output_sell_y[:n_row])
    print(y_exact)

    # check error
    csr_rel_error = np.linalg.norm(
        output_csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    ell_rel_error = np.linalg.norm(
        output_sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(ell_rel_error, 5)}.")

