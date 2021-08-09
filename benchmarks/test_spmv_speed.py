import math
import time

import numpy as np

from code.debug_tools import find_instr
from code.matrix_tools import csr_to_sellpack
from code.matrix_tools import random_spmatrix
from code.matrix_tools import spmatrix_to_csr
from code.spmv_kernel import csr_spmv_multi_thread
from code.spmv_kernel import sliced_ellpack_spmv
from code.spmv_kernel import array_parallel_test


def random_data_test(n_row, n_col, per_nnz, slice_height, t):
    """
    Test SpMV performance
    """

    # generate a sparse matrix fill with random value
    sp_matrix, nnz_count, row_max_nnz = random_spmatrix(n_row, n_col, per_nnz)
    nnz_per = (nnz_count / (n_row * n_col)) * 100
    avg_nnz = nnz_count / n_row
    print(str(nnz_count) +
          " non-zero elements in this sparse matrix (" + str(nnz_per) + "%).")
    print("Row average non-zero elements: " + str(avg_nnz) +
          ", row max non-zero elements: " + str(row_max_nnz))
    # print(sp_matrix)
    # convert sparse matrix to CSR format
    print("Convert sparse matrix to CSR format...")
    csr_rowptr, csr_colidx, csr_val = spmatrix_to_csr(sp_matrix)
    # print(csr_rowptr)
    # print(csr_colidx)
    # print(csr_val)
    # convert CSR to Sliced ELLPACK format
    print("Convert CSR to Sliced ELLPACK format...")
    ell_colidx, ell_sliceptr, ell_val = csr_to_sellpack(
        csr_rowptr, csr_colidx, csr_val, slice_height)
    # print(ell_colidx)
    # print(ell_sliceptr)
    # print(ell_val)
    # generate x array
    x = np.ones(n_col, dtype='float32')
    x *= 1.23
    # generate data
    csr_y = np.zeros(n_row, dtype='float32')
    slice_count = math.ceil(n_row / slice_height)

    # performance test
    print("Start performance test...")
    csr_y = csr_spmv_multi_thread(csr_y, n_row, csr_rowptr,
                                  csr_colidx, csr_val, x)
    # start test
    csr_start = time.time()
    for i in range(t):
        csr_spmv_multi_thread(csr_y, n_row, csr_rowptr,
                              csr_colidx, csr_val, x)
    csr_end = time.time()
    print("CSR format runtime: ", (csr_end - csr_start) / t)
    # performance test
    ell_y = sliced_ellpack_spmv(slice_count, ell_sliceptr,
                                ell_colidx, ell_val, x, slice_height)
    # start test
    ell_start = time.time()
    for i in range(t):
        sliced_ellpack_spmv(slice_count, ell_sliceptr,
                            ell_colidx, ell_val, x, slice_height)
    ell_end = time.time()
    print("Sliced ELLPACK format runtime: ", (ell_end - ell_start) / t)
    # print(csr_y)
    # print(ell_y)
    check_number = sum(csr_y - ell_y[:n_row])
    print("Check number = ", check_number, "(should be 0 if result correct)")
    # print(sliced_ellpack_spmv.parallel_diagnostics(level=4))
    # print(sliced_ellpack_spmv.inspect_asm()
    #       [list(sliced_ellpack_spmv.inspect_asm().keys())[0]])
    find_instr(sliced_ellpack_spmv, keyword='add')


def data_set_test(sp_matrix, slice_height, t):
    """
    Test SpMV performance
    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    # print("Sparse matrix row: ", n_row, ", column: ", n_col)

    # convert sparse matrix to CSR format
    # print("Convert sparse matrix to CSR format...")
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data

    # convert CSR to Sliced ELLPACK format
    # print("Convert CSR to Sliced ELLPACK format...")
    ell_colidx, ell_sliceptr, ell_val = csr_to_sellpack(
        csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate x array
    x = np.ones(n_col, dtype='float32')
    x *= 1.23
    # generate data
    csr_y = np.zeros(n_row, dtype='float32')
    slice_count = math.ceil(n_row / slice_height)

    # performance test
    # print("Start performance test...")
    csr_y = csr_spmv_multi_thread(csr_y, n_row, csr_rowptr,
                                  csr_colidx, csr_val, x)
    # start test
    csr_start = time.time()
    for i in range(t):
        csr_spmv_multi_thread(csr_y, n_row, csr_rowptr,
                              csr_colidx, csr_val, x)
    csr_end = time.time()
    print("CSR format runtime: ", (csr_end - csr_start) / t)
    # performance test
    ell_y = sliced_ellpack_spmv(slice_count, ell_sliceptr,
                                ell_colidx, ell_val, x, slice_height)
    # start test
    ell_start = time.time()
    for i in range(t):
        sliced_ellpack_spmv(slice_count, ell_sliceptr,
                            ell_colidx, ell_val, x, slice_height)
    ell_end = time.time()
    print("Sliced ELLPACK format runtime: ", (ell_end - ell_start) / t, "(slice:", slice_height, ")")
    check_number = sum(csr_y - ell_y[:n_row])
    # print("Check number = ", check_number, "(should be 0 if result correct)")
    # print(sliced_ellpack_spmv.parallel_diagnostics(level=4))
    # print(sliced_ellpack_spmv.inspect_asm()
    #       [list(sliced_ellpack_spmv.inspect_asm().keys())[0]])
    find_instr(sliced_ellpack_spmv, keyword='mul')


def avx_test():
    a = np.random.rand(10)
    b = np.random.rand(10)
    x32 = np.linspace(1, 2, 10000, dtype='float32')
    y32 = np.linspace(1, 2, 10000, dtype='float32')
    array_parallel_test(x32, y32)

    print(array_parallel_test.inspect_asm()
          [list(array_parallel_test.inspect_asm().keys())[0]])
    find_instr(array_parallel_test, keyword='mul')
