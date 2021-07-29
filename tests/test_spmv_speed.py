import time

import numpy as np

from code.matrix_tools import random_spmatrix
from code.matrix_tools import spmatrix_to_CSR
from code.matrix_tools import CSR_to_SELLPACK

from code.spmv_kernel import csr_spmv_multi_thread
from code.spmv_kernel import sliced_ellpack_spmv_multi_thread


def speed_test(n_row, n_col, per_nnz, slice_height):
    """
    Test SpMV performance.
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
    csr_rowptr, csr_colidx, csr_val = spmatrix_to_CSR(sp_matrix)

    # convert CSR to Sliced ELLPACK
    ell_colidx, ell_sliceptr, ell_val = CSR_to_SELLPACK(
        csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate x array
    x = np.ones(n_col, dtype=np.float32)
    x *= 1.23
    # generate data
    csr_y = np.zeros(n_row, dtype=np.float32)
    ell_y = np.zeros(n_row, dtype=np.float32)
    slice_count = int(n_row / slice_height)

    # speed test
    csr_start = time.time()
    for i in range(100):
        csr_spmv_multi_thread(csr_y, n_row, csr_rowptr,
                              csr_colidx, csr_val, x)
    csr_end = time.time()
    print("CSR format runtime: ", (csr_end - csr_start) / 100)
    # speed test
    ell_start = time.time()
    for i in range(100):
        sliced_ellpack_spmv_multi_thread(
            ell_y, slice_count, ell_sliceptr,
            ell_colidx, ell_val, x, slice_height)
    ell_end = time.time()
    print("Sliced ELLPACK format runtime: ", (ell_end - ell_start) / 100)