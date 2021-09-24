import numpy as np
import pytest
from scipy.sparse import csr_matrix

from FasterSpMV.matrix_tools import csr_to_sell, random_spmatrix, spmatrix_to_csr
from FasterSpMV.numba_spmv import numba_csr_spmv, numba_sell_spmv


def test_spmv():
    # define matrix parameters
    n_row = n_col = 10
    slice_height = 2

    # generate a sparse matrix fill with random value
    sp_matrix, nnz_count, row_max_nnz = random_spmatrix(n_row, n_col, 10)

    # convert sparse matrix to CSR format
    csr_rowptr, csr_colidx, csr_val =  spmatrix_to_csr(sp_matrix)

    # convert CSR to Sliced ELLPACK format
    slice_count, ell_colidx, ell_sliceptr, _, ell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # run CSR SpMV
    csr_y = numba_csr_spmv(n_row, csr_rowptr, csr_colidx, csr_val, x)

    # run Sliced ELLPACK SpMV
    sell_y = numba_sell_spmv(slice_count, ell_sliceptr,
                             ell_colidx, ell_val, x, slice_height)

    # check the result
    assert y_exact == pytest.approx(csr_y, rel=1e-6, abs=1e-12)
    assert y_exact == pytest.approx(sell_y, rel=1e-6, abs=1e-12)
