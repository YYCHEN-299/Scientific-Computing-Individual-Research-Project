import pytest
from scipy.sparse import csr_matrix

from FasterSpMV.matrix_tools import *
from FasterSpMV.numba_spmv import *


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

    csr_y = np.empty(n_row, dtype=np.float32)
    # run CSR SpMV
    numba_csr_spmv(csr_y, n_row, csr_rowptr, csr_colidx, csr_val, x)

    sell_y = np.empty(slice_count * slice_height, dtype=np.float32)
    # run Sliced ELLPACK SpMV
    numba_sell_spmv(sell_y, slice_count, ell_sliceptr,
                    ell_colidx, ell_val, x, slice_height)
    sell_y = sell_y[:n_row]

    # check the result
    assert y_exact == pytest.approx(csr_y, rel=1e-6, abs=1e-12)
    assert y_exact == pytest.approx(sell_y, rel=1e-6, abs=1e-12)
