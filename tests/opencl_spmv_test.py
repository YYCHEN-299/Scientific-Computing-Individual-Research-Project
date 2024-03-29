import pytest
from scipy.sparse import csr_matrix

from FasterSpMV.matrix_tools import *
from FasterSpMV.opencl_spmv import *


def test_spmv():
    # define matrix parameters
    n_row = n_col = 10
    slice_height = 2

    # generate a sparse matrix fill with random value
    sp_matrix, nnz_count, row_max_nnz = random_spmatrix(n_row, n_col, 10)

    # convert sparse matrix to CSR format
    csr_rowptr, csr_colidx, csr_val =  spmatrix_to_csr(sp_matrix)

    # convert CSR to Sliced ELLPACK format
    slice_count, sell_colidx, sell_sliceptr, sell_slicecol, sell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # run CSR SpMV
    csr_spmv = OclCSRSpMV(n_row, np.array(csr_rowptr), np.array(csr_colidx),
                          np.array(csr_val, dtype=np.float32), 'CPU')
    csr_y = csr_spmv.run(x)

    # run Sliced ELLPACK SpMV
    sell_spmv = OclSELLSpMV(n_row, slice_count, sell_sliceptr, sell_slicecol,
                            sell_colidx, sell_val, slice_height, 'CPU')
    sell_y = sell_spmv.run(x)

    # check the result
    assert y_exact == pytest.approx(csr_y, rel=1e-6, abs=1e-12)
    assert y_exact == pytest.approx(sell_y, rel=1e-6, abs=1e-12)
