import math
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from numpy.testing import assert_allclose

from FasterSpMV.matrix_tools import csr_to_sellpack
from FasterSpMV.matrix_tools import random_spmatrix
from FasterSpMV.matrix_tools import spmatrix_to_csr
from FasterSpMV.opencl_spmv import BaseSELLSpMV
from FasterSpMV.opencl_spmv import CSRSpMV


def test_spmv():
    # define matrix parameters
    n_row = n_col = 10
    slice_height = 2

    # generate a sparse matrix fill with random value
    sp_matrix, nnz_count, row_max_nnz = random_spmatrix(n_row, n_col, 10)

    # convert sparse matrix to CSR format
    csr_rowptr, csr_colidx, csr_val =  spmatrix_to_csr(sp_matrix)

    # convert CSR to Sliced ELLPACK format
    ell_colidx, ell_sliceptr, ell_slicecol, ell_val = \
        csr_to_sellpack(csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # generate data
    slice_count = math.ceil(n_row / slice_height)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # run CSR SpMV
    csr_spmv = CSRSpMV(n_row, np.array(csr_rowptr), np.array(csr_colidx),
                       np.array(csr_val, dtype=np.float32), x)
    csr_spmv.run()
    csr_y = csr_spmv.get_result()

    # run Sliced ELLPACK SpMV
    sell_spmv = BaseSELLSpMV(n_row, slice_count, ell_sliceptr,  ell_slicecol,
                             ell_colidx, ell_val, x, slice_height)
    sell_spmv.run()
    sell_y = sell_spmv.get_result()

    # check the result
    assert y_exact == pytest.approx(csr_y, rel=1e-6, abs=1e-12)
    assert y_exact == pytest.approx(sell_y, rel=1e-6, abs=1e-12)
