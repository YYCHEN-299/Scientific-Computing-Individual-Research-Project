import pytest
from scipy.sparse import csr_matrix

from FasterSpMV.cuda_spmv import *
from FasterSpMV.matrix_tools import *


def test_spmv():
    # define matrix parameters
    n_row = n_col = 10
    slice_height = 2

    # generate a sparse matrix fill with random value
    sp_matrix, nnz_count, row_max_nnz = random_spmatrix(n_row, n_col, 10)

    # convert sparse matrix to CSR format
    csr_rowptr, csr_colidx, csr_val =  spmatrix_to_csr(sp_matrix)

    # convert CSR to Sliced ELLPACK format
    slice_count, sell_colidx, sell_sliceptr, _, sell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # CSR test
    nblocks = (n_row,)  # global blocks
    nthreads = (1,)  # threads per block, better be a multiple of 32

    # CUDA buffer
    bf_csr_rowptr = cuda.to_device(csr_rowptr)
    bf_csr_colidx = cuda.to_device(csr_colidx)
    bf_csr_val = cuda.to_device(csr_val)
    bf_x = cuda.to_device(x)
    bf_csr_y = cuda.device_array(n_row, dtype=np.float32)

    cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
                                     bf_csr_val, bf_x, bf_csr_y)
    csr_y = bf_csr_y.copy_to_host()

    # Sliced ELLPACK test
    nblocks = (slice_count,)  # global blocks
    nthreads = (slice_height,)  # threads per block, better be a multiple of 32

    # CUDA buffer
    bf_sell_y = cuda.device_array(slice_height * slice_count, dtype=np.float32)
    bf_sell_sliceptr = cuda.to_device(sell_sliceptr)
    bf_sell_colidx = cuda.to_device(sell_colidx)
    bf_sell_val = cuda.to_device(sell_val)
    bf_x = cuda.to_device(x)

    cuda_sell_spmv[nblocks, nthreads](bf_sell_sliceptr,
                                      bf_sell_colidx,
                                      bf_sell_val, bf_x,
                                      slice_height,
                                      bf_sell_y)
    sell_y = bf_sell_y.copy_to_host()

    # check the result
    assert y_exact == pytest.approx(csr_y, rel=1e-6, abs=1e-12)
    assert y_exact == pytest.approx(sell_y, rel=1e-6, abs=1e-12)
