import time

from timeit import repeat
from numba import cuda
from scipy.sparse import csr_matrix

from FasterSpMV.benchmark_tools import find_instr
from FasterSpMV.matrix_tools import *
from FasterSpMV.numba_spmv import *
from FasterSpMV.opencl_spmv import OclSELLSpMV, OclCSRSpMV, OclSELL4SpMV, OclSELL8SpMV, OclSELLRdSpMV
from FasterSpMV.cuda_spmv import cuda_csr_spmv, cuda_sell_spmv, cuda_rd_sell_spmv
from FasterSpMV.spmv import SpMVOperator


def numba_random_data_test(n_row, n_col, per_nnz, slice_height, t):
    """

    Parameters
    ----------
    n_row
    n_col
    per_nnz
    slice_height
    t

    Returns
    -------

    """

    # generate a sparse matrix fill with random value
    sp_matrix, nnz_count, row_max_nnz = random_spmatrix(n_row, n_col, per_nnz)
    nnz_per = (nnz_count / (n_row * n_col)) * 100
    avg_nnz = nnz_count / n_row
    print(str(nnz_count) +
          " non-zero elements in this sparse matrix (" + str(nnz_per) + "%).")
    print("Row average non-zero elements:" + str(avg_nnz) +
          ", row max non-zero elements:" + str(row_max_nnz))

    # convert sparse matrix to CSR format
    csr_rowptr, csr_colidx, csr_val = spmatrix_to_csr(sp_matrix)

    # convert CSR to Sliced ELLPACK format
    slice_count, ell_colidx, ell_sliceptr, ell_slicecol, ell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # performance test
    y = np.zeros(n_row, dtype=np.float32)
    csr_y = numba_csr_spmv(y, n_row, csr_rowptr, csr_colidx, csr_val, x)
    # start test
    csr_start = time.perf_counter()
    for _ in range(t):
        numba_csr_spmv(y, n_row, csr_rowptr, csr_colidx, csr_val, x)
    csr_end = time.perf_counter()
    print("CSR format runtime:", (csr_end - csr_start) / t)

    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    # performance test
    y = np.zeros(slice_count * slice_height, dtype=np.float32)
    ell_y = numba_sell_spmv(y, slice_count, ell_sliceptr,
                            ell_colidx, ell_val, x, slice_height)
    # start test
    ell_start = time.perf_counter()
    for _ in range(t):
        numba_sell_spmv(y, slice_count, ell_sliceptr,
                        ell_colidx, ell_val, x, slice_height)
    ell_end = time.perf_counter()
    print("Sliced ELLPACK format runtime:", (ell_end - ell_start) / t)

    ell_rel_error = np.linalg.norm(
        ell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(ell_rel_error, 5)}.")

    # print(sliced_ellpack_spmv.parallel_diagnostics(level=4))
    # print(sliced_ellpack_spmv.inspect_asm()
    #       [list(sliced_ellpack_spmv.inspect_asm().keys())[0]])
    find_instr(numba_sell_spmv, key='add')


def numba_performance_benchmark(sp_matrix, slice_height, t):
    """

    Parameters
    ----------
    sp_matrix
    slice_height
    t

    Returns
    -------

    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # convert sparse matrix to CSR format
    csr_rowptr = sp_matrix.indptr.astype(np.uint32)
    csr_colidx = sp_matrix.indices.astype(np.uint32)
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    slice_count, ell_colidx, ell_sliceptr, ell_slicecol, ell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # CSR test
    n_row = np.uint32(n_row)
    csr_y = np.zeros(n_row, dtype=np.float32)
    numba_csr_spmv(csr_y, n_row, csr_rowptr, csr_colidx, csr_val, x)
    # check mem
    # smem = proc.memory_info().rss
    # start test
    csr_perf_start = time.perf_counter()
    for _ in range(t):
        numba_csr_spmv(csr_y, n_row, csr_rowptr, csr_colidx, csr_val, x)
    csr_perf_end = time.perf_counter()
    print("CSR perf count:", (csr_perf_end - csr_perf_start) / t)

    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    # Sliced ELLPACK test
    slice_height = np.uint32(slice_height)
    sell_y = np.empty((slice_count, slice_height), dtype=np.float32)
    numba_sell_spmv(sell_y, slice_count, ell_sliceptr,
                    ell_colidx, ell_val, x, slice_height)
    # start test
    ell_perf_start = time.perf_counter()
    for _ in range(t):
        numba_sell_spmv(sell_y, slice_count, ell_sliceptr,
                        ell_colidx, ell_val, x, slice_height)
    ell_perf_end = time.perf_counter()
    print("SELL perf count:", (ell_perf_end - ell_perf_start) / t)
    sell_y = sell_y.reshape((slice_count * slice_height))

    # check error
    ell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(ell_rel_error, 5)}.")

    # print(sliced_ellpack_spmv.parallel_diagnostics(level=4))
    # print(sliced_ellpack_spmv.inspect_asm()
    #       [list(sliced_ellpack_spmv.inspect_asm().keys())[0]])
    find_instr(numba_sell_spmv, key='mul')
    print('---')
    find_instr(numba_csr_spmv, key='mul')
    print('---')
    print(numba_sell_spmv.signatures)
    print('---')
    print(numba_csr_spmv.signatures)


def opencl_performance_benchmark(sp_matrix, slice_height, t):
    """

    Parameters
    ----------
    sp_matrix
    slice_height
    t

    Returns
    -------

    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # convert sparse matrix to CSR format
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    slice_count, ell_colidx, ell_sliceptr, ell_slicecol, ell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr),
                      shape=(n_row, n_col), dtype=np.float32)
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # CSR test
    csr_spmv = OclCSRSpMV(n_row, np.array(csr_rowptr), np.array(csr_colidx),
                          np.array(csr_val, dtype=np.float32))
    csr_y = csr_spmv.run(x)

    csr_perf_start = time.perf_counter()
    for _ in range(t):
        csr_spmv.run(x)
    csr_perf_end = time.perf_counter()
    print("CSR perf time: ", (csr_perf_end - csr_perf_start) / t)

    # Base Sliced ELLPACK test
    base_sell_spmv = OclSELLSpMV(n_row, slice_count,
                                 ell_sliceptr, ell_slicecol,
                                 ell_colidx, ell_val, slice_height)
    bsell_y = base_sell_spmv.run(x)
    bsell_perf_start = time.perf_counter()
    for _ in range(t):
        base_sell_spmv.run(x)
    bsell_perf_end = time.perf_counter()
    print("Base SELL perf time:", (bsell_perf_end - bsell_perf_start) / t)

    # check error
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    bsell_rel_error = np.linalg.norm(
        bsell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(bsell_rel_error, 5)}.")


def cuda_performance_benchmark(sp_matrix, slice_height, t):
    """

    Parameters
    ----------
    sp_matrix
    slice_height
    t

    Returns
    -------

    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # convert sparse matrix to CSR format
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    slice_count, ell_colidx, ell_sliceptr, ell_slicecol, ell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr),
                      shape=(n_row, n_col), dtype=np.float32)
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # CSR test
    nblocks = (n_row,)  # global blocks
    nthreads = (1,)  # threads per block, better be a multiple of 32
    csr_y = np.zeros(n_row, dtype=np.float32)

    # CUDA buffer
    bf_csr_rowptr = cuda.to_device(csr_rowptr)
    bf_csr_colidx = cuda.to_device(csr_colidx)
    bf_csr_val = cuda.to_device(csr_val)
    bf_x = cuda.to_device(x)
    bf_csr_y = cuda.device_array(n_row, dtype=np.float32)

    cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
                                     bf_csr_val, bf_x, bf_csr_y)
    csr_perf_start = time.perf_counter()
    for _ in range(t):
        bf_x = cuda.to_device(x)
        cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
                                         bf_csr_val, bf_x, bf_csr_y)
        output_csr_y = bf_csr_y.copy_to_host()
    csr_perf_end = time.perf_counter()
    print("CSR perf count:", (csr_perf_end - csr_perf_start) / t)
    output_csr_y = bf_csr_y.copy_to_host()

    # Sliced ELLPACK test
    nblocks = (slice_count,)  # global blocks
    nthreads = (slice_height,)  # threads per block, better be a multiple of 32
    # sell_y = np.zeros(slice_height * slice_count, dtype=np.float32)

    # CUDA buffer
    bf_sell_y = cuda.device_array(slice_height * slice_count, dtype=np.float32)
    bf_ell_sliceptr = cuda.to_device(ell_sliceptr)
    bf_ell_colidx = cuda.to_device(ell_colidx)
    bf_ell_val = cuda.to_device(ell_val)
    bf_x = cuda.to_device(x)

    # first run
    cuda_sell_spmv[nblocks, nthreads](bf_ell_sliceptr,
                                      bf_ell_colidx,
                                      bf_ell_val, bf_x,
                                      slice_height,
                                      bf_sell_y)

    # calculate running time
    ell_perf_start = time.perf_counter()
    for _ in range(t):
        bf_x = cuda.to_device(x)
        cuda_sell_spmv[nblocks, nthreads](bf_ell_sliceptr,
                                          bf_ell_colidx,
                                          bf_ell_val, bf_x,
                                          slice_height,
                                          bf_sell_y)
        output_sell_y = bf_sell_y.copy_to_host()
    ell_perf_end = time.perf_counter()
    print("SELL perf count:", (ell_perf_end - ell_perf_start) / t)
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


def test_operator_class(sp_matrix, slice_height, t):
    """

    Parameters
    ----------
    sp_matrix
    slice_height
    t

    Returns
    -------

    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # convert sparse matrix to CSR format
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr),
                      shape=(n_row, n_col), dtype=np.float32)
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # CSR test
    csrspmvop = SpMVOperator('cuda', 'csr', n_row, n_col,
                             csr_rowptr, csr_colidx, csr_val, slice_height)
    perf_start = time.perf_counter()
    output_csr_y = csrspmvop.run(x)
    perf_end = time.perf_counter()
    print("CSR perf count:", (perf_end - perf_start))

    # Sliced ELLPACK test
    sellspmvop = SpMVOperator('cuda', 'sell', n_row, n_col,
                              csr_rowptr, csr_colidx, csr_val, slice_height)
    perf_start = time.perf_counter()
    output_sell_y = sellspmvop.run(x)
    perf_end = time.perf_counter()
    print("SELL perf count:", (perf_end - perf_start))

    # check error
    csr_rel_error = np.linalg.norm(
        output_csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    ell_rel_error = np.linalg.norm(
        output_sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(ell_rel_error, 5)}.")


def test_numba_explicit_parallel(sp_matrix):
    """

    Parameters
    ----------
    sp_matrix

    Returns
    -------

    """

    slice_height = 4
    t = 100
    n_row, n_col = sp_matrix.shape
    print("Sparse matrix row:", n_row, ", column:", n_col)

    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    slice_count, ell_colidx, ell_sliceptr, ell_slicecol, ell_val = \
        csr_to_ocl_sell4(n_row, csr_rowptr, csr_colidx, csr_val)

    # convert CSR to Sliced ELLPACK format
    slice_count1, ell_colidx1, ell_sliceptr1, ell_slicecol1, ell_val1 = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # performance test
    csr_y = np.empty(n_row, dtype=np.float32)
    numba_csr_spmv(csr_y, n_row, csr_rowptr, csr_colidx, csr_val, x)
    # start test
    csr_start = time.perf_counter()
    for _ in range(t):
        numba_csr_spmv(csr_y, n_row, csr_rowptr, csr_colidx, csr_val, x)
    csr_end = time.perf_counter()
    print("CSR format runtime:", (csr_end - csr_start) / t)

    print(min(repeat(
        lambda: numba_csr_spmv(
            csr_y, n_row, csr_rowptr, csr_colidx, csr_val, x),
        number=50, repeat=5)))

    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    # performance test
    slice_height = np.uint64(4)
    ell_y = np.empty((slice_count, 4), dtype=np.float32)
    numba_sell4_spmv(ell_y, slice_count,
                     ell_slicecol, ell_colidx, ell_val, x)
    # start test
    ell_start = time.perf_counter()
    for _ in range(t):
        numba_sell4_spmv(ell_y, slice_count,
                         ell_slicecol, ell_colidx, ell_val, x)
    ell_end = time.perf_counter()
    print("Sliced ELLPACK format runtime:", (ell_end - ell_start) / t)

    print(min(repeat(
        lambda: numba_sell4_spmv(
            ell_y, slice_count, ell_slicecol, ell_colidx, ell_val, x),
        number=50, repeat=5)))

    ell_y = ell_y.reshape(-1, )
    ell_rel_error = np.linalg.norm(
        ell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(ell_rel_error, 5)}.")

    # print(sliced_ellpack_spmv.parallel_diagnostics(level=4))
    # print(sliced_ellpack_spmv.inspect_asm()
    #       [list(sliced_ellpack_spmv.inspect_asm().keys())[0]])
    find_instr(numba_sell4_spmv, key='mul')


def test_2d_cuda(sp_matrix, slice_height, t):
    """

    Parameters
    ----------
    sp_matrix
    slice_height
    t

    Returns
    -------

    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # convert sparse matrix to CSR format
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    slice_count, ell_colidx, ell_sliceptr, ell_slicecol, ell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # slice_count1, ell_colidx1, ell_sliceptr1, ell_slicecol1, ell_val1 = \
    #     csr_to_2d_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr),
                      shape=(n_row, n_col), dtype=np.float32)
    y_exact = sp_A.dot(x)  # SciPy SpMV

    # Sliced ELLPACK test
    nblocks = (slice_count,)  # global blocks
    nthreads = (slice_height,)  # threads per block, better be a multiple of 32
    # sell_y = np.zeros(slice_height * slice_count, dtype=np.float32)

    # CUDA buffer
    bf_sell_y = cuda.device_array(slice_height * slice_count, dtype=np.float32)
    bf_ell_sliceptr = cuda.to_device(ell_sliceptr)
    bf_ell_colidx = cuda.to_device(ell_colidx)
    bf_ell_val = cuda.to_device(ell_val)
    bf_x = cuda.to_device(x)

    # first run
    cuda_sell_spmv[nblocks, nthreads](bf_ell_sliceptr,
                                      bf_ell_colidx,
                                      bf_ell_val, bf_x,
                                      np.int32(slice_height),
                                      bf_sell_y)

    # calculate running time
    ell_perf_start = time.perf_counter()
    #ell_cuda_start = cuda.event.record()
    for _ in range(t):
        bf_x = cuda.to_device(x)
        cuda_sell_spmv[nblocks, nthreads](bf_ell_sliceptr,
                                          bf_ell_colidx,
                                          bf_ell_val, bf_x,
                                          np.int32(slice_height),
                                          bf_sell_y)
        output_sell_y = bf_sell_y.copy_to_host()
    ell_perf_end = time.perf_counter()
    #ell_cuda_end = cuda.event.record()
    print("SELL perf count:", (ell_perf_end - ell_perf_start) / t)
    #print("SELL CUDA count:",
    #      cuda.event_elapsed_time(ell_cuda_end, ell_cuda_start) / t)
    output_sell_y = bf_sell_y.copy_to_host()

    def sell_time_t():
        # bf_x = cuda.to_device(x)
        cuda_sell_spmv[nblocks, nthreads](bf_ell_sliceptr,
                                          bf_ell_colidx,
                                          bf_ell_val, bf_x,
                                          np.int32(slice_height),
                                          bf_sell_y)
        output_sell_y = bf_sell_y.copy_to_host()

    print(min(repeat(lambda: sell_time_t(), number=50, repeat=5)))

    # rd Sliced ELLPACK test
    row_th = 4
    ell_val2, ell_colidx2, ell_slicecol2, ell_sliceptr2 = \
        csr_to_align_sell(sp_A, row_th, slice_height)

    nblocks = (slice_count,)  # global blocks
    nthreads = (slice_height * row_th,)  # threads per block, better be a multiple of 32
    # sell_y = np.zeros(slice_height * slice_count, dtype=np.float32)

    # CUDA buffer
    bf_rdsell_y = cuda.device_array(slice_height * slice_count, dtype=np.float32)
    bf_rdsell_sliceptr = cuda.to_device(ell_sliceptr2)
    bf_rdsell_colidx = cuda.to_device(ell_colidx2)
    bf_rdsell_slicecol = cuda.to_device(ell_slicecol2)
    bf_rdsell_val = cuda.to_device(ell_val2)
    bf_x = cuda.to_device(x)

    # first run
    align = int(row_th * slice_height)
    cuda_rd_sell_spmv[nblocks, nthreads, None, align](bf_rdsell_sliceptr,
                                                      bf_rdsell_slicecol,
                                                      bf_rdsell_colidx,
                                                      bf_rdsell_val, bf_x,
                                                      np.int32(slice_height),
                                                      np.int32(row_th),
                                                      bf_rdsell_y)

    # calculate running time
    rdell_perf_start = time.perf_counter()
    #rdell_cuda_start = cuda.event.record()
    for _ in range(t):
        bf_x = cuda.to_device(x)
        cuda_rd_sell_spmv[nblocks, nthreads, None, align](bf_rdsell_sliceptr,
                                                          bf_rdsell_slicecol,
                                                          bf_rdsell_colidx,
                                                          bf_rdsell_val, bf_x,
                                                          np.int32(slice_height),
                                                          np.int32(row_th),
                                                          bf_rdsell_y)
        output_rdsell_y = bf_rdsell_y.copy_to_host()
    rdell_perf_end = time.perf_counter()
    #rdell_cuda_end = cuda.event.record()
    print("rd SELL perf count:", (rdell_perf_end - rdell_perf_start) / t)
    #print("rd SELL CUDA count:",
    #      cuda.event_elapsed_time(rdell_cuda_end, rdell_cuda_start) / t)
    output_rdsell_y = bf_rdsell_y.copy_to_host()

    def sell_time_t():
        # bf_x = cuda.to_device(x)
        cuda_rd_sell_spmv[nblocks, nthreads, None, align](bf_rdsell_sliceptr,
                                                          bf_rdsell_slicecol,
                                                          bf_rdsell_colidx,
                                                          bf_rdsell_val, bf_x,
                                                          np.int32(slice_height),
                                                          np.int32(row_th),
                                                          bf_rdsell_y)
        output_rdsell_y = bf_rdsell_y.copy_to_host()

    print(min(repeat(lambda: sell_time_t(), number=50, repeat=5)))

    # # 2D Sliced ELLPACK test
    # nblocks = (slice_count1,)  # global blocks
    # nthreads = (slice_height,)  # threads per block, better be a multiple of 32
    #
    # # CUDA buffer
    # bf_2dsell_y = cuda.device_array((slice_count1, slice_height),
    #                                 dtype=np.float32)
    # bf_2dell_slicecol = cuda.to_device(ell_slicecol1)
    # bf_2dell_colidx = cuda.to_device(ell_colidx1)
    # bf_2dell_val = cuda.to_device(ell_val1)
    # bf_x = cuda.to_device(x)
    #
    # # first run
    # cuda_2d_sell_spmv[nblocks, nthreads](bf_2dell_slicecol,
    #                                      bf_2dell_colidx,
    #                                      bf_2dell_val, bf_x,
    #                                      bf_2dsell_y)
    #
    # # calculate running time
    # sell2_perf_start = time.perf_counter()
    # sell2_cuda_start = cuda.event.record()
    # for _ in range(t):
    #     bf_x = cuda.to_device(x)
    #     cuda_2d_sell_spmv[nblocks, nthreads](bf_2dell_slicecol,
    #                                          bf_2dell_colidx,
    #                                          bf_2dell_val, bf_x,
    #                                          bf_2dsell_y)
    #     output_2dsell_y = bf_2dsell_y.copy_to_host()
    # sell2_perf_end = time.perf_counter()
    # sell2_cuda_end = cuda.event.record()
    # print("2d SELL perf count: ", (sell2_perf_end - sell2_perf_start) / t)
    # print("2d SELL CUDA count:",
    #       cuda.event_elapsed_time(sell2_cuda_start, sell2_cuda_end) / t)
    # output_2dsell_y = bf_2dsell_y.copy_to_host()
    #
    # def dsell_time_t():
    #     # bf_x = cuda.to_device(x)
    #     cuda_2d_sell_spmv[nblocks, nthreads](bf_2dell_slicecol,
    #                                          bf_2dell_colidx,
    #                                          bf_2dell_val, bf_x,
    #                                          bf_2dsell_y)
    #     output_2dsell_y = bf_2dsell_y.copy_to_host()
    #
    # print(min(repeat(lambda: dsell_time_t(), number=50, repeat=5)))

    # CSR test
    nblocks = (n_row,)  # global blocks
    nthreads = (1,)  # threads per block, better be a multiple of 32
    csr_y = np.zeros(n_row, dtype=np.float32)

    # CUDA buffer
    csr_rowptr = csr_rowptr.astype(np.uint32)
    csr_colidx = csr_colidx.astype(np.uint32)
    bf_csr_rowptr = cuda.to_device(csr_rowptr)
    bf_csr_colidx = cuda.to_device(csr_colidx)
    bf_csr_val = cuda.to_device(csr_val)
    bf_x = cuda.to_device(x)
    bf_csr_y = cuda.device_array(n_row, dtype=np.float32)

    cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
                                     bf_csr_val, bf_x, bf_csr_y)
    csr_perf_start = time.perf_counter()
    #csr_cuda_start = cuda.event.record()
    for _ in range(t):
        bf_x = cuda.to_device(x)
        cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
                                         bf_csr_val, bf_x, bf_csr_y)
        output_csr_y = bf_csr_y.copy_to_host()
    csr_perf_end = time.perf_counter()
    #csr_cuda_end = cuda.event.record()
    print("CSR perf count:", (csr_perf_end - csr_perf_start) / t)
    #print("CSR CUDA count:",
    #      cuda.event_elapsed_time(csr_cuda_start, csr_cuda_end) / t)
    output_csr_y = bf_csr_y.copy_to_host()

    def csr_time_t():
        # bf_x = cuda.to_device(x)
        cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
                                         bf_csr_val, bf_x, bf_csr_y)
        output_csr_y = bf_csr_y.copy_to_host()

    print(min(repeat(lambda: csr_time_t(), number=50, repeat=5)))

    # check error
    csr_rel_error = np.linalg.norm(
        output_csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    ell_rel_error = np.linalg.norm(
        output_sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(ell_rel_error, 5)}.")

    ell_rel_error = np.linalg.norm(
        output_rdsell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"rd SELL Error: {round(ell_rel_error, 5)}.")

    # output_2dsell_y = output_2dsell_y.reshape(-1, )
    # dell_rel_error = np.linalg.norm(
    #     output_2dsell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    # print(f"SELL Error: {round(dell_rel_error, 5)}.")


def opencl_exp_performance_benchmark(sp_matrix, slice_height, t):
    """

    Parameters
    ----------
    sp_matrix
    slice_height
    t

    Returns
    -------

    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # convert sparse matrix to CSR format
    csr_rowptr = sp_matrix.indptr
    csr_colidx = sp_matrix.indices
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    slice_count, ell_colidx, ell_sliceptr, ell_slicecol, ell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # convert CSR to Sliced ELLPACK format
    slice_count1, ell_colidx1, ell_sliceptr1, ell_slicecol1, ell_val1 = \
        csr_to_ocl_sell4(n_row, csr_rowptr, csr_colidx, csr_val)

    # generate a random vector
    rand = np.random.RandomState(0)
    x = rand.randn(n_col).astype(np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr),
                      shape=(n_row, n_col), dtype=np.float32)
    y_exact = sp_A.dot(x)  # SciPy SpMV

    row_th = 4
    align = row_th * slice_height
    ell_val2, ell_colidx2, ell_slicecol2, ell_sliceptr2 = \
        csr_to_align_sell(sp_A, row_th, slice_height, align)

    # Sliced ELLPACK test
    sell_spmv = OclSELLSpMV(n_row, slice_count, ell_sliceptr,
                            ell_slicecol, ell_colidx, ell_val, slice_height)
    sell_y = sell_spmv.run(x)
    sell_perf_start = time.perf_counter()
    for _ in range(t):
        sell_spmv.run(x)
    sell_perf_end = time.perf_counter()
    print("Base SELL perf time:", (sell_perf_end - sell_perf_start) / t)
    print(min(repeat(lambda: sell_spmv.run(x), number=50, repeat=5)))

    # Slice ELLPACK 4 test
    sell4_spmv = OclSELL4SpMV(n_row, slice_count1,
                              ell_slicecol1, ell_colidx1, ell_val1)
    sell4_y = sell4_spmv.run(x)
    sell4_perf_start = time.perf_counter()
    for _ in range(t):
        sell4_spmv.run(x)
    sell4_perf_end = time.perf_counter()
    print("SELL 4 perf time:", (sell4_perf_end - sell4_perf_start) / t)
    print(min(repeat(lambda: sell4_spmv.run(x), number=50, repeat=5)))

    # Slice ELLPACK rd test
    sellrd_spmv = OclSELLRdSpMV(n_row, slice_count, ell_sliceptr2,
                                ell_slicecol2, ell_colidx2,
                                ell_val2, slice_height, row_th)
    sellrd_y = sellrd_spmv.run(x)
    sellrd_perf_start = time.perf_counter()
    for _ in range(t):
        sellrd_spmv.run(x)
    sellrd_perf_end = time.perf_counter()
    print("SELL rd perf time:", (sellrd_perf_end - sellrd_perf_start) / t)
    print(min(repeat(lambda: sellrd_spmv.run(x), number=50, repeat=5)))

    # CSR test
    csr_spmv = OclCSRSpMV(n_row, np.array(csr_rowptr),
                          np.array(csr_colidx),
                          np.array(csr_val, dtype=np.float32))
    csr_y = csr_spmv.run(x)
    csr_perf_start = time.perf_counter()
    for _ in range(t):
        csr_spmv.run(x)
    csr_perf_end = time.perf_counter()
    print("CSR perf time:", (csr_perf_end - csr_perf_start) / t)
    print(min(repeat(lambda: csr_spmv.run(x), number=50, repeat=5)))

    # check error
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR Error: {round(csr_rel_error, 5)}.")

    bsell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL Error: {round(bsell_rel_error, 5)}.")

    sell4_rel_error = np.linalg.norm(
        sell4_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL 4 Error: {round(sell4_rel_error, 5)}.")

    print(sellrd_y[:n_row])
    print(y_exact)
    sellrd_rel_error = np.linalg.norm(
        sellrd_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL rd Error: {round(sellrd_rel_error, 5)}.")
