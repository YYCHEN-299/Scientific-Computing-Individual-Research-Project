import time

from scipy.sparse import csr_matrix

from FasterSpMV.benchmark_tools import find_instr
from FasterSpMV.matrix_tools import *
from FasterSpMV.numba_spmv import *
from FasterSpMV.opencl_spmv import *
from FasterSpMV.cuda_spmv import *


def spmv_cpu_benchmark(sp_matrix, slice_height, row_th, t):
    """

    Parameters
    ----------
    sp_matrix
    slice_height
    row_th
    t

    Returns
    -------

    """

    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # get CSR data
    csr_rowptr = sp_matrix.indptr.astype(np.int32)
    csr_colidx = sp_matrix.indices.astype(np.int32)
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    slice_count, sell_colidx, sell_sliceptr, sell_slicecol, sell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # vector to multiply
    x = np.ones(n_col, dtype=np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    print("Run CPU SpMV benchmark...")

    # ========================================================================
    # Numba SpMV
    print("Numba SpMV benchmark...")
    print("")

    # === CSR test ===
    csr_y = np.zeros(n_row, dtype=np.float32)
    numba_csr_spmv(csr_y, n_row, csr_rowptr, csr_colidx, csr_val, x)

    csr_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        numba_csr_spmv(csr_y, n_row, csr_rowptr, csr_colidx, csr_val, x)
        end = time.perf_counter()
        csr_time_list.append((end - start))

    print("CSR avg run time:", np.mean(csr_time_list))
    print("CSR min run time:", np.min(csr_time_list))
    print("CSR run time std:", np.std(csr_time_list))
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR result error: {round(csr_rel_error, 2)}.")
    print("")

    # === Sliced ELLPACK test ===
    sell_y = np.empty((slice_count * slice_height), dtype=np.float32)
    numba_sell_spmv(sell_y, slice_count, sell_sliceptr,
                    sell_colidx, sell_val, x, slice_height)

    sell_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        numba_sell_spmv(sell_y, slice_count, sell_sliceptr,
                        sell_colidx, sell_val, x, slice_height)
        end = time.perf_counter()
        sell_time_list.append((end - start))

    print("SELL avg run time:", np.mean(sell_time_list))
    print("SELL min run time:", np.min(sell_time_list))
    print("SELL run time std:", np.std(sell_time_list))
    sell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL result error: {round(sell_rel_error, 2)}.")
    print("")

    # === Sliced ELLPACK explicit slice height (4) test ===
    # get data
    slice4_count, sell4_colidx, _, sell4_slicecol, sell4_val = \
        csr_to_2d_sell(n_row, csr_rowptr, csr_colidx, csr_val, 4)
    sell4_y = np.empty((slice4_count, 4), dtype=np.float32)
    numba_sell4_spmv(sell4_y, slice4_count,
                     sell4_slicecol, sell4_colidx, sell4_val, x)

    sell4_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        numba_sell4_spmv(sell4_y, slice4_count,
                         sell4_slicecol, sell4_colidx, sell4_val, x)
        end = time.perf_counter()
        sell4_time_list.append((end - start))

    print("SELL-4 avg run time:", np.mean(sell4_time_list))
    print("SELL-4 min run time:", np.min(sell4_time_list))
    print("SELL-4 run time std:", np.std(sell4_time_list))
    sell4_y = sell4_y.reshape(-1,)
    sell4_rel_error = np.linalg.norm(
        sell4_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL-4 result error: {round(sell4_rel_error, 2)}.")
    print("")

    # === find instructions ===
    # get instructions include mul
    print("SELL mul instructions:")
    find_instr(numba_sell_spmv, key='mul')
    print("")
    print("SELL-4 mul instructions:")
    find_instr(numba_sell4_spmv, key='mul')
    print("")
    print("CSR mul instructions:")
    find_instr(numba_csr_spmv, key='mul')
    print("")

    # get instructions include add
    print("SELL add instructions:")
    find_instr(numba_sell_spmv, key='add')
    print("")
    print("SELL-4 add instructions:")
    find_instr(numba_sell4_spmv, key='add')
    print("")
    print("CSR add instructions:")
    find_instr(numba_csr_spmv, key='add')
    print("")

    # ========================================================================
    # OpenCL CPU SpMV
    print("OpenCL SpMV benchmark...")
    print("")

    # === CSR test ===
    csr_spmv = OclCSRSpMV(n_row, csr_rowptr, csr_colidx, csr_val, 'CPU')
    csr_y = csr_spmv.run(x)

    csr_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        csr_spmv.run(x)
        end = time.perf_counter()
        csr_time_list.append((end - start))

    print("CSR avg run time:", np.mean(csr_time_list))
    print("CSR min run time:", np.min(csr_time_list))
    print("CSR run time std:", np.std(csr_time_list))
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR result error: {round(csr_rel_error, 2)}.")
    print("")

    # === Sliced ELLPACK test ===
    ocl_sell_spmv = OclSELLSpMV(n_row, slice_count,
                                sell_sliceptr, sell_slicecol,
                                sell_colidx, sell_val, slice_height, 'CPU')
    sell_y = ocl_sell_spmv.run(x)

    sell_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        ocl_sell_spmv.run(x)
        end = time.perf_counter()
        sell_time_list.append((end - start))

    print("SELL avg run time:", np.mean(sell_time_list))
    print("SELL min run time:", np.min(sell_time_list))
    print("SELL run time std:", np.std(sell_time_list))
    sell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL result error: {round(sell_rel_error, 2)}.")
    print("")

    # === Sliced ELLPACK explicit slice height (4) test ===
    # get data
    sliceocl4_count, sellocl4_colidx, _, sellocl4_slicecol, sellocl4_val = \
        csr_to_ocl_sell4(n_row, csr_rowptr, csr_colidx, csr_val)

    ocl_sell4_spmv = OclSELL4SpMV(n_row, sliceocl4_count, sellocl4_slicecol,
                                  sellocl4_colidx, sellocl4_val)
    sell4_y = ocl_sell4_spmv.run(x)

    sell4_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        ocl_sell4_spmv.run(x)
        end = time.perf_counter()
        sell4_time_list.append((end - start))

    print("SELL-4 avg run time:", np.mean(sell4_time_list))
    print("SELL-4 min run time:", np.min(sell4_time_list))
    print("SELL-4 run time std:", np.std(sell4_time_list))
    sell4_y = sell4_y.reshape(-1,)
    sell4_rel_error = np.linalg.norm(
        sell4_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL-4 result error: {round(sell4_rel_error, 2)}.")
    print("")

    # === Sliced ELLPACK with reduction test ===
    # get data
    sellrd_val, sellrd_colidx, sellrd_slicecol, sellrd_sliceptr = \
        csr_to_align_sell(sp_matrix, row_th, slice_height)

    sellrd_spmv = OclSELLRdSpMV(n_row, slice_count, sellrd_sliceptr,
                                sellrd_slicecol, sellrd_colidx,
                                sellrd_val, slice_height, row_th)
    sellrd_y = sellrd_spmv.run(x)

    sellrd_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        sellrd_spmv.run(x)
        end = time.perf_counter()
        sellrd_time_list.append((end - start))

    print("SELL-rd avg run time:", np.mean(sellrd_time_list))
    print("SELL-rd min run time:", np.min(sellrd_time_list))
    print("SELL-rd run time std:", np.std(sellrd_time_list))
    sellrd_rel_error = np.linalg.norm(
        sellrd_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL-rd result error: {round(sellrd_rel_error, 2)}.")
    print("")


def spmv_gpu_benchmark(sp_matrix, slice_height, row_th, t):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)
    print("Sparse matrix row:", n_row, ", column:", n_col)

    # get CSR data
    csr_rowptr = sp_matrix.indptr.astype(np.int32)
    csr_colidx = sp_matrix.indices.astype(np.int32)
    csr_val = sp_matrix.data.astype(np.float32)

    # convert CSR to Sliced ELLPACK format
    slice_count, sell_colidx, sell_sliceptr, sell_slicecol, sell_val = \
        csr_to_sell(n_row, csr_rowptr, csr_colidx, csr_val, slice_height)

    # vector to multiply
    x = np.ones(n_col, dtype=np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    print("Run GPU SpMV benchmark...")

    # ========================================================================
    # OpenCL GPU SpMV
    print("OpenCL SpMV benchmark...")
    print("")

    # === CSR test ===
    csr_spmv = OclCSRSpMV(n_row, csr_rowptr, csr_colidx, csr_val, 'GPU')
    csr_y = csr_spmv.run(x)

    csr_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        csr_spmv.run(x)
        end = time.perf_counter()
        csr_time_list.append((end - start))

    print("CSR avg run time:", np.mean(csr_time_list))
    print("CSR min run time:", np.min(csr_time_list))
    print("CSR run time std:", np.std(csr_time_list))
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR result error: {round(csr_rel_error, 2)}.")
    print("")

    # === Sliced ELLPACK test ===
    ocl_sell_spmv = OclSELLSpMV(n_row, slice_count,
                                sell_sliceptr, sell_slicecol,
                                sell_colidx, sell_val, slice_height, 'GPU')
    sell_y = ocl_sell_spmv.run(x)

    sell_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        ocl_sell_spmv.run(x)
        end = time.perf_counter()
        sell_time_list.append((end - start))

    print("SELL avg run time:", np.mean(sell_time_list))
    print("SELL min run time:", np.min(sell_time_list))
    print("SELL run time std:", np.std(sell_time_list))
    sell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL result error: {round(sell_rel_error, 2)}.")
    print("")

    # ========================================================================
    # CUDA SpMV
    print("CUDA SpMV benchmark...")
    print("")

    # === CSR test ===
    nblocks = (n_row,)  # global blocks
    nthreads = (1,)  # threads per block

    # CUDA buffer
    stream = cuda.stream()
    bf_csr_rowptr = cuda.to_device(csr_rowptr, stream=stream)
    bf_csr_colidx = cuda.to_device(csr_colidx, stream=stream)
    bf_csr_val = cuda.to_device(csr_val, stream=stream)
    bf_x = cuda.to_device(x, stream=stream)
    _csr_y = np.empty(n_row, dtype=np.float32)
    # bf_csr_y = cuda.device_array(n_row, dtype=np.float32, stream=stream)
    bf_csr_y = cuda.to_device(_csr_y, stream=stream)

    # first run
    cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
                                     bf_csr_val, bf_x, bf_csr_y)

    csr_y = np.empty_like(_csr_y, dtype=np.float32)
    csr_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        bf_x = cuda.to_device(x)
        cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
                                         bf_csr_val, bf_x, bf_csr_y)
        csr_y = bf_csr_y.copy_to_host()
        end = time.perf_counter()
        csr_time_list.append((end - start))

    print("CSR avg run time:", np.mean(csr_time_list))
    print("CSR min run time:", np.min(csr_time_list))
    print("CSR run time std:", np.std(csr_time_list))
    csr_y = bf_csr_y.copy_to_host()
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"CSR result error: {round(csr_rel_error, 2)}.")
    print("")

    # === Sliced ELLPACK test ===
    nblocks = (slice_count,)  # global blocks
    nthreads = (slice_height,)  # threads per block, better be multiple of 32

    # CUDA buffer
    stream = cuda.stream()
    _sell_y = np.empty(slice_height * slice_count, dtype=np.float32)
    # bf_sell_y = cuda.device_array(slice_height * slice_count,
    #                               dtype=np.float32)
    bf_sell_y = cuda.to_device(_sell_y, stream=stream)
    bf_sell_sliceptr = cuda.to_device(sell_sliceptr, stream=stream)
    bf_sell_colidx = cuda.to_device(sell_colidx, stream=stream)
    bf_sell_val = cuda.to_device(sell_val, stream=stream)
    bf_x = cuda.to_device(x, stream=stream)

    # first run
    cuda_sell_spmv[nblocks, nthreads](bf_sell_sliceptr,
                                      bf_sell_colidx,
                                      bf_sell_val, bf_x,
                                      slice_height,
                                      bf_sell_y)

    sell_y = np.empty_like(_sell_y, dtype=np.float32)
    sell_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        bf_x = cuda.to_device(x)
        cuda_sell_spmv[nblocks, nthreads](bf_sell_sliceptr,
                                          bf_sell_colidx,
                                          bf_sell_val, bf_x,
                                          slice_height,
                                          bf_sell_y)
        sell_y = bf_sell_y.copy_to_host()
        end = time.perf_counter()
        sell_time_list.append((end - start))

    print("SELL avg run time:", np.mean(sell_time_list))
    print("SELL min run time:", np.min(sell_time_list))
    print("SELL run time std:", np.std(sell_time_list))
    sell_y = bf_sell_y.copy_to_host()
    sell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL result error: {round(sell_rel_error, 2)}.")
