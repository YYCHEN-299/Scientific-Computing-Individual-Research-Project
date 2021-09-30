import time

from scipy.sparse import csr_matrix

from FasterSpMV.benchmark_tools import find_instr
from FasterSpMV.cuda_spmv import *
from FasterSpMV.matrix_tools import *
from FasterSpMV.numba_spmv import *
from FasterSpMV.opencl_spmv import *
from FasterSpMV.pycuda_spmv import *


def spmv_scipy_benchmark(sp_matrix, t):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)

    # get CSR data
    csr_rowptr = sp_matrix.indptr.astype(np.int32)
    csr_colidx = sp_matrix.indices.astype(np.int32)
    csr_val = sp_matrix.data.astype(np.float32)

    x = np.ones(n_col, dtype=np.float32)
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))

    time_list = []
    for _ in range(t):
        start = time.perf_counter()
        y = sp_A.dot(x)  # SciPy SpMV
        end = time.perf_counter()
        time_list.append((end - start))

    print("SciPy avg run time:", np.mean(time_list))
    print("SciPy min run time:", np.min(time_list))
    print("SciPy run time std:", np.std(time_list))


def spmv_cpu_csr_benchmark(sp_matrix, t):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)

    # get CSR data
    csr_rowptr = sp_matrix.indptr.astype(np.int32)
    csr_colidx = sp_matrix.indices.astype(np.int32)
    csr_val = sp_matrix.data.astype(np.float32)

    # vector to multiply
    x = np.ones(n_col, dtype=np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    print("Run CPU CSR SpMV benchmark...")

    # ========================================================================
    # Numba SpMV
    print("Numba SpMV benchmark...")
    print("")

    # === CSR test ===
    csr_y = np.empty(n_row, dtype=np.float32)
    numba_csr_spmv(csr_y, n_row, csr_rowptr, csr_colidx, csr_val, x)

    csr_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        numba_csr_spmv(csr_y, n_row, csr_rowptr, csr_colidx, csr_val, x)
        end = time.perf_counter()
        csr_time_list.append((end - start))

    print("Numba CSR avg run time:", np.mean(csr_time_list))
    print("Numba CSR min run time:", np.min(csr_time_list))
    print("Numba CSR run time std:", np.std(csr_time_list))
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"Numba CSR result error: {round(csr_rel_error, 2)}.")
    print("")

    # === find instructions ===
    # print("CSR mul instructions:")
    # find_instr(numba_csr_spmv, key='mul')
    # print("")
    # print("CSR add instructions:")
    # find_instr(numba_csr_spmv, key='add')
    # print("")

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

    print("OpenCL CSR avg run time:", np.mean(csr_time_list))
    print("OpenCL CSR min run time:", np.min(csr_time_list))
    print("OpenCL CSR run time std:", np.std(csr_time_list))
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"OpenCL CSR result error: {round(csr_rel_error, 2)}.")
    print("")


def spmv_cpu_sell_benchmark(sp_matrix, slice_height, t):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)

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

    print("Run CPU SELL SpMV benchmark...")

    # ========================================================================
    # Numba SpMV
    print("Numba SpMV benchmark...")
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

    print("Slice height:", slice_height)
    print("Numba SELL avg run time:", np.mean(sell_time_list))
    print("Numba SELL min run time:", np.min(sell_time_list))
    print("Numba SELL run time std:", np.std(sell_time_list))
    sell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"Numba SELL result error: {round(sell_rel_error, 2)}.")
    print("")

    # === find instructions ===
    # get instructions include mul
    # print("SELL mul instructions:")
    # find_instr(numba_sell_spmv, key='mul')
    # print("")
    # # get instructions include add
    # print("SELL add instructions:")
    # find_instr(numba_sell_spmv, key='add')
    # print("")

    # ========================================================================
    # OpenCL CPU SpMV
    print("OpenCL SpMV benchmark...")
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

    print("Slice height:", slice_height)
    print("OpenCL SELL avg run time:", np.mean(sell_time_list))
    print("OpenCL SELL min run time:", np.min(sell_time_list))
    print("OpenCL SELL run time std:", np.std(sell_time_list))
    sell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"OpenCL SELL result error: {round(sell_rel_error, 2)}.")
    print("")


def spmv_cpu_exp_sell_benchmark(sp_matrix, t):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)

    # get CSR data
    csr_rowptr = sp_matrix.indptr.astype(np.int32)
    csr_colidx = sp_matrix.indices.astype(np.int32)
    csr_val = sp_matrix.data.astype(np.float32)

    # vector to multiply
    x = np.ones(n_col, dtype=np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    print("Run CPU SELL SpMV benchmark...")
    # ========================================================================
    # OpenCL CPU SpMV
    print("OpenCL SpMV benchmark...")
    print("")

    # === Sliced ELLPACK explicit slice height (2) test ===
    # get data
    sliceocl2_count, sellocl2_colidx, _, sellocl2_slicecol, sellocl2_val = \
        csr_to_ocl_sell2(n_row, csr_rowptr, csr_colidx, csr_val)

    ocl_sell2_spmv = OclSELL2SpMV(n_row, sliceocl2_count, sellocl2_slicecol,
                                  sellocl2_colidx, sellocl2_val)
    sell2_y = ocl_sell2_spmv.run(x)

    sell2_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        ocl_sell2_spmv.run(x)
        end = time.perf_counter()
        sell2_time_list.append((end - start))

    print("SELL-2 avg run time:", np.mean(sell2_time_list))
    print("SELL-2 min run time:", np.min(sell2_time_list))
    print("SELL-2 run time std:", np.std(sell2_time_list))
    sell2_y = sell2_y.reshape(-1,)
    sell2_rel_error = np.linalg.norm(
        sell2_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL-2 result error: {round(sell2_rel_error, 2)}.")
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

    # === Sliced ELLPACK explicit slice height (8) test ===
    # get data
    sliceocl8_count, sellocl8_colidx, _, sellocl8_slicecol, sellocl8_val = \
        csr_to_ocl_sell8(n_row, csr_rowptr, csr_colidx, csr_val)

    ocl_sell8_spmv = OclSELL8SpMV(n_row, sliceocl8_count, sellocl8_slicecol,
                                  sellocl8_colidx, sellocl8_val)
    sell8_y = ocl_sell8_spmv.run(x)

    sell8_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        ocl_sell8_spmv.run(x)
        end = time.perf_counter()
        sell8_time_list.append((end - start))

    print("SELL-8 avg run time:", np.mean(sell8_time_list))
    print("SELL-8 min run time:", np.min(sell8_time_list))
    print("SELL-8 run time std:", np.std(sell8_time_list))
    sell8_y = sell8_y.reshape(-1,)
    sell8_rel_error = np.linalg.norm(
        sell8_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL-4 result error: {round(sell8_rel_error, 2)}.")
    print("")

    # === Sliced ELLPACK explicit slice height (16) test ===
    # get data
    sliceocl16_count, sellocl16_colidx, _, \
        sellocl16_slicecol, sellocl16_val = \
        csr_to_ocl_sell16(n_row, csr_rowptr, csr_colidx, csr_val)

    ocl_sell16_spmv = OclSELL16SpMV(n_row, sliceocl16_count,
                                    sellocl16_slicecol,
                                    sellocl16_colidx, sellocl16_val)
    sell16_y = ocl_sell16_spmv.run(x)

    sell16_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        ocl_sell16_spmv.run(x)
        end = time.perf_counter()
        sell16_time_list.append((end - start))

    print("SELL-16 avg run time:", np.mean(sell16_time_list))
    print("SELL-16 min run time:", np.min(sell16_time_list))
    print("SELL-16 run time std:", np.std(sell16_time_list))
    sell16_y = sell16_y.reshape(-1,)
    sell16_rel_error = np.linalg.norm(
        sell16_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"SELL-16 result error: {round(sell16_rel_error, 2)}.")
    print("")


def spmv_gpu_csr_benchmark(sp_matrix, t):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)

    # get CSR data
    csr_rowptr = sp_matrix.indptr.astype(np.int32)
    csr_colidx = sp_matrix.indices.astype(np.int32)
    csr_val = sp_matrix.data.astype(np.float32)

    # vector to multiply
    x = np.ones(n_col, dtype=np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    print("Run GPU CSR SpMV benchmark...")

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

    print("OpenCL CSR avg run time:", np.mean(csr_time_list))
    print("OpenCL CSR min run time:", np.min(csr_time_list))
    print("OpenCL CSR run time std:", np.std(csr_time_list))
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"OpenCL CSR result error: {round(csr_rel_error, 2)}.")
    print("")

    # # ========================================================================
    # # CUDA SpMV
    # print("CUDA SpMV benchmark...")
    # print("")
    #
    # # === CSR test ===
    # nblocks = (n_row,)  # global blocks
    # nthreads = (1,)  # threads per block
    #
    # # CUDA buffer
    # csr_y = np.empty(n_row, dtype=np.float32)
    # bf_csr_rowptr = cuda.to_device(csr_rowptr)
    # bf_csr_colidx = cuda.to_device(csr_colidx)
    # bf_csr_val = cuda.to_device(csr_val)
    # bf_x = cuda.to_device(x)
    # # _csr_y = np.empty(n_row, dtype=np.float32)
    # # bf_csr_y = cuda.to_device(csr_y, stream=stream)
    # bf_csr_y = cuda.device_array(n_row, dtype=np.float32)
    # # cuda.driver.host_to_device()
    # # first run
    # cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
    #                                  bf_csr_val, bf_x, bf_csr_y)
    #
    # csr_time_list = []
    # for _ in range(t):
    #     start = time.perf_counter()
    #     bf_x = cuda.to_device(x)
    #     cuda_csr_spmv[nblocks, nthreads](bf_csr_rowptr, bf_csr_colidx,
    #                                      bf_csr_val, bf_x, bf_csr_y)
    #     bf_csr_y.copy_to_host(csr_y)
    #     end = time.perf_counter()
    #     csr_time_list.append((end - start))
    #
    # print("CUDA CSR avg run time:", np.mean(csr_time_list))
    # print("CUDA CSR min run time:", np.min(csr_time_list))
    # print("CUDA CSR run time std:", np.std(csr_time_list))
    # bf_csr_y.copy_to_host(csr_y)
    # csr_rel_error = np.linalg.norm(
    #     csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    # print(f"CUDA CSR result error: {round(csr_rel_error, 2)}.")
    # print("")


def spmv_gpu_sell_benchmark(sp_matrix, slice_height, t):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)

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

    print("Slice height:", slice_height)
    print("OpenCL SELL avg run time:", np.mean(sell_time_list))
    print("OpenCL SELL min run time:", np.min(sell_time_list))
    print("OpenCL SELL run time std:", np.std(sell_time_list))
    sell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"OpenCL SELL result error: {round(sell_rel_error, 2)}.")
    print("")

    # # ========================================================================
    # # CUDA SpMV
    # print("CUDA SpMV benchmark...")
    # print("")
    #
    # # === Sliced ELLPACK test ===
    # nblocks = (slice_count,)  # global blocks
    # nthreads = (slice_height,)  # threads per block, better be multiple of 32
    #
    # # CUDA buffer
    # # _sell_y = np.empty(slice_height * slice_count, dtype=np.float32)
    # # bf_sell_y = cuda.to_device(_sell_y)
    # bf_sell_y = cuda.device_array(slice_height * slice_count,
    #                               dtype=np.float32)
    # sell_y = np.empty(slice_height * slice_count, dtype=np.float32)
    # bf_sell_sliceptr = cuda.to_device(sell_sliceptr)
    # bf_sell_colidx = cuda.to_device(sell_colidx)
    # bf_sell_val = cuda.to_device(sell_val)
    # bf_x = cuda.to_device(x)
    # # bf_x = cuda.pinned(x)
    #
    # # first run
    # cuda_sell_spmv[nblocks, nthreads](bf_sell_sliceptr,
    #                                   bf_sell_colidx,
    #                                   bf_sell_val, bf_x,
    #                                   slice_height,
    #                                   bf_sell_y)
    #
    # sell_time_list = []
    # for _ in range(t):
    #     start = time.perf_counter()
    #     bf_x = cuda.to_device(x)
    #     cuda_sell_spmv[nblocks, nthreads](bf_sell_sliceptr,
    #                                       bf_sell_colidx,
    #                                       bf_sell_val, bf_x,
    #                                       slice_height,
    #                                       bf_sell_y)
    #     bf_sell_y.copy_to_host(sell_y)
    #     end = time.perf_counter()
    #     sell_time_list.append((end - start))
    #
    # print("Slice height:", slice_height)
    # print("CUDA SELL avg run time:", np.mean(sell_time_list))
    # print("CUDA SELL min run time:", np.min(sell_time_list))
    # print("CUDA SELL run time std:", np.std(sell_time_list))
    # bf_sell_y.copy_to_host(sell_y)
    # sell_rel_error = np.linalg.norm(
    #     sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    # print(f"CUDA SELL result error: {round(sell_rel_error, 2)}.")
    # print("")


def spmv_pycuda_csr_benchmark(sp_matrix, t):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)

    # get CSR data
    csr_rowptr = sp_matrix.indptr.astype(np.int32)
    csr_colidx = sp_matrix.indices.astype(np.int32)
    csr_val = sp_matrix.data.astype(np.float32)

    # vector to multiply
    x = np.ones(n_col, dtype=np.float32)

    # get exact y
    sp_A = csr_matrix((csr_val, csr_colidx, csr_rowptr), shape=(n_row, n_col))
    y_exact = sp_A.dot(x)  # SciPy SpMV

    csr_spmv = PyCUDACSRSpMV(n_row, csr_rowptr, csr_colidx, csr_val)
    csr_y = csr_spmv.run(x)

    csr_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        csr_spmv.run(x)
        end = time.perf_counter()
        csr_time_list.append((end - start))

    print("PyCUDA CSR avg run time:", np.mean(csr_time_list))
    print("PyCUDA CSR min run time:", np.min(csr_time_list))
    print("PyCUDA CSR run time std:", np.std(csr_time_list))
    csr_rel_error = np.linalg.norm(
        csr_y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"PyCUDA CSR result error: {round(csr_rel_error, 2)}.")
    print("")


def spmv_pycuda_sell_benchmark(sp_matrix, slice_height, t):
    # get sparse matrix shape
    n_row, n_col = sp_matrix.shape
    n_row = int(n_row)
    n_col = int(n_col)

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

    sell_spmv = PyCUDASELLSpMV(n_row, slice_height,
                               slice_count, sell_colidx,
                               sell_sliceptr, sell_slicecol, sell_val)
    sell_y = sell_spmv.run(x)

    sell_time_list = []
    for _ in range(t):
        start = time.perf_counter()
        sell_spmv.run(x)
        end = time.perf_counter()
        sell_time_list.append((end - start))

    print("Slice height:", slice_height)
    print("PyCUDA SELL avg run time:", np.mean(sell_time_list))
    print("PyCUDA SELL min run time:", np.min(sell_time_list))
    print("PyCUDA SELL run time std:", np.std(sell_time_list))
    sell_rel_error = np.linalg.norm(
        sell_y[:n_row] - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
    print(f"PyCUDA SELL result error: {round(sell_rel_error, 2)}.")
    print("")
