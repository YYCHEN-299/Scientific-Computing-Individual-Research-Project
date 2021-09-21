import numba
import numpy as np
from numba import cuda


@cuda.jit(fastmath=True)
def cuda_csr_spmv(rowptr, colidx, val, x, y):

    row = np.uint32(cuda.blockIdx.x)
    ptr1 = rowptr[row]
    ptr2 = rowptr[row + 1]
    row_data = 0.0
    for j in range(ptr1, ptr2):
        row_data += val[j] * x[colidx[j]]
    y[row] = row_data


@cuda.jit(fastmath=True)
def cuda_sliced_ellpack_spmv_1d(slice_ptr, colidx, val, x, slice_height, y):

    slice_id = np.uint32(cuda.blockIdx.x)
    slice_row_id = np.uint32(cuda.threadIdx.x)
    local_idx = cuda.shared.array(2, np.uint32)
    local_y = cuda.shared.array(32, np.float32)
    row_data = 0.0
    if slice_row_id == 0:
        local_idx[0] = slice_ptr[slice_id]
        local_idx[1] = slice_ptr[slice_id + 1]
    cuda.syncthreads()
    for i in range(local_idx[0] + slice_row_id, local_idx[1], slice_height):
        row_data += x[colidx[i]] * val[i]
    local_y[slice_row_id] = row_data
    cuda.syncthreads()
    if slice_row_id == 31:
        for k in range(32):
            y[slice_id * slice_height + k] = local_y[k]


@cuda.jit(fastmath=True)
def cuda_sliced_ellpack_spmv_2d(slice_col, colidx, val, x, y):

    slice_id = np.uint32(cuda.blockIdx.x)
    row_id = np.uint32(cuda.threadIdx.x)
    local_idx = cuda.shared.array(2, np.uint32)
    local_y = cuda.shared.array(32, np.float32)
    row_data = 0.0
    if row_id == 0:
        local_idx[0] = slice_col[slice_id]
        local_idx[1] = slice_col[slice_id + 1]
    cuda.syncthreads()
    for i in range(local_idx[0], local_idx[1]):
        row_data += x[colidx[i, row_id]] * val[i, row_id]
    local_y[row_id] = row_data
    cuda.syncthreads()
    if row_id == 31:
        for k in range(32):
            y[slice_id, k] = local_y[k]


@cuda.jit(fastmath=True)
def cuda_sliced_ellpack_spmv_warp(slice_count, slice_ptr,
                                  colidx, val, x, slice_height, y):

    slice_per_block = np.uint32(cuda.blockDim.x / slice_height)
    slice_id = np.uint32(cuda.threadIdx.x % slice_height)
    slice_warp_count = np.uint32(slice_per_block * cuda.gridDim.x)
    slice_warp_id = np.uint32(slice_per_block *
                              cuda.blockIdx.x +
                              cuda.threadIdx.x / slice_height)
    for slice_idx in range(slice_warp_id, slice_count, slice_warp_count):
        row_data = 0.0
        row = np.uint32(slice_idx * slice_height + slice_id)
        offset = slice_ptr[slice_idx]
        next_offset = slice_ptr[slice_idx + 1]
        num_columns = np.uint32((next_offset - offset) / slice_height)
        for item_id in range(num_columns):
            idx = offset + item_id * slice_height + slice_id
            row_data += x[colidx[idx]] * val[idx]
        y[row] = sum
