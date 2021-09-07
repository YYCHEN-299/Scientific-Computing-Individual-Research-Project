from numba import cuda


@cuda.jit
def cuda_csr_spmv(rowptr, colidx, val, x, y):

    row = cuda.blockIdx.x
    row_data = 0.0
    for j in range(rowptr[row], rowptr[row + 1]):
        row_data += val[j] * x[colidx[j]]
    y[row] = row_data


@cuda.jit
def cuda_sliced_ellpack_spmv(slice_ptr, colidx, val, x, slice_height, y):

    slice_id = cuda.blockIdx.x
    slice_row_id = cuda.threadIdx.x
    row_data = 0.0
    for i in range(slice_ptr[slice_id] + slice_row_id, slice_ptr[slice_id + 1], slice_height):
        row_data += x[colidx[i]] * val[i]
    y[slice_id * slice_height + slice_row_id] = row_data


@cuda.jit
def cuda_sliced_ellpack_spmv_mark2(slice_count, slice_ptr,
                                   colidx, val, x, slice_height, y):

    slice_per_block = int(cuda.blockDim.x / slice_height)
    slice_id = int(cuda.threadIdx.x % slice_height)
    slice_warp_count = int(slice_per_block * cuda.gridDim.x)
    slice_warp_id = int(slice_per_block *
                        cuda.blockIdx.x + cuda.threadIdx.x / slice_height)
    for slice_idx in range(slice_warp_id, slice_count, slice_warp_count):
        row_data = 0.0
        row = int(slice_idx * slice_height + slice_id)
        offset = slice_ptr[slice_idx]
        next_offset = slice_ptr[slice_idx + 1]
        num_columns = int((next_offset - offset) / slice_height)
        for item_id in range(num_columns):
            idx = offset + item_id * slice_height + slice_id
            row_data += x[colidx[idx]] * val[idx]
        y[row] = sum
