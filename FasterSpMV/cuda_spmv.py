import numpy as np
from numba import cuda


@cuda.jit(fastmath=True)
def cuda_csr_spmv(rowptr, colidx, val, x, y):
    """
    CUDA CSR SpMV.

    Parameters
    ----------
    rowptr : ndarrays
        Row pointer
    colidx : ndarrays
        Column index array
    val : ndarrays
        Value array
    x : ndarrays
        Vector to multiply with matrix
    y : ndarrays
        SpMV result

    Returns
    -------
    Nothing
    """

    row = np.int32(cuda.blockIdx.x)
    ptr1 = rowptr[row]
    ptr2 = rowptr[row + 1]
    row_data = 0.0
    for j in range(ptr1, ptr2):
        row_data += val[j] * x[colidx[j]]
    y[row] = row_data


@cuda.jit(fastmath=True)
def cuda_sell_spmv(slice_ptr, colidx, val, x, slice_height, y):
    """
    CUDA Sliced ELLPACK SpMV.

    Parameters
    ----------
    slice_ptr : ndarrays
        Slice pointer
    colidx : ndarrays
        Column index array
    val : ndarrays
        Value array
    x : ndarrays
        Vector to multiply with matrix
    slice_height : int
        Slice height
    y : ndarrays
        SpMV result

    Returns
    -------
    Nothing
    """

    slice_id = np.int32(cuda.blockIdx.x)
    slice_row_id = np.int32(cuda.threadIdx.x)
    local_idx = cuda.shared.array(2, np.int32)
    local_y = cuda.shared.array(cuda.blockDim.x, np.float32)
    row_data = 0.0
    if slice_row_id == 0:
        local_idx[0] = slice_ptr[slice_id]
        local_idx[1] = slice_ptr[slice_id + 1]
    cuda.syncthreads()
    for i in range(local_idx[0] + slice_row_id, local_idx[1], slice_height):
        row_data += x[colidx[i]] * val[i]
    local_y[slice_row_id] = row_data
    cuda.syncthreads()
    if slice_row_id == slice_height - 1:
        for k in range(slice_height):
            y[slice_id * slice_height + k] = local_y[k]


@cuda.jit(fastmath=True)
def cuda_rd_sell_spmv(slice_ptr, slice_col, colidx,
                      val, x, slice_height, row_th, y):
    """
    CUDA Sliced ELLPACK SpMV with reduction.

    Parameters
    ----------
    slice_ptr : ndarrays
        Slice pointer
    slice_col : ndarrays
        Column length of each slice
    colidx : ndarrays
        Column index array
    val : ndarrays
        Value array
    x : ndarrays
        Vector to multiply with matrix
    slice_height : int
        Slice height
    row_th : int
        Number of threads to process a row
    y : ndarrays
        SpMV result

    Returns
    -------
    Nothing
    """

    local_y = cuda.shared.array(cuda.blockDim.x, np.float32)
    local_id = cuda.threadIdx.x
    local_th = int(local_id % row_th)
    slice_id = cuda.blockIdx.x
    global_th = int(slice_id * slice_height * row_th + local_id)
    row_id = int(global_th / row_th)

    sub_data = 0.0
    row_len = int((slice_col[row_id] + row_th - 1) / row_th)

    for i in range(row_len):
        idx = i * slice_height * row_th + slice_ptr[slice_id] + local_id
        sub_data += x[colidx[idx]] * val[idx]
    local_y[local_id] = sub_data
    cuda.syncthreads()

    s = int(row_th / 2)
    while s > 0:
        if local_th < s:
            local_y[local_id] += local_y[local_id + s]
            cuda.syncthreads()
        s >>= 1

    if local_th == 0:
        y[row_id] = local_y[local_id] + local_y[local_id + 1]


@cuda.jit(fastmath=True)
def cuda_2d_sell_spmv(slice_col, colidx, val, x, y):
    """
    CUDA 2d Sliced ELLPACK SpMV.

    Parameters
    ----------
    slice_col : ndarrays
        Column length of each slice
    colidx : ndarrays
        Column index array
    val : ndarrays
        Value array
    x : ndarrays
        Vector to multiply with matrix
    y : ndarrays
        SpMV result

    Returns
    -------
    Nothing
    """

    slice_id = np.int32(cuda.blockIdx.x)
    row_id = np.int32(cuda.threadIdx.x)
    local_idx = cuda.shared.array(2, np.int32)
    local_y = cuda.shared.array(cuda.blockDim.x, np.float32)
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
def cuda_warp_sell_spmv(slice_count, slice_ptr,
                        colidx, val, x, slice_height, y):
    """
    CUDA Sliced ELLPACK SpMV adapt to warp number.

    Parameters
    ----------
    slice_count : int
        Slices number
    slice_ptr : ndarrays
        Slice pointer
    colidx : ndarrays
        Column index array
    val : ndarrays
        Value array
    x : ndarrays
        Vector to multiply with matrix
    slice_height : int
        Slice height
    y : ndarrays
        SpMV result

    Returns
    -------
    Nothing
    """

    slice_per_block = np.int32(cuda.blockDim.x / slice_height)
    slice_id = np.int32(cuda.threadIdx.x % slice_height)
    slice_warp_count = np.int32(slice_per_block * cuda.gridDim.x)
    slice_warp_id = np.int32(slice_per_block *
                              cuda.blockIdx.x +
                              cuda.threadIdx.x / slice_height)
    for slice_idx in range(slice_warp_id, slice_count, slice_warp_count):
        row_data = 0.0
        row = np.int32(slice_idx * slice_height + slice_id)
        offset = slice_ptr[slice_idx]
        next_offset = slice_ptr[slice_idx + 1]
        num_columns = np.int32((next_offset - offset) / slice_height)
        for item_id in range(num_columns):
            idx = offset + item_id * slice_height + slice_id
            row_data += x[colidx[idx]] * val[idx]
        y[row] = sum
