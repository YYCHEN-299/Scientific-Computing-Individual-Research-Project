import numba
import numpy as np
from numba import jit, prange, guvectorize


# import llvmlite.binding as llvm
# llvm.set_option('', '--debug-only=loop-vectorize')


@jit(nopython=True, parallel=True, fastmath=True)
def numba_csr_spmv(y, num_row, rowptr, colidx, val, x):
    """
    This function is multi thread CSR format based SpMV (y=Ax).

    Parameters
    ----------
    y : ndarrays
        Result of SpMV
    num_row : int
        Number of rows
    rowptr : ndarrays
        Row pointer of CSR format
    colidx : ndarrays
        Column index of CSR format
    val : ndarrays
        None zero elements value of CSR format
    x : ndarrays
        Vector x

    Returns
    -------
    y : ndarrays
        Result of SpMV

    Examples
    --------
    >>>
    """

    for i in prange(num_row):
        row_data = 0.0
        for j in range(rowptr[i], rowptr[i + 1]):
            row_data += val[j] * x[colidx[j]]
        y[i] = row_data


@jit(nopython=True, parallel=True, fastmath=True)
def numba_sliced_ellpack_spmv(y, slice_count, slice_ptr,
                              colidx, val, x, slice_height):
    """
    This function is multi thread Sliced ELLPACK format based SpMV (y=Ax).

    Parameters
    ----------
    y : ndarrays
        Result of SpMV
    slice_count : int
        Number of slices
    slice_ptr : ndarrays
        Slice pointer of Sliced ELLPACK format
    slice_col : ndarrays
        Column length of slices
    colidx : ndarrays
        Column index of Sliced ELLPACK format
    val : ndarrays
        None zero elements value of Sliced ELLPACK format
    x : ndarrays
        Vector x
    slice_height : int
        Slice height of Sliced ELLPACK format

    Returns
    -------
    y : ndarrays
        Result of SpMV

    Examples
    --------
    >>>
    """

    for s in prange(slice_count):
        ptr_start = slice_ptr[s]
        ptr_end = slice_ptr[s + 1]
        row_data = y[s, :]
        for k in range(slice_height):
            row = 0.0
            for i in range(ptr_start + k, ptr_end, slice_height):
                row += x[colidx[i]] * val[i]
            row_data[k] = row


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def numba_sell_spmv(slice_count, slice_ptr, colidx, val, x, y):

    for s in prange(slice_count):
        ptr_start = slice_ptr[s]
        ptr_end = slice_ptr[s + 1]
        row_idx = np.uint32(s * 4)
        for r in range(4):
            row_data = 0.0
            for i in range(ptr_start + r, ptr_end, 4):
                row_data += x[colidx[i]] * val[i]
            y[row_idx + r] = row_data


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def numba_sell_spmv_mark1(y, slice_count, slice_ptr, colidx, val, x, slice_height):

    for s in prange(slice_count):
        now_ptr = slice_ptr[s]
        next_ptr = slice_ptr[s + 1]
        ptr_len = next_ptr - now_ptr
        inner_colidx = colidx[now_ptr:next_ptr]
        inner_val = val[now_ptr:next_ptr]
        for r in range(slice_height):
            row_data = 0.0
            for i in range(r, ptr_len, slice_height):
                row_data += x[inner_colidx[i]] * inner_val[i]
            y[s * slice_height + r] = row_data

    return y


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def numba_sliced_ellpack_spmv_mark2(slice_count, slice_ptr,
                                    colidx, val, x, slice_height):

    y = np.zeros(slice_count * slice_height, dtype='float32')
    for s in prange(slice_count):
        inner_y1 = 0.0
        inner_y2 = 0.0
        inner_y3 = 0.0
        inner_y4 = 0.0
        for idx in range(slice_ptr[s], slice_ptr[s + 1], slice_height):
            inner_y1 += x[colidx[idx + 0]] * val[idx + 0]
            inner_y2 += x[colidx[idx + 1]] * val[idx + 1]
            inner_y3 += x[colidx[idx + 2]] * val[idx + 2]
            inner_y4 += x[colidx[idx + 3]] * val[idx + 3]
        y[s * slice_height + 0] = inner_y1
        y[s * slice_height + 1] = inner_y2
        y[s * slice_height + 2] = inner_y3
        y[s * slice_height + 3] = inner_y4

    return y


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def numba_sliced_ellpack_spmv_h4(y, slice_count, slice_col, colidx, val, x):

    for s in prange(slice_count):
        now_ptr = slice_col[s]
        next_ptr = slice_col[s + 1]
        row_data = np.zeros(4, dtype=np.float32)
        lm_x = np.empty(4, dtype=np.float32)
        for i in range(now_ptr, next_ptr):
            for k in range(4):
                lm_x[k] = x[colidx[i, k]]
            row_data += lm_x * val[i]
        y[s] = row_data
