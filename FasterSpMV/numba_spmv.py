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
        for k in range(slice_height):
            row_data = 0.0
            for i in range(ptr_start + k, ptr_end, slice_height):
                row_data += x[colidx[i]] * val[i]
            y[s * slice_height + k] = row_data


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def numba_sell_spmv(slice_count, slice_ptr, colidx, val, x, y):

    for s in prange(slice_count):
        ptr_start = slice_ptr[s]
        ptr_end = slice_ptr[s + 1]
        for r in range(4):
            row_data = 0.0
            for i in range(ptr_start + r, ptr_end, 4):
                row_data += x[colidx[i]] * val[i]
            y[s * 4 + r] = row_data


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def numba_sell_spmv_h4(y, slice_count, slice_col, colidx, val, x):

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
