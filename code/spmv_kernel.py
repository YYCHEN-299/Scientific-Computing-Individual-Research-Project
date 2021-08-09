import numba
import numpy as np
from numba import jit


# import llvmlite.binding as llvm
# llvm.set_option('', '--debug-only=loop-vectorize')


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def csr_spmv_multi_thread(y, num_row, rowptr, colidx, val, x):
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

    for i in numba.prange(num_row):
        row_data = 0.0
        for j in range(rowptr[i], rowptr[i + 1]):
            row_data += val[j] * x[colidx[j]]
        y[i] = row_data

    return y


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def sliced_ellpack_spmv(slice_count, slice_ptr,
                        colidx, val, x, slice_height):
    """
    This function is multi thread Sliced ELLPACK format based SpMV (y=Ax).

    Parameters
    ----------
    y : ndarrays
        Row pointer of CSR format
    slice_count : ndarrays
        Number of slices
    slice_ptr : ndarrays
        Slice pointer of Sliced ELLPACK format
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

    y = np.zeros(slice_count * slice_height, dtype='float32')

    for s in numba.prange(slice_count):
        for r in range(slice_height):
            row_data = 0.0
            for i in range(slice_ptr[s] + r, slice_ptr[s + 1], slice_height):
                row_data += x[colidx[i]] * val[i]
            y[s * slice_height + r] = row_data

    return y


@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def array_parallel_test(a, b):
    c = 0.
    for i in numba.prange(a.shape[0]):
        c += a[i] * b[i]

    return c
