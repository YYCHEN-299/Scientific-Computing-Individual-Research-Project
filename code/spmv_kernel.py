import numpy as np

import numba
from numba import jit


@jit(nopython=True, parallel=True, nogil=True)
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


@jit(nopython=True, parallel=True, nogil=True)
def sliced_ellpack_spmv_multi_thread(y, slice_count, slice_ptr,
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

    for s in numba.prange(slice_count):
        for r in numba.prange(slice_height):
            row_data = 0.0
            for i in range(slice_ptr[s] + r, slice_ptr[s + 1], slice_height):
                rd = x[colidx[i]] * val[i]
                row_data += rd
            y[s * slice_height + r] = row_data

    return y
