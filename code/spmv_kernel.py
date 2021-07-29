import numpy as np

import numba
from numba import jit


def csr_spmv_single_thread(rowptr, colidx, val, x):
    """
    This function is single thread CSR format based SpMV (y=Ax).

    Parameters
    ----------
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

    num_row = len(rowptr) - 1
    y = np.zeros(num_row, dtype=np.float32)

    for i in range(num_row):
        for j in range(rowptr[i], rowptr[i + 1]):
            y[i] += val[j] * x[colidx[j]]

    return y


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def csr_spmv_multi_thread(y, num_row, rowptr, colidx, val, x):
    """
    This function is multi thread CSR format based SpMV (y=Ax).

    Parameters
    ----------
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


def sliced_ellpack_spmv_single_thread(N, slice_ptr, colidx,
                                      val, x, slice_height):
    """
    This function is multi thread Sliced ELLPACK format based SpMV (y=Ax).

    Parameters
    ----------
    N : int
        Number of rows
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

    y = np.zeros(N, dtype=np.float32)
    slice_count = int(N / slice_height)

    for s in range(slice_count):
        row_idx = s * slice_height
        for r in range(row_idx, row_idx + slice_height):
            for i in range(slice_ptr[s] + r - row_idx,
                           slice_ptr[s + 1], slice_height):
                Ax_data = x[colidx[i]] * val[i]
                y[r] += Ax_data

    return y


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
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
        row_idx = s * slice_height
        for r in range(slice_height):
            for i in range(slice_ptr[s] + r, slice_ptr[s + 1], slice_height):
                y[row_idx + r] += x[colidx[i]] * val[i]

    return y