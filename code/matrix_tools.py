import math
import random

import numpy as np


def random_spmatrix(n_row, n_col, per_nnz):
    """
    This function output a random sparse matrix.

    Parameters
    ----------
    n_row : int
        Number of rows
    n_col : int
        Number of columns
    per_nnz : int
        Percentage of none zero elements

    Returns
    -------
    sp_matrix : list
        Sparse matrix without any storage format
    nnz_count : int
        Total none zero elements number
    row_max_nnz : int
        Row wise max none zero elements number

    Examples
    --------
    >>>
    """

    sp_matrix = []
    nnz_count = 0
    row_max_nnz = 0

    for i in range(n_row):
        row_data = []
        row_nnz_count = 0
        for j in range(n_col):
            r_val = random.randint(0, 100)
            if r_val < per_nnz:
                row_data.append(0)
            else:
                nnz_count += 1
                row_nnz_count += 1
                row_data.append(r_val)
        row_max_nnz = max(row_max_nnz, row_nnz_count)
        sp_matrix.append(row_data)

    return sp_matrix, nnz_count, row_max_nnz


def spmatrix_to_csr(sp_matrix):
    """
    This function convert sparse matrix to CSR format.

    Parameters
    ----------
    sp_matrix : list or ndarrays
        Sparse matrix without any storage format

    Returns
    -------
    rowptr : ndarrays
        Row pointer of CSR format
    colidx : ndarrays
        Column index of CSR format
    val : ndarrays
        None zero elements value of CSR format

    Examples
    --------
    >>>
    """

    n_row = len(sp_matrix)
    n_col = len(sp_matrix[0])

    rowptr = []
    colidx = []
    val = []
    nnz_count = 0

    for i in range(n_row):
        rowptr.append(nnz_count)
        for j in range(n_col):
            if sp_matrix[i][j] != 0:
                nnz_count += 1
                colidx.append(j)
                val.append(sp_matrix[i][j])
    rowptr.append(nnz_count)

    return np.array(rowptr), np.array(colidx), np.array(val, dtype='float32')


def csr_to_sellpack(rowptr, colidx, val, slice_height):
    """
    This fucntion convert CSR format to Sliced ELLPACK format.

    Parameters
    ----------
    rowptr : ndarrays
        Row pointer of CSR format
    colidx : ndarrays
        Column index of CSR format
    val : ndarrays
        None zero elements value of CSR format
    slice_height : int
        Slice height of Sliced ELLPACK format

    Returns
    -------
    ell_colidx : ndarrays
        Column index of Sliced ELLPACK format
    ell_sliceptr : ndarrays
        Slice pointer of Sliced ELLPACK format
    ell_val : ndarrays
        None zero elements value of CSR format

    Examples
    --------
    >>>
    """
    # TODO check code and fix bug

    N = len(rowptr) - 1  # number of rows
    slice_number = math.floor(N / slice_height)  # number of slices
    nnz_count = 0

    ell_colidx = []
    ell_sliceptr = []
    ell_val = []

    for i in range(slice_number):
        max_nnz = 0
        for s in range(slice_height):
            col_count = rowptr[i * slice_height + s + 1] - \
                        rowptr[i * slice_height + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        for j in range(max_nnz):
            for k in range(slice_height):
                idx = i * slice_height + k  # row index
                now_ptr = rowptr[idx]  # start index of this row
                next_ptr = rowptr[idx + 1]  # start index of next row
                nnz_count += 1  # count non-zero number
                if now_ptr + j < next_ptr:
                    ell_colidx.append(colidx[now_ptr + j])
                    ell_val.append(val[now_ptr + j])
                else:
                    ell_colidx.append(0)
                    ell_val.append(0)  # padded zero

    if N % slice_height != 0:  # if have remainder
        now_row = slice_number * slice_height
        remain_rows = N - now_row
        max_nnz = 0
        for s in range(remain_rows):
            col_count = rowptr[now_row + s + 1] - rowptr[now_row + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        for j in range(max_nnz):  # column
            for k in range(slice_height):  # row
                nnz_count += 1  # count non-zero number
                if k >= remain_rows:
                    ell_colidx.append(0)
                    ell_val.append(0)  # padded zero
                else:
                    idx = now_row + k  # row index
                    now_ptr = rowptr[idx]  # start index of this row
                    next_ptr = rowptr[idx + 1]  # start index of next row
                    if now_ptr + j < next_ptr:
                        ell_colidx.append(colidx[now_ptr + j])
                        ell_val.append(val[now_ptr + j])
                    else:
                        ell_colidx.append(0)
                        ell_val.append(0)  # padded zero
    ell_sliceptr.append(nnz_count)

    return np.array(ell_colidx), np.array(
        ell_sliceptr), np.array(ell_val, dtype='float32')
