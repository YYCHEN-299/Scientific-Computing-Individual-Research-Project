import math
import random
import itertools

import numpy as np
from pyopencl import cltypes


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
    >>> n_row = 5
    >>> n_col = 5
    >>> per_nnz = 10
    >>> random_spmatrix(n_row, n_col, per_nnz)
    >>> # return a list fill with random data and 10% zero
    """

    if n_row < 0 or n_col < 0:
        raise ValueError('The number of rows or columns must > 0')

    if per_nnz < 0 or per_nnz > 100:
        raise ValueError('The percentage of nonzeros must between 0 - 100')

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
    sp_matrix : list
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
    >>> sp_matrix = [[0, 1],[2, 3]]
    >>> spmatrix_to_csr(sp_matrix)
    >>> rowptr
        [0, 1]
    >>> colidx
        [1, 0, 1]
    >>> val
        [1, 2, 3]
    """

    if not isinstance(sp_matrix, list):
        raise ValueError('The input matrix must be a list')

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

    return np.array(rowptr, dtype=np.uint32), \
        np.array(colidx, dtype=np.uint32), \
        np.array(val, dtype=np.float32)


def csr_to_sell(n_row, rowptr, colidx, val, slice_height):
    """
    This fucntion convert CSR format to Sliced ELLPACK format.

    Parameters
    ----------
    n_row : int
        Number of rows
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
    slice_count : int
        Number of slices
    ell_colidx : ndarrays
        Column index of Sliced ELLPACK format
    ell_sliceptr : ndarrays
        Slice pointer of Sliced ELLPACK format
    ell_slicecol : ndarrays
        Column length of a slice
    ell_val : ndarrays
        None zero elements value of CSR format
    """

    if slice_height < 0:
        raise ValueError('The slice height must > 0')

    N = len(rowptr) - 1  # number of rows
    slice_number = math.floor(N / slice_height)  # number of slices
    nnz_count = 0

    ell_colidx = []
    ell_sliceptr = []
    ell_slicecol = []
    ell_val = []

    for i in range(slice_number):
        max_nnz = 0
        for s in range(slice_height):
            col_count = rowptr[i * slice_height + s + 1] - \
                        rowptr[i * slice_height + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        ell_slicecol.append(max_nnz)
        pre_idx = 0
        for j in range(max_nnz):  # column scan
            for k in range(slice_height):  # row scan
                idx = i * slice_height + k  # row index
                now_ptr = rowptr[idx]  # start index of this row
                next_ptr = rowptr[idx + 1]  # start index of next row
                nnz_count += 1  # count non-zero number
                if now_ptr + j < next_ptr:
                    pre_idx = colidx[now_ptr + j]
                    ell_colidx.append(colidx[now_ptr + j])
                    ell_val.append(val[now_ptr + j])
                else:
                    ell_colidx.append(pre_idx)
                    ell_val.append(0)  # padded zero

    if N % slice_height != 0:  # if have remainder
        now_row = slice_number * slice_height
        remain_rows = N - now_row
        max_nnz = 0
        for s in range(remain_rows):
            col_count = rowptr[now_row + s + 1] - rowptr[now_row + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        ell_slicecol.append(max_nnz)
        pre_idx = 0
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
                        pre_idx = colidx[now_ptr + j]
                        ell_colidx.append(colidx[now_ptr + j])
                        ell_val.append(val[now_ptr + j])
                    else:
                        ell_colidx.append(pre_idx)
                        ell_val.append(0)  # padded zero
    ell_sliceptr.append(nnz_count)

    slice_count = math.ceil(n_row / slice_height)
    return slice_count, \
        np.array(ell_colidx, dtype=np.int32), \
        np.array(ell_sliceptr, dtype=np.int32), \
        np.array(ell_slicecol, dtype=np.int32), \
        np.array(ell_val, dtype=np.float32)


def csr_to_ocl_sell2(n_row, rowptr, colidx, val):
    """
    Convert CSR format to Sliced ELLPACK format and slice height = 2.

    Parameters
    ----------
    n_row : int
        Number of rows
    rowptr : ndarrays
        Row pointer of CSR format
    colidx : ndarrays
        Column index of CSR format
    val : ndarrays
        None zero elements value of CSR format

    Returns
    -------
    slice_count : int
        Number of slices
    ell_colidx : ndarrays
        Column index of Sliced ELLPACK format
    ell_sliceptr : ndarrays
        Slice pointer of Sliced ELLPACK format
    ell_slicecol : ndarrays
        Column length of a slice
    ell_val : ndarrays
        None zero elements value of CSR format
    """

    slice_height = 2
    slice_number = math.floor(n_row / slice_height)  # number of full slices
    slice_count = math.ceil(n_row / slice_height)  # real number of slices
    nnz_count = 0
    total_col_count = 0

    ell_colidx = []
    ell_sliceptr = []
    ell_slicecol = [0]
    ell_val = []

    for i in range(slice_number):
        max_nnz = 0
        for s in range(slice_height):
            col_count = rowptr[i * slice_height + s + 1] - \
                        rowptr[i * slice_height + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        total_col_count += max_nnz
        ell_slicecol.append(total_col_count)
        pre_idx = 0
        for j in range(max_nnz):  # column scan
            slice_row_val = []
            slice_row_colidx = []
            for k in range(slice_height):  # row scan
                idx = i * slice_height + k  # row index
                now_ptr = rowptr[idx]  # start index of this row
                next_ptr = rowptr[idx + 1]  # start index of next row
                nnz_count += 1  # count non-zero number
                if now_ptr + j < next_ptr:
                    pre_idx = colidx[now_ptr + j]
                    slice_row_colidx.append(colidx[now_ptr + j])
                    slice_row_val.append(val[now_ptr + j])
                else:
                    slice_row_colidx.append(pre_idx)
                    slice_row_val.append(0)  # padded zero

            # convert to vector int
            int2_slice_row_colidx = cltypes.make_int2(slice_row_colidx[0],
                                                      slice_row_colidx[1])

            # convert to vector float
            float2_slice_row_val = cltypes.make_float2(slice_row_val[0],
                                                       slice_row_val[1])
            ell_colidx.append(int2_slice_row_colidx)
            ell_val.append(float2_slice_row_val)

    if n_row % slice_height != 0:  # if have remainder
        now_row = slice_number * slice_height
        remain_rows = n_row - now_row
        max_nnz = 0
        for s in range(remain_rows):
            col_count = rowptr[now_row + s + 1] - rowptr[now_row + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        total_col_count += max_nnz
        ell_slicecol.append(total_col_count)
        pre_idx = 0
        for j in range(max_nnz):  # column
            slice_row_val = []
            slice_row_colidx = []
            for k in range(slice_height):  # row
                nnz_count += 1  # count non-zero number
                if k >= remain_rows:
                    slice_row_colidx.append(0)
                    slice_row_val.append(0)  # padded zero
                else:
                    idx = now_row + k  # row index
                    now_ptr = rowptr[idx]  # start index of this row
                    next_ptr = rowptr[idx + 1]  # start index of next row
                    if now_ptr + j < next_ptr:
                        pre_idx = colidx[now_ptr + j]
                        slice_row_colidx.append(colidx[now_ptr + j])
                        slice_row_val.append(val[now_ptr + j])
                    else:
                        slice_row_colidx.append(pre_idx)
                        slice_row_val.append(0)  # padded zero

            # convert to vector int
            int2_slice_row_colidx = cltypes.make_int2(slice_row_colidx[0],
                                                      slice_row_colidx[1])

            # convert to vector float
            float2_slice_row_val = cltypes.make_float2(slice_row_val[0],
                                                       slice_row_val[1])
            ell_colidx.append(int2_slice_row_colidx)
            ell_val.append(float2_slice_row_val)

    ell_sliceptr.append(nnz_count)
    return slice_count, \
        np.array(ell_colidx), \
        np.array(ell_sliceptr, dtype=np.int32), \
        np.array(ell_slicecol, dtype=np.int32), \
        np.array(ell_val)


def csr_to_ocl_sell4(n_row, rowptr, colidx, val):
    """
    Convert CSR format to Sliced ELLPACK format and slice height = 4.

    Parameters
    ----------
    n_row : int
        Number of rows
    rowptr : ndarrays
        Row pointer of CSR format
    colidx : ndarrays
        Column index of CSR format
    val : ndarrays
        None zero elements value of CSR format

    Returns
    -------
    slice_count : int
        Number of slices
    ell_colidx : ndarrays
        Column index of Sliced ELLPACK format
    ell_sliceptr : ndarrays
        Slice pointer of Sliced ELLPACK format
    ell_slicecol : ndarrays
        Column length of a slice
    ell_val : ndarrays
        None zero elements value of CSR format
    """

    slice_height = 4
    slice_number = math.floor(n_row / slice_height)  # number of full slices
    slice_count = math.ceil(n_row / slice_height)  # real number of slices
    nnz_count = 0
    total_col_count = 0

    ell_colidx = []
    ell_sliceptr = []
    ell_slicecol = [0]
    ell_val = []

    for i in range(slice_number):
        max_nnz = 0
        for s in range(slice_height):
            col_count = rowptr[i * slice_height + s + 1] - \
                        rowptr[i * slice_height + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        total_col_count += max_nnz
        ell_slicecol.append(total_col_count)
        pre_idx = 0
        for j in range(max_nnz):  # column scan
            slice_row_val = []
            slice_row_colidx = []
            for k in range(slice_height):  # row scan
                idx = i * slice_height + k  # row index
                now_ptr = rowptr[idx]  # start index of this row
                next_ptr = rowptr[idx + 1]  # start index of next row
                nnz_count += 1  # count non-zero number
                if now_ptr + j < next_ptr:
                    pre_idx = colidx[now_ptr + j]
                    slice_row_colidx.append(colidx[now_ptr + j])
                    slice_row_val.append(val[now_ptr + j])
                else:
                    slice_row_colidx.append(pre_idx)
                    slice_row_val.append(0)  # padded zero

            # convert to vector int
            int4_slice_row_colidx = cltypes.make_int4(slice_row_colidx[0],
                                                      slice_row_colidx[1],
                                                      slice_row_colidx[2],
                                                      slice_row_colidx[3])

            # convert to vector float
            float4_slice_row_val = cltypes.make_float4(slice_row_val[0],
                                                       slice_row_val[1],
                                                       slice_row_val[2],
                                                       slice_row_val[3])
            ell_colidx.append(int4_slice_row_colidx)
            ell_val.append(float4_slice_row_val)

    if n_row % slice_height != 0:  # if have remainder
        now_row = slice_number * slice_height
        remain_rows = n_row - now_row
        max_nnz = 0
        for s in range(remain_rows):
            col_count = rowptr[now_row + s + 1] - rowptr[now_row + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        total_col_count += max_nnz
        ell_slicecol.append(total_col_count)
        pre_idx = 0
        for j in range(max_nnz):  # column
            slice_row_val = []
            slice_row_colidx = []
            for k in range(slice_height):  # row
                nnz_count += 1  # count non-zero number
                if k >= remain_rows:
                    slice_row_colidx.append(0)
                    slice_row_val.append(0)  # padded zero
                else:
                    idx = now_row + k  # row index
                    now_ptr = rowptr[idx]  # start index of this row
                    next_ptr = rowptr[idx + 1]  # start index of next row
                    if now_ptr + j < next_ptr:
                        pre_idx = colidx[now_ptr + j]
                        slice_row_colidx.append(colidx[now_ptr + j])
                        slice_row_val.append(val[now_ptr + j])
                    else:
                        slice_row_colidx.append(pre_idx)
                        slice_row_val.append(0)  # padded zero

            # convert to vector int
            int4_slice_row_colidx = cltypes.make_int4(slice_row_colidx[0],
                                                      slice_row_colidx[1],
                                                      slice_row_colidx[2],
                                                      slice_row_colidx[3])

            # convert to vector float
            float4_slice_row_val = cltypes.make_float4(slice_row_val[0],
                                                       slice_row_val[1],
                                                       slice_row_val[2],
                                                       slice_row_val[3])
            ell_colidx.append(int4_slice_row_colidx)
            ell_val.append(float4_slice_row_val)

    ell_sliceptr.append(nnz_count)
    return slice_count, \
        np.array(ell_colidx), \
        np.array(ell_sliceptr, dtype=np.int32), \
        np.array(ell_slicecol, dtype=np.int32), \
        np.array(ell_val)


def csr_to_ocl_sell8(n_row, rowptr, colidx, val):
    """
    Convert CSR format to Sliced ELLPACK format and slice height = 8.

    Parameters
    ----------
    n_row : int
        Number of rows
    rowptr : ndarrays
        Row pointer of CSR format
    colidx : ndarrays
        Column index of CSR format
    val : ndarrays
        None zero elements value of CSR format

    Returns
    -------
    slice_count : int
        Number of slices
    ell_colidx : ndarrays
        Column index of Sliced ELLPACK format
    ell_sliceptr : ndarrays
        Slice pointer of Sliced ELLPACK format
    ell_slicecol : ndarrays
        Column length of a slice
    ell_val : ndarrays
        None zero elements value of CSR format
    """

    slice_height = 8
    slice_number = math.floor(n_row / slice_height)  # number of full slices
    slice_count = math.ceil(n_row / slice_height)  # real number of slices
    nnz_count = 0
    total_col_count = 0

    ell_colidx = []
    ell_sliceptr = []
    ell_slicecol = [0]
    ell_val = []

    for i in range(slice_number):
        max_nnz = 0
        for s in range(slice_height):
            col_count = rowptr[i * slice_height + s + 1] - \
                        rowptr[i * slice_height + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        total_col_count += max_nnz
        ell_slicecol.append(total_col_count)
        pre_idx = 0
        for j in range(max_nnz):  # column scan
            slice_row_val = []
            slice_row_colidx = []
            for k in range(slice_height):  # row scan
                idx = i * slice_height + k  # row index
                now_ptr = rowptr[idx]  # start index of this row
                next_ptr = rowptr[idx + 1]  # start index of next row
                nnz_count += 1  # count non-zero number
                if now_ptr + j < next_ptr:
                    pre_idx = colidx[now_ptr + j]
                    slice_row_colidx.append(colidx[now_ptr + j])
                    slice_row_val.append(val[now_ptr + j])
                else:
                    slice_row_colidx.append(pre_idx)
                    slice_row_val.append(0)  # padded zero

            # convert to vector int
            int8_slice_row_colidx = cltypes.make_int8(slice_row_colidx[0],
                                                      slice_row_colidx[1],
                                                      slice_row_colidx[2],
                                                      slice_row_colidx[3],
                                                      slice_row_colidx[4],
                                                      slice_row_colidx[5],
                                                      slice_row_colidx[6],
                                                      slice_row_colidx[7])

            # convert to vector float
            float8_slice_row_val = cltypes.make_float8(slice_row_val[0],
                                                       slice_row_val[1],
                                                       slice_row_val[2],
                                                       slice_row_val[3],
                                                       slice_row_val[4],
                                                       slice_row_val[5],
                                                       slice_row_val[6],
                                                       slice_row_val[7])
            ell_colidx.append(int8_slice_row_colidx)
            ell_val.append(float8_slice_row_val)

    if n_row % slice_height != 0:  # if have remainder
        now_row = slice_number * slice_height
        remain_rows = n_row - now_row
        max_nnz = 0
        for s in range(remain_rows):
            col_count = rowptr[now_row + s + 1] - rowptr[now_row + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        total_col_count += max_nnz
        ell_slicecol.append(total_col_count)
        pre_idx = 0
        for j in range(max_nnz):  # column
            slice_row_val = []
            slice_row_colidx = []
            for k in range(slice_height):  # row
                nnz_count += 1  # count non-zero number
                if k >= remain_rows:
                    slice_row_colidx.append(0)
                    slice_row_val.append(0)  # padded zero
                else:
                    idx = now_row + k  # row index
                    now_ptr = rowptr[idx]  # start index of this row
                    next_ptr = rowptr[idx + 1]  # start index of next row
                    if now_ptr + j < next_ptr:
                        pre_idx = colidx[now_ptr + j]
                        slice_row_colidx.append(colidx[now_ptr + j])
                        slice_row_val.append(val[now_ptr + j])
                    else:
                        slice_row_colidx.append(pre_idx)
                        slice_row_val.append(0)  # padded zero

            # convert to vector int
            int8_slice_row_colidx = cltypes.make_int8(slice_row_colidx[0],
                                                      slice_row_colidx[1],
                                                      slice_row_colidx[2],
                                                      slice_row_colidx[3],
                                                      slice_row_colidx[4],
                                                      slice_row_colidx[5],
                                                      slice_row_colidx[6],
                                                      slice_row_colidx[7])

            # convert to vector float
            float8_slice_row_val = cltypes.make_float8(slice_row_val[0],
                                                       slice_row_val[1],
                                                       slice_row_val[2],
                                                       slice_row_val[3],
                                                       slice_row_val[4],
                                                       slice_row_val[5],
                                                       slice_row_val[6],
                                                       slice_row_val[7])
            ell_colidx.append(int8_slice_row_colidx)
            ell_val.append(float8_slice_row_val)

    ell_sliceptr.append(nnz_count)
    return slice_count, \
        np.array(ell_colidx), \
        np.array(ell_sliceptr, dtype=np.int32), \
        np.array(ell_slicecol, dtype=np.int32), \
        np.array(ell_val)


def csr_to_ocl_sell16(n_row, rowptr, colidx, val):
    """
    Convert CSR format to Sliced ELLPACK format and slice height = 16.

    Parameters
    ----------
    n_row : int
        Number of rows
    rowptr : ndarrays
        Row pointer of CSR format
    colidx : ndarrays
        Column index of CSR format
    val : ndarrays
        None zero elements value of CSR format

    Returns
    -------
    slice_count : int
        Number of slices
    ell_colidx : ndarrays
        Column index of Sliced ELLPACK format
    ell_sliceptr : ndarrays
        Slice pointer of Sliced ELLPACK format
    ell_slicecol : ndarrays
        Column length of a slice
    ell_val : ndarrays
        None zero elements value of CSR format
    """

    slice_height = 16
    slice_number = math.floor(n_row / slice_height)  # number of full slices
    slice_count = math.ceil(n_row / slice_height)  # real number of slices
    nnz_count = 0
    total_col_count = 0

    ell_colidx = []
    ell_sliceptr = []
    ell_slicecol = [0]
    ell_val = []

    for i in range(slice_number):
        max_nnz = 0
        for s in range(slice_height):
            col_count = rowptr[i * slice_height + s + 1] - \
                        rowptr[i * slice_height + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        total_col_count += max_nnz
        ell_slicecol.append(total_col_count)
        pre_idx = 0
        for j in range(max_nnz):  # column scan
            slice_row_val = []
            slice_row_colidx = []
            for k in range(slice_height):  # row scan
                idx = i * slice_height + k  # row index
                now_ptr = rowptr[idx]  # start index of this row
                next_ptr = rowptr[idx + 1]  # start index of next row
                nnz_count += 1  # count non-zero number
                if now_ptr + j < next_ptr:
                    pre_idx = colidx[now_ptr + j]
                    slice_row_colidx.append(colidx[now_ptr + j])
                    slice_row_val.append(val[now_ptr + j])
                else:
                    slice_row_colidx.append(pre_idx)
                    slice_row_val.append(0)  # padded zero

            # convert to vector int
            int16_slice_row_colidx = cltypes.make_int16(slice_row_colidx[0],
                                                        slice_row_colidx[1],
                                                        slice_row_colidx[2],
                                                        slice_row_colidx[3],
                                                        slice_row_colidx[4],
                                                        slice_row_colidx[5],
                                                        slice_row_colidx[6],
                                                        slice_row_colidx[7],
                                                        slice_row_colidx[8],
                                                        slice_row_colidx[9],
                                                        slice_row_colidx[10],
                                                        slice_row_colidx[11],
                                                        slice_row_colidx[12],
                                                        slice_row_colidx[13],
                                                        slice_row_colidx[14],
                                                        slice_row_colidx[15])

            # convert to vector float
            float16_slice_row_val = cltypes.make_float16(slice_row_val[0],
                                                         slice_row_val[1],
                                                         slice_row_val[2],
                                                         slice_row_val[3],
                                                         slice_row_val[4],
                                                         slice_row_val[5],
                                                         slice_row_val[6],
                                                         slice_row_val[7],
                                                         slice_row_val[8],
                                                         slice_row_val[9],
                                                         slice_row_val[10],
                                                         slice_row_val[11],
                                                         slice_row_val[12],
                                                         slice_row_val[13],
                                                         slice_row_val[14],
                                                         slice_row_val[15])
            ell_colidx.append(int16_slice_row_colidx)
            ell_val.append(float16_slice_row_val)

    if n_row % slice_height != 0:  # if have remainder
        now_row = slice_number * slice_height
        remain_rows = n_row - now_row
        max_nnz = 0
        for s in range(remain_rows):
            col_count = rowptr[now_row + s + 1] - rowptr[now_row + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        total_col_count += max_nnz
        ell_slicecol.append(total_col_count)
        pre_idx = 0
        for j in range(max_nnz):  # column
            slice_row_val = []
            slice_row_colidx = []
            for k in range(slice_height):  # row
                nnz_count += 1  # count non-zero number
                if k >= remain_rows:
                    slice_row_colidx.append(0)
                    slice_row_val.append(0)  # padded zero
                else:
                    idx = now_row + k  # row index
                    now_ptr = rowptr[idx]  # start index of this row
                    next_ptr = rowptr[idx + 1]  # start index of next row
                    if now_ptr + j < next_ptr:
                        pre_idx = colidx[now_ptr + j]
                        slice_row_colidx.append(colidx[now_ptr + j])
                        slice_row_val.append(val[now_ptr + j])
                    else:
                        slice_row_colidx.append(pre_idx)
                        slice_row_val.append(0)  # padded zero

            # convert to vector int
            int16_slice_row_colidx = cltypes.make_int16(slice_row_colidx[0],
                                                        slice_row_colidx[1],
                                                        slice_row_colidx[2],
                                                        slice_row_colidx[3],
                                                        slice_row_colidx[4],
                                                        slice_row_colidx[5],
                                                        slice_row_colidx[6],
                                                        slice_row_colidx[7],
                                                        slice_row_colidx[8],
                                                        slice_row_colidx[9],
                                                        slice_row_colidx[10],
                                                        slice_row_colidx[11],
                                                        slice_row_colidx[12],
                                                        slice_row_colidx[13],
                                                        slice_row_colidx[14],
                                                        slice_row_colidx[15])

            # convert to vector float
            float16_slice_row_val = cltypes.make_float16(slice_row_val[0],
                                                         slice_row_val[1],
                                                         slice_row_val[2],
                                                         slice_row_val[3],
                                                         slice_row_val[4],
                                                         slice_row_val[5],
                                                         slice_row_val[6],
                                                         slice_row_val[7],
                                                         slice_row_val[8],
                                                         slice_row_val[9],
                                                         slice_row_val[10],
                                                         slice_row_val[11],
                                                         slice_row_val[12],
                                                         slice_row_val[13],
                                                         slice_row_val[14],
                                                         slice_row_val[15])
            ell_colidx.append(int16_slice_row_colidx)
            ell_val.append(float16_slice_row_val)

    ell_sliceptr.append(nnz_count)
    return slice_count, \
        np.array(ell_colidx), \
        np.array(ell_sliceptr, dtype=np.int32), \
        np.array(ell_slicecol, dtype=np.int32), \
        np.array(ell_val)


def csr_to_2d_sell(n_row, rowptr, colidx, val, slice_height):
    """
    Convert CSR format to Sliced ELLPACK format.
    The colidx and val will be stored as 2d array.

    Parameters
    ----------
    n_row : int
        Number of rows
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
    slice_count : int
        Number of slices
    ell_colidx : ndarrays
        Column index of Sliced ELLPACK format
    ell_sliceptr : ndarrays
        Slice pointer of Sliced ELLPACK format
    ell_slicecol : ndarrays
        Column length of a slice
    ell_val : ndarrays
        None zero elements value of CSR format
    """

    if slice_height < 0:
        raise ValueError('The slice height must > 0')

    slice_number = math.floor(n_row / slice_height)  # number of full slices
    slice_count = math.ceil(n_row / slice_height)  # real number of slices
    nnz_count = 0
    total_col_count = 0

    ell_colidx = []
    ell_sliceptr = []
    ell_slicecol = [0]
    ell_val = []

    for i in range(slice_number):
        max_nnz = 0
        for s in range(slice_height):
            col_count = rowptr[i * slice_height + s + 1] - \
                        rowptr[i * slice_height + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        total_col_count += max_nnz
        ell_slicecol.append(total_col_count)
        pre_idx = 0
        for j in range(max_nnz):  # column scan
            slice_row_val = []
            slice_row_colidx = []
            for k in range(slice_height):  # row scan
                idx = i * slice_height + k  # row index
                now_ptr = rowptr[idx]  # start index of this row
                next_ptr = rowptr[idx + 1]  # start index of next row
                nnz_count += 1  # count non-zero number
                if now_ptr + j < next_ptr:
                    pre_idx = colidx[now_ptr + j]
                    slice_row_colidx.append(colidx[now_ptr + j])
                    slice_row_val.append(val[now_ptr + j])
                else:
                    slice_row_colidx.append(pre_idx)
                    slice_row_val.append(0)  # padded zero
            ell_colidx.append(slice_row_colidx)
            ell_val.append(slice_row_val)

    if n_row % slice_height != 0:  # if have remainder
        now_row = slice_number * slice_height
        remain_rows = n_row - now_row
        max_nnz = 0
        for s in range(remain_rows):
            col_count = rowptr[now_row + s + 1] - rowptr[now_row + s]
            max_nnz = max(max_nnz, col_count)

        ell_sliceptr.append(nnz_count)
        total_col_count += max_nnz
        ell_slicecol.append(total_col_count)
        pre_idx = 0
        for j in range(max_nnz):  # column
            slice_row_val = []
            slice_row_colidx = []
            for k in range(slice_height):  # row
                nnz_count += 1  # count non-zero number
                if k >= remain_rows:
                    slice_row_colidx.append(0)
                    slice_row_val.append(0)  # padded zero
                else:
                    idx = now_row + k  # row index
                    now_ptr = rowptr[idx]  # start index of this row
                    next_ptr = rowptr[idx + 1]  # start index of next row
                    if now_ptr + j < next_ptr:
                        pre_idx = colidx[now_ptr + j]
                        slice_row_colidx.append(colidx[now_ptr + j])
                        slice_row_val.append(val[now_ptr + j])
                    else:
                        slice_row_colidx.append(pre_idx)
                        slice_row_val.append(0)  # padded zero
            ell_colidx.append(slice_row_colidx)
            ell_val.append(slice_row_val)

    ell_sliceptr.append(nnz_count)
    return slice_count, \
        np.array(ell_colidx, dtype=np.int32), \
        np.array(ell_sliceptr, dtype=np.int32), \
        np.array(ell_slicecol, dtype=np.int32), \
        np.array(ell_val, dtype=np.float32)


def adapt_row_th(data, row_th):
    max_len = max([len(i) for i in data])
    flag = max_len % row_th
    if flag:  # if not 0
        max_len += row_th - flag

    return [l + [0, ] * (max_len - len(l)) for l in data]


def get_col_list(data, row_th):
    gp = [zip(*[iter(i)]*row_th) for i in data]
    col_list = zip(*gp)

    return list(itertools.chain.from_iterable(
        itertools.chain.from_iterable(col_list)))


def csr_to_sell_rd(m, row_th, slice_height):
    """
    Convert CSR to Sliced ELLPACK for parallel reduction.

    Parameters
    ----------
    m : csr matrix
        Sparse matrix in SciPy CSR format
    row_th : int
        Number of Threads to calculate a row
    slice_height : int
        Slice height of Sliced ELLPACK

    Returns
    -------
    val : ndarrays
        Value array
    colidx : ndarrays
        Column index
    row_len : ndarrays
        Column length of each row
    slice_ptr : ndarrays
        Slice pointer
    """

    if slice_height < 0:
        raise ValueError('The slice height must > 0')

    align = int(slice_height * row_th)
    row_len = []
    val = []
    colidx = []
    for idx in range(m.shape[0]):
        row = m.getrow(idx)
        csr_val = row.data.tolist()
        csr_colidx = row.indices.tolist()
        col_count = len(csr_val)
        val.append(csr_val)
        colidx.append(csr_colidx)
        row_len.append(col_count)

    row_num = len(row_len)
    gap = row_num % slice_height
    if not gap == 0:
        add_num = slice_height - gap
        val.extend([[]] * add_num)
        colidx.extend([[]] * add_num)
        row_len.extend([0] * add_num)

    output_val = []
    output_colidx = []
    slice_ptr = [0]
    for gr in zip(*[iter(val)]*slice_height):
        gr = adapt_row_th(gr, row_th)
        output_val.extend(get_col_list(gr, row_th))
        slice_ptr.append(len(output_val))

    for gr in zip(*[iter(colidx)]*slice_height):
        gr = adapt_row_th(gr, row_th)
        output_colidx.extend(get_col_list(gr, row_th))

    shift = 0
    for idx in range(1, len(slice_ptr)):
        slice_ptr[idx] += shift
        ptr1 = slice_ptr[idx - 1]
        ptr2 = slice_ptr[idx]
        gap = align - (ptr2 - ptr1)
        if gap < 0:
            continue
        shift += gap
        for _ in range(gap):
            output_val.insert(ptr2, 0)
            output_colidx.insert(ptr2, 0)
            slice_ptr[idx] += 1

    return np.array(output_val, dtype=np.float32), \
        np.array(output_colidx, dtype=np.int32), \
        np.array(row_len, dtype=np.int32), \
        np.array(slice_ptr, dtype=np.int32)
