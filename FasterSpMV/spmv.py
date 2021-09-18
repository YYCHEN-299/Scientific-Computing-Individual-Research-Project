import time

import numpy as np

from numba import cuda

from FasterSpMV.matrix_tools import *
from FasterSpMV.numba_spmv import numba_csr_spmv, numba_sliced_ellpack_spmv
from FasterSpMV.opencl_spmv import BaseSELLSpMV, SELLSpMV, CSRSpMV
from FasterSpMV.cuda_spmv import cuda_csr_spmv, cuda_sliced_ellpack_spmv_1d


class SpMVOperator:
    def __init__(self, method, format, n_row, n_col,
                 csr_rowptr, csr_colidx, csr_val, slice_height):
        self.n_row = n_row
        self.n_col = n_col
        self.csr_rowptr = csr_rowptr
        self.csr_colidx = csr_colidx
        self.csr_val = csr_val
        self.slice_height = slice_height

        if format == 'csr':
            if method == 'opencl':
                self._ocl_csr = CSRSpMV(n_row, csr_rowptr, csr_colidx, csr_val)
                self.kernel = self._opencl_csr
            elif method == 'numba':
                self.y = np.zeros(n_row, dtype=np.float32)
                self.kernel = self._numba_csr
            elif method == 'cuda':
                self.nblocks = (n_row,)  # global blocks
                self.nthreads = (1,)  # threads per block
                # CUDA buffer
                self.bf_csr_rowptr = cuda.to_device(csr_rowptr)
                self.bf_csr_colidx = cuda.to_device(csr_colidx)
                self.bf_csr_val = cuda.to_device(csr_val)
                self.bf_csr_y = cuda.device_array(n_row, dtype=np.float32)
                self.kernel = self._cuda_csr
        elif format == 'sell':
            slice_count, ell_colidx, ell_sliceptr, slice_col, ell_val = \
                csr_to_sellpack(n_row, csr_rowptr,
                                csr_colidx, csr_val, slice_height)
            self.slice_count = slice_count
            self.ell_colidx = ell_colidx
            self.ell_sliceptr = ell_sliceptr
            self.slice_col = slice_col
            self.ell_val = ell_val
            if method == 'opencl':
                self._ocl_sell = BaseSELLSpMV(n_row, slice_count,
                                              ell_sliceptr,
                                              slice_col, ell_colidx,
                                              ell_val, slice_height)
                self.kernel = self._opencl_sell
            elif method == 'numba':
                self.y = np.zeros(slice_count *
                                  slice_height, dtype=np.float32)
                self.kernel = self._numba_sell
            elif method == 'cuda':
                self.nblocks = (slice_count,)  # global blocks
                self.nthreads = (slice_height,)  # threads per block
                # CUDA buffer
                self.bf_sell_y = cuda.device_array(slice_height *
                                                   slice_count,
                                                   dtype=np.float32)
                self.bf_ell_sliceptr = cuda.to_device(ell_sliceptr)
                self.bf_ell_colidx = cuda.to_device(ell_colidx)
                self.bf_ell_val = cuda.to_device(ell_val)
                self.kernel = self._cuda_sell

        # first run kernel
        x = np.ones(n_col, dtype=np.float32)
        self.kernel(x)

    def run(self, v):
        y = self.kernel(v)
        return y

    def _numba_csr(self, v):
        return numba_csr_spmv(self.y, self.n_row, self.csr_rowptr,
                              self.csr_colidx, self.csr_val, v)

    def _numba_sell(self, v):
        return numba_sliced_ellpack_spmv(self.y, self.slice_count,
                                         self.ell_sliceptr, self.slice_col,
                                         self.ell_colidx, self.ell_val,
                                         v, self.slice_height)

    def _opencl_csr(self, v):
        return self._ocl_csr.run(v)

    def _opencl_sell(self, v):
        return self._ocl_sell.run(v)

    def _cuda_csr(self, v):
        bf_x = cuda.to_device(v)
        cuda_csr_spmv[self.nblocks, self.nthreads](self.bf_csr_rowptr,
                                                   self.bf_csr_colidx,
                                                   self.bf_csr_val,
                                                   bf_x, self.bf_csr_y)
        output_csr_y = self.bf_csr_y.copy_to_host()
        return output_csr_y

    def _cuda_sell(self, v):
        bf_x = cuda.to_device(v)
        cuda_sliced_ellpack_spmv_1d[self.nblocks,
                                    self.nthreads](self.bf_ell_sliceptr,
                                                self.bf_ell_colidx,
                                                self.bf_ell_val, bf_x,
                                                self.slice_height,
                                                self.bf_sell_y)
        output_sell_y = self.bf_sell_y.copy_to_host()
        return output_sell_y
