import time

import numpy as np
import pyopencl as cl

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
        x = np.ones(n_col, dtype=np.float32)
        if format == 'csr':
            if method == 'opencl':
                self.ctx = cl.create_some_context()
                self.queue = cl.CommandQueue(self.ctx)
                # read in the OpenCL source file as a string
                f = open('FasterSpMV/clkernels/ClCSRKernel.cl', 'r')
                fstr = ''.join(f.readlines())
                # create the program
                self.program = cl.Program(self.ctx, fstr).build()
                # create OpenCL buffers
                mf = cl.mem_flags
                self._y = np.zeros(self.n_row, dtype=np.float32)
                self.rowptr_buf = cl.Buffer(self.ctx,
                                            mf.READ_ONLY | mf.COPY_HOST_PTR,
                                            hostbuf=csr_rowptr)
                self.colidx_buf = cl.Buffer(self.ctx,
                                            mf.READ_ONLY | mf.COPY_HOST_PTR,
                                            hostbuf=csr_colidx)
                self.val_buf = cl.Buffer(self.ctx,
                                         mf.READ_ONLY | mf.COPY_HOST_PTR,
                                         hostbuf=csr_val)
                self.opencl_csr(x)
            elif method == 'numba':
                self.numba_csr(x)
            elif method == 'cuda':
                self.nblocks = (n_row,)  # global blocks
                self.nthreads = (1,)  # threads per block
                # CUDA buffer
                self.bf_csr_rowptr = cuda.to_device(csr_rowptr)
                self.bf_csr_colidx = cuda.to_device(csr_colidx)
                self.bf_csr_val = cuda.to_device(csr_val)
                self.cuda_csr(x)
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
                self.ctx = cl.create_some_context()
                self.queue = cl.CommandQueue(self.ctx)
                # read in the OpenCL source file as a string
                f = open('FasterSpMV/clkernels/ClBaseSELLKernel.cl', 'r')
                fstr = ''.join(f.readlines())
                # create the program
                self.program = cl.Program(self.ctx, fstr).build()
                # create OpenCL buffers
                self.n_row = n_row
                self.slice_count = slice_count
                self.slice_height = slice_height
                mf = cl.mem_flags
                self._y = np.zeros(slice_count * slice_height, dtype=np.float32)
                self.slice_ptr_buf = cl.Buffer(self.ctx,
                                               mf.READ_ONLY | mf.COPY_HOST_PTR,
                                               hostbuf=ell_sliceptr)
                self.slice_col_buf = cl.Buffer(self.ctx,
                                               mf.READ_ONLY | mf.COPY_HOST_PTR,
                                               hostbuf=slice_col)
                self.colidx_buf = cl.Buffer(self.ctx,
                                            mf.READ_ONLY | mf.COPY_HOST_PTR,
                                            hostbuf=ell_colidx)
                self.val_buf = cl.Buffer(self.ctx,
                                         mf.READ_ONLY | mf.COPY_HOST_PTR,
                                         hostbuf=ell_val)
                self.opencl_sell(x)
            elif method == 'numba':
                self.numba_sell(x)
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
                self.cuda_sell(x)

    def numba_csr(self, v):
        return numba_csr_spmv(self.n_row, self.csr_rowptr,
                              self.csr_colidx, self.csr_val, v)

    def numba_sell(self, v):
        return numba_sliced_ellpack_spmv(self.slice_count,
                                         self.ell_sliceptr, self.ell_colidx,
                                         self.ell_val, v, self.slice_height)

    def opencl_csr(self, v):
        mf = cl.mem_flags
        x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v)
        y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._y.nbytes)
        self.program.csr_spmv(self.queue, (self.n_row,), None,
                              self.rowptr_buf, self.colidx_buf,
                              self.val_buf, x_buf, y_buf)
        csr_y = np.empty_like(self._y)
        cl.enqueue_copy(self.queue, csr_y, y_buf)
        return csr_y

    def opencl_sell(self, v):
        mf = cl.mem_flags
        x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v)
        y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._y.nbytes)
        self.program.bsell_spmv(self.queue,
                                (self.slice_count * self.slice_height,),
                                (self.slice_height,), self.slice_ptr_buf,
                                self.slice_col_buf, self.colidx_buf,
                                self.val_buf, x_buf,
                                np.int32(self.slice_height), y_buf)
        bsell_y = np.empty_like(self._y)
        cl.enqueue_copy(self.queue, bsell_y, self.y_buf)
        return bsell_y

    def cuda_csr(self, v):
        bf_x = cuda.to_device(v)
        bf_csr_y = cuda.device_array(self.n_row, dtype=np.float32)
        cuda_csr_spmv[self.nblocks, self.nthreads](self.bf_csr_rowptr,
                                                   self.bf_csr_colidx,
                                                   self.bf_csr_val,
                                                   bf_x, bf_csr_y)
        output_csr_y = bf_csr_y.copy_to_host()
        return output_csr_y

    def cuda_sell(self, v):
        bf_x = cuda.to_device(v)
        bf_sell_y = cuda.device_array(self.slice_height *
                                      self.slice_count, dtype=np.float32)
        cuda_sliced_ellpack_spmv_1d[self.nblocks,
                                    self.nthreads](self.bf_ell_sliceptr,
                                                self.bf_ell_colidx,
                                                self.bf_ell_val, bf_x,
                                                self.slice_height, bf_sell_y)
        output_sell_y = bf_sell_y.copy_to_host()
        return output_sell_y
