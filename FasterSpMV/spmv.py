from FasterSpMV.matrix_tools import *
from FasterSpMV.numba_spmv import *
from FasterSpMV.opencl_spmv import *
from FasterSpMV.pycuda_spmv import *


class SpMVOperator:
    def __init__(self, method, format, dev, n_row, n_col,
                 csr_rowptr, csr_colidx, csr_val, slice_height):
        self.n_row = n_row
        self.n_col = n_col
        self.csr_rowptr = csr_rowptr
        self.csr_colidx = csr_colidx
        self.csr_val = csr_val
        self.slice_height = slice_height

        if format == 'csr':
            if method == 'opencl':
                self._ocl_csr = OclCSRSpMV(n_row, csr_rowptr,
                                           csr_colidx, csr_val, dev)
                self.kernel = self._opencl_csr
            elif method == 'numba':
                self.y = np.zeros(n_row, dtype=np.float32)
                self.kernel = self._numba_csr
            elif method == 'cuda':
                self._cu_csr = PyCUDACSRSpMV(n_row, csr_rowptr,
                                             csr_colidx, csr_val)
                self.kernel = self._cuda_csr
        elif format == 'sell':
            slice_count, sell_colidx, sell_sliceptr, slice_col, sell_val = \
                csr_to_sell(n_row, csr_rowptr,
                            csr_colidx, csr_val, slice_height)
            self.slice_count = slice_count
            self.sell_colidx = sell_colidx
            self.sell_sliceptr = sell_sliceptr
            self.slice_col = slice_col
            self.sell_val = sell_val
            if method == 'opencl':
                if dev == 'CPU':
                    slice2_count, sell2_colidx, _, \
                        sell2_slicecol, sell2_val = \
                        csr_to_ocl_sell2(n_row, csr_rowptr,
                                         csr_colidx, csr_val)
                    self._ocl_sell = OclSELL2SpMV(n_row, slice2_count,
                                                  sell2_slicecol,
                                                  sell2_colidx,
                                                  sell2_val, dev)
                elif dev == 'GPU':
                    self._ocl_sell = OclSELLSpMV(n_row, slice_count,
                                                 sell_sliceptr, slice_col,
                                                 sell_colidx, sell_val,
                                                 slice_height, dev)
                self.kernel = self._opencl_sell
            elif method == 'numba':
                self.y = np.zeros(slice_count * slice_height,
                                  dtype=np.float32)
                self.kernel = self._numba_sell
            elif method == 'cuda':
                self._cu_sell = PyCUDASELLSpMV(n_row, slice_height,
                                               slice_count, sell_colidx,
                                               sell_sliceptr, slice_col,
                                               sell_val)
                self.kernel = self._cuda_sell

        # first run the kernel
        x = np.ones(n_col, dtype=np.float32)
        self.kernel(x)

    def run_matvec(self, v):
        return self.kernel(v)

    def _numba_csr(self, v):
        numba_csr_spmv(self.y, self.n_row,
                       self.csr_rowptr, self.csr_colidx, self.csr_val, v)
        return self.y

    def _numba_sell(self, v):
        numba_sell_spmv(self.y, self.slice_count,
                        self.sell_sliceptr, self.sell_colidx,
                        self.sell_val, v, self.slice_height)
        return self.y[:self.n_row]

    def _opencl_csr(self, v):
        return self._ocl_csr.run(v)

    def _opencl_sell(self, v):
        _y = self._ocl_sell.run(v)
        return _y[:self.n_row]

    def _cuda_csr(self, v):
        return self._cu_csr.run(v)

    def _cuda_sell(self, v):
        return self._cu_sell.run(v)
