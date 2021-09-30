import numpy as np
import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule

_ = pycuda.autoinit.cuda


class PyCUDASELLSpMV:
    def __init__(self, n_row, slice_height, slice_count,
                 sell_colidx, sell_sliceptr, sell_slicecol, sell_val):
        # CUDA Sliced ELLPACK SpMV
        mod = SourceModule("""
        __global__ void sell_spmv(const int * slice_ptr,
                                  const int * slice_col,
                                  const int * colidx,
                                  const float * val,
                                  const float * x,
                                  const int slice_height,
                                  float * y)
        {
            int i, j, k, idx;
            float row_data = 0.0f;
            i = blockIdx.x;
            j = threadIdx.x;
            for (k = 0; k < slice_col[i]; k++) {
                idx = slice_ptr[i] + k * slice_height + j;
                row_data = fmaf(x[colidx[idx]], val[idx], row_data);
            }
            y[i * slice_height + j] = row_data;
        }
        """)
        self.n_row = n_row
        self.slice_height = slice_height
        self.slice_count = slice_count
        self.prg = mod.get_function("sell_spmv")
        self.dev_sell_sliceptr = drv.to_device(sell_sliceptr)
        self.dev_sell_slicecol = drv.to_device(sell_slicecol)
        self.dev_sell_colidx = drv.to_device(sell_colidx)
        self.dev_sell_val = drv.to_device(sell_val)

    def run(self, x):
        dev_x = drv.to_device(x)
        sell_y = np.empty(self.slice_height *
                          self.slice_count, dtype=np.float32)
        self.prg(self.dev_sell_sliceptr, self.dev_sell_slicecol,
                 self.dev_sell_colidx, self.dev_sell_val, dev_x,
                 np.int32(self.slice_height), drv.Out(sell_y),
                 block=(self.slice_height, 1, 1), grid=(self.slice_count, 1))
        return sell_y[:self.n_row]


class PyCUDACSRSpMV:
    def __init__(self, n_row, rowptr, colidx, val):
        # CUDA CSR SpMV
        mod = SourceModule("""
        __global__ void csr_spmv(const int * rowptr,
                                 const int * colidx,
                                 const float * val,
                                 const float * x,
                                 float * y)
        {
            int i, k;
            i = blockIdx.x;
            float row_data = 0.0f;
            for (k = rowptr[i]; k < rowptr[i + 1]; k++) {
                row_data = fmaf(x[colidx[k]], val[k], row_data);
            }
            y[i] = row_data;
        }
        """)
        self.n_row = n_row
        self.prg = mod.get_function("csr_spmv")
        self.dev_rowptr = drv.to_device(rowptr)
        self.dev_colidx = drv.to_device(colidx)
        self.dev_val = drv.to_device(val)

    def run(self, x):
        dev_x = drv.to_device(x)
        csr_y = np.empty(self.n_row, dtype=np.float32)
        self.prg(self.dev_rowptr, self.dev_colidx,
                 self.dev_val, dev_x, drv.Out(csr_y),
                 block=(1, 1, 1), grid=(self.n_row, 1))
        return csr_y
