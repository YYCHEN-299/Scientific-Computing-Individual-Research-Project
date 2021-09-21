import numpy as np
import pyopencl as cl


class BaseSELLSpMV:
    def __init__(self, n_row, slice_count, slice_ptr,
                 slice_col, colidx, val, slice_height):
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
        self._y = np.empty(slice_count * slice_height, dtype=np.float32)
        self.slice_ptr_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mf.USE_HOST_PTR,
                                       hostbuf=slice_ptr)
        self.slice_col_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mf.USE_HOST_PTR,
                                       hostbuf=slice_col)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mf.USE_HOST_PTR,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mf.USE_HOST_PTR,
                                 hostbuf=val)

        self.y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._y.nbytes)
        lm = np.zeros(slice_height, dtype=np.int32)
        self.local_mem = cl.LocalMemory(lm.nbytes)

    def run(self, x):
        mf = cl.mem_flags
        x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=x)
        self.program.bsell_spmv(self.queue,
                                (self.slice_count * self.slice_height,),
                                (self.slice_height,), self.slice_ptr_buf,
                                self.slice_col_buf, self.colidx_buf,
                                self.val_buf, x_buf,
                                np.int32(self.slice_height), self.y_buf)
        bsell_y = np.empty_like(self._y)
        cl.enqueue_copy(self.queue, bsell_y, self.y_buf)
        return bsell_y


class SELL4SpMV:
    def __init__(self, n_row, slice_count, slice_col, colidx, val):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/clkernels/ClSELL4Kernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # create OpenCL buffers
        self.n_row = n_row
        self.slice_count = slice_count
        self.slice_height = slice_height = 4
        mf = cl.mem_flags
        self._y = np.empty((slice_count, slice_height), dtype=np.float32)
        self.slice_col_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mf.USE_HOST_PTR,
                                       hostbuf=slice_col)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mf.USE_HOST_PTR,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mf.USE_HOST_PTR,
                                 hostbuf=val)

        self.y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._y.nbytes)
        shared_x = np.empty_like(val, dtype=np.float32)
        self.shared_x_buf = cl.Buffer(self.ctx,
                                      mf.READ_WRITE | mf.USE_HOST_PTR,
                                      hostbuf=shared_x)

    def run(self, x):
        mf = cl.mem_flags
        x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=x)
        self.program.sell4_spmv(self.queue, (self.slice_count,), None,
                                self.slice_col_buf, self.colidx_buf,
                                self.val_buf, x_buf, self.y_buf)
        bsell4_y = np.empty_like(self._y)
        cl.enqueue_copy(self.queue, bsell4_y, self.y_buf)
        return bsell4_y


class SELL8SpMV:
    def __init__(self, n_row, slice_count, slice_col, colidx, val):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/clkernels/ClSELL8Kernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # create OpenCL buffers
        self.n_row = n_row
        self.slice_count = slice_count
        self.slice_height = slice_height = 16
        mf = cl.mem_flags
        self._y = np.empty((slice_count, slice_height), dtype=np.float32)
        self.slice_col_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mf.USE_HOST_PTR,
                                       hostbuf=slice_col)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mf.USE_HOST_PTR,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mf.USE_HOST_PTR,
                                 hostbuf=val)

        self.y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._y.nbytes)
        shared_x = np.empty_like(val, dtype=np.float32)
        self.shared_x_buf = cl.Buffer(self.ctx,
                                      mf.READ_WRITE | mf.USE_HOST_PTR,
                                      hostbuf=shared_x)

    def run(self, x):
        mf = cl.mem_flags
        x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=x)
        self.program.sell8_spmv(self.queue, (self.slice_count,), None,
                                self.slice_col_buf, self.colidx_buf,
                                self.val_buf, x_buf, self.y_buf)
        sell8_y = np.empty_like(self._y)
        cl.enqueue_copy(self.queue, sell8_y, self.y_buf)
        return sell8_y


class SELLSpMV:
    def __init__(self, n_row, slice_count, slice_ptr,
                 slice_col, colidx, val, slice_height):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/clkernels/ClSELLGPUKernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # parameters
        self.row_threads = 2
        self.n_row = n_row
        self.total_row = slice_count * slice_height
        self.slice_count = slice_count
        self.slice_height = slice_height
        local_size = self.row_threads * slice_height
        self.local_size = int(local_size)
        global_size = self.total_row * self.row_threads
        self.global_size = int(global_size)

        # create OpenCL buffers
        mf = cl.mem_flags
        self._y = np.empty(self.total_row, dtype=np.float32)
        self.slice_ptr_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mf.USE_HOST_PTR,
                                       hostbuf=slice_ptr)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mf.USE_HOST_PTR,
                                    hostbuf=colidx)
        self.slice_col_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mf.USE_HOST_PTR,
                                       hostbuf=slice_col)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mf.USE_HOST_PTR,
                                 hostbuf=val)

        self.y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._y.nbytes)
        lm = np.zeros(slice_height, dtype=np.int32)
        self.local_mem = cl.LocalMemory(lm.nbytes)

    def run(self, x):
        mf = cl.mem_flags
        x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=x)
        self.program.sell_spmv(self.queue, (256 * 256,), (256,),
                               self.slice_ptr_buf, self.colidx_buf,
                               self.val_buf, x_buf,
                               np.int32(self.slice_height),
                               np.int32(self.slice_count), self.y_buf)
        sell_y = np.empty_like(self._y)
        cl.enqueue_copy(self.queue, sell_y, self.y_buf)
        return sell_y


class CSRSpMV:
    def __init__(self, num_row, rowptr, colidx, val):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/clkernels/ClCSRKernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # create OpenCL buffers
        self.num_row = num_row
        mf = cl.mem_flags
        self._y = np.empty(num_row, dtype=np.float32)
        self.rowptr_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mf.USE_HOST_PTR,
                                    hostbuf=rowptr)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mf.USE_HOST_PTR,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mf.USE_HOST_PTR,
                                 hostbuf=val)

        self.y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._y.nbytes)

    def run(self, x):
        mf = cl.mem_flags
        x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=x)
        self.program.csr_spmv(self.queue, (self.num_row,), None,
                              self.rowptr_buf, self.colidx_buf,
                              self.val_buf, x_buf, self.y_buf)
        csr_y = np.empty_like(self._y)
        cl.enqueue_copy(self.queue, csr_y, self.y_buf)
        return csr_y
