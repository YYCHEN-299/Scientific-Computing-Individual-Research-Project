import numpy as np
import pyopencl as cl


class OclSELLSpMV:
    def __init__(self, n_row, slice_count, slice_ptr,
                 slice_col, colidx, val, slice_height, dev='CPU'):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/oclkernels/oclSELLKernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # set data
        self.n_row = n_row
        self.slice_count = slice_count
        self.slice_height = np.int32(slice_height)
        self.global_size = int(slice_count * slice_height)
        self.local_size = int(slice_height)

        # set memory flag
        self.dev = dev
        self.mf = mf = cl.mem_flags
        mem_flag = mf.USE_HOST_PTR
        if dev == 'GPU':
            mem_flag = mf.COPY_HOST_PTR
        self.mem_flag = mem_flag

        # create OpenCL buffers
        self._y = np.empty(slice_count * slice_height, dtype=np.float32)
        self.slice_ptr_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mem_flag,
                                       hostbuf=slice_ptr)
        self.slice_col_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mem_flag,
                                       hostbuf=slice_col)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mem_flag,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mem_flag,
                                 hostbuf=val)
        # self.y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._y.nbytes)
        self.y_buf = cl.Buffer(self.ctx,
                               mf.WRITE_ONLY | mem_flag,
                               hostbuf=self._y)

    def run(self, x):
        x_buf = cl.Buffer(self.ctx,
                          self.mf.READ_ONLY | self.mem_flag,
                          hostbuf=x)
        self.program.sell_spmv(self.queue, (self.global_size,),
                               (self.local_size,), self.slice_ptr_buf,
                               self.slice_col_buf, self.colidx_buf,
                               self.val_buf, x_buf,
                               self.slice_height, self.y_buf)
        sell_y = np.empty_like(self._y, dtype=np.float32)
        cl.enqueue_copy(self.queue, sell_y, self.y_buf)
        return sell_y


class OclSELL2SpMV:
    def __init__(self, n_row, slice_count, slice_col, colidx, val, dev='CPU'):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/oclkernels/oclSELL2Kernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # set data
        self.n_row = n_row
        self.slice_count = slice_count
        self.slice_height = slice_height = 2

        # set memory flag
        self.dev = dev
        self.mf = mf = cl.mem_flags
        mem_flag = mf.USE_HOST_PTR
        if dev == 'GPU':
            mem_flag = mf.COPY_HOST_PTR
        self.mem_flag = mem_flag

        # create OpenCL buffers
        self._y = np.empty((slice_count, slice_height), dtype=np.float32)
        self.slice_col_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mem_flag,
                                       hostbuf=slice_col)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mem_flag,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mem_flag,
                                 hostbuf=val)
        self.y_buf = cl.Buffer(self.ctx,
                               mf.WRITE_ONLY | mem_flag,
                               hostbuf=self._y)

    def run(self, x):
        x_buf = cl.Buffer(self.ctx,
                          self.mf.READ_ONLY | self.mem_flag,
                          hostbuf=x)
        self.program.sell2_spmv(self.queue, (self.slice_count,), None,
                                self.slice_col_buf, self.colidx_buf,
                                self.val_buf, x_buf, self.y_buf)
        sell2_y = np.empty_like(self._y, dtype=np.float32)
        cl.enqueue_copy(self.queue, sell2_y, self.y_buf)
        return sell2_y.reshape(-1,)


class OclSELL4SpMV:
    def __init__(self, n_row, slice_count, slice_col, colidx, val, dev='CPU'):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/oclkernels/oclSELL4Kernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # set data
        self.n_row = n_row
        self.slice_count = slice_count
        self.slice_height = slice_height = 4

        # set memory flag
        self.dev = dev
        self.mf = mf = cl.mem_flags
        mem_flag = mf.USE_HOST_PTR
        if dev == 'GPU':
            mem_flag = mf.COPY_HOST_PTR
        self.mem_flag = mem_flag

        # create OpenCL buffers
        self._y = np.empty((slice_count, slice_height), dtype=np.float32)
        self.slice_col_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mem_flag,
                                       hostbuf=slice_col)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mem_flag,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mem_flag,
                                 hostbuf=val)
        self.y_buf = cl.Buffer(self.ctx,
                               mf.WRITE_ONLY | mem_flag,
                               hostbuf=self._y)

    def run(self, x):
        x_buf = cl.Buffer(self.ctx,
                          self.mf.READ_ONLY | self.mem_flag,
                          hostbuf=x)
        self.program.sell4_spmv(self.queue, (self.slice_count,), None,
                                self.slice_col_buf, self.colidx_buf,
                                self.val_buf, x_buf, self.y_buf)
        sell4_y = np.empty_like(self._y, dtype=np.float32)
        cl.enqueue_copy(self.queue, sell4_y, self.y_buf)
        return sell4_y.reshape(-1,)


class OclSELL8SpMV:
    def __init__(self, n_row, slice_count, slice_col, colidx, val, dev='CPU'):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/oclkernels/oclSELL8Kernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # set data
        self.n_row = n_row
        self.slice_count = slice_count
        self.slice_height = slice_height = 8

        # set memory flag
        self.dev = dev
        self.mf = mf = cl.mem_flags
        mem_flag = mf.USE_HOST_PTR
        if dev == 'GPU':
            mem_flag = mf.COPY_HOST_PTR
        self.mem_flag = mem_flag

        # create OpenCL buffers
        self._y = np.empty((slice_count, slice_height), dtype=np.float32)
        self.slice_col_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mem_flag,
                                       hostbuf=slice_col)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mem_flag,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mem_flag,
                                 hostbuf=val)
        self.y_buf = cl.Buffer(self.ctx,
                               mf.WRITE_ONLY | mem_flag,
                               hostbuf=self._y)

    def run(self, x):
        x_buf = cl.Buffer(self.ctx,
                          self.mf.READ_ONLY | self.mem_flag,
                          hostbuf=x)
        self.program.sell8_spmv(self.queue, (self.slice_count,), None,
                                self.slice_col_buf, self.colidx_buf,
                                self.val_buf, x_buf, self.y_buf)
        sell8_y = np.empty_like(self._y, dtype=np.float32)
        cl.enqueue_copy(self.queue, sell8_y, self.y_buf)
        return sell8_y.reshape(-1,)


class OclSELL16SpMV:
    def __init__(self, n_row, slice_count, slice_col, colidx, val, dev='CPU'):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/oclkernels/oclSELL16Kernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # set data
        self.n_row = n_row
        self.slice_count = slice_count
        self.slice_height = slice_height = 16

        # set memory flag
        self.dev = dev
        self.mf = mf = cl.mem_flags
        mem_flag = mf.USE_HOST_PTR
        if dev == 'GPU':
            mem_flag = mf.COPY_HOST_PTR
        self.mem_flag = mem_flag

        # create OpenCL buffers
        self._y = np.empty((slice_count, slice_height), dtype=np.float32)
        self.slice_col_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mem_flag,
                                       hostbuf=slice_col)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mem_flag,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mem_flag,
                                 hostbuf=val)
        self.y_buf = cl.Buffer(self.ctx,
                               mf.WRITE_ONLY | mem_flag,
                               hostbuf=self._y)

    def run(self, x):
        x_buf = cl.Buffer(self.ctx,
                          self.mf.READ_ONLY | self.mem_flag,
                          hostbuf=x)
        self.program.sell16_spmv(self.queue, (self.slice_count,), None,
                                 self.slice_col_buf, self.colidx_buf,
                                 self.val_buf, x_buf, self.y_buf)
        sell16_y = np.empty_like(self._y, dtype=np.float32)
        cl.enqueue_copy(self.queue, sell16_y, self.y_buf)
        return sell16_y.reshape(-1,)


class OclSELLRdSpMV:
    def __init__(self, n_row, slice_count, slice_ptr,
                 slice_col, colidx, val, slice_height, row_th, dev='CPU'):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/oclkernels/oclSELLRdKernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # set data
        self.row_th = np.int32(row_th)
        self.n_row = np.int32(n_row)
        self.slice_count = np.int32(slice_count)
        self.slice_height = np.int32(slice_height)
        self.local_size = int(slice_height * row_th)
        self.global_size = int(slice_count * slice_height * row_th)

        # set memory flag
        self.dev = dev
        self.mf = mf = cl.mem_flags
        mem_flag = mf.USE_HOST_PTR
        if dev == 'GPU':
            mem_flag = mf.COPY_HOST_PTR
        self.mem_flag = mem_flag

        # create OpenCL buffers
        self._y = np.empty(slice_count * slice_height, dtype=np.float32)
        self.slice_ptr_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mem_flag,
                                       hostbuf=slice_ptr)
        self.slice_col_buf = cl.Buffer(self.ctx,
                                       mf.READ_ONLY | mem_flag,
                                       hostbuf=slice_col)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mem_flag,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mem_flag,
                                 hostbuf=val)
        # self.y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._y.nbytes)
        self.y_buf = cl.Buffer(self.ctx,
                               mf.WRITE_ONLY | mem_flag,
                               hostbuf=self._y)
        lm = np.zeros(self.local_size, dtype=np.float32)
        self.local_mem = cl.LocalMemory(lm.nbytes)

    def run(self, x):
        x_buf = cl.Buffer(self.ctx,
                          self.mf.READ_ONLY | self.mem_flag,
                          hostbuf=x)
        self.program.sell_rd_spmv(self.queue, (self.global_size,),
                                  (self.local_size,), self.slice_ptr_buf,
                                  self.slice_col_buf, self.colidx_buf,
                                  self.val_buf, x_buf, self.slice_height,
                                  self.slice_count, self.row_th,
                                  self.local_mem, self.y_buf)
        sellrd_y = np.empty_like(self._y, dtype=np.float32)
        cl.enqueue_copy(self.queue, sellrd_y, self.y_buf)
        return sellrd_y


class OclCSRSpMV:
    def __init__(self, n_row, rowptr, colidx, val, dev='CPU'):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # read in the OpenCL source file as a string
        f = open('FasterSpMV/oclkernels/oclCSRKernel.cl', 'r')
        fstr = ''.join(f.readlines())

        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

        # set data
        self.n_row = n_row

        # set memory flag
        self.dev = dev
        self.mf = mf = cl.mem_flags
        mem_flag = mf.USE_HOST_PTR
        if dev == 'GPU':
            mem_flag = mf.COPY_HOST_PTR
        self.mem_flag = mem_flag

        # create OpenCL buffers
        mf = cl.mem_flags
        self._y = np.empty(n_row, dtype=np.float32)
        self.rowptr_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mem_flag,
                                    hostbuf=rowptr)
        self.colidx_buf = cl.Buffer(self.ctx,
                                    mf.READ_ONLY | mem_flag,
                                    hostbuf=colidx)
        self.val_buf = cl.Buffer(self.ctx,
                                 mf.READ_ONLY | mem_flag,
                                 hostbuf=val)
        # self.y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._y.nbytes)
        self.y_buf = cl.Buffer(self.ctx,
                               mf.WRITE_ONLY | mem_flag,
                               hostbuf=self._y)

    def run(self, x):
        x_buf = cl.Buffer(self.ctx,
                          self.mf.READ_ONLY | self.mem_flag,
                          hostbuf=x)
        self.program.csr_spmv(self.queue, (self.n_row,), None,
                              self.rowptr_buf, self.colidx_buf,
                              self.val_buf, x_buf, self.y_buf)
        csr_y = np.empty_like(self._y, dtype=np.float32)
        cl.enqueue_copy(self.queue, csr_y, self.y_buf)
        return csr_y
