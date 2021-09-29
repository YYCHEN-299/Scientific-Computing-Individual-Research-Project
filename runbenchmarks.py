from scipy.io import mmread

from benchmarks.linearoperator_benchmark import *
from benchmarks.spmv_benchmark import *


def run_benchmark():
    datasets = ['data/cant.mtx', 'data/consph.mtx',
                'data/cop20k_A.mtx', 'data/pdb1HYS.mtx', 'data/rma10.mtx']
    t = 200

    # slice_heights = [2, 4, 8, 16]
    # os.environ['PYOPENCL_CTX'] = '2'
    # for d in datasets:
    #     print("matrix name:", d)
    #     matrix_data = mmread(d).tocsr()
    #     spmv_scipy_benchmark(matrix_data, t)
    #     spmv_cpu_csr_benchmark(matrix_data, t)
    #     spmv_cpu_exp_sell_benchmark(matrix_data, t)
    #     for s in slice_heights:
    #         spmv_cpu_sell_benchmark(matrix_data, s, t)

    slice_heights = [32, 64, 128, 256, 512]
    os.environ['PYOPENCL_CTX'] = '0'
    for d in datasets:
        print("matrix name:", d)
        matrix_data = mmread(d).tocsr()
        # spmv_gpu_csr_benchmark(matrix_data, t)
        spmv_pycuda_csr_benchmark(matrix_data, t)
        for s in slice_heights:
            # spmv_gpu_sell_benchmark(matrix_data, s, t)
            spmv_pycuda_sell_benchmark(matrix_data, s, t)


def run_op_benchmark():
    matrix_data = mmread('data/cant.mtx').tocsr()
    solver_gpu_benchmark(matrix_data)


if __name__ == "__main__":
    run_benchmark()
    # run_op_benchmark()
