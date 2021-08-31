from setuptools import setup, find_packages

setup(
    name='SpMV',
    version='1.1.0',
    packages=find_packages(),
    py_modules=['code.spmv_kernel', 'code.matrix_tools',
                'tests.test_matrix_tools', 'tests.test_spmv_speed'],
    description='Sparse matrix-vector multiplication',
    url="https://github.com/YYCHEN-299/Scientific-Computing-Individual-Research-Project",
    author='Yangyuan Chen',
    entry_points={
    }
)