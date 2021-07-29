from setuptools import setup, find_packages

setup(
    name='FasterSpMV',
    version='0.1.0',
    packages=find_packages(),
    py_modules=['code.spmv_kernel', 'code.matrix_tools',
                'tests.test_matrix_tools', 'tests.test_spmv_speed'],
    description='',
    author='Yangyuan Chen',
    entry_points={

    }
)