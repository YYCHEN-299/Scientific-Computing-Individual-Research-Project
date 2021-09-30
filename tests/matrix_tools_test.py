import pytest

from FasterSpMV.matrix_tools import *


def test_setup_tools():
    with pytest.raises(ValueError):
        random_spmatrix(-5, 5, 5)
    with pytest.raises(ValueError):
        random_spmatrix(5, -5, 5)
    with pytest.raises(ValueError):
        random_spmatrix(5, 5, -5)
    with pytest.raises(ValueError):
        random_spmatrix(5, 5, 500)


def test_csr_converter():
    with pytest.raises(ValueError):
        spmatrix_to_csr(123)


def test_sell_converter():
    with pytest.raises(ValueError):
        csr_to_sell(1, [5], [5], [5], -5)


def test_sell_rd_converter():
    with pytest.raises(ValueError):
        csr_to_2d_sell(1, [2], [2], [2], -2)


def test_2d_sell_converter():
    with pytest.raises(ValueError):
        csr_to_sell_rd([], 2, -1)
