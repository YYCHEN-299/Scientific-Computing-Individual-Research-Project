import pytest

from FasterSpMV.matrix_tools import csr_to_sell, random_spmatrix, spmatrix_to_csr


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
