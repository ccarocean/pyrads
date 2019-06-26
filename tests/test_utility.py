import pytest
from rads._utility import fortran_float


def test_fortran_float():
    assert fortran_float('3.14e10') == pytest.approx(3.14e10)
    assert fortran_float('3.14E10') == pytest.approx(3.14e10)
    assert fortran_float('3.14d10') == pytest.approx(3.14e10)
    assert fortran_float('3.14D10') == pytest.approx(3.14e10)
    assert fortran_float('3.14e+10') == pytest.approx(3.14e10)
    assert fortran_float('3.14E+10') == pytest.approx(3.14e10)
    assert fortran_float('3.14d+10') == pytest.approx(3.14e10)
    assert fortran_float('3.14D+10') == pytest.approx(3.14e10)
    assert fortran_float('3.14e-10') == pytest.approx(3.14e-10)
    assert fortran_float('3.14E-10') == pytest.approx(3.14e-10)
    assert fortran_float('3.14d-10') == pytest.approx(3.14e-10)
    assert fortran_float('3.14D-10') == pytest.approx(3.14e-10)
    assert fortran_float('3.14+100') == pytest.approx(3.14e100)
    assert fortran_float('3.14-100') == pytest.approx(3.14e-100)
    with pytest.raises(ValueError):
        fortran_float('not a float')