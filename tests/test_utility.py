import io

import pytest  # type: ignore

from rads.utility import (
    contains_sublist,
    delete_sublist,
    ensure_open,
    fortran_float,
    isio,
    merge_sublist,
    xor,
)


def test_ensure_open_closeio_default():
    file = io.StringIO("content")
    with ensure_open(file) as f:
        assert not f.closed
    assert not f.closed


def test_ensure_open_closeio_true():
    file = io.StringIO("content")
    with ensure_open(file, closeio=True) as f:
        assert not f.closed
    assert f.closed


def test_ensure_open_closeio_false():
    file = io.StringIO("content")
    with ensure_open(file, closeio=False) as f:
        assert not f.closed
    assert not f.closed


def test_isio(mocker):
    assert isio(io.StringIO("content"))
    assert not isio("string is not io")
    m = mocker.Mock()
    m.read.return_value = "duck typing not accepted"
    assert not isio(m)


def test_isio_read(mocker):
    assert isio(io.StringIO("content"), read=True)
    assert not isio("string is not io", read=True)
    m = mocker.Mock(spec=["read"])
    m.read.return_value = "duck typing is accepted"
    assert isio(m, read=True)
    m = mocker.Mock(spec=["write"])
    m.write.return_value = "duck typing is accepted"
    assert not isio(m, read=True)


def test_isio_write(mocker):
    assert isio(io.StringIO("content"), write=True)
    assert not isio("string is not io", write=True)
    m = mocker.Mock(spec=["read"])
    m.read.return_value = "duck typing is accepted"
    assert not isio(m, write=True)
    m = mocker.Mock(spec=["write"])
    m.write.return_value = "duck typing is accepted"
    assert isio(m, write=True)


def test_xor():
    assert not xor(True, True)
    assert xor(True, False)
    assert xor(False, True)
    assert not xor(False, False)


def test_contains_sublist():
    assert contains_sublist([1, 2, 3, 4], [1, 2])
    assert contains_sublist([1, 2, 3, 4], [2, 3])
    assert contains_sublist([1, 2, 3, 4], [3, 4])
    assert contains_sublist([1, 2, 3, 4], [1, 2, 3, 4])
    assert not contains_sublist([1, 2, 3, 4], [2, 1])
    assert not contains_sublist([1, 2, 3, 4], [3, 2])
    assert not contains_sublist([1, 2, 3, 4], [4, 3])
    # while the empty list is technically a sublist of any list for this
    # function [] is never a sublist
    assert not contains_sublist([1, 2, 3, 4], [])


def test_merge_sublist():
    assert merge_sublist([1, 2, 3, 4], []) == [1, 2, 3, 4]
    assert merge_sublist([1, 2, 3, 4], [1, 2]) == [1, 2, 3, 4]
    assert merge_sublist([1, 2, 3, 4], [2, 3]) == [1, 2, 3, 4]
    assert merge_sublist([1, 2, 3, 4], [3, 4]) == [1, 2, 3, 4]
    assert merge_sublist([1, 2, 3, 4], [0, 1]) == [1, 2, 3, 4, 0, 1]
    assert merge_sublist([1, 2, 3, 4], [4, 5]) == [1, 2, 3, 4, 4, 5]
    assert merge_sublist([1, 2, 3, 4], [1, 1]) == [1, 2, 3, 4, 1, 1]


def test_delete_sublist():
    assert delete_sublist([1, 2, 3, 4], []) == [1, 2, 3, 4]
    assert delete_sublist([1, 2, 3, 4], [1, 2]) == [3, 4]
    assert delete_sublist([1, 2, 3, 4], [2, 3]) == [1, 4]
    assert delete_sublist([1, 2, 3, 4], [3, 4]) == [1, 2]
    assert delete_sublist([1, 2, 3, 4], [0, 1]) == [1, 2, 3, 4]
    assert delete_sublist([1, 2, 3, 4], [4, 5]) == [1, 2, 3, 4]
    assert delete_sublist([1, 2, 3, 4], [1, 1]) == [1, 2, 3, 4]


def test_fortran_float():
    assert fortran_float("3.14e10") == pytest.approx(3.14e10)
    assert fortran_float("3.14E10") == pytest.approx(3.14e10)
    assert fortran_float("3.14d10") == pytest.approx(3.14e10)
    assert fortran_float("3.14D10") == pytest.approx(3.14e10)
    assert fortran_float("3.14e+10") == pytest.approx(3.14e10)
    assert fortran_float("3.14E+10") == pytest.approx(3.14e10)
    assert fortran_float("3.14d+10") == pytest.approx(3.14e10)
    assert fortran_float("3.14D+10") == pytest.approx(3.14e10)
    assert fortran_float("3.14e-10") == pytest.approx(3.14e-10)
    assert fortran_float("3.14E-10") == pytest.approx(3.14e-10)
    assert fortran_float("3.14d-10") == pytest.approx(3.14e-10)
    assert fortran_float("3.14D-10") == pytest.approx(3.14e-10)
    assert fortran_float("3.14+100") == pytest.approx(3.14e100)
    assert fortran_float("3.14-100") == pytest.approx(3.14e-100)
    with pytest.raises(ValueError):
        fortran_float("not a float")
