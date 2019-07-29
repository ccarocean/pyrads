import pytest  # type: ignore

from rads._utility import (
    contains_sublist,
    delete_sublist,
    fortran_float,
    merge_sublist,
    xor,
)


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
