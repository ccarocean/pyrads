import warnings
from datetime import datetime
from typing import MutableSequence, MutableMapping

import math
import numpy as np  # type: ignore
import pytest  # type: ignore

from rads.rpn import (NumberOrArray, StackUnderflowError, Literal, PI, E,
                      Variable, Operator, SUB, ADD, MUL, POP, NEG, ABS, INV,
                      SQRT, SQR, EXP, LOG, LOG10, SIN, COS, TAN, SIND, COSD,
                      TAND, SINH, COSH, TANH, ASIN, ACOS, ATAN, ASIND, ACOSD,
                      ATAND, ASINH, ACOSH, ATANH, ISNAN, ISAN, RINT, NINT,
                      CEIL, CEILING, FLOOR, D2R, R2D, YMDHMS, SUM, DIF, DUP,
                      DIV, POW, FMOD, MIN, MAX, ATAN2, HYPOT, R2, EQ, NE, LT,
                      LE, GT, GE, NAN, AND, OR, IAND, IOR, BTEST, AVG, DXDY,
                      EXCH, INRANGE, BOXCAR, GAUSS)

GOLDEN_RATIO = math.log((1 + math.sqrt(5)) / 2)


class TestLiteral:

    def test_init(self):
        Literal(3)
        Literal(3.14)
        with pytest.raises(TypeError):
            Literal('not a number')  # type: ignore

    def test_value(self):
        assert Literal(3).value == 3
        assert Literal(3.14).value == 3.14

    def test_call(self):
        stack: MutableSequence[NumberOrArray] = []
        environment: MutableMapping[str, NumberOrArray] = {}
        assert Literal(3.14)(stack, environment) is None
        assert Literal(2.71)(stack, environment) is None
        assert stack == [3.14, 2.71]
        assert environment == {}

    def test_eq(self):
        assert Literal(3.14) == Literal(3.14)
        assert not Literal(3.14) == Literal(2.71)
        assert not Literal(3.14) == 3.14

    def test_ne(self):
        assert Literal(3.14) != Literal(2.71)
        assert not Literal(3.14) != Literal(3.14)
        assert Literal(3.14) != 3.14

    def test_lt(self):
        assert Literal(2.71) < Literal(3.14)
        assert not Literal(3.14) < Literal(2.71)
        with pytest.raises(TypeError):
            Literal(2.71) < 3.14
        with pytest.raises(TypeError):
            2.71 < Literal(3.14)

    def test_le(self):
        assert Literal(2.71) <= Literal(3.14)
        assert Literal(3.14) <= Literal(3.14)
        assert not Literal(3.14) <= Literal(2.71)
        with pytest.raises(TypeError):
            Literal(2.71) <= 3.14
        with pytest.raises(TypeError):
            2.71 <= Literal(3.14)

    def test_gt(self):
        assert Literal(3.14) > Literal(2.71)
        assert not Literal(2.71) > Literal(3.14)
        with pytest.raises(TypeError):
            Literal(3.14) > 2.71
        with pytest.raises(TypeError):
            3.14 > Literal(2.71)

    def test_ge(self):
        assert Literal(3.14) >= Literal(2.71)
        assert Literal(3.14) >= Literal(3.14)
        assert not Literal(2.71) >= Literal(3.14)
        with pytest.raises(TypeError):
            Literal(3.14) >= 2.71
        with pytest.raises(TypeError):
            3.14 >= Literal(2.71)

    def test_repr(self):
        assert repr(Literal(3)) == 'Literal(3)'
        assert repr(Literal(3.14)) == 'Literal(3.14)'

    def test_str(self):
        assert str(Literal(3)) == '3'
        assert str(Literal(3.14)) == '3.14'

    def test_pi(self):
        assert PI.value == pytest.approx(np.pi)

    def test_e(self):
        assert E.value == pytest.approx(np.e)


class TestVariable:

    def test_init(self):
        Variable('alt')
        with pytest.raises(ValueError):
            Variable('3')
        with pytest.raises(ValueError):
            Variable('3name')
        with pytest.raises(TypeError):
            Variable(3)  # type: ignore
        with pytest.raises(TypeError):
            Variable(3.14)  # type: ignore

    def test_name(self):
        assert Variable('alt').name == 'alt'

    def test_call(self):
        stack: MutableSequence[NumberOrArray] = []
        environment = {
            'alt': np.array([1, 2, 3]),
            'dry_tropo': 4,
            'wet_tropo': 5}
        assert Variable('wet_tropo')(stack, environment) is None
        assert Variable('alt')(stack, environment) is None
        assert len(stack) == 2
        assert stack[0] == 5
        assert np.all(stack[1] == np.array([1, 2, 3]))
        assert len(environment) == 3
        assert 'alt' in environment
        assert 'dry_tropo' in environment
        assert 'wet_tropo' in environment
        assert np.all(environment['alt'] == np.array([1, 2, 3]))
        assert environment['dry_tropo'] == 4
        assert environment['wet_tropo'] == 5
        with pytest.raises(KeyError):
            assert Variable('alt')(stack, {}) is None
        assert len(stack) == 2
        assert stack[0] == 5
        assert np.all(stack[1] == np.array([1, 2, 3]))

    def test_eq(self):
        assert Variable('alt') == Variable('alt')
        assert not Variable('alt') == Variable('dry_tropo')
        assert not Variable('alt') == 'alt'

    def test_ne(self):
        assert Variable('alt') != Variable('dry_tropo')
        assert not Variable('alt') != Variable('alt')
        assert Variable('alt') != 'alt'

    def test_repr(self):
        assert repr(Variable('alt')) == "Variable('alt')"

    def test_str(self):
        assert str(Variable('alt')) == 'alt'


def contains_array(stack: MutableSequence[NumberOrArray]) -> bool:
    for item in stack:
        if isinstance(item, np.ndarray):
            return True
    return False


def contains_nan(stack: MutableSequence[NumberOrArray]) -> bool:
    for item in stack:
        try:
            if math.isnan(item):
                return True
        except TypeError:
            pass
    return False


def assert_operator(operator: Operator,
                    pre_stack: MutableSequence[NumberOrArray],
                    post_stack: MutableSequence[NumberOrArray],
                    *, approx: bool = False,
                    rtol: float = 1e-15, atol: float = 1e-16) -> None:
    """Assert that an operator modifies the stack properly.

    Parameters
    ----------
    operator
        Operator to test.
    pre_stack
        Stack state before calling the operator.
    post_stack
        Desired stack state after calling the operator.
    approx
        Set to true to use approximate equality instead of exact.
    rtol
        Relative tolerance.  Only used if :paramref:`approx` is True.
    atol
        Absolute tolerance.  Only used if :paramref:`approx` is True.

    Raises
    ------
    AssertionError
        If the operator does not produce the proper post stack state or the
        environment parameter is changed.

    """
    stack = pre_stack
    environment = {'dont_touch': 5}
    operator(stack, environment)
    # environment should be unchanged
    assert environment == {'dont_touch': 5}
    # check stack
    if approx or contains_nan(post_stack) or contains_array(post_stack):
        assert len(stack) == len(post_stack)
        for a, b in zip(stack, post_stack):
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if approx:
                    np.testing.assert_allclose(
                        a, b, rtol=rtol, atol=atol, equal_nan=True)
                else:
                    np.testing.assert_equal(a, b)
            else:
                if math.isnan(b):
                    assert math.isnan(a)
                elif approx:
                    assert a == pytest.approx(b, rel=rtol, abs=atol)
                else:
                    assert a == b
    else:
        assert stack == post_stack


class TestOperator:

    def test_sub(self):
        assert repr(SUB) == 'SUB'
        assert_operator(SUB, [2, 4], [-2])
        assert_operator(SUB, [2, np.array([4, 1])], [np.array([-2, 1])])
        assert_operator(SUB, [np.array([4, 1]), 2], [np.array([2, -1])])
        assert_operator(
            SUB, [np.array([4, 1]), np.array([1, 4])], [np.array([3, -3])])
        # extra stack elements
        assert_operator(SUB, [0, 2, 4], [0, -2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SUB([], {})
        with pytest.raises(StackUnderflowError):
            SUB([1], {})

    def test_add(self):
        assert repr(ADD) == 'ADD'
        assert_operator(ADD, [2, 4], [6])
        assert_operator(ADD, [2, np.array([4, 1])], [np.array([6, 3])])
        assert_operator(ADD, [np.array([4, 1]), 2], [np.array([6, 3])])
        assert_operator(
            ADD, [np.array([4, 1]), np.array([1, 4])], [np.array([5, 5])])
        # extra stack elements
        assert_operator(ADD, [0, 2, 4], [0, 6])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ADD([], {})
        with pytest.raises(StackUnderflowError):
            ADD([1], {})

    def test_mul(self):
        assert repr(MUL) == 'MUL'
        assert_operator(MUL, [2, 4], [8])
        assert_operator(MUL, [2, np.array([4, 1])], [np.array([8, 2])])
        assert_operator(MUL, [np.array([4, 1]), 2], [np.array([8, 2])])
        assert_operator(
            MUL, [np.array([4, 1]), np.array([1, 4])], [np.array([4, 4])])
        # extra stack elements
        assert_operator(MUL, [0, 2, 4], [0, 8])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            MUL([], {})
        with pytest.raises(StackUnderflowError):
            MUL([1], {})

    def test_pop(self):
        assert repr(POP) == 'POP'
        assert_operator(POP, [1], [])
        assert_operator(POP, [1, 2], [1])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            POP([], {})

    def test_neg(self):
        assert repr(NEG) == 'NEG'
        assert_operator(NEG, [2], [-2])
        assert_operator(NEG, [-2], [2])
        assert_operator(NEG, [np.array([4, -1])], [np.array([-4, 1])])
        assert_operator(NEG, [np.array([-4, 1])], [np.array([4, -1])])
        # extra stack elements
        assert_operator(NEG, [0, 2], [0, -2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            NEG([], {})

    def test_abs(self):
        assert repr(ABS) == 'ABS'
        assert_operator(ABS, [2], [2])
        assert_operator(ABS, [-2], [2])
        assert_operator(ABS, [np.array([4, -1])], [np.array([4, 1])])
        assert_operator(ABS, [np.array([-4, 1])], [np.array([4, 1])])
        # extra stack elements
        assert_operator(ABS, [0, -2], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ABS([], {})

    def test_inv(self):
        assert repr(INV) == 'INV'
        assert_operator(INV, [2], [0.5])
        assert_operator(INV, [-2], [-0.5])
        assert_operator(INV, [np.array([4, -1])], [np.array([0.25, -1])])
        assert_operator(INV, [np.array([-4, 1])], [np.array([-0.25, 1])])
        # extra stack elements
        assert_operator(INV, [0, 2], [0, 0.5])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            INV([], {})

    def test_sqrt(self):
        assert repr(SQRT) == 'SQRT'
        assert_operator(SQRT, [4], [2])
        assert_operator(SQRT, [np.array([4, 16])], [np.array([2, 4])])
        # extra stack elements
        assert_operator(SQRT, [0, 4], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SQRT([], {})

    def test_sqr(self):
        assert repr(SQR) == 'SQR'
        assert_operator(SQR, [2], [4])
        assert_operator(SQR, [-2], [4])
        assert_operator(SQR, [np.array([4, -1])], [np.array([16, 1])])
        assert_operator(SQR, [np.array([-4, 1])], [np.array([16, 1])])
        # extra stack elements
        assert_operator(SQR, [0, -2], [0, 4])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SQR([], {})

    def test_exp(self):
        assert repr(EXP) == 'EXP'
        assert_operator(EXP, [math.log(1)], [1.0], approx=True)
        assert_operator(EXP, [math.log(2)], [2.0], approx=True)
        assert_operator(
            EXP, [np.array([np.log(4), np.log(1)])], [np.array([4.0, 1.0])],
            approx=True)
        # extra stack elements
        assert_operator(EXP, [0, np.log(1)], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            EXP([], {})

    def test_log(self):
        assert repr(LOG) == 'LOG'
        assert_operator(LOG, [math.e], [1.0], approx=True)
        assert_operator(LOG, [math.e ** 2], [2.0], approx=True)
        assert_operator(LOG, [math.e ** -2], [-2.0], approx=True)
        assert_operator(
            LOG, [np.array([np.e ** 4, np.e ** -1])], [np.array([4.0, -1.0])],
            approx=True)
        assert_operator(
            LOG, [np.array([np.e ** -4, np.e ** 1])], [np.array([-4.0, 1.0])],
            approx=True)
        # extra stack elements
        assert_operator(LOG, [0, np.e], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            LOG([], {})

    def test_log10(self):
        assert repr(LOG10) == 'LOG10'
        assert_operator(LOG10, [10], [1.0], approx=True)
        assert_operator(LOG10, [10 ** 2], [2.0], approx=True)
        assert_operator(LOG10, [10 ** -2], [-2.0], approx=True)
        assert_operator(
            LOG10, [np.array([10 ** 4, 10 ** -1])], [np.array([4.0, -1.0])],
            approx=True)
        assert_operator(
            LOG10, [np.array([10 ** -4, 10 ** 1])], [np.array([-4.0, 1.0])],
            approx=True)
        # extra stack elements
        assert_operator(LOG10, [0, 10], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            LOG10([], {})

    def test_sin(self):
        assert repr(SIN) == 'SIN'
        assert_operator(SIN, [0.0], [0.0], approx=True)
        assert_operator(SIN, [math.pi / 6], [1 / 2], approx=True)
        assert_operator(SIN, [math.pi / 4], [1 / math.sqrt(2)], approx=True)
        assert_operator(SIN, [math.pi / 3], [math.sqrt(3) / 2], approx=True)
        assert_operator(SIN, [math.pi / 2], [1.0], approx=True)
        assert_operator(
            SIN,
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            approx=True)
        assert_operator(
            SIN,
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [-np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            approx=True)
        # extra stack elements
        assert_operator(SIN, [0, math.pi / 2], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SIN([], {})

    def test_cos(self):
        assert repr(COS) == 'COS'
        assert_operator(COS, [0.0], [1.0], approx=True)
        assert_operator(COS, [math.pi / 6], [math.sqrt(3) / 2], approx=True)
        assert_operator(COS, [math.pi / 4], [1 / math.sqrt(2)], approx=True)
        assert_operator(COS, [math.pi / 3], [1 / 2], approx=True)
        assert_operator(COS, [math.pi / 2], [0.0], approx=True)
        assert_operator(
            COS,
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            approx=True)
        assert_operator(
            COS,
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            approx=True)
        # extra stack elements
        assert_operator(COS, [0, math.pi / 2], [0, 0.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            COS([], {})

    def test_tan(self):
        assert repr(TAN) == 'TAN'
        assert_operator(TAN, [0.0], [0.0], approx=True)
        assert_operator(TAN, [math.pi / 6], [1 / math.sqrt(3)], approx=True)
        assert_operator(TAN, [math.pi / 4], [1.0], approx=True)
        assert_operator(TAN, [math.pi / 3], [math.sqrt(3)], approx=True)
        assert_operator(
            TAN,
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])],
            [np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            approx=True)
        assert_operator(
            TAN,
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])],
            [-np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            approx=True)
        # extra stack elements
        assert_operator(TAN, [0, math.pi / 4], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            TAN([], {})

    def test_sind(self):
        assert repr(SIND) == 'SIND'
        assert_operator(SIND, [0], [0.0], approx=True)
        assert_operator(SIND, [30], [1 / 2], approx=True)
        assert_operator(SIND, [45], [1 / math.sqrt(2)], approx=True)
        assert_operator(SIND, [60], [math.sqrt(3) / 2], approx=True)
        assert_operator(SIND, [90], [1.0], approx=True)
        assert_operator(
            SIND,
            [np.array([0, 30, 45, 60, 90])],
            [np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            approx=True)
        assert_operator(
            SIND,
            [-np.array([0, 30, 45, 60, 90])],
            [-np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            approx=True)
        # extra stack elements
        assert_operator(SIND, [0, 90], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SIND([], {})

    def test_cosd(self):
        assert repr(COSD) == 'COSD'
        assert_operator(COSD, [0], [1.0], approx=True)
        assert_operator(COSD, [30], [math.sqrt(3) / 2], approx=True)
        assert_operator(COSD, [45], [1 / math.sqrt(2)], approx=True)
        assert_operator(COSD, [60], [1 / 2], approx=True)
        assert_operator(COSD, [90], [0.0], approx=True)
        assert_operator(
            COSD,
            [np.array([0, 30, 45, 60, 90])],
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            approx=True)
        assert_operator(
            COSD,
            [-np.array([0, 30, 45, 60, 90])],
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            approx=True)
        # extra stack elements
        assert_operator(COSD, [0, 90], [0, 0.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            COSD([], {})

    def test_tand(self):
        assert repr(TAND) == 'TAND'
        assert_operator(TAND, [0], [0], approx=True)
        assert_operator(TAND, [30], [1 / math.sqrt(3)], approx=True)
        assert_operator(TAND, [45], [1.0], approx=True)
        assert_operator(TAND, [60], [math.sqrt(3)], approx=True)
        assert_operator(
            TAND,
            [np.array([0, 30, 45, 60])],
            [np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            approx=True)
        assert_operator(
            TAND,
            [-np.array([0, 30, 45, 60])],
            [-np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            approx=True)
        # extra stack elements
        assert_operator(TAND, [0, 45], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            TAND([], {})

    def test_sinh(self):
        assert repr(SINH) == 'SINH'
        assert_operator(SINH, [0.0], [0.0], approx=True)
        assert_operator(SINH, [GOLDEN_RATIO], [0.5], approx=True)
        assert_operator(
            SINH,
            [np.array([0.0, GOLDEN_RATIO])],
            [np.array([0.0, 0.5])],
            approx=True)
        # extra stack elements
        assert_operator(SINH, [0, GOLDEN_RATIO], [0, 0.5], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SINH([], {})

    def test_cosh(self):
        assert repr(COSH) == 'COSH'
        assert_operator(COSH, [0.0], [1.0], approx=True)
        assert_operator(COSH, [GOLDEN_RATIO], [math.sqrt(5) / 2], approx=True)
        assert_operator(
            COSH,
            [np.array([0.0, GOLDEN_RATIO])],
            [np.array([1.0, np.sqrt(5) / 2])],
            approx=True)
        # extra stack elements
        assert_operator(
            COSH, [0, GOLDEN_RATIO], [0, math.sqrt(5) / 2], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            COSH([], {})

    def test_tanh(self):
        assert repr(TANH) == 'TANH'
        assert_operator(TANH, [0.0], [0.0], approx=True)
        assert_operator(TANH, [GOLDEN_RATIO], [math.sqrt(5) / 5], approx=True)
        assert_operator(
            TANH,
            [np.array([0.0, GOLDEN_RATIO])],
            [np.array([0.0, np.sqrt(5) / 5])],
            approx=True)
        # extra stack elements
        assert_operator(
            TANH, [0, GOLDEN_RATIO], [0, math.sqrt(5) / 5], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            TANH([], {})

    def test_asin(self):
        assert repr(ASIN) == 'ASIN'
        assert_operator(ASIN, [0.0], [0.0], approx=True)
        assert_operator(ASIN, [1 / 2], [math.pi / 6], approx=True)
        assert_operator(ASIN, [1 / math.sqrt(2)], [math.pi / 4], approx=True)
        assert_operator(ASIN, [math.sqrt(3) / 2], [math.pi / 3], approx=True)
        assert_operator(ASIN, [1.0], [math.pi / 2], approx=True)
        assert_operator(
            ASIN,
            [np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            approx=True)
        assert_operator(
            ASIN,
            [-np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            approx=True)
        # extra stack elements
        assert_operator(ASIN, [0, 1.0], [0, math.pi / 2], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ASIN([], {})

    def test_acos(self):
        assert repr(ACOS) == 'ACOS'
        assert_operator(ACOS, [1.0], [0.0], approx=True)
        assert_operator(ACOS, [math.sqrt(3) / 2], [math.pi / 6], approx=True)
        assert_operator(ACOS, [1 / math.sqrt(2)], [math.pi / 4], approx=True)
        assert_operator(ACOS, [1 / 2], [math.pi / 3], approx=True)
        assert_operator(ACOS, [0.0], [math.pi / 2], approx=True)
        assert_operator(
            ACOS,
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            approx=True)
        # extra stack elements
        assert_operator(ACOS, [0, 0.0], [0, math.pi / 2], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ACOS([], {})

    def test_atan(self):
        assert repr(ATAN) == 'ATAN'
        assert_operator(ATAN, [0.0], [0.0], approx=True)
        assert_operator(ATAN, [1 / math.sqrt(3)], [math.pi / 6], approx=True)
        assert_operator(ATAN, [1.0], [math.pi / 4], approx=True)
        assert_operator(ATAN, [math.sqrt(3)], [math.pi / 3], approx=True)
        assert_operator(
            ATAN,
            [np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])],
            approx=True)
        assert_operator(
            ATAN,
            [-np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])],
            approx=True)
        # extra stack elements
        assert_operator(ATAN, [0, 1.0], [0, math.pi / 4], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ATAN([], {})

    def test_asind(self):
        assert repr(ASIND) == 'ASIND'
        assert_operator(ASIND, [0.0], [0], approx=True)
        assert_operator(ASIND, [1 / 2], [30], approx=True)
        assert_operator(ASIND, [1 / math.sqrt(2)], [45], approx=True)
        assert_operator(ASIND, [math.sqrt(3) / 2], [60], approx=True)
        assert_operator(ASIND, [1.0], [90], approx=True)
        assert_operator(
            ASIND,
            [np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            [np.array([0, 30, 45, 60, 90])],
            approx=True)
        assert_operator(
            ASIND,
            [-np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            [-np.array([0, 30, 45, 60, 90])],
            approx=True)
        # extra stack elements
        assert_operator(ASIND, [0, 1.0], [0, 90], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ASIND([], {})

    def test_acosd(self):
        assert repr(ACOSD) == 'ACOSD'
        assert_operator(ACOSD, [1.0], [0], approx=True)
        assert_operator(ACOSD, [math.sqrt(3) / 2], [30], approx=True)
        assert_operator(ACOSD, [1 / math.sqrt(2)], [45], approx=True)
        assert_operator(ACOSD, [1 / 2], [60], approx=True)
        assert_operator(ACOSD, [0.0], [90], approx=True)
        assert_operator(
            ACOSD,
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            [np.array([0, 30, 45, 60, 90])],
            approx=True)
        # extra stack elements
        assert_operator(ACOSD, [0, 0.0], [0, 90], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ACOSD([], {})

    def test_atand(self):
        assert repr(ATAND) == 'ATAND'
        assert_operator(ATAND, [0.0], [0], approx=True)
        assert_operator(ATAND, [1 / math.sqrt(3)], [30], approx=True)
        assert_operator(ATAND, [1.0], [45], approx=True)
        assert_operator(ATAND, [math.sqrt(3)], [60], approx=True)
        assert_operator(
            ATAND,
            [np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            [np.array([0, 30, 45, 60])],
            approx=True)
        assert_operator(
            ATAND,
            [-np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            [-np.array([0, 30, 45, 60])],
            approx=True)
        # extra stack elements
        assert_operator(ATAND, [0, 1.0], [0, 45], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ATAND([], {})

    def test_asinh(self):
        assert repr(ASINH) == 'ASINH'
        assert_operator(ASINH, [0.0], [0.0], approx=True)
        assert_operator(ASINH, [0.5], [GOLDEN_RATIO], approx=True)
        assert_operator(
            ASINH,
            [np.array([0.0, 0.5])],
            [np.array([0.0, GOLDEN_RATIO])],
            approx=True)
        # extra stack elements
        assert_operator(ASINH, [0, 0.5], [0, GOLDEN_RATIO], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ASINH([], {})

    def test_acosh(self):
        assert repr(ACOSH) == 'ACOSH'
        assert_operator(ACOSH, [1.0], [0.0], approx=True)
        assert_operator(ACOSH, [math.sqrt(5) / 2], [GOLDEN_RATIO], approx=True)
        assert_operator(
            ACOSH,
            [np.array([1.0, np.sqrt(5) / 2])],
            [np.array([0.0, GOLDEN_RATIO])],
            approx=True)
        # extra stack elements
        assert_operator(
            ACOSH, [0, math.sqrt(5) / 2], [0, GOLDEN_RATIO], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ACOSH([], {})

    def test_atanh(self):
        assert repr(ATANH) == 'ATANH'
        assert_operator(ATANH, [0.0], [0.0], approx=True)
        assert_operator(ATANH, [math.sqrt(5) / 5], [GOLDEN_RATIO], approx=True)
        assert_operator(
            ATANH,
            [np.array([0.0, np.sqrt(5) / 5])],
            [np.array([0.0, GOLDEN_RATIO])],
            approx=True)
        # extra stack elements
        assert_operator(
            ATANH, [0, math.sqrt(5) / 5], [0, GOLDEN_RATIO], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ATANH([], {})

    def test_isnan(self):
        assert repr(ISNAN) == 'ISNAN'
        assert_operator(ISNAN, [2], [False])
        assert_operator(ISNAN, [float('nan')], [True])
        assert_operator(
            ISNAN, [np.array([4, np.nan])], [np.array([False, True])])
        assert_operator(
            ISNAN, [np.array([np.nan, 1])], [np.array([True, False])])
        # extra stack elements
        assert_operator(ISNAN, [0, float('nan')], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ISNAN([], {})

    def test_isan(self):
        assert repr(ISAN) == 'ISAN'
        assert_operator(ISAN, [2], [True])
        assert_operator(ISAN, [float('nan')], [False])
        assert_operator(
            ISAN, [np.array([4, np.nan])], [np.array([True, False])])
        assert_operator(
            ISAN, [np.array([np.nan, 1])], [np.array([False, True])])
        # extra stack elements
        assert_operator(ISAN, [0, 2], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ISAN([], {})

    def test_rint(self):
        assert repr(RINT) == 'RINT'
        assert_operator(RINT, [1.6], [2])
        assert_operator(RINT, [2.4], [2])
        assert_operator(RINT, [-1.6], [-2])
        assert_operator(RINT, [-2.4], [-2])
        assert_operator(RINT, [np.array([1.6, 2.4])], [np.array([2, 2])])
        assert_operator(RINT, [np.array([-1.6, -2.4])], [np.array([-2, -2])])
        # extra stack elements
        assert_operator(RINT, [0, 1.6], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            RINT([], {})

    def test_nint(self):
        assert repr(NINT) == 'NINT'
        assert_operator(NINT, [1.6], [2])
        assert_operator(NINT, [2.4], [2])
        assert_operator(NINT, [-1.6], [-2])
        assert_operator(NINT, [-2.4], [-2])
        assert_operator(NINT, [np.array([1.6, 2.4])], [np.array([2, 2])])
        assert_operator(NINT, [np.array([-1.6, -2.4])], [np.array([-2, -2])])
        # extra stack elements
        assert_operator(NINT, [0, 1.6], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            NINT([], {})

    def test_ceil(self):
        assert repr(CEIL) == 'CEIL'
        assert_operator(CEIL, [1.6], [2])
        assert_operator(CEIL, [2.4], [3])
        assert_operator(CEIL, [-1.6], [-1])
        assert_operator(CEIL, [-2.4], [-2])
        assert_operator(CEIL, [np.array([1.6, 2.4])], [np.array([2, 3])])
        assert_operator(CEIL, [np.array([-1.6, -2.4])], [np.array([-1, -2])])
        # extra stack elements
        assert_operator(CEIL, [0, 1.2], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            CEIL([], {})

    def test_ceiling(self):
        assert repr(CEILING) == 'CEILING'
        assert_operator(CEILING, [1.6], [2])
        assert_operator(CEILING, [2.4], [3])
        assert_operator(CEILING, [-1.6], [-1])
        assert_operator(CEILING, [-2.4], [-2])
        assert_operator(CEILING, [np.array([1.6, 2.4])], [np.array([2, 3])])
        assert_operator(
            CEILING, [np.array([-1.6, -2.4])], [np.array([-1, -2])])
        # extra stack elements
        assert_operator(CEILING, [0, 1.2], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            CEILING([], {})

    def test_floor(self):
        assert repr(FLOOR) == 'FLOOR'
        assert_operator(FLOOR, [1.6], [1])
        assert_operator(FLOOR, [2.4], [2])
        assert_operator(FLOOR, [-1.6], [-2])
        assert_operator(FLOOR, [-2.4], [-3])
        assert_operator(FLOOR, [np.array([1.6, 2.4])], [np.array([1, 2])])
        assert_operator(FLOOR, [np.array([-1.6, -2.4])], [np.array([-2, -3])])
        # extra stack elements
        assert_operator(FLOOR, [0, 1.8], [0, 1])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            FLOOR([], {})

    def test_d2r(self):
        assert repr(D2R) == 'D2R'
        assert_operator(D2R, [0], [0.0], approx=True)
        assert_operator(D2R, [30], [math.pi / 6], approx=True)
        assert_operator(D2R, [45], [math.pi / 4], approx=True)
        assert_operator(D2R, [60], [math.pi / 3], approx=True)
        assert_operator(D2R, [90], [math.pi / 2], approx=True)
        assert_operator(
            D2R,
            [np.array([0, 30, 45, 60, 90])],
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            approx=True)
        assert_operator(
            D2R,
            [-np.array([0, 30, 45, 60, 90])],
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            approx=True)
        # extra stack elements
        assert_operator(D2R, [0, 90], [0, math.pi / 2], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            D2R([], {})

    def test_r2d(self):
        assert repr(R2D) == 'R2D'
        assert_operator(R2D, [0.0], [0], approx=True)
        assert_operator(R2D, [math.pi / 6], [30], approx=True)
        assert_operator(R2D, [math.pi / 4], [45], approx=True)
        assert_operator(R2D, [math.pi / 3], [60], approx=True)
        assert_operator(R2D, [math.pi / 2], [90], approx=True)
        assert_operator(
            R2D,
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [np.array([0, 30, 45, 60, 90])],
            approx=True)
        assert_operator(
            R2D,
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [-np.array([0, 30, 45, 60, 90])],
            approx=True)
        # extra stack elements
        assert_operator(R2D, [0, math.pi / 2], [0, 90], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            R2D([], {})

    def test_ymdhms(self):
        assert repr(YMDHMS) == 'YMDHMS'
        epoch = datetime(1985, 1, 1, 0, 0, 0, 0)
        date1 = datetime(2008, 7, 4, 12, 19, 19, 570865)
        date2 = datetime(2019, 6, 26, 12, 31, 6, 930575)
        seconds1 = (date1 - epoch).total_seconds()
        seconds2 = (date2 - epoch).total_seconds()
        assert_operator(YMDHMS, [seconds1], [80704121919.570865], approx=True)
        assert_operator(YMDHMS, [seconds2], [190626123106.930575], approx=True)
        assert_operator(
            YMDHMS,
            [np.array([seconds1, seconds2])],
            [np.array([80704121919.570865, 190626123106.930575])],
            approx=True)
        # extra stack elements
        assert_operator(
            YMDHMS, [0, seconds1], [0, 80704121919.570865], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            YMDHMS([], {})

    def test_sum(self):
        assert repr(SUM) == 'SUM'
        assert_operator(SUM, [2], [2])
        assert_operator(SUM, [-2], [-2])
        assert_operator(SUM, [float('nan')], [0])
        assert_operator(SUM, [np.array([4, -1])], [3])
        assert_operator(SUM, [np.array([-4, 1])], [-3])
        assert_operator(SUM, [np.array([1, np.nan, 3])], [4])
        assert_operator(SUM, [np.array([np.nan])], [0])
        # extra stack elements
        assert_operator(SUM, [0, 2], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SUM([], {})

    def test_diff(self):
        assert repr(DIF) == 'DIF'
        assert_operator(DIF, [2], [np.array([np.nan])])
        assert_operator(DIF, [np.array([1, 2])], [np.array([np.nan, 1])])
        assert_operator(DIF, [np.array([1, 2, 5])], [np.array([np.nan, 1, 3])])
        assert_operator(
            DIF,
            [np.array([1, np.nan, 5])],
            [np.array([np.nan, np.nan, np.nan])])
        # extra stack elements
        assert_operator(DIF, [0, 2], [0, np.array([np.nan])])
        with pytest.raises(StackUnderflowError):
            DIF([], {})

    def test_dup(self):
        assert repr(DUP) == 'DUP'
        assert_operator(DUP, [2], [2, 2])
        assert_operator(
            DUP, [np.array([4, -1])], [np.array([4, -1]), np.array([4, -1])])
        # extra stack elements
        assert_operator(DUP, [0, 2], [0, 2, 2])
        with pytest.raises(StackUnderflowError):
            DUP([], {})

    def test_div(self):
        assert repr(DIV) == 'DIV'
        assert_operator(DIV, [10, 2], [5])
        assert_operator(DIV, [10, np.array([2, 5])], [np.array([5, 2])])
        assert_operator(DIV, [np.array([10, 4]), 2], [np.array([5, 2])])
        assert_operator(
            DIV, [np.array([8, 16]), np.array([2, 4])], [np.array([4, 4])])
        # extra stack elements
        assert_operator(DIV, [0, 10, 2], [0, 5])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            DIV([], {})
        with pytest.raises(StackUnderflowError):
            DIV([1], {})

    def test_pow(self):
        assert repr(POW) == 'POW'
        assert_operator(POW, [1, 2], [1])
        assert_operator(POW, [2, 2], [4])
        assert_operator(POW, [2, 4], [16])
        assert_operator(POW, [2, np.array([1, 2, 3])], [np.array([2, 4, 8])])
        assert_operator(POW, [np.array([1, 2, 3]), 2], [np.array([1, 4, 9])])
        assert_operator(
            POW, [np.array([2, 3]), np.array([5, 6])], [np.array([32, 729])])
        # extra stack elements
        assert_operator(POW, [0, 2, 4], [0, 16])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            POW([], {})
        with pytest.raises(StackUnderflowError):
            POW([1], {})

    def test_fmod(self):
        assert repr(FMOD) == 'FMOD'
        assert_operator(FMOD, [1, 2], [1])
        assert_operator(FMOD, [2, 10], [2])
        assert_operator(FMOD, [12, 10], [2])
        assert_operator(FMOD, [13, np.array([10, 100])], [np.array([3, 13])])
        assert_operator(FMOD, [np.array([7, 15]), 10], [np.array([7, 5])])
        assert_operator(
            FMOD, [np.array([7, 15]), np.array([10, 5])], [np.array([7, 0])])
        # extra stack elements
        assert_operator(FMOD, [0, 12, 10], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            FMOD([], {})
        with pytest.raises(StackUnderflowError):
            FMOD([1], {})

    def test_min(self):
        assert repr(MIN) == 'MIN'
        assert_operator(MIN, [2, 3], [2])
        assert_operator(MIN, [3, 2], [2])
        assert_operator(MIN, [2, np.array([1, 3])], [np.array([1, 2])])
        assert_operator(MIN, [np.array([1, 3]), 2], [np.array([1, 2])])
        assert_operator(
            MIN, [np.array([2, 3]), np.array([3, 2])], [np.array([2, 2])])
        # # extra stack elements
        assert_operator(MIN, [0, 2, 3], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            MIN([], {})
        with pytest.raises(StackUnderflowError):
            MIN([1], {})

    def test_max(self):
        assert repr(MAX) == 'MAX'
        assert_operator(MAX, [2, 3], [3])
        assert_operator(MAX, [3, 2], [3])
        assert_operator(MAX, [2, np.array([1, 3])], [np.array([2, 3])])
        assert_operator(MAX, [np.array([1, 3]), 2], [np.array([2, 3])])
        assert_operator(
            MAX, [np.array([2, 3]), np.array([3, 2])], [np.array([3, 3])])
        # # extra stack elements
        assert_operator(MAX, [0, 2, 3], [0, 3])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            MAX([], {})
        with pytest.raises(StackUnderflowError):
            MAX([1], {})

    def test_atan2(self):
        assert repr(ATAN2) == 'ATAN2'
        # NOTE: second parameter is x, first is y
        assert_operator(ATAN2, [0, 1], [0], approx=True)
        assert_operator(ATAN2, [1, math.sqrt(3)], [math.pi / 6], approx=True)
        assert_operator(ATAN2, [1, 1], [math.pi / 4], approx=True)
        assert_operator(ATAN2, [math.sqrt(3), 1], [math.pi / 3], approx=True)
        assert_operator(ATAN2, [1, 0], [math.pi / 2], approx=True)
        assert_operator(
            ATAN2,
            [math.sqrt(3), -1],
            [math.pi / 2 + math.pi / 6],
            approx=True)
        assert_operator(
            ATAN2, [1, -1], [math.pi / 2 + math.pi / 4], approx=True)
        assert_operator(
            ATAN2,
            [1, -math.sqrt(3)],
            [math.pi / 2 + math.pi / 3],
            approx=True)
        assert_operator(
            ATAN2, [0, -1], [math.pi / 2 + math.pi / 2], approx=True)
        assert_operator(
            ATAN2,
            [np.array([0, 1, 1, np.sqrt(3), 1, np.sqrt(3), 1, 1, 0]),
             np.array([1, np.sqrt(3), 1, 1, 0, -1, -1, -np.sqrt(3), -1])],
            [np.array(
                [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2,
                 np.pi / 2 + np.pi / 6, np.pi / 2 + np.pi / 4,
                 np.pi / 2 + np.pi / 3, np.pi / 2 + np.pi / 2])],
            approx=True)
        # extra stack elements
        assert_operator(ATAN2, [0, 1, 1], [0, math.pi / 4], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ATAN2([], {})

    def test_hypot(self):
        assert repr(HYPOT) == 'HYPOT'
        assert_operator(HYPOT, [1, 1], [math.sqrt(2)], approx=True)
        assert_operator(HYPOT, [math.sqrt(3), 1], [2], approx=True)
        assert_operator(
            HYPOT,
            [1, np.array([np.sqrt(3), 1])],
            [np.array([2, np.sqrt(2)])],
            approx=True)
        assert_operator(
            HYPOT,
            [np.array([np.sqrt(3), 1]), 1],
            [np.array([2, np.sqrt(2)])],
            approx=True)
        assert_operator(
            HYPOT,
            [np.array([np.sqrt(3), 1]), np.array([1, 1])],
            [np.array([2, np.sqrt(2)])],
            approx=True)
        # extra stack elements
        assert_operator(HYPOT, [0, math.sqrt(3), 1], [0, 2], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            HYPOT([], {})
        with pytest.raises(StackUnderflowError):
            HYPOT([1], {})

    def test_r2(self):
        assert repr(R2) == 'R2'
        assert_operator(R2, [2, 3], [13])
        assert_operator(R2, [2, np.array([3, 4])], [np.array([13, 20])])
        assert_operator(R2, [np.array([3, 4]), 2], [np.array([13, 20])])
        assert_operator(
            R2, [np.array([1, 2]), np.array([3, 4])], [np.array([10, 20])])
        # extra stack elements
        assert_operator(R2, [0, 2, 3], [0, 13], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            R2([], {})
        with pytest.raises(StackUnderflowError):
            R2([1], {})

    def test_eq(self):
        assert repr(EQ) == 'EQ'
        assert_operator(EQ, [2, 2], [True])
        assert_operator(EQ, [2, 3], [False])
        assert_operator(
            EQ,
            [2, np.array([1, np.nan, 2])],
            [np.array([False, False, True])])
        assert_operator(
            EQ,
            [np.array([1, np.nan, 2]), 2],
            [np.array([False, False, True])])
        assert_operator(
            EQ,
            [np.array([1, np.nan, 3, 3]), np.array([1, np.nan, 2, 3])],
            [np.array([True, False, False, True])])
        # extra stack elements
        assert_operator(EQ, [0, 2, 2], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            EQ([], {})
        with pytest.raises(StackUnderflowError):
            EQ([1], {})

    def test_ne(self):
        assert repr(NE) == 'NE'
        assert_operator(NE, [2, 2], [False])
        assert_operator(NE, [2, 3], [True])
        assert_operator(
            NE,
            [2, np.array([1, np.nan, 2])],
            [np.array([True, True, False])])
        assert_operator(
            NE,
            [np.array([1, np.nan, 2]), 2],
            [np.array([True, True, False])])
        assert_operator(
            NE,
            [np.array([1, np.nan, 3, 3]), np.array([1, np.nan, 2, 3])],
            [np.array([False, True, True, False])])
        # extra stack elements
        assert_operator(NE, [0, 2, 2], [0, False])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            NE([], {})
        with pytest.raises(StackUnderflowError):
            NE([1], {})

    def test_lt(self):
        assert repr(LT) == 'LT'
        assert_operator(LT, [2, 3], [True])
        assert_operator(LT, [2, 2], [False])
        assert_operator(LT, [3, 2], [False])
        assert_operator(
            LT, [2, np.array([1, 2, 3])], [np.array([False, False, True])])
        assert_operator(
            LT, [np.array([1, 2, 3]), 2], [np.array([True, False, False])])
        assert_operator(
            LT,
            [np.array([1, 2, 3]), np.array([3, 2, 1])],
            [np.array([True, False, False])])
        # extra stack elements
        assert_operator(LT, [0, 2, 3], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            LT([], {})
        with pytest.raises(StackUnderflowError):
            LT([1], {})

    def test_le(self):
        assert repr(LE) == 'LE'
        assert_operator(LE, [2, 3], [True])
        assert_operator(LE, [2, 2], [True])
        assert_operator(LE, [3, 2], [False])
        assert_operator(
            LE, [2, np.array([1, 2, 3])], [np.array([False, True, True])])
        assert_operator(
            LE, [np.array([1, 2, 3]), 2], [np.array([True, True, False])])
        assert_operator(
            LE,
            [np.array([1, 2, 3]), np.array([3, 2, 1])],
            [np.array([True, True, False])])
        # # extra stack elements
        assert_operator(LE, [0, 2, 3], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            LE([], {})
        with pytest.raises(StackUnderflowError):
            LE([1], {})

    def test_gt(self):
        assert repr(GT) == 'GT'
        assert_operator(GT, [2, 3], [False])
        assert_operator(GT, [2, 2], [False])
        assert_operator(GT, [3, 2], [True])
        assert_operator(
            GT, [2, np.array([1, 2, 3])], [np.array([True, False, False])])
        assert_operator(
            GT, [np.array([1, 2, 3]), 2], [np.array([False, False, True])])
        assert_operator(
            GT,
            [np.array([1, 2, 3]), np.array([3, 2, 1])],
            [np.array([False, False, True])])
        # extra stack elements
        assert_operator(GT, [0, 2, 3], [0, False])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            GT([], {})
        with pytest.raises(StackUnderflowError):
            GT([1], {})

    def test_ge(self):
        assert repr(GE) == 'GE'
        assert_operator(GE, [2, 3], [False])
        assert_operator(GE, [2, 2], [True])
        assert_operator(GE, [3, 2], [True])
        assert_operator(
            GE, [2, np.array([1, 2, 3])], [np.array([True, True, False])])
        assert_operator(
            GE, [np.array([1, 2, 3]), 2], [np.array([False, True, True])])
        assert_operator(
            GE,
            [np.array([1, 2, 3]), np.array([3, 2, 1])],
            [np.array([False, True, True])])
        # extra stack elements
        assert_operator(GE, [0, 2, 3], [0, False])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            GE([], {})
        with pytest.raises(StackUnderflowError):
            GE([1], {})

    def test_nan(self):
        assert repr(NAN) == 'NAN'
        assert_operator(NAN, [2, 2], [float('nan')])
        assert_operator(NAN, [2, 3], [2])
        assert_operator(NAN, [2, np.array([2, 3])], [np.array([np.nan, 2])])
        assert_operator(NAN, [np.array([2, 3]), 2], [np.array([np.nan, 3])])
        assert_operator(
            NAN,
            [np.array([1, 2, 3]), np.array([3, 2, 1])],
            [np.array([1, np.nan, 3])])
        # as float
        assert_operator(
            NAN,
            [np.array([1.0, 2.0, 3.0]), np.array([3, 2, 1])],
            [np.array([1, np.nan, 3])],
            approx=True)
        # extra stack elements
        assert_operator(NAN, [0, 2, 2], [0, float('nan')])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            NAN([], {})
        with pytest.raises(StackUnderflowError):
            NAN([1], {})

    def test_and(self):
        assert repr(AND) == 'AND'
        assert_operator(AND, [2, 3], [2])
        assert_operator(AND, [float('nan'), 3], [3])
        assert_operator(
            AND, [float('nan'), np.array([2, 3])], [np.array([2, 3])])
        assert_operator(AND, [np.array([np.nan, 3]), 2], [np.array([2, 3])])
        assert_operator(
            AND,
            [np.array([10, np.nan, 30]), np.array([1, 2, 3])],
            [np.array([10, 2, 30])])
        # extra stack elements
        assert_operator(AND, [0, float('nan'), 3], [0, 3])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            AND([], {})
        with pytest.raises(StackUnderflowError):
            AND([1], {})

    def test_or(self):
        assert repr(OR) == 'OR'
        assert_operator(OR, [2, 3], [2])
        assert_operator(OR, [2, float('nan')], [float('nan')])
        assert_operator(
            OR, [2, np.array([3, np.nan])], [np.array([2, np.nan])])
        assert_operator(
            OR, [np.array([2, 3]), np.nan], [np.array([np.nan, np.nan])])
        assert_operator(
            OR,
            [np.array([1, 2, 3]), np.array([10, np.nan, 30])],
            [np.array([1, np.nan, 3])])
        # as float
        assert_operator(
            OR,
            [np.array([1.0, 2.0, 3.0]), np.array([10, np.nan, 30])],
            [np.array([1, np.nan, 3])])
        # extra stack elements
        assert_operator(OR, [0, 2, float('nan')], [0, float('nan')])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            OR([], {})
        with pytest.raises(StackUnderflowError):
            OR([1], {})

    def test_iand(self):
        assert repr(IAND) == 'IAND'
        assert_operator(IAND, [5, 3], [1])
        assert_operator(IAND, [15, 21], [5])
        assert_operator(IAND, [21, 15], [5])
        assert_operator(
            IAND, [15, np.array([9, 21, 35])], [np.array([9, 5, 3])])
        assert_operator(
            IAND, [np.array([9, 21, 35]), 15], [np.array([9, 5, 3])])
        assert_operator(
            IAND,
            [np.array([9, 21, 35]), np.array([3, 15, 127])],
            [np.array([1, 5, 35])])
        # extra stack elements
        assert_operator(IAND, [0, 15, 21], [0, 5])
        # floats are not supported
        with pytest.raises(TypeError):
            IAND([1.0, 2], {})
        with pytest.raises(TypeError):
            IAND([1, 2.0], {})
        with pytest.raises(TypeError):
            IAND([1, np.array([2.0, 3.0])], {})
        with pytest.raises(TypeError):
            IAND([np.array([2.0, 3.0]), 1], {})
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            IAND([], {})
        with pytest.raises(StackUnderflowError):
            IAND([1], {})

    def test_ior(self):
        assert repr(IOR) == 'IOR'
        assert_operator(IOR, [5, 3], [7])
        assert_operator(IOR, [15, 21], [31])
        assert_operator(IOR, [21, 15], [31])
        assert_operator(
            IOR, [15, np.array([9, 21, 35])], [np.array([15, 31, 47])])
        assert_operator(
            IOR, [np.array([9, 21, 35]), 15], [np.array([15, 31, 47])])
        assert_operator(
            IOR,
            [np.array([9, 21, 35]), np.array([3, 15, 127])],
            [np.array([11, 31, 127])])
        # extra stack elements
        assert_operator(IOR, [0, 15, 21], [0, 31])
        # floats are not supported
        with pytest.raises(TypeError):
            IOR([1.0, 2], {})
        with pytest.raises(TypeError):
            IOR([1, 2.0], {})
        with pytest.raises(TypeError):
            IOR([1, np.array([2.0, 3.0])], {})
        with pytest.raises(TypeError):
            IOR([np.array([2.0, 3.0]), 1], {})
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            IOR([], {})
        with pytest.raises(StackUnderflowError):
            IOR([1], {})

    def test_btest(self):
        assert repr(BTEST) == 'BTEST'
        assert_operator(BTEST, [9, 0], [True])
        assert_operator(BTEST, [9, 1], [False])
        assert_operator(BTEST, [9, 2], [False])
        assert_operator(BTEST, [9, 3], [True])
        assert_operator(BTEST, [9, 4], [False])
        assert_operator(
            BTEST,
            [9, np.array([0, 1, 2, 3, 4])],
            [np.array([True, False, False, True, False])])
        assert_operator(
            BTEST, [np.array([1, 3, 5]), 1], [np.array([False, True, False])])
        assert_operator(
            BTEST,
            [np.array([1, 3, 5]), np.array([1, 2, 0])],
            [np.array([False, False, True])])
        # extra stack elements
        assert_operator(BTEST, [0, 9, 3], [0, True])
        # floats are not supported
        with pytest.raises(TypeError):
            BTEST([1.0, 2], {})
        with pytest.raises(TypeError):
            BTEST([1, 2.0], {})
        with pytest.raises(TypeError):
            BTEST([1, np.array([2.0, 3.0])], {})
        with pytest.raises(TypeError):
            BTEST([np.array([2.0, 3.0]), 1], {})
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            BTEST([], {})
        with pytest.raises(StackUnderflowError):
            BTEST([1], {})

    def test_avg(self):
        assert repr(AVG) == 'AVG'
        assert_operator(AVG, [5, 11], [8])
        assert_operator(AVG, [float('nan'), 11], [11])
        assert_operator(AVG, [5, float('nan')], [5])
        assert_operator(
            AVG, [3, np.array([7, np.nan, 11])], [np.array([5, 3, 7])])
        assert_operator(
            AVG, [np.nan, np.array([1, 2, 3])], [np.array([1, 2, 3])])
        assert_operator(
            AVG, [np.array([7, np.nan, 11]), 3], [np.array([5, 3, 7])])
        assert_operator(
            AVG, [np.array([1, 2, 3]), np.nan], [np.array([1, 2, 3])])
        assert_operator(
            AVG,
            [np.array([3, np.nan, 11]), np.array([7, 2, np.nan])],
            [np.array([5, 2, 11])])
        # extra stack elements
        assert_operator(AVG, [0, 5, 11], [0, 8])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            AVG([], {})
        with pytest.raises(StackUnderflowError):
            AVG([1], {})

    def test_dxdy(self):
        assert repr(DXDY) == 'DXDY'
        assert_operator(DXDY, [5, 11], [float('nan')])
        assert_operator(
            DXDY, [3, np.array([5, 11])], [np.array([np.nan, np.nan])])
        assert_operator(
            DXDY, [3, np.array([5, 7, 11])], [np.array([np.nan, 0, np.nan])])
        assert_operator(
            DXDY,
            [3, np.array([5, 7, 8, 11])],
            [np.array([np.nan, 0, 0, np.nan])])
        with warnings.catch_warnings():  # divide by zero in numpy
            warnings.simplefilter('ignore')
            assert_operator(
                DXDY, [np.array([5, 11]), 3], [np.array([np.nan, np.nan])])
            assert_operator(
                DXDY,
                [np.array([5, 7, 11]), 3],
                [np.array([np.nan, np.inf, np.nan])])
            assert_operator(
                DXDY,
                [np.array([5, 7, 8, 11]), 3],
                [np.array([np.nan, np.inf, np.inf, np.nan])])
        assert_operator(
            DXDY,
            [np.array([0, 4, 9, 8]), np.array([5, 7, 8, 11])],
            [np.array([np.nan, 3, 1, np.nan])])
        # extra stack elements
        assert_operator(DXDY, [0, 5, 11], [0, float('nan')])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            DXDY([], {})
        with pytest.raises(StackUnderflowError):
            DXDY([1], {})

    def test_exch(self):
        assert repr(EXCH) == 'EXCH'
        assert_operator(EXCH, [5, 11], [11, 5])
        assert_operator(
            EXCH, [3, np.array([5, 11])], [np.array([5, 11]), 3])
        assert_operator(
            EXCH, [np.array([5, 11]), 3], [3, np.array([5, 11])])
        assert_operator(
            EXCH,
            [np.array([1, 2]), np.array([3, 4])],
            [np.array([3, 4]), np.array([1, 2])])
        # extra stack elements
        assert_operator(EXCH, [0, 5, 11], [0, 11, 5])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            EXCH([], {})
        with pytest.raises(StackUnderflowError):
            EXCH([1], {})

    def test_inrange(self):
        assert repr(INRANGE) == 'INRANGE'
        assert_operator(INRANGE, [0, 1, 3], [False])
        assert_operator(INRANGE, [1, 1, 3], [True])
        assert_operator(INRANGE, [2, 1, 3], [True])
        assert_operator(INRANGE, [3, 1, 3], [True])
        assert_operator(INRANGE, [4, 1, 3], [False])
        assert_operator(
            INRANGE,
            [np.array([0, 1, 2, 3, 4]), 1, 3],
            [np.array([False, True, True, True, False])])
        assert_operator(
            INRANGE,
            [2, np.array([1, 2, 3]), 3],
            [np.array([True, True, False])])
        assert_operator(
            INRANGE,
            [2, 1, np.array([1, 2, 3])],
            [np.array([False, True, True])])
        assert_operator(
            INRANGE,
            [np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])],
            [np.array([True, True, True])])
        # extra stack elements
        assert_operator(INRANGE, [0, 2, 1, 3], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            INRANGE([], {})
        with pytest.raises(StackUnderflowError):
            INRANGE([1], {})
        with pytest.raises(StackUnderflowError):
            INRANGE([1, 2], {})

    def test_boxcar(self):
        assert repr(BOXCAR) == 'BOXCAR'
        # returns value if scalar
        assert_operator(BOXCAR, [1, 2, 3], [1])
        # simple
        assert_operator(
            BOXCAR,
            [np.array([0, 1, 2, 3, 4]), 0, 3],
            [np.array([1 / 3, 1, 2, 3, 11 / 3])],
            approx=True)
        # window size of 1 should return original array
        assert_operator(
            BOXCAR,
            [np.array([0, 1, 2, 3, 4]), 0, 1],
            [np.array([0, 1, 2, 3, 4])],
            approx=True)
        # with nan it should base result on non nan values in window
        assert_operator(
            BOXCAR,
            [np.array([0, np.nan, 2, 3, 4]), 0, 3],
            [np.array([0, np.nan, 2.5, 3, 11 / 3])],
            approx=True)
        # multi-dimensional x
        assert_operator(
            BOXCAR,
            [np.array([[0, 1, 2, 3, 4], [0, 10, 20, 30, 40]]), 1, 3],
            [np.array([[1 / 3, 1, 2, 3, 11 / 3],
                       [10 / 3, 10, 20, 30, 110 / 3]])],
            approx=True)
        assert_operator(
            BOXCAR,
            [np.array([[0, 1, 2, 3, 4],
                       [0, 10, 20, 30, 40],
                       [0, 100, 200, 300, 400]]), 0, 3],
            [np.array([[0, 4, 8, 12, 16],
                       [0, 37, 74, 111, 148],
                       [0, 70, 140, 210, 280]])],
            approx=True)
        # extra stack elements
        assert_operator(BOXCAR, [0, 1, 2, 3], [0, 1])
        # y must be a scalar
        with pytest.raises(ValueError):
            BOXCAR([np.array([1, 2, 3]), np.array([1, 2]), 3], {})
        # z must be a scalar
        with pytest.raises(ValueError):
            BOXCAR([np.array([1, 2, 3]), 2, np.array([1, 2])], {})
        # x must have dimension y
        with pytest.raises(IndexError):
            BOXCAR([np.array([1, 2, 3]), 1, 3], {})
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            BOXCAR([], {})
        with pytest.raises(StackUnderflowError):
            BOXCAR([1], {})
        with pytest.raises(StackUnderflowError):
            BOXCAR([1, 2], {})

    def test_gauss(self):
        # NOTE: This operator is not tested thoroughly.
        assert repr(GAUSS) == 'GAUSS'
        # returns value if scalar
        assert_operator(GAUSS, [1, 2, 3], [1])
        # simple
        assert_operator(
            GAUSS,
            [np.array([0, 1, 2, 3, 4]), 0, 3],
            [np.array([1.06300638, 1.5089338, 2.0, 2.4910662, 2.93699362])],
            approx=True, rtol=1e-6, atol=1e-6)
        # with nan it should base result on non nan values in window
        assert_operator(
            GAUSS,
            [np.array([0, np.nan, 2, 3, 4]), 0, 3],
            [np.array(
                [1.07207303, np.nan, 2.14390036, 2.66876589, 3.10693751])],
            approx=True, rtol=1e-6, atol=1e-6)
        # multi-dimensional x
        assert_operator(
            GAUSS,
            [np.array([[0, 1, 2, 3, 4], [0, 10, 20, 30, 40]]), 1, 3],
            [np.array(
                [[1.06300638, 1.5089338, 2.0, 2.4910662, 2.93699362],
                 [10.63006385, 15.08933801, 20.0, 24.91066199, 29.36993615]])],
            approx=True, rtol=1e-6, atol=1e-6)
        assert_operator(
            GAUSS,
            [np.array([[0, 1, 2, 3, 4],
                       [0, 10, 20, 30, 40],
                       [0, 100, 200, 300, 400]]), 0, 3],
            [np.array(
                [[0, 32.59544683, 65.19089366, 97.78634049, 130.38178732],
                 [0, 45.11412619, 90.22825238, 135.34237857, 180.45650475],
                 [0, 58.21491651, 116.42983302, 174.64474953, 232.85966604]])],
            approx=True, rtol=1e-6, atol=1e-6)
        # extra stack elements
        assert_operator(GAUSS, [0, 1, 2, 3], [0, 1])
        # y must be a scalar
        with pytest.raises(ValueError):
            GAUSS([np.array([1, 2, 3]), np.array([1, 2]), 3], {})
        # z must be a scalar
        with pytest.raises(ValueError):
            GAUSS([np.array([1, 2, 3]), 2, np.array([1, 2])], {})
        # x must have dimension y
        with pytest.raises(IndexError):
            GAUSS([np.array([1, 2, 3]), 1, 3], {})
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            GAUSS([], {})
        with pytest.raises(StackUnderflowError):
            GAUSS([1], {})
        with pytest.raises(StackUnderflowError):
            GAUSS([1, 2], {})
