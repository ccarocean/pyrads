import math
import warnings
from copy import copy, deepcopy
from datetime import datetime
from typing import Mapping, MutableMapping, MutableSequence, Optional

import numpy as np  # type: ignore
import pytest  # type: ignore

from rads._typing import NumberOrArray
from rads.rpn import (
    ABS,
    ACOS,
    ACOSD,
    ACOSH,
    ADD,
    AND,
    ASIN,
    ASIND,
    ASINH,
    ATAN,
    ATAN2,
    ATAND,
    ATANH,
    AVG,
    BOXCAR,
    BTEST,
    CEIL,
    CEILING,
    COS,
    COSD,
    COSH,
    D2R,
    DIF,
    DIV,
    DUP,
    DXDY,
    EQ,
    EXCH,
    EXP,
    FLOOR,
    FMOD,
    GAUSS,
    GE,
    GT,
    HYPOT,
    IAND,
    INRANGE,
    INV,
    IOR,
    ISAN,
    ISNAN,
    LE,
    LOG,
    LOG10,
    LT,
    MAX,
    MIN,
    MUL,
    NAN,
    NE,
    NEG,
    NINT,
    OR,
    PI,
    POP,
    POW,
    R2,
    R2D,
    RINT,
    SIN,
    SIND,
    SINH,
    SQR,
    SQRT,
    SUB,
    SUM,
    TAN,
    TAND,
    TANH,
    YMDHMS,
    CompleteExpression,
    E,
    Expression,
    Literal,
    StackUnderflowError,
    Token,
    Variable,
    token,
)

GOLDEN_RATIO = math.log((1 + math.sqrt(5)) / 2)


class TestLiteral:
    def test_init(self):
        Literal(3)
        Literal(3.14)
        with pytest.raises(TypeError):
            Literal("not a number")  # type: ignore

    def test_pops(self):
        assert Literal(3).pops == 0

    def test_puts(self):
        assert Literal(3).puts == 1

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
        assert repr(Literal(3)) == "Literal(3)"
        assert repr(Literal(3.14)) == "Literal(3.14)"

    def test_str(self):
        assert str(Literal(3)) == "3"
        assert str(Literal(3.14)) == "3.14"

    def test_pi(self):
        assert PI.value == pytest.approx(np.pi)

    def test_e(self):
        assert E.value == pytest.approx(np.e)


class TestVariable:
    def test_init(self):
        Variable("alt")
        with pytest.raises(ValueError):
            Variable("3")
        with pytest.raises(ValueError):
            Variable("3name")
        with pytest.raises(TypeError):
            Variable(3)  # type: ignore
        with pytest.raises(TypeError):
            Variable(3.14)  # type: ignore

    def test_pops(self):
        assert Variable("alt").pops == 0

    def test_puts(self):
        assert Variable("alt").puts == 1

    def test_name(self):
        assert Variable("alt").name == "alt"

    def test_call(self):
        stack: MutableSequence[NumberOrArray] = []
        environment = {"alt": np.array([1, 2, 3]), "dry_tropo": 4, "wet_tropo": 5}
        assert Variable("wet_tropo")(stack, environment) is None
        assert Variable("alt")(stack, environment) is None
        assert len(stack) == 2
        assert stack[0] == 5
        assert np.all(stack[1] == np.array([1, 2, 3]))
        assert len(environment) == 3
        assert "alt" in environment
        assert "dry_tropo" in environment
        assert "wet_tropo" in environment
        assert np.all(environment["alt"] == np.array([1, 2, 3]))
        assert environment["dry_tropo"] == 4
        assert environment["wet_tropo"] == 5
        with pytest.raises(KeyError):
            assert Variable("alt")(stack, {}) is None
        assert len(stack) == 2
        assert stack[0] == 5
        assert np.all(stack[1] == np.array([1, 2, 3]))

    def test_eq(self):
        assert Variable("alt") == Variable("alt")
        assert not Variable("alt") == Variable("dry_tropo")
        assert not Variable("alt") == "alt"

    def test_ne(self):
        assert Variable("alt") != Variable("dry_tropo")
        assert not Variable("alt") != Variable("alt")
        assert Variable("alt") != "alt"

    def test_repr(self):
        assert repr(Variable("alt")) == "Variable('alt')"

    def test_str(self):
        assert str(Variable("alt")) == "alt"


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


def assert_token(
    operator: Token,
    pre_stack: MutableSequence[NumberOrArray],
    post_stack: MutableSequence[NumberOrArray],
    environment: Optional[Mapping[str, NumberOrArray]] = None,
    *,
    approx: bool = False,
    rtol: float = 1e-15,
    atol: float = 1e-16,
) -> None:
    """Assert that a token modifies the stack properly.

    Parameters
    ----------
    operator
        Operator to test.
    pre_stack
        Stack state before calling the operator.
    post_stack
        Desired stack state after calling the operator.
    environment
        Optional dictionary like object providing the environment for
        variable lookup.
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
    if not environment:
        environment = {"dont_touch": 5}
    original_environment = deepcopy(environment)
    stack = pre_stack
    operator(stack, environment)
    # environment should be unchanged
    assert environment == original_environment
    # check stack
    if approx or contains_nan(post_stack) or contains_array(post_stack):
        assert len(stack) == len(post_stack)
        for a, b in zip(stack, post_stack):
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if approx:
                    np.testing.assert_allclose(
                        a, b, rtol=rtol, atol=atol, equal_nan=True
                    )
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


class TestSUBOperator:
    def test_repr(self):
        assert repr(SUB) == "SUB"

    def test_pops(self):
        assert SUB.pops == 2

    def test_puts(self):
        assert SUB.puts == 1

    def test_no_copy(self):
        assert copy(SUB) is SUB
        assert deepcopy(SUB) is SUB

    def test_call(self):
        assert_token(SUB, [2, 4], [-2])
        assert_token(SUB, [2, np.array([4, 1])], [np.array([-2, 1])])
        assert_token(SUB, [np.array([4, 1]), 2], [np.array([2, -1])])
        assert_token(SUB, [np.array([4, 1]), np.array([1, 4])], [np.array([3, -3])])
        # extra stack elements
        assert_token(SUB, [0, 2, 4], [0, -2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SUB([], {})
        with pytest.raises(StackUnderflowError):
            SUB([1], {})


class TestADDOperator:
    def test_repr(self):
        assert repr(ADD) == "ADD"

    def test_pops(self):
        assert ADD.pops == 2

    def test_puts(self):
        assert ADD.puts == 1

    def test_no_copy(self):
        assert copy(ADD) is ADD
        assert deepcopy(ADD) is ADD

    def test_call(self):
        assert_token(ADD, [2, 4], [6])
        assert_token(ADD, [2, np.array([4, 1])], [np.array([6, 3])])
        assert_token(ADD, [np.array([4, 1]), 2], [np.array([6, 3])])
        assert_token(ADD, [np.array([4, 1]), np.array([1, 4])], [np.array([5, 5])])
        # extra stack elements
        assert_token(ADD, [0, 2, 4], [0, 6])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ADD([], {})
        with pytest.raises(StackUnderflowError):
            ADD([1], {})


class TestMULOperator:
    def test_repr(self):
        assert repr(MUL) == "MUL"

    def test_pops(self):
        assert MUL.pops == 2

    def test_puts(self):
        assert MUL.puts == 1

    def test_no_copy(self):
        assert copy(MUL) is MUL
        assert deepcopy(MUL) is MUL

    def test_call(self):
        assert_token(MUL, [2, 4], [8])
        assert_token(MUL, [2, np.array([4, 1])], [np.array([8, 2])])
        assert_token(MUL, [np.array([4, 1]), 2], [np.array([8, 2])])
        assert_token(MUL, [np.array([4, 1]), np.array([1, 4])], [np.array([4, 4])])
        # extra stack elements
        assert_token(MUL, [0, 2, 4], [0, 8])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            MUL([], {})
        with pytest.raises(StackUnderflowError):
            MUL([1], {})


class TestPOPOperator:
    def test_repr(self):
        assert repr(POP) == "POP"

    def test_pops(self):
        assert POP.pops == 1

    def test_puts(self):
        assert POP.puts == 0

    def test_no_copy(self):
        assert copy(POP) is POP
        assert deepcopy(POP) is POP

    def test_call(self):
        assert_token(POP, [1], [])
        assert_token(POP, [1, 2], [1])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            POP([], {})


class TestNEGOperator:
    def test_repr(self):
        assert repr(NEG) == "NEG"

    def test_pops(self):
        assert NEG.pops == 1

    def test_puts(self):
        assert NEG.puts == 1

    def test_no_copy(self):
        assert copy(NEG) is NEG
        assert deepcopy(NEG) is NEG

    def test_call(self):
        assert_token(NEG, [2], [-2])
        assert_token(NEG, [-2], [2])
        assert_token(NEG, [np.array([4, -1])], [np.array([-4, 1])])
        assert_token(NEG, [np.array([-4, 1])], [np.array([4, -1])])
        # extra stack elements
        assert_token(NEG, [0, 2], [0, -2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            NEG([], {})


class TestABSOperator:
    def test_repr(self):
        assert repr(ABS) == "ABS"

    def test_pops(self):
        assert ABS.pops == 1

    def test_puts(self):
        assert ABS.puts == 1

    def test_no_copy(self):
        assert copy(ABS) is ABS
        assert deepcopy(ABS) is ABS

    def test_call(self):
        assert_token(ABS, [2], [2])
        assert_token(ABS, [-2], [2])
        assert_token(ABS, [np.array([4, -1])], [np.array([4, 1])])
        assert_token(ABS, [np.array([-4, 1])], [np.array([4, 1])])
        # extra stack elements
        assert_token(ABS, [0, -2], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ABS([], {})


class TestINVOperator:
    def test_repr(self):
        assert repr(INV) == "INV"

    def test_pops(self):
        assert INV.pops == 1

    def test_puts(self):
        assert INV.puts == 1

    def test_no_copy(self):
        assert copy(INV) is INV
        assert deepcopy(INV) is INV

    def test_call(self):
        assert_token(INV, [2], [0.5])
        assert_token(INV, [-2], [-0.5])
        assert_token(INV, [np.array([4, -1])], [np.array([0.25, -1])])
        assert_token(INV, [np.array([-4, 1])], [np.array([-0.25, 1])])
        # extra stack elements
        assert_token(INV, [0, 2], [0, 0.5])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            INV([], {})


class TestSQRTOperator:
    def test_repr(self):
        assert repr(SQRT) == "SQRT"

    def test_pops(self):
        assert SQRT.pops == 1

    def test_puts(self):
        assert SQRT.puts == 1

    def test_no_copy(self):
        assert copy(SQRT) is SQRT
        assert deepcopy(SQRT) is SQRT

    def test_call(self):
        assert_token(SQRT, [4], [2])
        assert_token(SQRT, [np.array([4, 16])], [np.array([2, 4])])
        # extra stack elements
        assert_token(SQRT, [0, 4], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SQRT([], {})


class TestSQROperator:
    def test_repr(self):
        assert repr(SQR) == "SQR"

    def test_pops(self):
        assert SQR.pops == 1

    def test_puts(self):
        assert SQR.puts == 1

    def test_no_copy(self):
        assert copy(EXP) is EXP
        assert deepcopy(EXP) is EXP

    def test_call(self):
        assert_token(SQR, [2], [4])
        assert_token(SQR, [-2], [4])
        assert_token(SQR, [np.array([4, -1])], [np.array([16, 1])])
        assert_token(SQR, [np.array([-4, 1])], [np.array([16, 1])])
        # extra stack elements
        assert_token(SQR, [0, -2], [0, 4])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SQR([], {})


class TestEXPOperator:
    def test_repr(self):
        assert repr(EXP) == "EXP"

    def test_pops(self):
        assert EXP.pops == 1

    def test_puts(self):
        assert EXP.puts == 1

    def test_no_copy(self):
        assert copy(EXP) is EXP
        assert deepcopy(EXP) is EXP

    def test_call(self):
        assert_token(EXP, [math.log(1)], [1.0], approx=True)
        assert_token(EXP, [math.log(2)], [2.0], approx=True)
        assert_token(
            EXP, [np.array([np.log(4), np.log(1)])], [np.array([4.0, 1.0])], approx=True
        )
        # extra stack elements
        assert_token(EXP, [0, np.log(1)], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            EXP([], {})


class TestLOGOperator:
    def test_repr(self):
        assert repr(LOG) == "LOG"

    def test_pops(self):
        assert LOG.pops == 1

    def test_puts(self):
        assert LOG.puts == 1

    def test_no_copy(self):
        assert copy(LOG) is LOG
        assert deepcopy(LOG) is LOG

    def test_call(self):
        assert_token(LOG, [math.e], [1.0], approx=True)
        assert_token(LOG, [math.e ** 2], [2.0], approx=True)
        assert_token(LOG, [math.e ** -2], [-2.0], approx=True)
        assert_token(
            LOG,
            [np.array([np.e ** 4, np.e ** -1])],
            [np.array([4.0, -1.0])],
            approx=True,
        )
        assert_token(
            LOG,
            [np.array([np.e ** -4, np.e ** 1])],
            [np.array([-4.0, 1.0])],
            approx=True,
        )
        # extra stack elements
        assert_token(LOG, [0, np.e], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            LOG([], {})


class TestLOG10Operator:
    def test_repr(self):
        assert repr(LOG10) == "LOG10"

    def test_pops(self):
        assert LOG10.pops == 1

    def test_puts(self):
        assert LOG10.puts == 1

    def test_no_copy(self):
        assert copy(LOG10) is LOG10
        assert deepcopy(LOG10) is LOG10

    def test_call(self):
        assert_token(LOG10, [10], [1.0], approx=True)
        assert_token(LOG10, [10 ** 2], [2.0], approx=True)
        assert_token(LOG10, [10 ** -2], [-2.0], approx=True)
        assert_token(
            LOG10, [np.array([10 ** 4, 10 ** -1])], [np.array([4.0, -1.0])], approx=True
        )
        assert_token(
            LOG10, [np.array([10 ** -4, 10 ** 1])], [np.array([-4.0, 1.0])], approx=True
        )
        # extra stack elements
        assert_token(LOG10, [0, 10], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            LOG10([], {})


class TestSINOperator:
    def test_repr(self):
        assert repr(SIN) == "SIN"

    def test_pops(self):
        assert SIN.pops == 1

    def test_puts(self):
        assert SIN.puts == 1

    def test_no_copy(self):
        assert copy(SIN) is SIN
        assert deepcopy(SIN) is SIN

    def test_call(self):
        assert_token(SIN, [0.0], [0.0], approx=True)
        assert_token(SIN, [math.pi / 6], [1 / 2], approx=True)
        assert_token(SIN, [math.pi / 4], [1 / math.sqrt(2)], approx=True)
        assert_token(SIN, [math.pi / 3], [math.sqrt(3) / 2], approx=True)
        assert_token(SIN, [math.pi / 2], [1.0], approx=True)
        assert_token(
            SIN,
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            approx=True,
        )
        assert_token(
            SIN,
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [-np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            approx=True,
        )
        # extra stack elements
        assert_token(SIN, [0, math.pi / 2], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SIN([], {})


class TestCOSOperator:
    def test_repr(self):
        assert repr(COS) == "COS"

    def test_pops(self):
        assert COS.pops == 1

    def test_puts(self):
        assert COS.puts == 1

    def test_no_copy(self):
        assert copy(COS) is COS
        assert deepcopy(COS) is COS

    def test_call(self):
        assert_token(COS, [0.0], [1.0], approx=True)
        assert_token(COS, [math.pi / 6], [math.sqrt(3) / 2], approx=True)
        assert_token(COS, [math.pi / 4], [1 / math.sqrt(2)], approx=True)
        assert_token(COS, [math.pi / 3], [1 / 2], approx=True)
        assert_token(COS, [math.pi / 2], [0.0], approx=True)
        assert_token(
            COS,
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            approx=True,
        )
        assert_token(
            COS,
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            approx=True,
        )
        # extra stack elements
        assert_token(COS, [0, math.pi / 2], [0, 0.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            COS([], {})


class TestTANOperator:
    def test_repr(self):
        assert repr(TAN) == "TAN"

    def test_pops(self):
        assert TAN.pops == 1

    def test_puts(self):
        assert TAN.puts == 1

    def test_no_copy(self):
        assert copy(TAN) is TAN
        assert deepcopy(TAN) is TAN

    def test_call(self):
        assert_token(TAN, [0.0], [0.0], approx=True)
        assert_token(TAN, [math.pi / 6], [1 / math.sqrt(3)], approx=True)
        assert_token(TAN, [math.pi / 4], [1.0], approx=True)
        assert_token(TAN, [math.pi / 3], [math.sqrt(3)], approx=True)
        assert_token(
            TAN,
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])],
            [np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            approx=True,
        )
        assert_token(
            TAN,
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])],
            [-np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            approx=True,
        )
        # extra stack elements
        assert_token(TAN, [0, math.pi / 4], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            TAN([], {})


class TestSINDOperator:
    def test_repr(self):
        assert repr(SIND) == "SIND"

    def test_pops(self):
        assert SIND.pops == 1

    def test_puts(self):
        assert SIND.puts == 1

    def test_no_copy(self):
        assert copy(COSD) is COSD
        assert deepcopy(COSD) is COSD

    def test_call(self):
        assert_token(SIND, [0], [0.0], approx=True)
        assert_token(SIND, [30], [1 / 2], approx=True)
        assert_token(SIND, [45], [1 / math.sqrt(2)], approx=True)
        assert_token(SIND, [60], [math.sqrt(3) / 2], approx=True)
        assert_token(SIND, [90], [1.0], approx=True)
        assert_token(
            SIND,
            [np.array([0, 30, 45, 60, 90])],
            [np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            approx=True,
        )
        assert_token(
            SIND,
            [-np.array([0, 30, 45, 60, 90])],
            [-np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            approx=True,
        )
        # extra stack elements
        assert_token(SIND, [0, 90], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SIND([], {})


class TestCOSDOperator:
    def test_repr(self):
        assert repr(COSD) == "COSD"

    def test_pops(self):
        assert COSD.pops == 1

    def test_puts(self):
        assert COSD.puts == 1

    def test_no_copy(self):
        assert copy(COSD) is COSD
        assert deepcopy(COSD) is COSD

    def test_call(self):
        assert_token(COSD, [0], [1.0], approx=True)
        assert_token(COSD, [30], [math.sqrt(3) / 2], approx=True)
        assert_token(COSD, [45], [1 / math.sqrt(2)], approx=True)
        assert_token(COSD, [60], [1 / 2], approx=True)
        assert_token(COSD, [90], [0.0], approx=True)
        assert_token(
            COSD,
            [np.array([0, 30, 45, 60, 90])],
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            approx=True,
        )
        assert_token(
            COSD,
            [-np.array([0, 30, 45, 60, 90])],
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            approx=True,
        )
        # extra stack elements
        assert_token(COSD, [0, 90], [0, 0.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            COSD([], {})


class TestTANDOperator:
    def test_repr(self):
        assert repr(TAND) == "TAND"

    def test_pops(self):
        assert TAND.pops == 1

    def test_puts(self):
        assert TAND.puts == 1

    def test_no_copy(self):
        assert copy(TAND) is TAND
        assert deepcopy(TAND) is TAND

    def test_call(self):
        assert_token(TAND, [0], [0], approx=True)
        assert_token(TAND, [30], [1 / math.sqrt(3)], approx=True)
        assert_token(TAND, [45], [1.0], approx=True)
        assert_token(TAND, [60], [math.sqrt(3)], approx=True)
        assert_token(
            TAND,
            [np.array([0, 30, 45, 60])],
            [np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            approx=True,
        )
        assert_token(
            TAND,
            [-np.array([0, 30, 45, 60])],
            [-np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            approx=True,
        )
        # extra stack elements
        assert_token(TAND, [0, 45], [0, 1.0], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            TAND([], {})


class TestSINHOperator:
    def test_repr(self):
        assert repr(SINH) == "SINH"

    def test_pops(self):
        assert SINH.pops == 1

    def test_puts(self):
        assert SINH.puts == 1

    def test_no_copy(self):
        assert copy(SINH) is SINH
        assert deepcopy(SINH) is SINH

    def test_call(self):
        assert_token(SINH, [0.0], [0.0], approx=True)
        assert_token(SINH, [GOLDEN_RATIO], [0.5], approx=True)
        assert_token(
            SINH, [np.array([0.0, GOLDEN_RATIO])], [np.array([0.0, 0.5])], approx=True
        )
        # extra stack elements
        assert_token(SINH, [0, GOLDEN_RATIO], [0, 0.5], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SINH([], {})


class TestCOSHOperator:
    def test_repr(self):
        assert repr(COSH) == "COSH"

    def test_pops(self):
        assert COSH.pops == 1

    def test_puts(self):
        assert COSH.puts == 1

    def test_no_copy(self):
        assert copy(COSH) is COSH
        assert deepcopy(COSH) is COSH

    def test_call(self):
        assert_token(COSH, [0.0], [1.0], approx=True)
        assert_token(COSH, [GOLDEN_RATIO], [math.sqrt(5) / 2], approx=True)
        assert_token(
            COSH,
            [np.array([0.0, GOLDEN_RATIO])],
            [np.array([1.0, np.sqrt(5) / 2])],
            approx=True,
        )
        # extra stack elements
        assert_token(COSH, [0, GOLDEN_RATIO], [0, math.sqrt(5) / 2], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            COSH([], {})


class TestTANHOperator:
    def test_repr(self):
        assert repr(TANH) == "TANH"

    def test_pops(self):
        assert TANH.pops == 1

    def test_puts(self):
        assert TANH.puts == 1

    def test_no_copy(self):
        assert copy(TANH) is TANH
        assert deepcopy(TANH) is TANH

    def test_call(self):
        assert_token(TANH, [0.0], [0.0], approx=True)
        assert_token(TANH, [GOLDEN_RATIO], [math.sqrt(5) / 5], approx=True)
        assert_token(
            TANH,
            [np.array([0.0, GOLDEN_RATIO])],
            [np.array([0.0, np.sqrt(5) / 5])],
            approx=True,
        )
        # extra stack elements
        assert_token(TANH, [0, GOLDEN_RATIO], [0, math.sqrt(5) / 5], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            TANH([], {})


class TestASINOperator:
    def test_repr(self):
        assert repr(ASIN) == "ASIN"

    def test_pops(self):
        assert ASIN.pops == 1

    def test_puts(self):
        assert ASIN.puts == 1

    def test_no_copy(self):
        assert copy(ASIN) is ASIN
        assert deepcopy(ASIN) is ASIN

    def test_call(self):
        assert_token(ASIN, [0.0], [0.0], approx=True)
        assert_token(ASIN, [1 / 2], [math.pi / 6], approx=True)
        assert_token(ASIN, [1 / math.sqrt(2)], [math.pi / 4], approx=True)
        assert_token(ASIN, [math.sqrt(3) / 2], [math.pi / 3], approx=True)
        assert_token(ASIN, [1.0], [math.pi / 2], approx=True)
        assert_token(
            ASIN,
            [np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            approx=True,
        )
        assert_token(
            ASIN,
            [-np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            approx=True,
        )
        # extra stack elements
        assert_token(ASIN, [0, 1.0], [0, math.pi / 2], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ASIN([], {})


class TestACOSOperator:
    def test_repr(self):
        assert repr(ACOS) == "ACOS"

    def test_pops(self):
        assert ACOS.pops == 1

    def test_puts(self):
        assert ACOS.puts == 1

    def test_no_copy(self):
        assert copy(ACOS) is ACOS
        assert deepcopy(ACOS) is ACOS

    def test_call(self):
        assert_token(ACOS, [1.0], [0.0], approx=True)
        assert_token(ACOS, [math.sqrt(3) / 2], [math.pi / 6], approx=True)
        assert_token(ACOS, [1 / math.sqrt(2)], [math.pi / 4], approx=True)
        assert_token(ACOS, [1 / 2], [math.pi / 3], approx=True)
        assert_token(ACOS, [0.0], [math.pi / 2], approx=True)
        assert_token(
            ACOS,
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            approx=True,
        )
        # extra stack elements
        assert_token(ACOS, [0, 0.0], [0, math.pi / 2], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ACOS([], {})


class TestATANOperator:
    def test_repr(self):
        assert repr(ATAN) == "ATAN"

    def test_pops(self):
        assert ATAN.pops == 1

    def test_puts(self):
        assert ATAN.puts == 1

    def test_no_copy(self):
        assert copy(ATAN) is ATAN
        assert deepcopy(ATAN) is ATAN

    def test_call(self):
        assert_token(ATAN, [0.0], [0.0], approx=True)
        assert_token(ATAN, [1 / math.sqrt(3)], [math.pi / 6], approx=True)
        assert_token(ATAN, [1.0], [math.pi / 4], approx=True)
        assert_token(ATAN, [math.sqrt(3)], [math.pi / 3], approx=True)
        assert_token(
            ATAN,
            [np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])],
            approx=True,
        )
        assert_token(
            ATAN,
            [-np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])],
            approx=True,
        )
        # extra stack elements
        assert_token(ATAN, [0, 1.0], [0, math.pi / 4], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ATAN([], {})


class TestASINDOperator:
    def test_repr(self):
        assert repr(ASIND) == "ASIND"

    def test_pops(self):
        assert ASIND.pops == 1

    def test_puts(self):
        assert ASIND.puts == 1

    def test_no_copy(self):
        assert copy(ASIND) is ASIND
        assert deepcopy(ASIND) is ASIND

    def test_call(self):
        assert_token(ASIND, [0.0], [0], approx=True)
        assert_token(ASIND, [1 / 2], [30], approx=True)
        assert_token(ASIND, [1 / math.sqrt(2)], [45], approx=True)
        assert_token(ASIND, [math.sqrt(3) / 2], [60], approx=True)
        assert_token(ASIND, [1.0], [90], approx=True)
        assert_token(
            ASIND,
            [np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            [np.array([0, 30, 45, 60, 90])],
            approx=True,
        )
        assert_token(
            ASIND,
            [-np.array([0.0, 1 / 2, 1 / np.sqrt(2), np.sqrt(3) / 2, 1.0])],
            [-np.array([0, 30, 45, 60, 90])],
            approx=True,
        )
        # extra stack elements
        assert_token(ASIND, [0, 1.0], [0, 90], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ASIND([], {})


class TestACOSDOperator:
    def test_repr(self):
        assert repr(ACOSD) == "ACOSD"

    def test_pops(self):
        assert ACOSD.pops == 1

    def test_puts(self):
        assert ACOSD.puts == 1

    def test_no_copy(self):
        assert copy(ACOSD) is ACOSD
        assert deepcopy(ACOSD) is ACOSD

    def test_call(self):
        assert_token(ACOSD, [1.0], [0], approx=True)
        assert_token(ACOSD, [math.sqrt(3) / 2], [30], approx=True)
        assert_token(ACOSD, [1 / math.sqrt(2)], [45], approx=True)
        assert_token(ACOSD, [1 / 2], [60], approx=True)
        assert_token(ACOSD, [0.0], [90], approx=True)
        assert_token(
            ACOSD,
            [np.array([1.0, np.sqrt(3) / 2, 1 / np.sqrt(2), 1 / 2, 0.0])],
            [np.array([0, 30, 45, 60, 90])],
            approx=True,
        )
        # extra stack elements
        assert_token(ACOSD, [0, 0.0], [0, 90], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ACOSD([], {})


class TestATANDOperator:
    def test_repr(self):
        assert repr(ATAND) == "ATAND"

    def test_pops(self):
        assert ATAND.pops == 1

    def test_puts(self):
        assert ATAND.puts == 1

    def test_no_copy(self):
        assert copy(ATAND) is ATAND
        assert deepcopy(ATAND) is ATAND

    def test_call(self):
        assert_token(ATAND, [0.0], [0], approx=True)
        assert_token(ATAND, [1 / math.sqrt(3)], [30], approx=True)
        assert_token(ATAND, [1.0], [45], approx=True)
        assert_token(ATAND, [math.sqrt(3)], [60], approx=True)
        assert_token(
            ATAND,
            [np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            [np.array([0, 30, 45, 60])],
            approx=True,
        )
        assert_token(
            ATAND,
            [-np.array([0.0, 1 / np.sqrt(3), 1.0, np.sqrt(3)])],
            [-np.array([0, 30, 45, 60])],
            approx=True,
        )
        # extra stack elements
        assert_token(ATAND, [0, 1.0], [0, 45], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ATAND([], {})


class TestASINHOperator:
    def test_repr(self):
        assert repr(ASINH) == "ASINH"

    def test_pops(self):
        assert ASINH.pops == 1

    def test_puts(self):
        assert ASINH.puts == 1

    def test_no_copy(self):
        assert copy(ASINH) is ASINH
        assert deepcopy(ASINH) is ASINH

    def test_call(self):
        assert_token(ASINH, [0.0], [0.0], approx=True)
        assert_token(ASINH, [0.5], [GOLDEN_RATIO], approx=True)
        assert_token(
            ASINH, [np.array([0.0, 0.5])], [np.array([0.0, GOLDEN_RATIO])], approx=True
        )
        # extra stack elements
        assert_token(ASINH, [0, 0.5], [0, GOLDEN_RATIO], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ASINH([], {})


class TestACOSHOperator:
    def test_repr(self):
        assert repr(ACOSH) == "ACOSH"

    def test_pops(self):
        assert ACOSH.pops == 1

    def test_puts(self):
        assert ACOSH.puts == 1

    def test_no_copy(self):
        assert copy(ACOSH) is ACOSH
        assert deepcopy(ACOSH) is ACOSH

    def test_call(self):
        assert_token(ACOSH, [1.0], [0.0], approx=True)
        assert_token(ACOSH, [math.sqrt(5) / 2], [GOLDEN_RATIO], approx=True)
        assert_token(
            ACOSH,
            [np.array([1.0, np.sqrt(5) / 2])],
            [np.array([0.0, GOLDEN_RATIO])],
            approx=True,
        )
        # extra stack elements
        assert_token(ACOSH, [0, math.sqrt(5) / 2], [0, GOLDEN_RATIO], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ACOSH([], {})


class TestATANHOperator:
    def test_repr(self):
        assert repr(ATANH) == "ATANH"

    def test_pops(self):
        assert ATANH.pops == 1

    def test_puts(self):
        assert ATANH.puts == 1

    def test_no_copy(self):
        assert copy(ATANH) is ATANH
        assert deepcopy(ATANH) is ATANH

    def test_call(self):
        assert_token(ATANH, [0.0], [0.0], approx=True)
        assert_token(ATANH, [math.sqrt(5) / 5], [GOLDEN_RATIO], approx=True)
        assert_token(
            ATANH,
            [np.array([0.0, np.sqrt(5) / 5])],
            [np.array([0.0, GOLDEN_RATIO])],
            approx=True,
        )
        # extra stack elements
        assert_token(ATANH, [0, math.sqrt(5) / 5], [0, GOLDEN_RATIO], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ATANH([], {})


class TestISNANOperator:
    def test_repr(self):
        assert repr(ISNAN) == "ISNAN"

    def test_pops(self):
        assert ISNAN.pops == 1

    def test_puts(self):
        assert ISNAN.puts == 1

    def test_no_copy(self):
        assert copy(ISNAN) is ISNAN
        assert deepcopy(ISNAN) is ISNAN

    def test_call(self):
        assert_token(ISNAN, [2], [False])
        assert_token(ISNAN, [float("nan")], [True])
        assert_token(ISNAN, [np.array([4, np.nan])], [np.array([False, True])])
        assert_token(ISNAN, [np.array([np.nan, 1])], [np.array([True, False])])
        # extra stack elements
        assert_token(ISNAN, [0, float("nan")], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ISNAN([], {})


class TestISANOperator:
    def test_repr(self):
        assert repr(ISAN) == "ISAN"

    def test_pops(self):
        assert ISAN.pops == 1

    def test_puts(self):
        assert ISAN.puts == 1

    def test_no_copy(self):
        assert copy(ISAN) is ISAN
        assert deepcopy(ISAN) is ISAN

    def test_call(self):
        assert_token(ISAN, [2], [True])
        assert_token(ISAN, [float("nan")], [False])
        assert_token(ISAN, [np.array([4, np.nan])], [np.array([True, False])])
        assert_token(ISAN, [np.array([np.nan, 1])], [np.array([False, True])])
        # extra stack elements
        assert_token(ISAN, [0, 2], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ISAN([], {})


class TestRINTOperator:
    def test_repr(self):
        assert repr(RINT) == "RINT"

    def test_pops(self):
        assert RINT.pops == 1

    def test_puts(self):
        assert RINT.puts == 1

    def test_no_copy(self):
        assert copy(RINT) is RINT
        assert deepcopy(RINT) is RINT

    def test_call(self):
        assert_token(RINT, [1.6], [2])
        assert_token(RINT, [2.4], [2])
        assert_token(RINT, [-1.6], [-2])
        assert_token(RINT, [-2.4], [-2])
        assert_token(RINT, [np.array([1.6, 2.4])], [np.array([2, 2])])
        assert_token(RINT, [np.array([-1.6, -2.4])], [np.array([-2, -2])])
        # extra stack elements
        assert_token(RINT, [0, 1.6], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            RINT([], {})


class TestNINTOperator:
    def test_repr(self):
        assert repr(NINT) == "NINT"

    def test_pops(self):
        assert NINT.pops == 1

    def test_puts(self):
        assert NINT.puts == 1

    def test_no_copy(self):
        assert copy(NINT) is NINT
        assert deepcopy(NINT) is NINT

    def test_call(self):
        assert_token(NINT, [1.6], [2])
        assert_token(NINT, [2.4], [2])
        assert_token(NINT, [-1.6], [-2])
        assert_token(NINT, [-2.4], [-2])
        assert_token(NINT, [np.array([1.6, 2.4])], [np.array([2, 2])])
        assert_token(NINT, [np.array([-1.6, -2.4])], [np.array([-2, -2])])
        # extra stack elements
        assert_token(NINT, [0, 1.6], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            NINT([], {})


class TestCEILOperator:
    def test_repr(self):
        assert repr(CEIL) == "CEIL"

    def test_pops(self):
        assert CEIL.pops == 1

    def test_puts(self):
        assert CEIL.puts == 1

    def test_no_copy(self):
        assert copy(CEIL) is CEIL
        assert deepcopy(CEIL) is CEIL

    def test_call(self):
        assert_token(CEIL, [1.6], [2])
        assert_token(CEIL, [2.4], [3])
        assert_token(CEIL, [-1.6], [-1])
        assert_token(CEIL, [-2.4], [-2])
        assert_token(CEIL, [np.array([1.6, 2.4])], [np.array([2, 3])])
        assert_token(CEIL, [np.array([-1.6, -2.4])], [np.array([-1, -2])])
        # extra stack elements
        assert_token(CEIL, [0, 1.2], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            CEIL([], {})


class TestCEILINGOperator:
    def test_repr(self):
        assert repr(CEILING) == "CEILING"

    def test_pops(self):
        assert CEILING.pops == 1

    def test_puts(self):
        assert CEILING.puts == 1

    def test_no_copy(self):
        assert copy(CEILING) is CEILING
        assert deepcopy(CEILING) is CEILING

    def test_call(self):
        assert_token(CEILING, [1.6], [2])
        assert_token(CEILING, [2.4], [3])
        assert_token(CEILING, [-1.6], [-1])
        assert_token(CEILING, [-2.4], [-2])
        assert_token(CEILING, [np.array([1.6, 2.4])], [np.array([2, 3])])
        assert_token(CEILING, [np.array([-1.6, -2.4])], [np.array([-1, -2])])
        # extra stack elements
        assert_token(CEILING, [0, 1.2], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            CEILING([], {})


class TestFLOOROperator:
    def test_repr(self):
        assert repr(FLOOR) == "FLOOR"

    def test_pops(self):
        assert FLOOR.pops == 1

    def test_puts(self):
        assert FLOOR.puts == 1

    def test_no_copy(self):
        assert copy(FLOOR) is FLOOR
        assert deepcopy(FLOOR) is FLOOR

    def test_call(self):
        assert_token(FLOOR, [1.6], [1])
        assert_token(FLOOR, [2.4], [2])
        assert_token(FLOOR, [-1.6], [-2])
        assert_token(FLOOR, [-2.4], [-3])
        assert_token(FLOOR, [np.array([1.6, 2.4])], [np.array([1, 2])])
        assert_token(FLOOR, [np.array([-1.6, -2.4])], [np.array([-2, -3])])
        # extra stack elements
        assert_token(FLOOR, [0, 1.8], [0, 1])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            FLOOR([], {})


class TestD2ROperator:
    def test_repr(self):
        assert repr(D2R) == "D2R"

    def test_pops(self):
        assert D2R.pops == 1

    def test_puts(self):
        assert D2R.puts == 1

    def test_no_copy(self):
        assert copy(D2R) is D2R
        assert deepcopy(D2R) is D2R

    def test_call(self):
        assert_token(D2R, [0], [0.0], approx=True)
        assert_token(D2R, [30], [math.pi / 6], approx=True)
        assert_token(D2R, [45], [math.pi / 4], approx=True)
        assert_token(D2R, [60], [math.pi / 3], approx=True)
        assert_token(D2R, [90], [math.pi / 2], approx=True)
        assert_token(
            D2R,
            [np.array([0, 30, 45, 60, 90])],
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            approx=True,
        )
        assert_token(
            D2R,
            [-np.array([0, 30, 45, 60, 90])],
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            approx=True,
        )
        # extra stack elements
        assert_token(D2R, [0, 90], [0, math.pi / 2], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            D2R([], {})


class TestR2DOperator:
    def test_repr(self):
        assert repr(R2D) == "R2D"

    def test_pops(self):
        assert R2D.pops == 1

    def test_puts(self):
        assert R2D.puts == 1

    def test_no_copy(self):
        assert copy(R2D) is R2D
        assert deepcopy(R2D) is R2D

    def test_call(self):
        assert_token(R2D, [0.0], [0], approx=True)
        assert_token(R2D, [math.pi / 6], [30], approx=True)
        assert_token(R2D, [math.pi / 4], [45], approx=True)
        assert_token(R2D, [math.pi / 3], [60], approx=True)
        assert_token(R2D, [math.pi / 2], [90], approx=True)
        assert_token(
            R2D,
            [np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [np.array([0, 30, 45, 60, 90])],
            approx=True,
        )
        assert_token(
            R2D,
            [-np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])],
            [-np.array([0, 30, 45, 60, 90])],
            approx=True,
        )
        # extra stack elements
        assert_token(R2D, [0, math.pi / 2], [0, 90], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            R2D([], {})


class TestYMDHMSOperator:
    def test_repr(self):
        assert repr(YMDHMS) == "YMDHMS"

    def test_pops(self):
        assert YMDHMS.pops == 1

    def test_puts(self):
        assert YMDHMS.puts == 1

    def test_no_copy(self):
        assert copy(YMDHMS) is YMDHMS
        assert deepcopy(YMDHMS) is YMDHMS

    def test_call(self):
        epoch = datetime(1985, 1, 1, 0, 0, 0, 0)
        date1 = datetime(2008, 7, 4, 12, 19, 19, 570865)
        date2 = datetime(2019, 6, 26, 12, 31, 6, 930575)
        seconds1 = (date1 - epoch).total_seconds()
        seconds2 = (date2 - epoch).total_seconds()
        assert_token(YMDHMS, [seconds1], [80704121919.570865], approx=True)
        assert_token(YMDHMS, [seconds2], [190626123106.930575], approx=True)
        assert_token(
            YMDHMS,
            [np.array([seconds1, seconds2])],
            [np.array([80704121919.570865, 190626123106.930575])],
            approx=True,
        )
        # extra stack elements
        assert_token(YMDHMS, [0, seconds1], [0, 80704121919.570865], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            YMDHMS([], {})


class TestSUMOperator:
    def test_repr(self):
        assert repr(SUM) == "SUM"

    def test_pops(self):
        assert SUM.pops == 1

    def test_puts(self):
        assert SUM.puts == 1

    def test_no_copy(self):
        assert copy(SUM) is SUM
        assert deepcopy(SUM) is SUM

    def test_call(self):
        assert_token(SUM, [2], [2])
        assert_token(SUM, [-2], [-2])
        assert_token(SUM, [float("nan")], [0])
        assert_token(SUM, [np.array([4, -1])], [3])
        assert_token(SUM, [np.array([-4, 1])], [-3])
        assert_token(SUM, [np.array([1, np.nan, 3])], [4])
        assert_token(SUM, [np.array([np.nan])], [0])
        # extra stack elements
        assert_token(SUM, [0, 2], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            SUM([], {})


class TestDIFFOperator:
    def test_repr(self):
        assert repr(DIF) == "DIF"

    def test_pops(self):
        assert DIF.pops == 1

    def test_puts(self):
        assert DIF.puts == 1

    def test_no_copy(self):
        assert copy(DIF) is DIF
        assert deepcopy(DIF) is DIF

    def test_call(self):
        assert_token(DIF, [2], [np.array([np.nan])])
        assert_token(DIF, [np.array([1, 2])], [np.array([np.nan, 1])])
        assert_token(DIF, [np.array([1, 2, 5])], [np.array([np.nan, 1, 3])])
        assert_token(
            DIF, [np.array([1, np.nan, 5])], [np.array([np.nan, np.nan, np.nan])]
        )
        # extra stack elements
        assert_token(DIF, [0, 2], [0, np.array([np.nan])])
        with pytest.raises(StackUnderflowError):
            DIF([], {})


class TestDUPOperator:
    def test_repr(self):
        assert repr(DUP) == "DUP"

    def test_pops(self):
        assert DUP.pops == 1

    def test_puts(self):
        assert DUP.puts == 2

    def test_no_copy(self):
        assert copy(DUP) is DUP
        assert deepcopy(DUP) is DUP

    def test_call(self):
        assert_token(DUP, [2], [2, 2])
        assert_token(DUP, [np.array([4, -1])], [np.array([4, -1]), np.array([4, -1])])
        # extra stack elements
        assert_token(DUP, [0, 2], [0, 2, 2])
        with pytest.raises(StackUnderflowError):
            DUP([], {})


class TestDIVOperator:
    def test_repr(self):
        assert repr(DIV) == "DIV"

    def test_pops(self):
        assert DIV.pops == 2

    def test_puts(self):
        assert DIV.puts == 1

    def test_no_copy(self):
        assert copy(DIV) is DIV
        assert deepcopy(DIV) is DIV

    def test_call(self):
        assert_token(DIV, [10, 2], [5])
        assert_token(DIV, [10, np.array([2, 5])], [np.array([5, 2])])
        assert_token(DIV, [np.array([10, 4]), 2], [np.array([5, 2])])
        assert_token(DIV, [np.array([8, 16]), np.array([2, 4])], [np.array([4, 4])])
        # extra stack elements
        assert_token(DIV, [0, 10, 2], [0, 5])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            DIV([], {})
        with pytest.raises(StackUnderflowError):
            DIV([1], {})


class TestPOWOperator:
    def test_repr(self):
        assert repr(POW) == "POW"

    def test_pops(self):
        assert POW.pops == 2

    def test_puts(self):
        assert POW.puts == 1

    def test_no_copy(self):
        assert copy(POW) is POW
        assert deepcopy(POW) is POW

    def test_call(self):
        assert_token(POW, [1, 2], [1])
        assert_token(POW, [2, 2], [4])
        assert_token(POW, [2, 4], [16])
        assert_token(POW, [2, np.array([1, 2, 3])], [np.array([2, 4, 8])])
        assert_token(POW, [np.array([1, 2, 3]), 2], [np.array([1, 4, 9])])
        assert_token(POW, [np.array([2, 3]), np.array([5, 6])], [np.array([32, 729])])
        # extra stack elements
        assert_token(POW, [0, 2, 4], [0, 16])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            POW([], {})
        with pytest.raises(StackUnderflowError):
            POW([1], {})


class TestFMODOperator:
    def test_repr(self):
        assert repr(FMOD) == "FMOD"
        assert FMOD.pops == 2
        assert FMOD.puts == 1

    def test_pops(self):
        assert repr(FMOD) == "FMOD"
        assert FMOD.pops == 2
        assert FMOD.puts == 1

    def test_puts(self):
        assert repr(FMOD) == "FMOD"
        assert FMOD.pops == 2
        assert FMOD.puts == 1

    def test_no_copy(self):
        assert copy(FMOD) is FMOD
        assert deepcopy(FMOD) is FMOD

    def test_call(self):
        assert_token(FMOD, [1, 2], [1])
        assert_token(FMOD, [2, 10], [2])
        assert_token(FMOD, [12, 10], [2])
        assert_token(FMOD, [13, np.array([10, 100])], [np.array([3, 13])])
        assert_token(FMOD, [np.array([7, 15]), 10], [np.array([7, 5])])
        assert_token(FMOD, [np.array([7, 15]), np.array([10, 5])], [np.array([7, 0])])
        # extra stack elements
        assert_token(FMOD, [0, 12, 10], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            FMOD([], {})
        with pytest.raises(StackUnderflowError):
            FMOD([1], {})


class TestMINOperator:
    def test_repr(self):
        assert repr(MIN) == "MIN"

    def test_pops(self):
        assert MIN.pops == 2

    def test_puts(self):
        assert MIN.puts == 1

    def test_no_copy(self):
        assert copy(MIN) is MIN
        assert deepcopy(MIN) is MIN

    def test_call(self):
        assert_token(MIN, [2, 3], [2])
        assert_token(MIN, [3, 2], [2])
        assert_token(MIN, [2, np.array([1, 3])], [np.array([1, 2])])
        assert_token(MIN, [np.array([1, 3]), 2], [np.array([1, 2])])
        assert_token(MIN, [np.array([2, 3]), np.array([3, 2])], [np.array([2, 2])])
        # # extra stack elements
        assert_token(MIN, [0, 2, 3], [0, 2])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            MIN([], {})
        with pytest.raises(StackUnderflowError):
            MIN([1], {})


class TestMAXOperator:
    def test_repr(self):
        assert repr(MAX) == "MAX"

    def test_pops(self):
        assert MAX.pops == 2

    def test_puts(self):
        assert MAX.puts == 1

    def test_no_copy(self):
        assert copy(MAX) is MAX
        assert deepcopy(MAX) is MAX

    def test_call(self):
        assert_token(MAX, [2, 3], [3])
        assert_token(MAX, [3, 2], [3])
        assert_token(MAX, [2, np.array([1, 3])], [np.array([2, 3])])
        assert_token(MAX, [np.array([1, 3]), 2], [np.array([2, 3])])
        assert_token(MAX, [np.array([2, 3]), np.array([3, 2])], [np.array([3, 3])])
        # # extra stack elements
        assert_token(MAX, [0, 2, 3], [0, 3])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            MAX([], {})
        with pytest.raises(StackUnderflowError):
            MAX([1], {})


class TestATAN2Operator:
    def test_repr(self):
        assert repr(ATAN2) == "ATAN2"

    def test_pops(self):
        assert ATAN2.pops == 2

    def test_puts(self):
        assert ATAN2.puts == 1

    def test_no_copy(self):
        assert copy(ATAN2) is ATAN2
        assert deepcopy(ATAN2) is ATAN2

    def test_call(self):
        # NOTE: second parameter is x, first is y
        assert_token(ATAN2, [0, 1], [0], approx=True)
        assert_token(ATAN2, [1, math.sqrt(3)], [math.pi / 6], approx=True)
        assert_token(ATAN2, [1, 1], [math.pi / 4], approx=True)
        assert_token(ATAN2, [math.sqrt(3), 1], [math.pi / 3], approx=True)
        assert_token(ATAN2, [1, 0], [math.pi / 2], approx=True)
        assert_token(
            ATAN2, [math.sqrt(3), -1], [math.pi / 2 + math.pi / 6], approx=True
        )
        assert_token(ATAN2, [1, -1], [math.pi / 2 + math.pi / 4], approx=True)
        assert_token(
            ATAN2, [1, -math.sqrt(3)], [math.pi / 2 + math.pi / 3], approx=True
        )
        assert_token(ATAN2, [0, -1], [math.pi / 2 + math.pi / 2], approx=True)
        assert_token(
            ATAN2,
            [
                np.array([0, 1, 1, np.sqrt(3), 1, np.sqrt(3), 1, 1, 0]),
                np.array([1, np.sqrt(3), 1, 1, 0, -1, -1, -np.sqrt(3), -1]),
            ],
            [
                np.array(
                    [
                        0.0,
                        np.pi / 6,
                        np.pi / 4,
                        np.pi / 3,
                        np.pi / 2,
                        np.pi / 2 + np.pi / 6,
                        np.pi / 2 + np.pi / 4,
                        np.pi / 2 + np.pi / 3,
                        np.pi / 2 + np.pi / 2,
                    ]
                )
            ],
            approx=True,
        )
        # extra stack elements
        assert_token(ATAN2, [0, 1, 1], [0, math.pi / 4], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            ATAN2([], {})


class TestHYPOTOperator:
    def test_repr(self):
        assert repr(HYPOT) == "HYPOT"

    def test_pops(self):
        assert HYPOT.pops == 2

    def test_puts(self):
        assert HYPOT.puts == 1

    def test_no_copy(self):
        assert copy(HYPOT) is HYPOT
        assert deepcopy(HYPOT) is HYPOT

    def test_call(self):
        assert_token(HYPOT, [1, 1], [math.sqrt(2)], approx=True)
        assert_token(HYPOT, [math.sqrt(3), 1], [2], approx=True)
        assert_token(
            HYPOT,
            [1, np.array([np.sqrt(3), 1])],
            [np.array([2, np.sqrt(2)])],
            approx=True,
        )
        assert_token(
            HYPOT,
            [np.array([np.sqrt(3), 1]), 1],
            [np.array([2, np.sqrt(2)])],
            approx=True,
        )
        assert_token(
            HYPOT,
            [np.array([np.sqrt(3), 1]), np.array([1, 1])],
            [np.array([2, np.sqrt(2)])],
            approx=True,
        )
        # extra stack elements
        assert_token(HYPOT, [0, math.sqrt(3), 1], [0, 2], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            HYPOT([], {})
        with pytest.raises(StackUnderflowError):
            HYPOT([1], {})


class TestR2Operator:
    def test_repr(self):
        assert repr(R2) == "R2"

    def test_pops(self):
        assert R2.pops == 2

    def test_puts(self):
        assert R2.puts == 1

    def test_no_copy(self):
        assert copy(R2) is R2
        assert deepcopy(R2) is R2

    def test_call(self):
        assert_token(R2, [2, 3], [13])
        assert_token(R2, [2, np.array([3, 4])], [np.array([13, 20])])
        assert_token(R2, [np.array([3, 4]), 2], [np.array([13, 20])])
        assert_token(R2, [np.array([1, 2]), np.array([3, 4])], [np.array([10, 20])])
        # extra stack elements
        assert_token(R2, [0, 2, 3], [0, 13], approx=True)
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            R2([], {})
        with pytest.raises(StackUnderflowError):
            R2([1], {})


class TestEQOperator:
    def test_repr(self):
        assert repr(EQ) == "EQ"

    def test_pops(self):
        assert EQ.pops == 2

    def test_puts(self):
        assert EQ.puts == 1

    def test_no_copy(self):
        assert copy(EQ) is EQ
        assert deepcopy(EQ) is EQ

    def test_call(self):
        assert_token(EQ, [2, 2], [True])
        assert_token(EQ, [2, 3], [False])
        assert_token(
            EQ, [2, np.array([1, np.nan, 2])], [np.array([False, False, True])]
        )
        assert_token(
            EQ, [np.array([1, np.nan, 2]), 2], [np.array([False, False, True])]
        )
        assert_token(
            EQ,
            [np.array([1, np.nan, 3, 3]), np.array([1, np.nan, 2, 3])],
            [np.array([True, False, False, True])],
        )
        # extra stack elements
        assert_token(EQ, [0, 2, 2], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            EQ([], {})
        with pytest.raises(StackUnderflowError):
            EQ([1], {})


class TestNEOperator:
    def test_repr(self):
        assert repr(NE) == "NE"

    def test_pops(self):
        assert NE.pops == 2

    def test_puts(self):
        assert NE.puts == 1

    def test_no_copy(self):
        assert copy(NE) is NE
        assert deepcopy(NE) is NE

    def test_call(self):
        assert_token(NE, [2, 2], [False])
        assert_token(NE, [2, 3], [True])
        assert_token(NE, [2, np.array([1, np.nan, 2])], [np.array([True, True, False])])
        assert_token(NE, [np.array([1, np.nan, 2]), 2], [np.array([True, True, False])])
        assert_token(
            NE,
            [np.array([1, np.nan, 3, 3]), np.array([1, np.nan, 2, 3])],
            [np.array([False, True, True, False])],
        )
        # extra stack elements
        assert_token(NE, [0, 2, 2], [0, False])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            NE([], {})
        with pytest.raises(StackUnderflowError):
            NE([1], {})


class TestLTOperator:
    def test_repr(self):
        assert repr(LT) == "LT"

    def test_pops(self):
        assert LT.pops == 2

    def test_puts(self):
        assert LT.puts == 1

    def test_no_copy(self):
        assert copy(LT) is LT
        assert deepcopy(LT) is LT

    def test_call(self):
        assert_token(LT, [2, 3], [True])
        assert_token(LT, [2, 2], [False])
        assert_token(LT, [3, 2], [False])
        assert_token(LT, [2, np.array([1, 2, 3])], [np.array([False, False, True])])
        assert_token(LT, [np.array([1, 2, 3]), 2], [np.array([True, False, False])])
        assert_token(
            LT,
            [np.array([1, 2, 3]), np.array([3, 2, 1])],
            [np.array([True, False, False])],
        )
        # extra stack elements
        assert_token(LT, [0, 2, 3], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            LT([], {})
        with pytest.raises(StackUnderflowError):
            LT([1], {})


class TestLEOperator:
    def test_repr(self):
        assert repr(LE) == "LE"

    def test_pops(self):
        assert LE.pops == 2

    def test_puts(self):
        assert LE.puts == 1

    def test_no_copy(self):
        assert copy(LE) is LE
        assert deepcopy(LE) is LE

    def test_le(self):
        assert_token(LE, [2, 3], [True])
        assert_token(LE, [2, 2], [True])
        assert_token(LE, [3, 2], [False])
        assert_token(LE, [2, np.array([1, 2, 3])], [np.array([False, True, True])])
        assert_token(LE, [np.array([1, 2, 3]), 2], [np.array([True, True, False])])
        assert_token(
            LE,
            [np.array([1, 2, 3]), np.array([3, 2, 1])],
            [np.array([True, True, False])],
        )
        # # extra stack elements
        assert_token(LE, [0, 2, 3], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            LE([], {})
        with pytest.raises(StackUnderflowError):
            LE([1], {})


class TestGTOperator:
    def test_repr(self):
        assert repr(GT) == "GT"

    def test_pops(self):
        assert GT.pops == 2

    def test_puts(self):
        assert GT.puts == 1

    def test_no_copy(self):
        assert copy(GT) is GT
        assert deepcopy(GT) is GT

    def test_call(self):
        assert_token(GT, [2, 3], [False])
        assert_token(GT, [2, 2], [False])
        assert_token(GT, [3, 2], [True])
        assert_token(GT, [2, np.array([1, 2, 3])], [np.array([True, False, False])])
        assert_token(GT, [np.array([1, 2, 3]), 2], [np.array([False, False, True])])
        assert_token(
            GT,
            [np.array([1, 2, 3]), np.array([3, 2, 1])],
            [np.array([False, False, True])],
        )
        # extra stack elements
        assert_token(GT, [0, 2, 3], [0, False])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            GT([], {})
        with pytest.raises(StackUnderflowError):
            GT([1], {})


class TestGEOperator:
    def test_repr(self):
        assert repr(GE) == "GE"

    def test_pops(self):
        assert GE.pops == 2

    def test_puts(self):
        assert GE.puts == 1

    def test_no_copy(self):
        assert copy(GE) is GE
        assert deepcopy(GE) is GE

    def test_call(self):
        assert_token(GE, [2, 3], [False])
        assert_token(GE, [2, 2], [True])
        assert_token(GE, [3, 2], [True])
        assert_token(GE, [2, np.array([1, 2, 3])], [np.array([True, True, False])])
        assert_token(GE, [np.array([1, 2, 3]), 2], [np.array([False, True, True])])
        assert_token(
            GE,
            [np.array([1, 2, 3]), np.array([3, 2, 1])],
            [np.array([False, True, True])],
        )
        # extra stack elements
        assert_token(GE, [0, 2, 3], [0, False])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            GE([], {})
        with pytest.raises(StackUnderflowError):
            GE([1], {})


class TestNANOperator:
    def test_repr(self):
        assert repr(NAN) == "NAN"

    def test_pops(self):
        assert NAN.pops == 2

    def test_puts(self):
        assert NAN.puts == 1

    def test_no_copy(self):
        assert copy(NAN) is NAN
        assert deepcopy(NAN) is NAN

    def test_call(self):
        assert_token(NAN, [2, 2], [float("nan")])
        assert_token(NAN, [2, 3], [2])
        assert_token(NAN, [2, np.array([2, 3])], [np.array([np.nan, 2])])
        assert_token(NAN, [np.array([2, 3]), 2], [np.array([np.nan, 3])])
        assert_token(
            NAN, [np.array([1, 2, 3]), np.array([3, 2, 1])], [np.array([1, np.nan, 3])]
        )
        # as float
        assert_token(
            NAN,
            [np.array([1.0, 2.0, 3.0]), np.array([3, 2, 1])],
            [np.array([1, np.nan, 3])],
            approx=True,
        )
        # extra stack elements
        assert_token(NAN, [0, 2, 2], [0, float("nan")])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            NAN([], {})
        with pytest.raises(StackUnderflowError):
            NAN([1], {})


class TestANDOperator:
    def test_repr(self):
        assert repr(AND) == "AND"

    def test_pops(self):
        assert AND.pops == 2

    def test_puts(self):
        assert AND.puts == 1

    def test_no_copy(self):
        assert copy(AND) is AND
        assert deepcopy(AND) is AND

    def test_call(self):
        assert_token(AND, [2, 3], [2])
        assert_token(AND, [float("nan"), 3], [3])
        assert_token(AND, [float("nan"), np.array([2, 3])], [np.array([2, 3])])
        assert_token(AND, [np.array([np.nan, 3]), 2], [np.array([2, 3])])
        assert_token(
            AND,
            [np.array([10, np.nan, 30]), np.array([1, 2, 3])],
            [np.array([10, 2, 30])],
        )
        # extra stack elements
        assert_token(AND, [0, float("nan"), 3], [0, 3])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            AND([], {})
        with pytest.raises(StackUnderflowError):
            AND([1], {})


class TestOROperator:
    def test_repr(self):
        assert repr(OR) == "OR"

    def test_pops(self):
        assert OR.pops == 2

    def test_puts(self):
        assert OR.puts == 1

    def test_no_copy(self):
        assert copy(OR) is OR
        assert deepcopy(OR) is OR

    def test_call(self):
        assert_token(OR, [2, 3], [2])
        assert_token(OR, [2, float("nan")], [float("nan")])
        assert_token(OR, [2, np.array([3, np.nan])], [np.array([2, np.nan])])
        assert_token(OR, [np.array([2, 3]), np.nan], [np.array([np.nan, np.nan])])
        assert_token(
            OR,
            [np.array([1, 2, 3]), np.array([10, np.nan, 30])],
            [np.array([1, np.nan, 3])],
        )
        # as float
        assert_token(
            OR,
            [np.array([1.0, 2.0, 3.0]), np.array([10, np.nan, 30])],
            [np.array([1, np.nan, 3])],
        )
        # extra stack elements
        assert_token(OR, [0, 2, float("nan")], [0, float("nan")])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            OR([], {})
        with pytest.raises(StackUnderflowError):
            OR([1], {})


class TestIANDOperator:
    def test_repr(self):
        assert repr(IAND) == "IAND"

    def test_pops(self):
        assert IAND.pops == 2

    def test_puts(self):
        assert IAND.puts == 1

    def test_no_copy(self):
        assert copy(IAND) is IAND
        assert deepcopy(IAND) is IAND

    def test_call(self):
        assert_token(IAND, [5, 3], [1])
        assert_token(IAND, [15, 21], [5])
        assert_token(IAND, [21, 15], [5])
        assert_token(IAND, [15, np.array([9, 21, 35])], [np.array([9, 5, 3])])
        assert_token(IAND, [np.array([9, 21, 35]), 15], [np.array([9, 5, 3])])
        assert_token(
            IAND,
            [np.array([9, 21, 35]), np.array([3, 15, 127])],
            [np.array([1, 5, 35])],
        )
        # extra stack elements
        assert_token(IAND, [0, 15, 21], [0, 5])
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


class TestIOROperator:
    def test_repr(self):
        assert repr(IOR) == "IOR"

    def test_pops(self):
        assert IOR.pops == 2

    def test_puts(self):
        assert IOR.puts == 1

    def test_no_copy(self):
        assert copy(IOR) is IOR
        assert deepcopy(IOR) is IOR

    def test_call(self):
        assert_token(IOR, [5, 3], [7])
        assert_token(IOR, [15, 21], [31])
        assert_token(IOR, [21, 15], [31])
        assert_token(IOR, [15, np.array([9, 21, 35])], [np.array([15, 31, 47])])
        assert_token(IOR, [np.array([9, 21, 35]), 15], [np.array([15, 31, 47])])
        assert_token(
            IOR,
            [np.array([9, 21, 35]), np.array([3, 15, 127])],
            [np.array([11, 31, 127])],
        )
        # extra stack elements
        assert_token(IOR, [0, 15, 21], [0, 31])
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


class TestBTESTOperator:
    def test_repr(self):
        assert repr(BTEST) == "BTEST"

    def test_pops(self):
        assert BTEST.pops == 2

    def test_puts(self):
        assert BTEST.puts == 1

    def test_no_copy(self):
        assert copy(BTEST) is BTEST
        assert deepcopy(BTEST) is BTEST

    def test_call(self):
        assert_token(BTEST, [9, 0], [True])
        assert_token(BTEST, [9, 1], [False])
        assert_token(BTEST, [9, 2], [False])
        assert_token(BTEST, [9, 3], [True])
        assert_token(BTEST, [9, 4], [False])
        assert_token(
            BTEST,
            [9, np.array([0, 1, 2, 3, 4])],
            [np.array([True, False, False, True, False])],
        )
        assert_token(BTEST, [np.array([1, 3, 5]), 1], [np.array([False, True, False])])
        assert_token(
            BTEST,
            [np.array([1, 3, 5]), np.array([1, 2, 0])],
            [np.array([False, False, True])],
        )
        # extra stack elements
        assert_token(BTEST, [0, 9, 3], [0, True])
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


class TestAVGOperator:
    def test_repr(self):
        assert repr(AVG) == "AVG"

    def test_pops(self):
        assert AVG.pops == 2

    def test_puts(self):
        assert AVG.puts == 1

    def test_no_copy(self):
        assert copy(AVG) is AVG
        assert deepcopy(AVG) is AVG

    def test_call(self):
        assert repr(AVG) == "AVG"
        assert AVG.pops == 2
        assert AVG.puts == 1
        assert_token(AVG, [5, 11], [8])
        assert_token(AVG, [float("nan"), 11], [11])
        assert_token(AVG, [5, float("nan")], [5])
        assert_token(AVG, [3, np.array([7, np.nan, 11])], [np.array([5, 3, 7])])
        assert_token(AVG, [np.nan, np.array([1, 2, 3])], [np.array([1, 2, 3])])
        assert_token(AVG, [np.array([7, np.nan, 11]), 3], [np.array([5, 3, 7])])
        assert_token(AVG, [np.array([1, 2, 3]), np.nan], [np.array([1, 2, 3])])
        assert_token(
            AVG,
            [np.array([3, np.nan, 11]), np.array([7, 2, np.nan])],
            [np.array([5, 2, 11])],
        )
        # extra stack elements
        assert_token(AVG, [0, 5, 11], [0, 8])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            AVG([], {})
        with pytest.raises(StackUnderflowError):
            AVG([1], {})


class TestDXDYOperator:
    def test_repr(self):
        assert repr(DXDY) == "DXDY"

    def test_pops(self):
        assert DXDY.pops == 2

    def test_puts(self):
        assert DXDY.puts == 1

    def test_no_copy(self):
        assert copy(DXDY) is DXDY
        assert deepcopy(DXDY) is DXDY

    def test_call(self):
        assert_token(DXDY, [5, 11], [float("nan")])
        assert_token(DXDY, [3, np.array([5, 11])], [np.array([np.nan, np.nan])])
        assert_token(DXDY, [3, np.array([5, 7, 11])], [np.array([np.nan, 0, np.nan])])
        assert_token(
            DXDY, [3, np.array([5, 7, 8, 11])], [np.array([np.nan, 0, 0, np.nan])]
        )
        with warnings.catch_warnings():  # divide by zero in numpy
            warnings.simplefilter("ignore")
            assert_token(DXDY, [np.array([5, 11]), 3], [np.array([np.nan, np.nan])])
            assert_token(
                DXDY, [np.array([5, 7, 11]), 3], [np.array([np.nan, np.inf, np.nan])]
            )
            assert_token(
                DXDY,
                [np.array([5, 7, 8, 11]), 3],
                [np.array([np.nan, np.inf, np.inf, np.nan])],
            )
        assert_token(
            DXDY,
            [np.array([0, 4, 9, 8]), np.array([5, 7, 8, 11])],
            [np.array([np.nan, 3, 1, np.nan])],
        )
        # extra stack elements
        assert_token(DXDY, [0, 5, 11], [0, float("nan")])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            DXDY([], {})
        with pytest.raises(StackUnderflowError):
            DXDY([1], {})


class TestEXCHOperator:
    def test_rerpr(self):
        assert repr(EXCH) == "EXCH"

    def test_pops(self):
        assert EXCH.pops == 2

    def test_puts(self):
        assert EXCH.puts == 2

    def test_no_copy(self):
        assert copy(EXCH) is EXCH
        assert deepcopy(EXCH) is EXCH

    def test_call(self):
        assert_token(EXCH, [5, 11], [11, 5])
        assert_token(EXCH, [3, np.array([5, 11])], [np.array([5, 11]), 3])
        assert_token(EXCH, [np.array([5, 11]), 3], [3, np.array([5, 11])])
        assert_token(
            EXCH,
            [np.array([1, 2]), np.array([3, 4])],
            [np.array([3, 4]), np.array([1, 2])],
        )
        # extra stack elements
        assert_token(EXCH, [0, 5, 11], [0, 11, 5])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            EXCH([], {})
        with pytest.raises(StackUnderflowError):
            EXCH([1], {})


class TestINRANGEOperator:
    def test_repr(self):
        assert repr(INRANGE) == "INRANGE"

    def test_pops(self):
        assert INRANGE.pops == 3

    def test_puts(self):
        assert INRANGE.puts == 1

    def test_no_copy(self):
        assert copy(INRANGE) is INRANGE
        assert deepcopy(INRANGE) is INRANGE

    def test_call(self):
        assert_token(INRANGE, [0, 1, 3], [False])
        assert_token(INRANGE, [1, 1, 3], [True])
        assert_token(INRANGE, [2, 1, 3], [True])
        assert_token(INRANGE, [3, 1, 3], [True])
        assert_token(INRANGE, [4, 1, 3], [False])
        assert_token(
            INRANGE,
            [np.array([0, 1, 2, 3, 4]), 1, 3],
            [np.array([False, True, True, True, False])],
        )
        assert_token(
            INRANGE, [2, np.array([1, 2, 3]), 3], [np.array([True, True, False])]
        )
        assert_token(
            INRANGE, [2, 1, np.array([1, 2, 3])], [np.array([False, True, True])]
        )
        assert_token(
            INRANGE,
            [np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])],
            [np.array([True, True, True])],
        )
        # extra stack elements
        assert_token(INRANGE, [0, 2, 1, 3], [0, True])
        # not enough stack elements
        with pytest.raises(StackUnderflowError):
            INRANGE([], {})
        with pytest.raises(StackUnderflowError):
            INRANGE([1], {})
        with pytest.raises(StackUnderflowError):
            INRANGE([1, 2], {})


class TestBOXCAROperator:
    def test_repr(self):
        assert repr(BOXCAR) == "BOXCAR"

    def test_pops(self):
        assert BOXCAR.pops == 3

    def test_puts(self):
        assert BOXCAR.puts == 1

    def test_no_copy(self):
        assert copy(BOXCAR) is BOXCAR
        assert deepcopy(BOXCAR) is BOXCAR

    def test_call(self):
        # returns value if scalar
        assert_token(BOXCAR, [1, 2, 3], [1])
        # simple
        assert_token(
            BOXCAR,
            [np.array([0, 1, 2, 3, 4]), 0, 3],
            [np.array([1 / 3, 1, 2, 3, 11 / 3])],
            approx=True,
        )
        # window size of 1 should return original array
        assert_token(
            BOXCAR,
            [np.array([0, 1, 2, 3, 4]), 0, 1],
            [np.array([0, 1, 2, 3, 4])],
            approx=True,
        )
        # with nan it should base result on non nan values in window
        assert_token(
            BOXCAR,
            [np.array([0, np.nan, 2, 3, 4]), 0, 3],
            [np.array([0, np.nan, 2.5, 3, 11 / 3])],
            approx=True,
        )
        # multi-dimensional x
        assert_token(
            BOXCAR,
            [np.array([[0, 1, 2, 3, 4], [0, 10, 20, 30, 40]]), 1, 3],
            [np.array([[1 / 3, 1, 2, 3, 11 / 3], [10 / 3, 10, 20, 30, 110 / 3]])],
            approx=True,
        )
        assert_token(
            BOXCAR,
            [
                np.array(
                    [[0, 1, 2, 3, 4], [0, 10, 20, 30, 40], [0, 100, 200, 300, 400]]
                ),
                0,
                3,
            ],
            [
                np.array(
                    [[0, 4, 8, 12, 16], [0, 37, 74, 111, 148], [0, 70, 140, 210, 280]]
                )
            ],
            approx=True,
        )
        # extra stack elements
        assert_token(BOXCAR, [0, 1, 2, 3], [0, 1])
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


class TestGAUSSOperator:
    def test_repr(self):
        assert repr(GAUSS) == "GAUSS"

    def test_pops(self):
        assert GAUSS.pops == 3

    def test_puts(self):
        assert GAUSS.puts == 1

    def test_no_copy(self):
        assert copy(GAUSS) is GAUSS
        assert deepcopy(GAUSS) is GAUSS

    def test_call(self):
        # returns value if scalar
        assert_token(GAUSS, [1, 2, 3], [1])
        # simple
        assert_token(
            GAUSS,
            [np.array([0, 1, 2, 3, 4]), 0, 3],
            [np.array([1.06300638, 1.5089338, 2.0, 2.4910662, 2.93699362])],
            approx=True,
            rtol=1e-6,
            atol=1e-6,
        )
        # with nan it should base result on non nan values in window
        assert_token(
            GAUSS,
            [np.array([0, np.nan, 2, 3, 4]), 0, 3],
            [np.array([1.07207303, np.nan, 2.14390036, 2.66876589, 3.10693751])],
            approx=True,
            rtol=1e-6,
            atol=1e-6,
        )
        # multi-dimensional x
        assert_token(
            GAUSS,
            [np.array([[0, 1, 2, 3, 4], [0, 10, 20, 30, 40]]), 1, 3],
            [
                np.array(
                    [
                        [1.06300638, 1.5089338, 2.0, 2.4910662, 2.93699362],
                        [10.63006385, 15.08933801, 20.0, 24.91066199, 29.36993615],
                    ]
                )
            ],
            approx=True,
            rtol=1e-6,
            atol=1e-6,
        )
        assert_token(
            GAUSS,
            [
                np.array(
                    [[0, 1, 2, 3, 4], [0, 10, 20, 30, 40], [0, 100, 200, 300, 400]]
                ),
                0,
                3,
            ],
            [
                np.array(
                    [
                        [0, 32.59544683, 65.19089366, 97.78634049, 130.38178732],
                        [0, 45.11412619, 90.22825238, 135.34237857, 180.45650475],
                        [0, 58.21491651, 116.42983302, 174.64474953, 232.85966604],
                    ]
                )
            ],
            approx=True,
            rtol=1e-6,
            atol=1e-6,
        )
        # extra stack elements
        assert_token(GAUSS, [0, 1, 2, 3], [0, 1])
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


def test_token_keywords():
    assert token("SUB") == SUB
    assert token("ADD") == ADD
    assert token("MUL") == MUL
    assert token("PI") == PI
    assert token("E") == E


def test_token_literals():
    assert token("3") == Literal(3)
    assert token("3.14e10") == Literal(3.14e10)
    assert token("3.14E10") == Literal(3.14e10)
    assert token("3.14d10") == Literal(3.14e10)
    assert token("3.14D10") == Literal(3.14e10)
    assert token("3.14e+10") == Literal(3.14e10)
    assert token("3.14E+10") == Literal(3.14e10)
    assert token("3.14d+10") == Literal(3.14e10)
    assert token("3.14D+10") == Literal(3.14e10)
    assert token("3.14e-10") == Literal(3.14e-10)
    assert token("3.14E-10") == Literal(3.14e-10)
    assert token("3.14d-10") == Literal(3.14e-10)
    assert token("3.14D-10") == Literal(3.14e-10)
    assert token("3.14+100") == Literal(3.14e100)
    assert token("3.14-100") == Literal(3.14e-100)


def test_token_variables():
    assert token("alt") == Variable("alt")
    assert token("ref_frame_offset") == Variable("ref_frame_offset")
    with pytest.raises(ValueError) as excinfo:
        token("3name")
    assert str(excinfo.value) == "invalid RPN token '3name'"
    with pytest.raises(ValueError) as excinfo:
        token("3,")
    assert str(excinfo.value) == "invalid RPN token '3,'"


def test_token_wrong_type():
    with pytest.raises(TypeError):
        token(5)  # type: ignore


class TestExpression:
    def test_init_with_token_sequence(self):
        # complete expressions
        Expression([Literal(1)])
        Expression([Literal(1), Literal(2.5), ADD])
        Expression([Literal(1), Variable("a_var"), ADD])
        # incomplete expressions
        Expression([])
        Expression([Literal(4), POP])
        Expression([POP, POP])
        Expression([Literal(1), ADD])
        Expression([Literal(1), Literal(2.5), DUP])
        Expression([ADD, Literal(2), MUL, Variable("a_var"), DUP])

    def test_init_with_mixed_sequence(self):
        # complete expressions
        Expression([1])
        Expression([1, 2.5, ADD])
        Expression([1, "a_var", ADD])
        # incomplete expressions
        Expression([])
        Expression([4, POP])
        Expression([POP, POP])
        Expression([1, ADD])
        Expression([1, 2.5, DUP])
        Expression([ADD, 2, MUL, "a_var", DUP])
        # invalid tokens
        with pytest.raises(ValueError) as excinfo:
            Expression([1, "3name", ADD])
        assert str(excinfo.value) == "invalid RPN token '3name'"
        with pytest.raises(ValueError) as excinfo:
            Expression([1, "3,", ADD])
        assert str(excinfo.value) == "invalid RPN token '3,'"

    def test_init_with_token_string(self):
        # complete expressions
        Expression("1")
        Expression("1 2.5 ADD")
        Expression("1 a_var ADD")
        # incomplete expressions
        Expression("")
        Expression("4 POP")
        Expression("POP POP")
        Expression("1 ADD")
        Expression("1 2.5 DUP")
        Expression("ADD 2 MUL a_var DUP")
        # extra spaces
        Expression("1   a_var        ADD")
        # invalid tokens
        with pytest.raises(ValueError) as excinfo:
            Expression("1 3name ADD")
        assert str(excinfo.value) == "invalid RPN token '3name'"
        with pytest.raises(ValueError) as excinfo:
            Expression("1 3, ADD")
        assert str(excinfo.value) == "invalid RPN token '3,'"

    # equality tests are first because they make it so we no longer have to
    # check every constructor variation since we now know they are equivalent
    def test_eq_with_token_sequence(self):
        # complete expressions
        assert Expression([Literal(1)]) == Expression([Literal(1)])
        assert Expression([Literal(1), Literal(2.5), ADD]) == Expression(
            [Literal(1), Literal(2.5), ADD]
        )
        assert Expression([Literal(1), Variable("a_var"), ADD]) == Expression(
            [Literal(1), Variable("a_var"), ADD]
        )
        # incomplete expressions
        assert Expression([]) == Expression([])
        assert Expression([Literal(4), POP]) == Expression([Literal(4), POP])
        assert Expression([POP, POP]) == Expression([POP, POP])
        assert Expression([Literal(1), ADD]) == Expression([Literal(1), ADD])
        assert Expression([Literal(1), Literal(2.5), DUP]) == Expression(
            [Literal(1), Literal(2.5), DUP]
        )
        assert Expression([ADD, Literal(2), MUL, Variable("a_var"), DUP]) == Expression(
            [ADD, Literal(2), MUL, Variable("a_var"), DUP]
        )
        # no comparison
        assert not Expression([Literal(1)]) == 1
        assert Expression([1]) != 1

    def test_eq_with_mixed_sequence(self):
        # complete expressions
        assert Expression([1]) == Expression([Literal(1)])
        assert Expression([1, 2.5, ADD]) == Expression([Literal(1), Literal(2.5), ADD])
        assert Expression([1, "a_var", ADD]) == Expression(
            [Literal(1), Variable("a_var"), ADD]
        )
        # incomplete expressions
        assert Expression([]) == Expression([])
        assert Expression([4, POP]) == Expression([Literal(4), POP])
        assert Expression([POP, POP]) == Expression([POP, POP])
        assert Expression([1, ADD]) == Expression([Literal(1), ADD])
        assert Expression([1, 2.5, DUP]) == Expression([Literal(1), Literal(2.5), DUP])
        assert Expression([ADD, 2, MUL, "a_var", DUP]) == Expression(
            [ADD, Literal(2), MUL, Variable("a_var"), DUP]
        )
        # no comparison
        assert not Expression([1]) == 1
        assert Expression([1]) != 1

    def test_eq_with_token_string(self):
        # complete expressions
        assert Expression("1") == Expression([Literal(1)])
        assert Expression("1 2.5 ADD") == Expression([Literal(1), Literal(2.5), ADD])
        assert Expression("1 a_var ADD") == Expression(
            [Literal(1), Variable("a_var"), ADD]
        )
        # incomplete expressions
        assert Expression("") == Expression([])
        assert Expression("4 POP") == Expression([Literal(4), POP])
        assert Expression("POP POP") == Expression([POP, POP])
        assert Expression("1 ADD") == Expression([Literal(1), ADD])
        assert Expression("1 2.5 DUP") == Expression([Literal(1), Literal(2.5), DUP])
        assert Expression("ADD 2 MUL a_var DUP") == Expression(
            [ADD, Literal(2), MUL, Variable("a_var"), DUP]
        )
        # extra spaces
        assert Expression("1    a_var         ADD") == Expression(
            [Literal(1), Variable("a_var"), ADD]
        )
        # no comparison
        assert not Expression("1") == 1

    def test_eq_with_complete_expression(self):
        assert Expression("1") == CompleteExpression("1")
        assert Expression("1 2.5 ADD") == CompleteExpression("1 2.5 ADD")
        assert Expression("1 a_var ADD") == CompleteExpression("1 a_var ADD")

    def test_pops(self):
        # complete expressions
        assert Expression("1").pops == 0
        assert Expression("1 2.5 ADD").pops == 0
        assert Expression("1 a_var ADD").pops == 0
        # incomplete expressions
        assert Expression("").pops == 0
        assert Expression("4 POP").pops == 0
        assert Expression("POP POP").pops == 2
        assert Expression("1 ADD").pops == 1
        assert Expression("1 2.5 DUP").pops == 0
        assert Expression("ADD 2 MUL a_var DUP").pops == 2

    def test_puts(self):
        # complete expressions
        assert Expression("1").puts == 1
        assert Expression("1 2.5 ADD").puts == 1
        assert Expression("1 a_var ADD").puts == 1
        # incomplete expressions
        assert Expression("").puts == 0
        assert Expression("4 POP").puts == 0
        assert Expression("POP POP").puts == 0
        assert Expression("1 ADD").puts == 1
        assert Expression("1 2.5 DUP").puts == 3
        assert Expression("ADD 2 MUL a_var DUP").puts == 3

    def test_variables(self):
        # complete expressions
        assert Expression("1 2.5 ADD").variables == set()
        assert Expression("a_var 2.5 ADD").variables == {"a_var"}
        assert Expression("a_var b_var ADD").variables == {"a_var", "b_var"}
        # incomplete expressions
        assert Expression("").variables == set()
        assert Expression("a_var ADD").variables == {"a_var"}
        assert Expression("a_var b_var DUP").variables == {"a_var", "b_var"}

    def test_complete(self):
        # complete expressions
        assert Expression("1").complete() == CompleteExpression("1")
        assert isinstance(Expression("1").complete(), CompleteExpression)
        assert Expression("1 2.5 ADD").complete() == CompleteExpression("1 2.5 ADD")
        assert isinstance(Expression("1 2.5 ADD").complete(), CompleteExpression)
        assert Expression("1 a_var ADD").complete() == CompleteExpression("1 a_var ADD")
        assert isinstance(Expression("1 a_var ADD").complete(), CompleteExpression)
        # incomplete expressions
        with pytest.raises(ValueError) as excinfo:
            Expression("").complete()
        assert str(excinfo.value).splitlines() == [
            "expression does not produce a result"
        ]
        with pytest.raises(ValueError) as excinfo:
            Expression("4 POP").complete()
        assert str(excinfo.value).splitlines() == [
            "expression does not produce a result",
            "4 POP",
        ]
        with pytest.raises(ValueError) as excinfo:
            Expression("POP POP").complete()
        assert str(excinfo.value).splitlines() == [
            "'POP' takes 1 argument(s) but the stack will only have 0 " "element(s)",
            "POP POP",
            "^",
        ]
        with pytest.raises(ValueError) as excinfo:
            Expression("1 ADD").complete()
        assert str(excinfo.value).splitlines() == [
            "'ADD' takes 2 argument(s) but the stack will only have 1 " "element(s)",
            "1 ADD",
            "  ^",
        ]
        with pytest.raises(ValueError) as excinfo:
            Expression("1 2.5 DUP").complete()
        assert str(excinfo.value).splitlines() == [
            "expression produces too many results (3), expected 1",
            "1 2.5 DUP",
        ]
        with pytest.raises(ValueError) as excinfo:
            Expression("ADD 2 MUL a_var DUP").complete()
        assert str(excinfo.value).splitlines() == [
            "'ADD' takes 2 argument(s) but the stack will only have 0 " "element(s)",
            "ADD 2 MUL a_var DUP",
            "^",
        ]

    def test_is_complete(self):
        # complete expressions
        assert Expression("1").is_complete()
        assert Expression("1 2.5 ADD").is_complete()
        assert Expression("1 a_var ADD").is_complete()
        # incomplete expressions
        assert not Expression("").is_complete()
        assert not Expression("4 POP").is_complete()
        assert not Expression("POP POP").is_complete()
        assert not Expression("1 ADD").is_complete()
        assert not Expression("1 2.5 DUP").is_complete()
        assert not Expression("ADD 2 MUL a_var DUP").is_complete()

    # expressions can behave as a token
    def test_call(self):
        # complete expressions
        assert_token(Expression("1"), [], [1])
        assert_token(Expression("1 2.5 ADD"), [], [3.5])
        assert_token(Expression("1 a_var ADD"), [], [5], {"a_var": 4})
        # incomplete expressions
        assert_token(Expression(""), [], [])
        assert_token(Expression("4 POP"), [], [])
        assert_token(Expression("POP POP"), [2, 3], [])
        assert_token(Expression("1 ADD"), [2], [3])
        assert_token(Expression("1 2.5 DUP"), [], [1, 2.5, 2.5])
        assert_token(
            Expression("ADD 2 MUL a_var DUP"), [4, 7], [22, 4, 4], {"a_var": 4}
        )

    def test_contains(self):
        # token in expression
        assert Literal(1) in Expression("1 a_var ADD")
        assert Variable("a_var") in Expression("1 a_var ADD")
        assert ADD in Expression("1 a_var ADD")
        # token not in expression
        assert Literal(2) not in Expression("1 a_var ADD")
        assert Variable("b_var") not in Expression("1 a_var ADD")
        assert SUB not in Expression("1 a_var ADD")
        # not a token
        assert 1 not in Expression("1 a_var ADD")
        assert "a_var" not in Expression("1 a_var ADD")

    def test_getitem_int_index(self):
        expression = Expression("ADD 2 MUL a_var DUP")
        assert expression[0] == ADD
        assert expression[1] == Literal(2)
        assert expression[2] == MUL
        assert expression[3] == Variable("a_var")
        assert expression[4] == DUP
        with pytest.raises(IndexError):
            expression[5]

    def test_getitem_slice_index(self):
        expression = Expression("ADD 2 MUL a_var DUP")
        assert expression[:] == expression
        assert expression[1:] == Expression("2 MUL a_var DUP")
        assert expression[:-1] == Expression("ADD 2 MUL a_var")
        assert expression[::2] == Expression("ADD MUL DUP")

    def test_iter(self):
        assert [t for t in Expression("ADD 2 MUL a_var DUP")] == [
            ADD,
            Literal(2),
            MUL,
            Variable("a_var"),
            DUP,
        ]

    def test_len(self):
        # complete expressions
        assert len(Expression("1")) == 1
        assert len(Expression("1 2.5 ADD")) == 3
        assert len(Expression("1 a_var ADD")) == 3
        # incomplete expressions
        assert len(Expression("")) == 0
        assert len(Expression("4 POP")) == 2
        assert len(Expression("POP POP")) == 2
        assert len(Expression("1 ADD")) == 2
        assert len(Expression("1 2.5 DUP")) == 3
        assert len(Expression("ADD 2 MUL a_var DUP")) == 5

    def test_ne(self):
        # complete expressions
        assert not Expression("1") != Expression([Literal(1)])
        assert not Expression("1 2.5 ADD") != Expression(
            [Literal(1), Literal(2.5), ADD]
        )
        assert not Expression("1 a_var ADD") != Expression(
            [Literal(1), Variable("a_var"), ADD]
        )
        # incomplete expressions
        assert not Expression("") != Expression([])
        assert not Expression("4 POP") != Expression([Literal(4), POP])
        assert not Expression("POP POP") != Expression([POP, POP])
        assert not Expression("1 ADD") != Expression([Literal(1), ADD])
        assert not (
            Expression("1 2.5 DUP") != Expression([Literal(1), Literal(2.5), DUP])
        )
        assert not (
            Expression("ADD 2 MUL a_var DUP")
            != Expression([ADD, Literal(2), MUL, Variable("a_var"), DUP])
        )
        # no comparison
        assert Expression("1") != 1

    def test_ne_with_complete_expression(self):
        assert not Expression("1") != CompleteExpression("1")
        assert not Expression("1 2.5 ADD") != CompleteExpression("1 2.5 ADD")
        assert not Expression("1 a_var ADD") != CompleteExpression("1 a_var ADD")

    def test_add(self):
        # result is complete
        assert Expression("1 a_var ADD") + Expression("3 SUB") == Expression(
            "1 a_var ADD 3 SUB"
        )
        assert isinstance(
            Expression("1 a_var ADD") + Expression("3 SUB"), CompleteExpression
        )
        assert Expression("") + Expression("1") + Expression("") == Expression("1")
        assert isinstance(
            Expression("") + Expression("1") + Expression(""), CompleteExpression
        )
        # result is incomplete
        assert Expression("2 a_var SUM") + Expression("5 MUL") == Expression(
            "2 a_var SUM 5 MUL"
        )
        assert not isinstance(
            Expression("2 a_var SUM") + Expression("5 MUL"), CompleteExpression
        )
        # unsupported
        with pytest.raises(TypeError):
            Expression("1 2 ADD") + 5
        with pytest.raises(TypeError):
            5 + Expression("1 2 ADD")  # type: ignore
        with pytest.raises(TypeError):
            Expression("1 2 ADD") + "a_var"
        with pytest.raises(TypeError):
            "a_var" + Expression("1 2 ADD")  # type: ignore

    def test_repr(self):
        # complete expressions
        assert repr(Expression("1")) == "Expression([Literal(1)])"
        assert (
            repr(Expression("1 2.5 ADD"))
            == "Expression([Literal(1), Literal(2.5), ADD])"
        )
        assert (
            repr(Expression("1 a_var ADD"))
            == "Expression([Literal(1), Variable('a_var'), ADD])"
        )
        # incomplete expressions
        assert repr(Expression("")) == "Expression([])"
        assert repr(Expression("4 POP")) == "Expression([Literal(4), POP])"
        assert repr(Expression("POP POP")) == "Expression([POP, POP])"
        assert repr(Expression("1 ADD")) == "Expression([Literal(1), ADD])"
        assert (
            repr(Expression("1 2.5 DUP"))
            == "Expression([Literal(1), Literal(2.5), DUP])"
        )
        assert (
            repr(Expression("ADD 2 MUL a_var DUP"))
            == "Expression([ADD, Literal(2), MUL, Variable('a_var'), DUP])"
        )

    def test_str(self):
        # complete expressions
        assert str(Expression("1")) == "1"
        assert str(Expression("1 2.5 ADD")) == "1 2.5 ADD"
        assert str(Expression("1 a_var ADD")) == "1 a_var ADD"
        # incomplete expressions
        assert str(Expression("")) == ""
        assert str(Expression("4 POP")) == "4 POP"
        assert str(Expression("POP POP")) == "POP POP"
        assert str(Expression("1 ADD")) == "1 ADD"
        assert str(Expression("1 2.5 DUP")) == "1 2.5 DUP"
        assert str(Expression("ADD 2 MUL a_var DUP")) == "ADD 2 MUL a_var DUP"


class TestCompleteExpression:
    def test_init_with_token_sequence(self):
        CompleteExpression([Literal(1)])
        CompleteExpression([Literal(1), Literal(2.5), ADD])
        CompleteExpression([Literal(1), Variable("a_var"), ADD])

    def test_init_with_mixed_sequence(self):
        CompleteExpression([1])
        CompleteExpression([1, 2.5, ADD])
        CompleteExpression([1, "a_var", ADD])
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([1, "3name", ADD])
        assert str(excinfo.value) == "invalid RPN token '3name'"
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([1, "3,", ADD])
        assert str(excinfo.value) == "invalid RPN token '3,'"

    def test_init_with_token_string(self):
        CompleteExpression("1")
        CompleteExpression("1 2.5 ADD")
        CompleteExpression("1 a_var ADD")
        # extra spaces
        CompleteExpression("1   a_var        ADD")
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression("1 3name ADD")
        assert str(excinfo.value) == "invalid RPN token '3name'"
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression("1 3, ADD")
        assert str(excinfo.value) == "invalid RPN token '3,'"

    def test_init_no_results_with_token_sequence(self):
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([])
        assert str(excinfo.value).splitlines() == [
            "expression does not produce a result"
        ]
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([Literal(1), Variable("a_var"), ADD, POP])
        assert str(excinfo.value).splitlines() == [
            "expression does not produce a result",
            "1 a_var ADD POP",
        ]

    def test_init_no_results_with_mixed_sequence(self):
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([1, "a_var", ADD, POP])
        assert str(excinfo.value).splitlines() == [
            "expression does not produce a result",
            "1 a_var ADD POP",
        ]

    def test_init_no_results_with_token_string(self):
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression("")
        assert str(excinfo.value).splitlines() == [
            "expression does not produce a result"
        ]
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression("1 a_var ADD POP")
        assert str(excinfo.value).splitlines() == [
            "expression does not produce a result",
            "1 a_var ADD POP",
        ]

    def test_init_too_many_results_with_token_sequence(self):
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([Literal(1), Variable("a_var")])
        assert str(excinfo.value).splitlines() == [
            "expression produces too many results (2), expected 1",
            "1 a_var",
        ]
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([Literal(1), Variable("a_var"), Literal(2.5), ADD, DUP])
        assert str(excinfo.value).splitlines() == [
            "expression produces too many results (3), expected 1",
            "1 a_var 2.5 ADD DUP",
        ]

    def test_init_too_many_results_with_mixed_sequence(self):
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([1, "a_var"])
        assert str(excinfo.value).splitlines() == [
            "expression produces too many results (2), expected 1",
            "1 a_var",
        ]
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([1, "a_var", 2.5, ADD, DUP])
        assert str(excinfo.value).splitlines() == [
            "expression produces too many results (3), expected 1",
            "1 a_var 2.5 ADD DUP",
        ]

    def test_init_too_many_results_with_token_string(self):
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression("1 a_var")
        assert str(excinfo.value).splitlines() == [
            "expression produces too many results (2), expected 1",
            "1 a_var",
        ]
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression("1 a_var 2.5 ADD DUP")
        assert str(excinfo.value).splitlines() == [
            "expression produces too many results (3), expected 1",
            "1 a_var 2.5 ADD DUP",
        ]

    def test_init_stack_underflow_with_token_sequence(self):
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([Literal(1), ADD])
        assert str(excinfo.value).splitlines() == [
            "'ADD' takes 2 argument(s) but the stack will only have " "1 element(s)",
            "1 ADD",
            "  ^",
        ]
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([Literal(1), Variable("a_var"), ADD, POP, MUL])
        assert str(excinfo.value).splitlines() == [
            "'MUL' takes 2 argument(s) but the stack will only have " "0 element(s)",
            "1 a_var ADD POP MUL",
            "                ^",
        ]

    def test_init_stack_underflow_with_mixed_sequence(self):
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([1, ADD])
        assert str(excinfo.value).splitlines() == [
            "'ADD' takes 2 argument(s) but the stack will only have " "1 element(s)",
            "1 ADD",
            "  ^",
        ]
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression([1, "a_var", ADD, POP, MUL])
        assert str(excinfo.value).splitlines() == [
            "'MUL' takes 2 argument(s) but the stack will only have " "0 element(s)",
            "1 a_var ADD POP MUL",
            "                ^",
        ]

    def test_init_stack_underflow_with_token_string(self):
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression("1 ADD")
        assert str(excinfo.value).splitlines() == [
            "'ADD' takes 2 argument(s) but the stack will only have " "1 element(s)",
            "1 ADD",
            "  ^",
        ]
        with pytest.raises(ValueError) as excinfo:
            CompleteExpression("1 a_var ADD POP MUL")
        assert str(excinfo.value).splitlines() == [
            "'MUL' takes 2 argument(s) but the stack will only have " "0 element(s)",
            "1 a_var ADD POP MUL",
            "                ^",
        ]

    def test_complete(self):
        # returns self since it is always complete
        expression = CompleteExpression("1 a_var ADD")
        assert id(expression.complete()) == id(expression)

    def test_eval(self):
        assert CompleteExpression("1").eval() == 1
        assert CompleteExpression("1 2.5 ADD").eval() == 3.5
        assert CompleteExpression("1 a_var ADD").eval({"a_var": 10}) == 11
        # extra spaces
        with pytest.raises(KeyError):
            CompleteExpression("1 a_var ADD").eval()

    def test_eq_with_expresion(self):
        assert CompleteExpression("1") == Expression("1")
        assert CompleteExpression("1 2.5 ADD") == Expression("1 2.5 ADD")
        assert CompleteExpression("1 a_var ADD") == Expression("1 a_var ADD")

    def test_ne_with_expresion(self):
        assert not CompleteExpression("1") != Expression("1")
        assert not CompleteExpression("1 2.5 ADD") != Expression("1 2.5 ADD")
        assert not (CompleteExpression("1 a_var ADD") != Expression("1 a_var ADD"))
