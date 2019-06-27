r"""Reverse Polish Notation calculator.

Constants
---------

==============  =================
Keyword         Value
==============  =================
:data:`PI`      3.141592653589793
:data:`E`       2.718281828459045
==============  =================


Operators
---------

===============  ==============================================================
Keyword          Description
===============  ==============================================================
:data:`SUB`      a = x - y
:data:`ADD`      a = x + y
:data:`MUL`      a = x*y
:data:`POP`      remove top of stack
:data:`NEG`      a = −x
:data:`ABS`      a = \|x\|
:data:`INV`      a = 1/x
:data:`SQRT`     a = sqrt(x)
:data:`SQR`      a = x*x
:data:`EXP`      a = exp(x)
:data:`LOG`      a = ln(x)
:data:`LOG10`    a = log10(x)
:data:`SIN`      a = sin(x)
:data:`COS`      a = cos(x)
:data:`TAN`      a = tan(x)
:data:`SIND`     a = sin(x) [x in degrees]
:data:`COSD`     a = cos(x) [x in degrees]
:data:`TAND`     a = tan(x) [x in degrees]
:data:`SINH`     a = sinh(x)
:data:`COSH`     a = cosh(x)
:data:`TANH`     a = tanh(x)
:data:`ASIN`     a = arcsin(x)
:data:`ACOS`     a = arccos(x)
:data:`ATAN`     a = arctan(x)
:data:`ASIND`    a = arcsin(x) [a in degrees]
:data:`ACOSD`    a = arccos(x) [a in degrees]
:data:`ATAND`    a = arctan(x) [a in degrees]
:data:`ASINH`    a = arcsinh(x)
:data:`ACOSH`    a = arccosh(x)
:data:`ATANH`    a = arctanh(x)
:data:`ISNAN`    a = 1 if x is NaN; a = 0 otherwise
:data:`ISAN`     a = 0 if x is NaN; a = 1 otherwise
:data:`RINT`     a is nearest integer to x
:data:`NINT`     a is nearest integer to x
:data:`CEIL`     a is nearest integer greater or equal to x
:data:`CEILING`  a is nearest integer greater or equal to x
:data:`FLOOR`    a is nearest integer less or equal to x
:data:`D2R`      convert x from degrees to radians
:data:`R2D`      convert x from radian to degrees
:data:`YMDHMS`   convert from seconds since 1985 to YYMMDDHHMMSS format (float)
:data:`SUM`      a[i] = x[1] + ... + x[i] while skipping all NaN
:data:`DIF`      a[i] = x[i]-x[i-1]; a[1] = NaN
:data:`DUP`      duplicate the last item on the stack
:data:`DIV`      a = x/y
:data:`POW`      a = x**y
:data:`FMOD`     a = x modulo y
:data:`MIN`      a = the lesser of x and y [element wise]
:data:`MAX`      a = the greater of x and y [element wise]
:data:`ATAN2`    a = arctan2(x, y)
:data:`HYPOT`    a = sqrt(x*x+y*y)
:data:`R2`       a = x*x + y*y
:data:`EQ`       a = 1 if x == y; a = 0 otherwise
:data:`NE`       a = 0 if x == y; a = 1 otherwise
:data:`LT`       a = 1 if x < y; a = 0 otherwise
:data:`LE`       a = 1 if x ≤ y; a = 0 otherwise
:data:`GT`       a = 1 if x > y; a = 0 otherwise
:data:`GE`       a = 1 if x ≥ y; a = 0 otherwise
:data:`NAN`      a = NaN if x == y; a = x otherwise
:data:`AND`      a = y if x is NaN; a = x otherwise
:data:`OR`       a = NaN if y is NaN; a = x otherwise
:data:`IAND`     a = bitwise AND of x and y
:data:`IOR`      a = bitwise OR of x and y
:data:`BTEST`    a = 1 if bit y of x is set; a = 0 otherwise
:data:`AVG`      a = 0.5*(x+y) [when x or y is NaN a returns the other value]
:data:`DXDY`     a[i] = (x[i+1]-x[i-1])/(y[i+1]-y[i-1]); a[1] = a[n] = NaN
:data:`EXCH`     exchange the top two stack elements
:data:`INRANGE`  a = 1 if x is between y and z (inclusive); a = 0 otherwise
:data:`BOXCAR`   filter x along dimension y with boxcar of length z
:data:`GAUSS`    filter x along dimension y with Gaussian of width sigma z
===============  ==============================================================

"""
# pylint: disable=too-many-lines

from abc import ABC, abstractmethod
from datetime import timedelta
from numbers import Integral
from typing import Any, Union, MutableSequence, Mapping, Tuple, cast

import numpy as np  # type: ignore
from astropy.convolution import (  # type: ignore
    Box1DKernel, Gaussian1DKernel, convolve)

from . import EPOCH
from .datetime64util import ymdhmsus

NumberOrArray = Union[int, float, np.generic, np.ndarray]


class StackUnderflowError(Exception):
    """Raised when the stack is too small for the operation.

    When this is raised the stack will exist in the state that it was before
    the operation was attempted.  Therefore, it is not necessary to repair the
    stack.

    """


class Token(ABC):
    """Base class of all RPN tokens.

    .. seealso::

        Class :class:`Literal`
            A literal numeric/array value.

        Class :class:`Variable`
            A variable to be looked up from the environment.

        Class :class:`Operator`
            Base class of operators that modify the stack.

    """

    @abstractmethod
    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Perform token's action on the given :paramref:`stack`.

        The actions currently supported are:

            * Place literal value on the stack.
            * Place variable on the stack from the :paramref:`environment`.
            * Perform operation on the stack.

        .. note::

            This must be overriden for all tokens.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.

        """


class Literal(Token):
    """Literal value token.

    Parameters
    ----------
    value
        Value of the literal.

    Raises
    ------
    ValueError
        If :paramref:`value` is not a number.

    """

    @property
    def value(self) -> Union[int, float, bool]:
        """Value of the literal."""
        return self._value

    def __init__(self, value: Union[int, float, bool]) -> None:
        if not isinstance(value, (int, float, bool)):
            raise TypeError("'value' must be an int, float, or bool")
        self._value: Union[int, float, bool] = value

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Place literal value on top of the given :paramref:`stack`.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to place the value on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        """
        stack.append(self.value)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Literal) and self.value == other.value

    def __ne__(self, other: Any) -> bool:
        return not isinstance(other, Literal) or self.value != other.value

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Literal):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, Literal):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Literal):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, Literal):
            return NotImplemented
        return self.value >= other.value

    def __repr__(self) -> str:
        return f'{self.__class__.__qualname__}({repr(self._value)})'

    def __str__(self) -> str:
        return str(self._value)


class Variable(Token):
    """Environment variable token.

    This is a place holder to lookup and place a number/array from an
    environment mapping onto the stack.

    Parameters
    ----------
    name
        Name of the variable, this is what will be used to lookup the
        variables value in the environment mapping.

    """

    @property
    def name(self) -> str:
        """Name of the variable, used to lookup value in the environment."""
        return self._name

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"'name' must be a string, got '{type(name)}'")
        if not name.isidentifier():
            raise ValueError(
                f"'name' must be a valid identifier, got '{name}'")
        self._name = name

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Get variable value from :paramref:`environment` and place on stack.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to place value on.
        environment
            The dictionary like object to lookup the variable's value from.

        Raises
        ------
        KeyError
            If the variable cannot be found in the given
            paramref:`environment`.

        """
        stack.append(environment[self.name])

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Variable) and other.name == self.name

    def __ne__(self, other: Any) -> bool:
        return not isinstance(other, Variable) or other.name != self.name

    def __repr__(self) -> str:
        return f'{self.__class__.__qualname__}({repr(self._name)})'

    def __str__(self) -> str:
        return str(self._name)


class Operator(Token, ABC):
    """Base class of all RPN operators.

    Parameters
    ----------
    name
         Name of the operator.

    """

    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return self._name


# NOTE: The operators in this file are in the same order as they are in the
# RADS user manual.


class _SUBType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Subtract one number/array from another.

        x y SUB a
            a = x - y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x - y)


class _ADDType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Add two numbers/arrays.

        x y ADD a
            a = x + y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x + y)


class _MULType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Multiply two numbers/arrays.

        x y MUL a
            a = x*y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x * y)


class _POPType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Remove top of stack.

        x POP
            remove last item from stack

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        _get_x(stack)


class _NEGType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Negate number/array.

        x NEG a
            a = −x

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(-x)


class _ABSType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        r"""Absolute value of number/array.

        x ABS a
            a = \|x\|

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.absolute(x))


class _INVType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Invert number/array.

        x INV a
            a = 1/x

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(1 / x)


class _SQRTType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compute square root of number/array.

        x SQRT a
            a = sqrt(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.sqrt(x))


class _SQRType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Square number/array.

        x SQR a
            a = x*x

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.square(x))


class _EXPType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Exponential of number/array.

        x EXP a
            a = exp(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.exp(x))


class _LOGType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Natural logarithm of number/array.

        x LOG a
            a = ln(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.log(x))


class _LOG10Type(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compute base 10 logarithm of number/array.

        x LOG10 a
            a = log10(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.log10(x))


class _SINType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Sine of number/array [in radians].

        x SIN a
            a = sin(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.sin(x))


class _COSType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Cosine of number/array [in radians].

        x COS a
            a = cos(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.cos(x))


class _TANType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Tangent of number/array [in radians].

        x TAN a
            a = tan(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.tan(x))


class _SINDType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Sine of number/array [in degrees].

        x SIND a
            a = sin(x) [x in degrees]

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.sin(np.deg2rad(x)))


class _COSDType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Cosine of number/array [in degrees].

        x COSD a
            a = cos(x) [x in degrees]

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.cos(np.deg2rad(x)))


class _TANDType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Tangend of number/array [in degrees].

        x TAND a
            a = tan(x) [x in degrees]

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.tan(np.deg2rad(x)))


class _SINHType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Hyperbolic sine of number/array.

        x SINH a
            a = sinh(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.sinh(x))


class _COSHType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Hyperbolic cosine of number/array.

        x COSH a
            a = cosh(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.cosh(x))


class _TANHType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Hyperbolic tangent of number/array.

        x TANH a
            a = tanh(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.tanh(x))


class _ASINType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Inverse sine of number/array [in radians].

        x ASIN a
            a = arcsin(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.arcsin(x))


class _ACOSType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Inverse cosine of number/array [in radians].

        x ACOS a
            a = arccos(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.arccos(x))


class _ATANType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Inverse tangent of number/array [in radians].

        x ATAN a
            a = arctan(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.arctan(x))


class _ASINDType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Inverse sine of number/array [in degrees].

        x ASIND a
            a = arcsin(x) [a in degrees]

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.rad2deg(np.arcsin(x)))


class _ACOSDType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Inverse cosine of number/array [in degrees].

        x ACOSD a
            a = arccos(x) [a in degrees]

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.rad2deg(np.arccos(x)))


class _ATANDType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Inverse tangent of number/array [in degrees].

        x ATAND a
            a = arctan(x) [a in degrees]

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.rad2deg(np.arctan(x)))


class _ASINHType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Inverse hyperbolic sine of number/array.

        x ASINH a
            a = arcsinh(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.arcsinh(x))


class _ACOSHType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Inverse hyperbolic cosine of number/array.

        x ACOSH a
            a = arccosh(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.arccosh(x))


class _ATANHType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Inverse hyperbolic tangent of number/array.

        x ATANH a
            a = arctanh(x)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.arctanh(x))


class _ISNANType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Determine if number/array is NaN.

        x ISNAN a
            a = 1 if x is NaN; a = 0 otherwise

        .. note::

            Instead of using 1 and 0 pyrads uses True and False which behave
            as 1 and 0 when treated as numbers.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.isnan(x))


class _ISANType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Determine if number/array is not NaN.

        x ISAN a
            a = 0 if x is NaN; a = 1 otherwise

        .. note::

            Instead of using 1 and 0 pyrads uses True and False which behave
            as 1 and 0 when treated as numbers.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.logical_not(np.isnan(x)))


class _RINTType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Round number/array to nearest integer.

        x RINT a
            a is nearest integer to x

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.round(x))


class _CEILType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Round number/array up to nearest integer.

        x CEIL a
            a is nearest integer greater or equal to x

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.ceil(x))


class _FLOORType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Round number/array down to nearest integer.

        x FLOOR a
            a is nearest integer less or equal to x

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.floor(x))


class _D2RType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Convert number/array from degrees to radians.

        x D2R a
            convert x from degrees to radians

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.deg2rad(x))


class _R2DType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Convert number/array from radians to degrees.

        x R2D a
            convert x from radian to degrees

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.rad2deg(x))


class _YMDHMSType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Convert number/array from seconds since RADS epoch to YYMMDDHHMMSS.

        x YMDHMS a
            convert seconds of 1985 to format YYMMDDHHMMSS

        .. note::

            The top of the stack should be in seconds since the RADS epoch
            which is currently 1985-01-01 00:00:00 UTC

        .. note::

            The RADS documentation says this format uses a 4 digit year, but
            RADS uses a 2 digit year so that is what is used here.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        if isinstance(x, np.ndarray):
            time = np.datetime64(EPOCH) + (x * 1e6).astype('timedelta64[us]')
            year, month, day, hour, minute, second, microsecond = \
                ymdhmsus(time)
            a = ((year % 100) * 1e10 + month * 1e8 + day * 1e6 +
                 hour * 1e4 + minute * 1e2 + second + microsecond * 1e-6)
        else:
            time = (EPOCH + timedelta(seconds=x))
            a = ((time.year % 100) * 1e10 + time.month * 1e8 + time.day * 1e6 +
                 time.hour * 1e4 + time.minute * 1e2 + time.second +
                 time.microsecond * 1e-6)
        stack.append(a)


class _SUMType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compute sum over number/array [ignoring NaNs].

        x SUM a
            a[i] = x[1] + ... + x[i] while skipping all NaN

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.nansum(x))


class _DIFType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compute difference over number/array.

        x DIF a
            a[i] = x[i]-x[i-1]; a[1] = NaN

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(np.diff(np.ravel(x), prepend=np.nan))


class _DUPType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Duplicate top of stack.

        x DUP a b
            duplicate the last item on the stack

        .. note::

            This is duplication by reference, no copy is made.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 1 element.

        """
        x = _get_x(stack)
        stack.append(x)
        stack.append(x)


class _DIVType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Divide one number/array from another.

        x y DIV a
            a = x/y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x / y)


class _POWType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Raise a number/array to the power of another number/array.

        x y POW a
            a = x**y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(np.power(x, y))


class _FMODType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Remainder of dividing one number/array by another.

        x y FMOD a
            a = x modulo y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(np.fmod(x, y))


class _MINType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Minimum of two numbers/arrays [element wise].

        x y MIN a
            a = the lesser of x and y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(np.minimum(x, y))


class _MAXType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Maximum of two numbers/arrays [element wise].

        x y MAX a
            a = the greater of x and y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(np.maximum(x, y))


class _ATAN2Type(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Inverse tangent of two numbers/arrays giving x and y.

        x y ATAN2 a
            a = arctan2(x, y)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(np.arctan2(x, y))


class _HYPOTType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Hypotenuse from numbers/arrays giving legs.

        x y HYPOT a
            a = sqrt(x*x+y*y)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(np.hypot(x, y))


class _R2Type(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Sum of squares of two numbers/arrays.

        x y R2 a
            a = x*x + y*y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x ** 2 + y ** 2)


class _EQType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compare two numbers/arrays for equality [element wise].

        x y EQ a
            a = 1 if x == y; a = 0 otherwise

        .. note::

            Instead of using 1 and 0 pyrads uses True and False which behave
            as 1 and 0 when treated as numbers.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x == y)


class _NEType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compare two numbers/arrays for inequality [element wise].

        x y NE a
            a = 0 if x == y; a = 1 otherwise

        .. note::

            Instead of using 1 and 0 pyrads uses True and False which behave
            as 1 and 0 when treated as numbers.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x != y)


class _LTType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compare two numbers/arrays with < [element wise].

        x y LT a
            a = 1 if x < y; a = 0 otherwise

        .. note::

            Instead of using 1 and 0 pyrads uses True and False which behave
            as 1 and 0 when treated as numbers.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x < y)


class _LEType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compare two numbers/arrays with <= [element wise].

        x y LE a
            a = 1 if x ≤ y; a = 0 otherwise

        .. note::

            Instead of using 1 and 0 pyrads uses True and False which behave
            as 1 and 0 when treated as numbers.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x <= y)


class _GTType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compare two numbers/arrays with > [element wise].

        x y GT a
            a = 1 if x > y; a = 0 otherwise

        .. note::

            Instead of using 1 and 0 pyrads uses True and False which behave
            as 1 and 0 when treated as numbers.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x > y)


class _GEType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compare two numbers/arrays with >= [element wise].

        x y GE a
            a = 1 if x ≥ y; a = 0 otherwise

        .. note::

            Instead of using 1 and 0 pyrads uses True and False which behave
            as 1 and 0 when treated as numbers.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(x >= y)


class _NANType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Replace number/array with NaN where it is equal to another.

        x y NAN a
            a = NaN if x == y; a = x otherwise

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.broadcast_arrays(x, y)
            # upgrade integers to doubles
            if np.issubdtype(x.dtype, np.integer):
                a = np.copy(x).astype('double')
            else:
                a = np.copy(x)
            a[x == y] = np.nan
            stack.append(a)
        else:
            stack.append(np.nan if x == y else x)


class _ANDType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Fallback to second number/array when first is NaN [element wise].

        x y AND a
            a = y if x is NaN; a = x otherwise

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.broadcast_arrays(x, y)
            a = np.copy(x)
            isnan = np.isnan(x)
            a[isnan] = y[isnan]
            stack.append(a)
        else:
            stack.append(y if np.isnan(x) else x)


class _ORType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Replace number/array with NaN where second is NaN.

        x y OR a
            a = NaN if y is NaN; a = x otherwise

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.broadcast_arrays(x, y)
            # upgrade integers to doubles
            if np.issubdtype(x.dtype, np.integer):
                a = np.copy(x).astype('double')
            else:
                a = np.copy(x)
            a[np.isnan(y)] = np.nan
            isnan = np.isnan(x)
            a[isnan] = y[isnan]
            stack.append(a)
        else:
            stack.append(np.nan if np.isnan(y) else x)


class _IANDType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Bitwise AND of two numbers/arrays [element wise].

        x y IAND a
            a = bitwise AND of x and y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(np.bitwise_and(x, y))


class _IORType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Bitwise OR of two numbers/arrays [element wise].

        x y IOR a
            a = bitwise OR of x and y

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(np.bitwise_or(x, y))


class _BTESTType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Test bit, given by second number/array, in first [element wise].

        x y BTEST a
            a = 1 if bit y of x is set; a = 0 otherwise

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        if not _is_integer(x):
            raise TypeError("'x' must be an integer type")
        if not _is_integer(y):
            raise TypeError("'y' must be an integer type")
        stack.append(np.bitwise_and(cast(int, x), 1 << cast(int, y)) != 0)


class _AVGType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Average of two numbers/arrays ignoring NaNs [element wise].

        x y AVG a
            a = 0.5*(x+y) [when x or y is NaN a returns the other value]

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.broadcast_arrays(x, y)
            stacked = np.stack((x, y))
            non_nan = np.sum(~np.isnan(stacked), axis=0)
            stack.append(np.nansum(stacked, axis=0) / non_nan)
        else:
            if np.isnan(x):
                stack.append(y)
            elif np.isnan(y):
                stack.append(x)
            else:
                stack.append((x + y) / 2)


class _DXDYType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Compute dx/dy from two numbers/arrays.

        x y DXDY a
            a[i] = (x[i+1]-x[i-1])/(y[i+1]-y[i-1]); a[1] = a[n] = NaN

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.broadcast_arrays(x, y)
            x = np.ravel(x)
            y = np.ravel(y)
            if np.size(x) < 3:
                a = np.empty(np.shape(x))
                a[:] = np.nan
            else:
                dxdy = (x[2:] - x[:-2]) / (y[2:] - y[:-2])
                a = np.concatenate(([np.nan], dxdy, [np.nan]))
        else:
            a = float('nan')
        stack.append(a)


class _EXCHType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Exchange top two elements of stack.

        x y EXCH a b
            exchange the last two items on the stack (NaNs have no influence)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 2 elements.

        """
        x, y = _get_xy(stack)
        stack.append(y)
        stack.append(x)


class _INRANGEType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Determine if number/array is between two numbers [element wise].

        x y z INRANGE a
            a = 1 if x is between y and z (inclusive)
            a = 0 otherwise (also in case of any NaN)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 3 elements.

        """
        x, y, z = _get_xyz(stack)
        if any(isinstance(v, np.ndarray) for v in [x, y, z]):
            x, y, z = np.broadcast_arrays(x, y, z)
            a = np.logical_and(y <= x, x <= z)
        else:
            a = y <= x <= z
        stack.append(a)


class _BOXCARType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Filter number/array with a boxcar filter along a given dimension.

        x y z BOXCAR a
            a = filter x along monotonic dimension y with boxcar of length z
            (NaNs are skipped)

        .. note::

            This may behave slightly differently than the official RADS
            software at boundaries and at NaN values.

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 3 elements.
        IndexError
            If x does not have dimension y.
        ValueError
            If y or z is not a scalar.

        """
        x, y, z = _get_xyz(stack)
        if np.size(x) == 1:
            a = x
        else:
            if np.size(y) != 1:
                raise ValueError("'y' of 'x y z BOXCAR a' must be a scalar")
            if np.size(z) != 1:
                raise ValueError("'z' of 'x y z BOXCAR a' must be a scalar")
            if len(np.shape(x)) <= y:
                raise IndexError(f"requested filter along dimension {y} but "
                                 f"'x' has only {len(np.shape(x))} dimensions")
            kernel = Box1DKernel(z)
            # split into slices along dimension y
            tmp = np.moveaxis(x, y, -1)
            slices = []
            for slice_ in tmp.reshape(-1, np.shape(x)[y]):
                # filter each slice
                slices.append(convolve(
                    slice_, kernel, boundary='extend', preserve_nan=True))
            # recombine slices
            a = np.moveaxis(np.array(slices).reshape(tmp.shape), -1, y)
        stack.append(a)


class _GAUSSType(Operator):

    def __call__(self, stack: MutableSequence[NumberOrArray],
                 environment: Mapping[str, NumberOrArray]) -> None:
        """Filter number/array with a gaussian filter along a given dimension.

        x y z GAUSS a
            a = filter x along monotonic dimension y with Gauss function with
            sigma z (NaNs are skipped)

        Parameters
        ----------
        stack
            The stack of numbers/arrays to operate on.
        environment
            The dictionary like object providing the immutable environment.
            Not used by this method.

        Raises
        ------
        IndexError
            If :paramref:`stack` does not have at least 3 elements.
        IndexError
            If x does not have dimension y.
        ValueError
            If y or z is not a scalar.

        """
        x, y, z = _get_xyz(stack)
        if np.size(x) == 1:
            a = x
        else:
            if np.size(y) != 1:
                raise ValueError("'y' of 'x y z GAUSS a' must be a scalar")
            if np.size(z) != 1:
                raise ValueError("'z' of 'x y z GAUSS a' must be a scalar")
            if len(np.shape(x)) <= y:
                raise IndexError(f"requested filter along dimension {y} but "
                                 f"'x' has only {len(np.shape(x))} dimensions")
            kernel = Gaussian1DKernel(z)
            # split into slices along dimension y
            tmp = np.moveaxis(x, y, -1)
            slices = []
            for slice_ in tmp.reshape(-1, np.shape(x)[y]):
                # filter each slice
                slices.append(convolve(
                    slice_, kernel, boundary='extend', preserve_nan=True))
            # recombine slices
            a = np.moveaxis(np.array(slices).reshape(tmp.shape), -1, y)
        stack.append(a)


PI = Literal(np.pi)
E = Literal(np.e)

# operators


SUB = _SUBType('SUB')
"""Subtract one number/array from another.

x y SUB a
    a = x - y

"""

ADD = _ADDType('ADD')
"""Add two numbers/arrays.

x y ADD a
    a = x + y

"""

MUL = _MULType('MUL')
"""Multiply two numbers/arrays.

x y MUL a
    a = x*y

"""

POP = _POPType('POP')
"""Remove top of stack.

x POP
    remove last item from stack

"""

NEG = _NEGType('NEG')
"""Negate number/array.

x NEG a
    a = −x

"""

ABS = _ABSType('ABS')
r"""Absolute value of number/array.

x ABS a
    a = \|x\|

"""

INV = _INVType('INV')
"""Invert number/array.

x INV a
    a = 1/x

"""

SQRT = _SQRTType('SQRT')
"""Compute square root of number/array.

x SQRT a
    a = sqrt(x)

"""

SQR = _SQRType('SQR')
"""Square number/array.

x SQR a
    a = x*x

"""

EXP = _EXPType('EXP')
"""Exponential of number/array.

x EXP a
    a = exp(x)

"""

LOG = _LOGType('LOG')
"""Natural logarithm of number/array.

x LOG a
    a = ln(x)

"""

LOG10 = _LOG10Type('LOG10')
"""Compute base 10 logarithm of number/array.

x LOG10 a
    a = log10(x)

"""

SIN = _SINType('SIN')
"""Sine of number/array [in radians].

x SIN a
    a = sin(x)

"""

COS = _COSType('COS')
"""Cosine of number/array [in radians].

x COS a
    a = cos(x)

"""

TAN = _TANType('TAN')
"""Tangent of number/array [in radians].

x TAN a
    a = tan(x)

"""

SIND = _SINDType('SIND')
"""Sine of number/array [in degrees].

x SIND a
    a = sin(x) [x in degrees]

"""

COSD = _COSDType('COSD')
"""Cosine of number/array [in degrees].

x COSD a
    a = cos(x) [x in degrees]

"""

TAND = _TANDType('TAND')
"""Tangend of number/array [in degrees].

x TAND a
    a = tan(x) [x in degrees]

"""

SINH = _SINHType('SINH')
"""Hyperbolic sine of number/array.

x SINH a
    a = sinh(x)

"""

COSH = _COSHType('COSH')
"""Hyperbolic cosine of number/array.

x COSH a
    a = cosh(x)

"""

TANH = _TANHType('TANH')
"""Hyperbolic tangent of number/array.

x TANH a
    a = tanh(x)

"""

ASIN = _ASINType('ASIN')
"""Inverse sine of number/array [in radians].

x ASIN a
    a = arcsin(x)

"""

ACOS = _ACOSType('ACOS')
"""Inverse cosine of number/array [in radians].

x ACOS a
    a = arccos(x)

"""

ATAN = _ATANType('ATAN')
"""Inverse tangent of number/array [in radians].

x ATAN a
    a = arctan(x)

"""

ASIND = _ASINDType('ASIND')
"""Inverse sine of number/array [in degrees].

x ASIND a
    a = arcsin(x) [a in degrees]

"""

ACOSD = _ACOSDType('ACOSD')
"""Inverse cosine of number/array [in degrees].

x ACOSD a
    a = arccos(x) [a in degrees]

"""

ATAND = _ATANDType('ATAND')
"""Inverse tangent of number/array [in degrees].

x ATAND a
    a = arctan(x) [a in degrees]

"""

ASINH = _ASINHType('ASINH')
"""Inverse hyperbolic sine of number/array.

x ASINH a
    a = arcsinh(x)

"""

ACOSH = _ACOSHType('ACOSH')
"""Inverse hyperbolic cosine of number/array.

x ACOSH a
    a = arccosh(x)

"""

ATANH = _ATANHType('ATANH')
"""Inverse hyperbolic tangent of number/array.

x ATANH a
    a = arctanh(x)

"""

ISNAN = _ISNANType('ISNAN')
"""Determine if number/array is NaN.

x ISNAN a
    a = 1 if x is NaN; a = 0 otherwise

"""

ISAN = _ISANType('ISAN')
"""Determine if number/array is not NaN.

x ISAN a
    a = 0 if x is NaN; a = 1 otherwise

"""

RINT = _RINTType('RINT')
"""Round number/array to nearest integer.

x RINT a
    a is nearest integer to x

"""

NINT = _RINTType('NINT')
"""Round number/array to nearest integer.

x NINT a
    a is nearest integer to x

"""

CEIL = _CEILType('CEIL')
"""Round number/array up to nearest integer.

x CEIL a
    a is nearest integer greater or equal to x

"""

CEILING = _CEILType('CEILING')
"""Round number/array up to nearest integer.

x CEILING a
    a is nearest integer greater or equal to x

"""

FLOOR = _FLOORType('FLOOR')
"""Round number/array down to nearest integer.

x FLOOR a
    a is nearest integer less or equal to x

"""

D2R = _D2RType('D2R')
"""Convert number/array from degrees to radians.

x D2R a
    convert x from degrees to radians

"""

R2D = _R2DType('R2D')
"""Convert number/array from radians to degrees.

x R2D a
    convert x from radian to degrees

"""

YMDHMS = _YMDHMSType('YMDHMS')
"""Convert number/array from seconds since RADS epoch to YYMMDDHHMMSS.

x YMDHMS a
    convert seconds of 1985 to format YYMMDDHHMMSS

"""

SUM = _SUMType('SUM')
"""Compute sum over number/array [ignoring NaNs].

x SUM a
    a[i] = x[1] + ... + x[i] while skipping all NaN

"""

DIF = _DIFType('DIF')
"""Compute difference over number/array.

x DIF a
    a[i] = x[i]-x[i-1]; a[1] = NaN

"""

DUP = _DUPType('DUP')
"""Duplicate top of stack.

x DUP a b
    duplicate the last item on the stack

"""

DIV = _DIVType('DIV')
"""Divide one number/array from another.

x y DIV a
    a = x/y

"""

POW = _POWType('POW')
"""Raise a number/array to the power of another number/array.

x y POW a
    a = x**y

"""

FMOD = _FMODType('FMOD')
"""Remainder of dividing one number/array by another.

x y FMOD a
    a = x modulo y

"""

MIN = _MINType('MIN')
"""Minimum of two numbers/arrays [element wise].

x y MIN a
    a = the lesser of x and y

"""

MAX = _MAXType('MAX')
"""Maximum of two numbers/arrays [element wise].

x y MAX a
    a = the greater of x and y

"""

ATAN2 = _ATAN2Type('ATAN2')
"""Inverse tangent of two numbers/arrays giving x and y.

x y ATAN2 a
    a = arctan2(x, y)

"""

HYPOT = _HYPOTType('HYPOT')
"""Hypotenuse from numbers/arrays giving legs.

x y HYPOT a
    a = sqrt(x*x+y*y)

"""

R2 = _R2Type('R2')
"""Sum of squares of two numbers/arrays.

x y R2 a
    a = x*x + y*y

"""

EQ = _EQType('EQ')
"""Compare two numbers/arrays for equality [element wise].

x y EQ a
    a = 1 if x == y; a = 0 otherwise

"""

NE = _NEType('NE')
"""Compare two numbers/arrays for inequality [element wise].

x y NE a
    a = 0 if x == y; a = 1 otherwise

"""

LT = _LTType('LT')
"""Compare two numbers/arrays with < [element wise].

x y LT a
    a = 1 if x < y; a = 0 otherwise

"""

LE = _LEType('LE')
"""Compare two numbers/arrays with <= [element wise].

x y LE a
    a = 1 if x ≤ y; a = 0 otherwise

"""

GT = _GTType('GT')
"""Compare two numbers/arrays with > [element wise].

x y GT a
    a = 1 if x > y; a = 0 otherwise

"""

GE = _GEType('GE')
"""Compare two numbers/arrays with >= [element wise].

x y GE a
    a = 1 if x ≥ y; a = 0 otherwise

"""

NAN = _NANType('NAN')
"""Replace number/array with NaN where it is equal to another.

x y NAN a
    a = NaN if x == y; a = x otherwise

"""

AND = _ANDType('AND')
"""Fallback to second number/array when first is NaN [element wise].

x y AND a
    a = y if x is NaN; a = x otherwise

"""

OR = _ORType('OR')
"""Replace number/array with NaN where second is NaN.

x y OR a
    a = NaN if y is NaN; a = x otherwise

"""

IAND = _IANDType('IAND')
"""Bitwise AND of two numbers/arrays [element wise].

x y IAND a
    a = bitwise AND of x and y

"""

IOR = _IORType('IOR')
"""Bitwise OR of two numbers/arrays [element wise].

x y IOR a
    a = bitwise OR of x and y

"""

BTEST = _BTESTType('BTEST')
"""Test bit, given by second number/array, in first [element wise].

x y BTEST a
    a = 1 if bit y of x is set; a = 0 otherwise

"""

AVG = _AVGType('AVG')
"""Average of two numbers/arrays ignoring NaNs [element wise].

x y AVG a
    a = 0.5*(x+y) [when x or y is NaN a returns the other value]

"""

DXDY = _DXDYType('DXDY')
"""Compute dx/dy from two numbers/arrays.

x y DXDY a
    a[i] = (x[i+1]-x[i-1])/(y[i+1]-y[i-1]); a[1] = a[n] = NaN

"""

EXCH = _EXCHType('EXCH')
"""Exchange top two elements of stack.

x y EXCH a b
    exchange the last two items on the stack (NaNs have no influence)

"""

INRANGE = _INRANGEType('INRANGE')
"""Determine if number/array is between two numbers [element wise].

x y z INRANGE a
    a = 1 if x is between y and z (inclusive)
    a = 0 otherwise (also in case of any NaN)

"""

BOXCAR = _BOXCARType('BOXCAR')
"""Filter number/array with a boxcar filter along a given dimension.

x y z BOXCAR a
    a = filter x along monotonic dimension y with boxcar of length z
    (NaNs are skipped)

"""

GAUSS = _GAUSSType('GAUSS')
"""Filter number/array with a gaussian filter along a given dimension.

x y z GAUSS a
    a = filter x along monotonic dimension y with Gauss function with
    sigma z (NaNs are skipped)

"""

_KEYWORDS = {
    'SUB': SUB,
    'ADD': ADD,
    'MUL': MUL,
    'PI': PI,
    'E': E,
    'POP': POP,
    'NEG': NEG,
    'ABS': ABS,
    'INV': INV,
    'SQRT': SQRT,
    'SQR': SQR,
    'EXP': EXP,
    'LOG': LOG,
    'LOG10': LOG10,
    'SIN': SIN,
    'COS': COS,
    'TAN': TAN,
    'SIND': SIND,
    'COSD': COSD,
    'TAND': TAND,
    'SINH': SINH,
    'COSH': COSH,
    'TANH': TANH,
    'ASIN': ASIN,
    'ACOS': ACOS,
    'ATAN': ATAN,
    'ASIND': ASIND,
    'ACOSD': ACOSD,
    'ATAND': ATAND,
    'ASINH': ASINH,
    'ACOSH': ACOSH,
    'ATANH': ATANH,
    'ISNAN': ISNAN,
    'ISAN': ISAN,
    'RINT': RINT,
    'NINT': NINT,
    'CEIL': CEIL,
    'CEILING': CEILING,
    'FLOOR': FLOOR,
    'D2R': D2R,
    'R2D': R2D,
    'YMDHMS': YMDHMS,
    'SUM': SUM,
    'DIF': DIF,
    'DUP': DUP,
    'DIV': DIV,
    'POW': POW,
    'FMOD': FMOD,
    'MIN': MIN,
    'MAX': MAX,
    'ATAN2': ATAN2,
    'HYPOT': HYPOT,
    'R2': R2,
    'EQ': EQ,
    'NE': NE,
    'LT': LT,
    'LE': LE,
    'GT': GT,
    'GE': GE,
    'NAN': NAN,
    'AND': AND,
    'OR': OR,
    'IAND': IAND,
    'IOR': IOR,
    'BTEST': BTEST,
    'AVG': AVG,
    'DXDY': DXDY,
    'EXCH': EXCH,
    'INRANGE': INRANGE,
    'BOXCAR': BOXCAR,
    'GAUSS': GAUSS
}


def _is_integer(x: NumberOrArray) -> bool:
    """Determine if number is an integer or array of integers.

    Parameters
    ----------
    x
        A number or array of numbers to check.

    Returns
    -------
    bool
        True if x is an integer or array of integers, otherwise False.

    """
    if isinstance(x, (np.ndarray, np.generic)):
        return issubclass(x.dtype.type, Integral)
    return isinstance(x, Integral)


def _get_x(stack: MutableSequence[NumberOrArray]) -> NumberOrArray:
    if not stack:
        raise StackUnderflowError(
            "attempted to get element from the 'stack' but the stack is empty")
    return stack.pop()


def _get_xy(stack: MutableSequence[NumberOrArray]) \
        -> Tuple[NumberOrArray, NumberOrArray]:
    if not stack:
        raise StackUnderflowError(
            "attempted to get 2 elements from the 'stack' but the stack "
            "is empty")
    if len(stack) < 2:
        raise StackUnderflowError(
            f"attempted to get 2 elements from the 'stack' but the stack has "
            f"only {len(stack)} elements")
    y = stack.pop()
    x = stack.pop()
    return x, y


def _get_xyz(stack: MutableSequence[NumberOrArray]) \
        -> Tuple[NumberOrArray, NumberOrArray, NumberOrArray]:
    if not stack:
        raise StackUnderflowError(
            "attempted to get 3 elements from the 'stack' but the stack "
            "is empty")
    if len(stack) < 3:
        raise StackUnderflowError(
            f"attempted to get 3 elements from the 'stack' but the stack has "
            f"only {len(stack)} elements")
    z = stack.pop()
    y = stack.pop()
    x = stack.pop()
    return x, y, z
