"""Parsers for text within a tag.

The parsers in this file should all take a string and return a value or take
one or more parser functions and return a new parsing function.

If a parser deals with generic XML then it should be rads.config.xml_parsers
and if it returns a rads.config.xml_parsers.Parser but is not generic then it
belongs in rads.config.grammar.
"""

from datetime import datetime
from numbers import Real
from typing import Any, Callable, List, Sequence, TypeVar

import numpy as np  # type: ignore

from .tree import Compress, Cycles, Range, ReferencePass, Repeat, Unit

__all__ = ['list_of', 'range_of', 'types', 'compress', 'cycles', 'nop',
           'ref_pass', 'repeat', 'time', 'unit']

T = TypeVar('T')


# COMBINATORS
#
# This section contains parser combinators that can be used to glue multiple
# text parsers together and return a new text parser.


def list_of(parser: Callable[[str], T]) -> Callable[[str], List[T]]:
    def _parser(string: str) -> List[T]:
        return [parser(s) for s in string.split()]

    return _parser


def range_of(parser: Callable[[str], Real]) -> Callable[[str], Range]:
    def _parser(string: str) -> Range:
        minmax = [parser(s) for s in string.split()]
        if len(minmax) == 0:
            raise ValueError(
                'ranges require exactly 2 values, but none were given')
        elif len(minmax) == 1:
            raise ValueError(
                'ranges require exactly 2 values, but only 1 was given')
        elif len(minmax) > 2:
            raise ValueError(
                'ranges require exactly 2 values, '
                f'but {len(minmax)} were given')
        return Range(*minmax)

    return _parser


def types(parsers: Sequence[Callable[[str], Any]]) \
        -> Callable[[str], Any]:
    parser_types = ', '.join(parser.__qualname__ for parser in parsers)

    def _parser(string: str) -> Any:
        for parser in parsers:
            try:
                return parser(string)
            except (TypeError, ValueError):
                pass

        raise TypeError(f"cannot convert '{string}' to any of the following "
                        f"types: {parser_types}")

    return _parser


# PARSERS
#
# This section contains the actual text parsers that take a string and return
# a value.


def compress(string: str) -> Compress:
    parts = string.split()
    if len(parts) > 3:
        raise TypeError(
            "too many values given, expected only 'type', "
            "'scale_factor', and 'add_offset'")
    try:
        return Compress(
            *(f(s) for f, s in zip((_rads_type, float, float), parts)))
    except TypeError:
        raise TypeError("'missing 'type'")


def cycles(string: str) -> Cycles:
    try:
        return Cycles(*(int(s) for s in string.split()))
    except TypeError:
        num_values = len(string.split())
        if num_values == 0:
            raise TypeError("missing 'first' cycle")
        if num_values == 1:
            raise TypeError("missing 'last' cycle")
        raise TypeError(
            "too many cycles given, expected only 'first' and 'last'")


def nop(string: str) -> str:
    return string


def ref_pass(string: str) -> ReferencePass:
    parts = string.split()
    if len(parts) > 5:
        raise TypeError("too many values given, expected only 'time', "
                        "'longitude', 'cycle number', 'pass number', and "
                        "optionally 'absolute orbit number'")
    try:
        funcs: Sequence[Callable[[str], Any]] = (time, float, int, int, int)
        return ReferencePass(*(f(s) for f, s in zip(funcs, parts)))
    except TypeError:
        if not parts:
            raise TypeError("missing 'time' of reference pass")
        if len(parts) == 1:
            raise TypeError("missing 'longitude' of reference pass")
        if len(parts) == 2:
            raise TypeError("missing 'cycle number' of reference pass")
        # len(parts) == 3
        raise TypeError("missing 'pass number' of reference pass")
        # absolute orbit number is defaulted in ReferencePass


def repeat(string: str) -> Repeat:
    parts = string.split()
    if len(parts) > 3:
        raise TypeError(
            "too many values given, expected only 'days', "
            "'passes', and 'unknown'")
    try:
        return Repeat(*(f(s) for f, s in zip((float, int, float), parts)))
    except TypeError:
        if parts:
            raise TypeError("missing length of repeat cycle in 'passes'")
        raise TypeError("missing length of repeat cycle in 'days'")


def time(string: str) -> datetime:
    try:
        return datetime.strptime(string, '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        try:
            return datetime.strptime(string, '%Y-%m-%dT%H:%M')
        except ValueError:
            try:
                return datetime.strptime(string, '%Y-%m-%dT%H')
            except ValueError:
                try:
                    return datetime.strptime(string, '%Y-%m-%dT')
                except ValueError:
                    try:
                        return datetime.strptime(string, '%Y-%m-%d')
                    except ValueError:
                        # required to avoid 'unconverted data' message from
                        # strptime
                        raise ValueError(
                            "time data '{:s}' does not match format "
                            "'%Y-%m-%dT%H:%M:%S'".format(string))


def unit(string: str) -> Unit:
    try:
        return Unit(string)
    except ValueError:
        # TODO: Need better handling for dB and yymmddhhmmss units.
        return string.strip()


def _rads_type(string: str) -> type:
    switch = {
        'int1': np.int8,
        'int2': np.int16,
        'int4': np.int32,
        'real': np.float32,
        'dble': np.float64
    }
    try:
        return switch[string.lower()]
    except KeyError:
        raise TypeError(f"invalid type string '{string}'")
