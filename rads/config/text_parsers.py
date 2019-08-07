"""Parsers for text within a tag.

The parsers in this file should all take a string and a mapping of attributes
for the tag and return a value or take one or more parser functions and return
a new parsing function.

If a parser deals with generic XML then it should be rads.config.xml_parsers
and if it returns a rads.config.xml_parsers.Parser but is not generic then it
belongs in rads.config.grammar.
"""

import re
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np  # type: ignore
import regex  # type: ignore
from cf_units import Unit  # type: ignore

from .._utility import fortran_float
from ..rpn import Expression
from .tree import (
    Compress,
    Constant,
    Cycles,
    Flags,
    Grid,
    MultiBitFlag,
    N,
    NetCDFAttribute,
    NetCDFVariable,
    Range,
    ReferencePass,
    Repeat,
    SingleBitFlag,
    SurfaceType,
)

__all__ = [
    "TerminalTextParseError",
    "TextParseError",
    "lift",
    "list_of",
    "one_of",
    "range_of",
    "compress",
    "cycles",
    "data",
    "nop",
    "ref_pass",
    "repeat",
    "time",
    "unit",
]

_T = TypeVar("_T")


class TerminalTextParseError(Exception):
    """Raised to terminate text parsing with an error.

    This error is not allowed to be handled by a text parser.  It indicates
    that no recovery is possible.
    """


# inherits from TerminalTextParseError because any except clause that catches
# terminal errors should also catch non-terminal errors.
class TextParseError(TerminalTextParseError):
    """Raised to indicate that a text parsing error has occured.

    Unlike :class:`TerminalTextParseError` this one is allowed to be handled
    by a text parser.
    """


# COMBINATORS
#
# This section contains parser combinators that can be used to glue multiple
# text parsers together and return a new text parser.

if TYPE_CHECKING:
    from typing_extensions import Protocol

    class _SupportsFromString(Protocol):
        def __init__(self, string: str) -> None:
            ...


def lift(
    string_parser: "Union[Callable[[str], _T], Type[_SupportsFromString]]",
    *,
    terminal: bool = False,
) -> Callable[[str, Mapping[str, str]], _T]:
    r"""Lift a simple string parser to a text parser that accepts attributes.

    This is very similar to lifting a plain function into a monad.

    :param string_parser:
        A simple parser that takes a string and returns a value.  This can
        also by a type that can be constructed from a string.
    :param terminal:
        Set to True to use :class:`TerminalTextParseError`\ s instead of
        :class:`TextParseError`\ s.

    :return:
        The given `string_parser` with an added argument to accept and ignore
        the attributes for the text tag.

    :raises TextParseError:
        The resulting parser throws this if the given `string_parser` throws a
        TypeError, ValueError, or KeyError and `terminal` was False (the
        default).
    :raises TerminalTextParseError:
        The resulting parser throws this if the given `string_parser` throws a
        TypeError, ValueError, or KeyError and `terminal` was True.
    """

    def _parser(string: str, _: Mapping[str, str]) -> _T:
        try:
            return string_parser(string)  # type: ignore
        except (TypeError, ValueError, KeyError) as err:
            if terminal:
                raise TerminalTextParseError(str(err)) from err
            raise TextParseError(str(err)) from err

    _parser._lifted = string_parser.__qualname__  # type: ignore
    return _parser


def list_of(
    parser: Callable[[str, Mapping[str, str]], _T],
    *,
    sep: Optional[str] = None,
    terminal: bool = False,
) -> Callable[[str, Mapping[str, str]], List[_T]]:
    """Convert parser into a parser of lists.

    :param parser:
        Original parser.
    :param sep:
        Item delimiter.  Default is to separate by one or more spaces.
    :param terminal:
        If set to True it promotes any :class:`TextParseError` s raised by the
        given `parser` to a :class:`TerminalTextParseError`.

    :return:
        The new parser of delimited lists.
    """

    def _parser(string: str, attr: Mapping[str, str]) -> List[_T]:
        return [parser(s, attr) for s in string.split(sep=sep)]

    def _terminal_parser(string: str, attr: Mapping[str, str]) -> List[_T]:
        try:
            return _parser(string, attr)
        except TextParseError as err:
            raise TerminalTextParseError(str(err)) from err

    return _terminal_parser if terminal else _parser


def range_of(
    parser: Callable[[str, Mapping[str, str]], N], *, terminal: bool = False
) -> Callable[[str, Mapping[str, str]], Range[N]]:
    """Create a range parser from a given parser for each range element.

    The resulting parser will parse space separated lists of length 2 and use
    the given `parser` for both elements.

    :param parser:
        Parser to use for the min and max values.
    :param terminal:
        Set to True to use :class:`TerminalTextParseError` s instead of
        :class:`TextParseError` s.  Also promotes any :class:`TextParseError` s
        raised by the given `parser` to a :class:`TerminalTextParseError`.

    :return:
        New range parser.

    :raises TextParseError:
        Resulting parser raises this if given a string that does not contain
        two space separated elements and `terminal` was False (the default).
    :raises TerminalTextParseError:
        Resulting parser raises this if given a string that does not contain
        two space separated elements and `terminal` was True.
    """

    def _parser(string: str, attr: Mapping[str, str]) -> Range[N]:
        minmax = [parser(s, attr) for s in string.split()]
        if not minmax:
            raise TextParseError("ranges require exactly 2 values, but none were given")
        if len(minmax) == 1:
            raise TextParseError(
                "ranges require exactly 2 values, but only 1 was given"
            )
        if len(minmax) > 2:
            raise TextParseError(
                "ranges require exactly 2 values, " f"but {len(minmax)} were given"
            )
        return Range(*minmax)

    def _terminal_parser(string: str, attr: Mapping[str, str]) -> Range[N]:
        try:
            return _parser(string, attr)
        except TextParseError as err:
            raise TerminalTextParseError(str(err)) from err

    return _terminal_parser if terminal else _parser


def one_of(
    parsers: Iterable[Callable[[str, Mapping[str, str]], Any]],
    *,
    terminal: bool = False,
) -> Callable[[str, Mapping[str, str]], Any]:
    """Convert parsers into a parser that tries each one in sequence.

    .. note::

        Each parser will be tried in sequence.  The next parser will be tried
        if :class:`TextParseError` is raised.

    :param parsers:
        A sequence of parsers the new parser should try in order.
    :param terminal:
        Set to True to use :class:`TerminalTextParseError` s instead of
        :class:`TextParseError` s.

    :return:
        The new parser which tries each of the given `parsers` in order until
        one succeeds.

    :raises TextParseError:
        Resulting parser raises this if given a string that cannot be parsed by
        any of the given `parsers` and `terminal` was False (the default).
    :raises TerminalTextParseError:
        Resulting parser raises this if given a string that cannot be parsed by
        any of the given `parsers` and `terminal` was True.
    """

    def _parser(string: str, attr: Mapping[str, str]) -> Any:
        for parser in parsers:
            try:
                return parser(string, attr)
            except TextParseError:
                pass

        # build list of parser types
        parser_types = []
        for parser in parsers:
            try:
                parser_types.append(parser._lifted)  # type: ignore
            except AttributeError:
                parser_types.append(parser.__qualname__)

        err_str = (
            f"cannot convert '{string}' to any of the following "
            f"types: {', '.join(parser_types)}"
        )
        if terminal:
            raise TerminalTextParseError(err_str)
        raise TextParseError(err_str)

    return _parser


# PARSERS
#
# This section contains the actual text parsers that take a string and return
# a value.


def compress(string: str, _: Mapping[str, str]) -> Compress:
    """Parse a string into a :class:`rads.config.tree.Compress` object.

    :param string:
        String to parse into a :class:`rads.config.tree.Compress` object.  It
        should be in the following form:

            <type:type> [scale_factor:float] [add_offset:float]

        where only the first value is required and should be one of the
        following 4 character RADS data type values:

            * `int1` - maps to numpy.int8
            * `int2` - maps to numpy.int16
            * `int4` - maps to numpy.int32
            * `real` - maps to numpy.float32
            * `dble` - maps to numpy.float64

        The attribute mapping of the tag the string came from.  Not currently
        used by this function.
    :param _:
        Mapping of tag attributes.  Not used by this function.

    :return:
        A new :class:`rads.config.tree.Compress` object created from the parsed
        string.

    :raises TextParseError:
        If the <type> is not in the given `string` or if too many values are in
        the `string`.  Also, if one of the values cannot be converted.
    """
    parts = string.split()
    try:
        funcs: Iterable[Callable[[str], Any]] = (
            _rads_type,
            fortran_float,
            fortran_float,
            lambda x: x,
        )
        return Compress(*(f(s) for f, s in zip(funcs, parts)))
    except (KeyError, ValueError) as err:
        raise TextParseError(str(err)) from err
    except TypeError:
        if len(parts) > 3:
            raise TextParseError(
                "too many values given, expected only 'type', "
                "'scale_factor', and 'add_offset'"
            )
        raise TextParseError("'missing 'type'")


def cycles(string: str, _: Mapping[str, str]) -> Cycles:
    """Parse a string into a :class:`rads.config.tree.Cycles` object.

    :param string:
        String to parse into a :class:`rads.config.tree.Cycles` object.  It
        should be in the following form:

        .. code-block:: text

            <first cycle in phase> <last cycle in phase>
    :param _:
        Mapping of tag attributes.  Not used by this function.

    :return:
        A new :class:`rads.config.tree.Cycles` object created from the parsed
        string.

    :raises TextParseError:
        If the wrong number of values are given in the `string` or one of the
        values is not parsable to an integer.
    """
    try:
        return Cycles(*(int(s) for s in string.split()))
    except ValueError as err:
        raise TextParseError(str(err)) from err
    except TypeError:
        num_values = len(string.split())
        if num_values == 0:
            raise TextParseError("missing 'first' cycle")
        if num_values == 1:
            raise TextParseError("missing 'last' cycle")
        raise TextParseError("too many cycles given, expected only 'first' and 'last'")


def data(string: str, attr: Mapping[str, str]) -> Any:
    """Parse a string into one of the data objects list below.

        * :class:`rads.config.tree.Constant`
        * :class:`rads.rpn.Expression`
        * :class:`rads.config.tree.Flags`
        * :class:`rads.config.tree.Grid`
        * :class:`rads.config.tree.NetCDFAttribute`
        * :class:`rads.config.tree.NetCDFVariable`

    The parsing is done based on both the given `string` and the 'source' value
    in `attr` if it exists.

    .. note::

        This is a terminal parser, it will either succeed or raise
        :class:`TerminalTextParseError`.

    :param string:
        String to parse into a data object.
    :param attr:
        Mapping of tag attributes.  This parser can make use of the following
        key/value pairs if they exist:

            * "source" - explicitly specify the data source, this can be any
              of the following:

              * "flags"
              * "constant"
              * "grid"
              * "grid_l"
              * "grid_s"
              * "grid_c"
              * "grid_q"
              * "grid_n"
              * "nc"
              * "netcdf"
              * "math"
              * "branch" - used by some sources to specify an alternate directory

            * "x" - used by the grid sources to set the x dimension
            * "y" - used by the grid sources to set the y dimension

    :return:
        A new data object representing the given `string`.

    :raises TerminalTextParseError:
        If for any reason the given `string` and `attr` cannot be parsed into
        one of the data objects listed above.
    """
    attr_ = {k: v.strip() for k, v in attr.items()}
    return one_of((_flags, _constant, _grid, _netcdf, _math, _invalid_data))(
        string.strip(), attr_
    )


def nop(string: str, _: Mapping[str, str]) -> str:
    """No operation parser, returns given string unchanged.

    This exists primarily as a default for when no parser is given as the use
    of `lift(str)` is recommended when the parsed value is supposed to be a
    string.

    :param string:
        String to return.
    :param _:
        Mapping of tag attributes.  Not used by this function.

    :return:
        The given `string`.
    """
    return string


def ref_pass(string: str, _: Mapping[str, str]) -> ReferencePass:
    """Parse a string into a :class:`rads.config.tree.ReferencePass` object.

    :param string:
        String to parse into a:class:`rads.config.tree.ReferencePass` object.
        It should be in the following form:

        .. code-block:: text

            <yyyy-mm-ddTHH:MM:SS> <lon> <cycle> <pass> [absolute orbit number]

        where the last element is optional and defaults to 1.  The date can
        also be missing seconds, minutes, and hours.
    :param _:
        Mapping of tag attributes.  Not used by this function.

    :return:
        A new :class:`rads.config.tree.ReferencePass` object created from the
        parsed string.

    :raises TextParseError:
        If the wrong number of values are given in the `string` or one of the
        values is not parsable.
    """
    parts = string.split()
    try:
        funcs: Sequence[Callable[[str], Any]] = (
            _time,
            float,
            int,
            int,
            int,
            lambda x: x,
        )
        return ReferencePass(*(f(s) for f, s in zip(funcs, parts)))
    except ValueError as err:
        raise TextParseError(str(err)) from err
    except TypeError:
        if not parts:
            raise TextParseError("missing 'time' of reference pass")
        if len(parts) == 1:
            raise TextParseError(
                "missing equator crossing 'longitude' of reference pass"
            )
        if len(parts) == 2:
            raise TextParseError("missing 'cycle number' of reference pass")
        if len(parts) == 3:
            raise TextParseError("missing 'pass number' of reference pass")
        # absolute orbit number is defaulted in ReferencePass
        raise TextParseError(
            "too many values given, expected only 'time', "
            "'longitude', 'cycle number', 'pass number', and "
            "optionally 'absolute orbit number'"
        )


def repeat(string: str, _: Mapping[str, str]) -> Repeat:
    """Parse a string into a :class:`rads.config.tree.Repeat` object.

    :param string:
        String to parse into a :class:`rads.config.tree.Repeat` object. It
        should be in the following form:

        .. code-block:: text

            <days:float> <passes:int> [longitude drift per cycle:float]

        where the last value is optional.
    :param _:
        Mapping of tag attributes.  Not used by this function.

    :return:
        A new :class:`rads.config.tree.Repeat` object created from the parsed
        string.

    :raises TextParseError:
        If the wrong number of values are given in the `string` or
        one of the values is not parsable.
    """
    parts = string.split()
    try:
        funcs: Sequence[Callable[[str], Any]] = (float, int, float, lambda x: x)
        return Repeat(*(f(s) for f, s in zip(funcs, parts)))
    except ValueError as err:
        raise TextParseError(str(err)) from err
    except TypeError:
        if not parts:
            raise TextParseError("missing length of repeat cycle in 'days'")
        if len(parts) == 1:
            raise TextParseError("missing length of repeat cycle in 'passes'")
        raise TextParseError(
            "too many values given, expected only 'days', "
            "'passes', and 'longitude_drift'"
        )


def time(string: str, _: Mapping[str, str]) -> datetime:
    """Parse a string into a :class:`datetime.datetime` object.

    :param string:
        String to parse into a :class:`datetime.datetime` object. It should be
        in one of the following forms:

            * yyyy-mm-ddTHH:MM:SS
            * yyyy-mm-ddTHH:MM
            * yyyy-mm-ddTHH
            * yyyy-mm-ddT
            * yyyy-mm-dd
    :param _:
        Mapping of tag attributes.  Not used by this function.

    :return:
        A new :class:`datetime.datetime` object created from the parsed string.

    :raises TextParseError:
        If the date/time `string` cannot be parsed.
    """
    try:
        return _time(string)
    except ValueError as err:
        raise TextParseError(str(err)) from err


def unit(string: str, _: Mapping[str, str]) -> Unit:
    """Parse a string into a :class:`cf_units.Unit` object.

    .. _cf_units: https://github.com/SciTools/cf-units

    .. _`issue 30`: https://github.com/SciTools/cf-units/issues/30

    :param string:
        String to parse into a :class:`cf_units.Unit` object.  See the
        cf_units_ package for supported units.  If given 'dB' or 'decibel' a
        no_unit object will be returned and if given 'yymmddhhmmss' an unknown
        unit will be returned.
    :param _:
        Mapping of tag attributes.  Not used by this function.

    :return:
        A new :class:`cf_units.Unit` object created from the parsed string.
        In the case of 'dB' and 'decibel' this will be a no_unit and in the
        case of 'yymmddhhmmss' it will be 'unknown'.  See `issue 30`_.

    :raises ValueError:
        If the given `string` does not represent a valid unit.
    """
    try:
        return Unit(string)
    except ValueError:
        string = string.strip()
        # TODO: Remove this hack when
        #  https://github.com/SciTools/cf-units/issues/30 is fixed.
        if string in ("dB", "decibel"):
            return Unit("no unit")
        if string == "yymmddhhmmss":
            return Unit("unknown")
        raise TextParseError(f"failed to parse unit '{string}'")


def _constant(string: str, attr: Mapping[str, str]) -> Constant:
    try:
        if "source" not in attr or attr["source"] == "constant":
            return Constant(one_of((lift(int), lift(float)))(string, attr))
    except TextParseError:
        if "source" in attr:  # constant, so hard fail
            raise TerminalTextParseError(f"invalid numerical constant '{string}'")
        raise  # pass on parsing
    raise TextParseError(f"'{string}' does not represent a constant")


# regex is a loose match in order to allow more specific error message to take
# precedence
_FLAGS_RE = re.compile(r"([\+\-\d\.]+)(?:\s+([\+\-\d\.]+))?")


def _flags(string: str, attr: Mapping[str, str]) -> Flags:
    if not attr.get("source") == "flags":
        raise TextParseError(f"'{string}' does not represent a flags source")
    if string == "surface_type":
        return SurfaceType()
    match = _FLAGS_RE.fullmatch(string)
    if match is None:
        raise TerminalTextParseError(f"'{string}' does not represent a flags source")
    bit, length = match.groups()
    length = "1" if length is None else length
    # import pdb; pdb.set_trace()
    try:
        bit_ = int(bit)
        length_ = int(length)
    except ValueError as err:
        raise TerminalTextParseError(err) from err
    if bit_ < 0:
        raise TerminalTextParseError(f"bit index '{bit_}' cannot be negative")
    if length_ == 1:
        return SingleBitFlag(bit=bit_)
    if length_ >= 2:
        return MultiBitFlag(bit=bit_, length=length_)
    raise TerminalTextParseError(
        "multi bit flags must have length 2 or greater, " f"length is '{length_}'"
    )


def _grid(string: str, attr: Mapping[str, str]) -> Grid:
    method = None
    try:
        if attr["source"] in ("grid", "grid_l"):
            method = "linear"
        elif attr["source"] in ("grid_s", "grid_c"):
            method = "spline"
        elif attr["source"] in ("grid_q", "grid_n"):
            method = "nearest"
    except KeyError:
        if re.fullmatch(r"\S[\S ]*\.nc", string):
            method = "linear"
    if method:
        return Grid(
            file=string, x=attr.get("x", "lon"), y=attr.get("y", "lat"), method=method
        )
    # pass on parsing
    raise TextParseError(f"'{string}' does not represent a grid")


def _invalid_data(string: str, _: Mapping[str, str]) -> NoReturn:
    raise TerminalTextParseError(f"invalid <data> tag value '{string}'")


_MATH_RE = re.compile(r"((\S+\s+)+)\S+")


def _math(string: str, attr: Mapping[str, str]) -> Expression:
    # NOTE: This parser is more aggressive about terminal errors, this is
    # because the math source is the most likely to have errors and is never
    # explicitly indicated in practice so the math error messages must be
    # terminal most of the time to aid in configuration file debugging.
    #
    # NOTE: This uses Expression instead of CompleteExpression.  This is
    # because checking for a complete expression must be delayed until AST
    # evaluation due to append, delete, and merge.
    try:
        if (
            "source" not in attr
            and _MATH_RE.fullmatch(string)
            or attr.get("source") == "math"
        ):
            return Expression(string)
    except ValueError as err:
        raise TerminalTextParseError(str(err)) from err
    raise TextParseError(f"'{string}' does not represent a math expression")


_NETCDF_RE = regex.compile(
    r"(?|([a-zA-Z][a-zA-Z0-9_]*)?:([a-zA-Z][a-zA-Z0-9_]+)|" r"([a-zA-Z][a-zA-Z0-9_]+))"
)


def _netcdf(
    string: str, attr: Mapping[str, str]
) -> Union[NetCDFVariable, NetCDFAttribute]:
    if attr.get("source", "nc") not in ("nc", "netcdf"):
        raise TextParseError(f"'{string}' does not represent a flags source")
    match = _NETCDF_RE.fullmatch(string)
    if match is None:
        str_ = f"'{string}' does not represent a netcdf attribute"
        if attr.get("source") in ("nc", "netcdf"):
            raise TerminalTextParseError(str_)
        raise TextParseError(str_)
    variable, attribute = match.groups()
    variable = variable if variable else None
    branch = attr.get("branch", None)
    if attribute:
        return NetCDFAttribute(name=attribute, variable=variable, branch=branch)
    return NetCDFVariable(name=variable, branch=branch)


def _rads_type(string: str) -> type:
    switch: Dict[str, type] = {
        "int1": np.int8,
        "int2": np.int16,
        "int4": np.int32,
        "real": np.float32,
        "dble": np.float64,
    }
    try:
        return switch[string.lower()]
    except KeyError:
        raise ValueError(f"invalid RADS type string '{string}'")


def _time(string: str) -> datetime:
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H",
        "%Y-%m-%dT",
        "%Y-%m-%d",
    ]
    for format_ in formats:
        try:
            return datetime.strptime(string, format_)
        except ValueError:
            pass
    # required to avoid 'unconverted data' message from strptime
    raise ValueError(f"time data '{string}' does not match format '%Y-%m-%dT%H:%M:%S'")
