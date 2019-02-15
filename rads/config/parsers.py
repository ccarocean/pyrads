"""Parser combinators for reading XML files.

This module is heavily based on PEGTL_, a parser combinator library for C++.

.. _PEGTL: https://github.com/taocpp/PEGTL

"""
import typing
from typing import (Callable, List, Any, Tuple, NoReturn, Optional,
                    MutableSequence, cast)
from abc import abstractmethod, ABC

import yzal

from rads.xml.base import Element


class ParseFailure(Exception):
    """Base class of parse errors.

    Parameters
    ----------
    file
        Name of file that was being parsed when the error occurred.
    line
        Line number in the :paramref:`file` that was being parsed when the
        error occurred.
    message
        An optional message (instead of the default 'parsing failed')
        detailing why the parse failed.

    Attributes
    ----------
    file : str
        Filename where the error occurred.
    line : int
        Line number in the :attr:`file` where the error occurred.
    message : str
        The message provided when the error was constructed.

    """

    def __init__(self, file: Optional[str], line: Optional[int],
                 message: str = 'parsing failed') -> None:
        super().__init__()
        self.file = file
        self.line = line
        self.message = message

    def __str__(self) -> str:
        """Convert error to a string.

        The format is:


        Returns
        -------
        str
            Error as a string in the following format:

            .. code:: text

                <filename>:<line number>: <message>

        """
        # the reason this is not done in the constructor is to speed up
        # exception handling when reporting is not necessary.
        file = self.file if self.file else ''
        line = str(self.line) if self.line else ''
        return '{:s}:{:s}: {:s}'.format(file, line, self.message)


class GlobalParseFailure(ParseFailure):
    """A parse error that should fail parsing of the entire file."""


class LocalParseFailure(ParseFailure):
    """A parse error that signals that a given parser failed.

    Unlike :class:`GlobalParseFailure` is expected and simply signals that the
    parser did not match and another parser should be tried.
    """

    def to_global(self, message: Optional[str] = None) -> GlobalParseFailure:
        """Raise this local failure to a global failure.

        Parameters
        ----------
        message
            Optionally a new message.

        Returns
        -------
        GlobalParseFailure
            A global parse failure with the same file and line, and possibly
            message as this exception.

        """
        if message:
            return GlobalParseFailure(self.file, self.line, message)
        return GlobalParseFailure(self.file, self.line, self.message)


@yzal.lazy
def next_element(pos: Element) -> Element:
    """Get next element lazily.

    Parameters
    ----------
    pos : Element

    Returns
    -------
    Element
        Next sibling XML element.

    Raises
    ------
    LocalParserFailure
        If there is no next sibling element.

    """
    try:
        return pos.next()
    except StopIteration:
        raise LocalParseFailure(
            pos.file, pos.closing_line, 'No more elements.')


@yzal.lazy
def first_child(pos: Element) -> Element:
    """Get first child of element, lazily."""
    try:
        return pos.down()
    except StopIteration:
        raise LocalParseFailure(
            pos.file, pos.opening_line,
            '<{:s}> has no children'.format(pos.tag))


class Parser(ABC):
    """Base parser combinator."""

    @abstractmethod
    def __call__(self, position: Element) -> Tuple[Any, Element]:
        """Call the parser, trying to match at the given :paramref:`position`.

        If the match fails a :class:`LocalParseFailure` will be raised.  This
        call will only return if the parser matches at the given
        :paramref:`position`.

        Parameters
        ----------
        position
            An XML element that the parser should attempt to match at.

        Returns
        -------
        object
            The value result of the match, depends on the particular parser.
        Element
            The next XML element to match at.  This can be the same element as
            given in :paramref:`position` (a non consuming parser) or any later
            sibling element.

            Further, it will actually be a :class:`yzal.Thunk` and will
            therefore delay it's construction until it is needed.  Therefore,
            any :class:`LocalParseFailure` that may be generated by moving to a
            later element will occur when the returned element is used.

        Raises
        ------
        LocalParseFailure
            If the parser does not match at the given :paramref:`position`.
        GlobalParseFailure
            If the parser encounters an unrecoverable error.

        """

    def __add__(self, other: 'Parser') -> 'Sequence':
        """Combine two parsers, matching the first followed by the second.

        Multiple consecutive uses of '+' will result in a single
        :class:`Sequence` because the :class:`Sequence` class automatically
        flattens itself.

        Parameters
        ----------
        other
            The parser to match after this one.

        Returns
        -------
        Sequence
            A new parser that will match this parser followed by the
            :paramref:`other` parser (if the this parser matched).

        """
        return Sequence(self, other)

    def __or__(self, other: 'Parser') -> 'Alternate':
        """Combine two parsers, matching the first or the second.

        Multiple consecutive uses of '|' will result in a single
        :class:`Alternate` because the :class:`Alternate` class automatically
        flattens itself.

        Parameters
        ----------
        other
            The parser to match if this parser does not.

        Returns
        -------
        Alternate
            A new parser that will match either this parser or the
            :paramref:`other` parser (if the this parser did not match).

        """
        return Alternate(self, other)

    def __xor__(self, func: Callable[[Any], Any]) -> 'Apply':
        """Apply a function to the value result of this parser.

        Parameters
        ----------
        func
            The function to apply to the value result of matching this parser.

            .. note::

                This will not be ran until this parser is matched.

        Returns
        -------
        Apply
            A new parser that will match this parser and upon a successful
            match apply the given :paramref:`func` to the value result.

        """
        return Apply(self, func)

    def __invert__(self) -> 'Not':
        """Invert this parser.

        If this parser would match a given position, now it will not.  If it
        would not match now it will, but it will not consume any elements.

        Returns
        -------
        Not
            A new parser that will not match whenever this parser does, and
            will match whenever this parser does not.  However, it will not
            consume any elements.

        """
        return Not(self)

    def __lshift__(self, message: str) -> 'Must':
        """Require the parser to succeed.

        This will convert all :class:`LocalParseFailure` s to
        :class:`GlobalParseFailure` s.

        Parameters
        ----------
        message
            The message that will be raised if the parser does not match.

        Returns
        -------
        Must
            A new parser that will elevate any local parse failures to global
            failures and overwrite their message with :paramref:`message`.

        """
        return Must(self, message)


class Apply(Parser):
    """Apply a function to the value result of the parser.

    Parameters
    ----------
    parser
        The parser whose value result to apply the given :paramref:`func` to
        the value result of.
    func
        The function to apply
    """

    def __init__(self, parser: Parser, func: Callable[[Any], Any]) -> None:
        self._parser = parser
        self._function = func

    def __call__(self, position: Element) -> Tuple[Any, Element]:  # noqa: D102
        value, position = self._parser(position)
        return self._function(value), position


class Lazy(Parser):
    """Delay construction of parser until evaluated.

    .. note::

        This lazy behavior is useful when constructing recursive parsers in
        order to avoid infinite recursion.

    Parameters
    ----------
    parser_func
        A zero argument function that returns a parser when called.  This will
        be used to delay construction of the parser.
    """

    def __init__(self, parser_func: Callable[[], Parser]):
        self._parser_func = parser_func
        self._parser: Optional[Parser] = None

    def __call__(self, position: Element) -> Tuple[Any, Element]:  # noqa: D102
        if self._parser is None:
            self._parser = self._parser_func()
        return self._parser(position)


class Must(Parser):
    """Raise a LocalParseFailure to a GlobalParseFailure ending parsing.

    Parameters
    ----------
    parser
        Parser that must match.
    message
        New message to apply to the :class:`GlobalParserFailure` if the parser
        does not match.

    """

    def __init__(self, parser: Parser, message: Optional[str] = None) -> None:
        self._parser = parser
        self._message = message

    def __call__(self, position: Element) -> Tuple[Any, Element]:  # noqa: D102
        try:
            return self._parser(position)
        except LocalParseFailure as err:
            raise err.to_global(self._message)


class At(Parser):
    """Match a parser, consuming nothing.

    Parameters
    ----------
    parser
        Parser to match.

    """

    def __init__(self, parser: Parser) -> None:
        self._parser = parser

    def __call__(self, position: Element) -> Tuple[Any, Element]:  # noqa: D102
        value, _ = self._parser(position)
        return value, position


class Not(Parser):
    """Invert a parser match, consuming nothing.

    Parameters
    ----------
    parser
        Parser to invert the match of.

    """

    def __init__(self, parser: Parser) -> None:
        self._parser = parser

    def __call__(self, position: Element) -> \
            Tuple[None, Element]:  # noqa: D102
        try:
            self._parser(position)
        except LocalParseFailure:
            return None, position
        raise LocalParseFailure(position.file, position.opening_line)


class Repeat(Parser):
    """Match a parser zero or more times (greedily).

    Parameters
    ----------
    parser
        Parser to match repeatedly.

    """

    def __init__(self, parser: Parser) -> None:
        self._parser = parser

    def __call__(self, position: Element) \
            -> Tuple[List[Any], Element]:  # noqa: D102
        values = []
        try:
            while True:  # loop until parse failure
                value, position = self._parser(position)
                values.append(value)
        except LocalParseFailure:
            pass  # no more matches
        return values, position


class MultiParser(Parser, ABC):
    """Base class of multiple parser combinators.

    .. note::

        Consecutive MultiParser's of the same subtype are automatically
        flattened.

    Parameters
    ----------
    subtype
        The type of the child parser (the type of parser that subclasses this).
    *parsers
        Parsers to store in the multi parser.

    """

    def __init__(self, subtype: type, *parsers: Parser) -> None:
        assert issubclass(subtype, MultiParser)
        self._subtype = subtype
        self._parsers: List[Parser] = []
        for parser in parsers:
            if isinstance(parser, subtype):
                self._parsers.extend(cast(MultiParser, parser)._parsers)
            else:
                self._parsers.append(parser)

    def _append(self, other: Parser) -> None:
        if isinstance(other, self._subtype):
            self._parsers.extend(cast(MultiParser, other)._parsers)
        else:
            self._parsers.append(other)


class Sequence(MultiParser):
    """Chain parsers together, succeeding only if all succeed in order.

    .. note::

        Consecutive Sequence's are automatically flattened.

    Parameters
    ----------
    *parsers
        Parsers to match in sequence.

    """

    def __init__(self, *parsers: Parser) -> None:
        super().__init__(Sequence, *parsers)

    def __call__(self, position: Element) \
            -> Tuple[List[Any], Element]:  # noqa: D102
        values = []
        for parser in self._parsers:
            value, position = parser(position)
            values.append(value)
        return values, position

    def __add__(self, other: Parser) -> 'Sequence':
        """Combine this sequence and a parser, returning a new sequence.

        .. note::

            If the :paramref:`other` parser is a :class:`Sequence` then the
            parsers in the :paramref:`other` :class:`Sequence` will be
            unwrapped and appended individually.

        Parameters
        ----------
        other
            The parser to combine with this sequence to form the new sequence.

        Returns
        -------
        Sequence
            A new sequence which matches this sequence followed by the given
            parser (if the sequence matched).

        """
        return Sequence(*self._parsers, other)

    def __iadd__(self, other: Parser) -> 'Sequence':
        """Combine this sequence with the given parser (in place).

        .. note::

            If the :paramref:`other` parser is a :class:`Sequence` then the
            parsers in the :paramref:`other` :class:`Sequence` will be
            unwrapped and appended to this sequence individually.

        Parameters
        ----------
        other
            The parser to combine with (append to) this sequence.

        Returns
        -------
        Sequence
            This sequence parser.

        """
        self._append(other)
        return self


class Alternate(MultiParser):
    """Match any one of the parsers, stops on first match.

    .. note::

        Consecutive Alternate's are automatically flattened.

    Parameters
    ----------
    *parsers
        Pool of parsers to find a match in.

    """

    def __init__(self, *parsers: Parser) -> None:
        super().__init__(Alternate, *parsers)

    def __call__(self, position: Element) -> Tuple[Any, Element]:  # noqa: D102
        for parser in self._parsers:
            try:
                return parser(position)
            except LocalParseFailure:
                pass
        raise LocalParseFailure(position.file, position.opening_line)

    def __or__(self, other: Parser) -> 'Alternate':
        """Combine this alternate and a parser, returning a new alternate.

        .. note::

            If the :paramref:`other` parser is a :class:`Alternate` then the
            parsers in the :paramref:`other` :class:`Alternate` will be
            unwrapped and added individually.

        Parameters
        ----------
        other
            The parser to combine with this alternate to form the new
            alternate.

        Returns
        -------
        Sequence
            A new alternate which matches any parser from this alternate or the
            given parser (if no parser of this alternate matches).

        """
        return Alternate(*self._parsers, other)

    def __ior__(self, other: Parser) -> 'Alternate':
        """Combine this alternate with the given parser (in place).

        .. note::

            If the :paramref:`other` parser is a :class:`Alternate` then the
            parsers in the :paramref:`other` :class:`Alternate` will be
            unwrapped and added to this alternate individually.

        Parameters
        ----------
        other
            The parser to combine with (append to) this alternate.

        Returns
        -------
        Sequence
            This alternate parser.

        """
        self._append(other)
        return self


class Success(Parser):
    """Parser that always succeeds, consuming nothing."""

    def __call__(self, position: Element) \
            -> Tuple[None, Element]:  # noqa: D102
        return None, position


class Failure(Parser):
    """Parser that always fails, consuming nothing."""

    def __call__(self, position: Element) -> NoReturn:  # noqa: D102
        raise LocalParseFailure(position.file, position.opening_line)


class Start(Parser):
    """Match start of an element, consuming nothing."""

    def __call__(self, position: Element) \
            -> Tuple[None, Element]:  # noqa: D102
        try:
            prev = position.prev()
            raise LocalParseFailure(
                prev.file, prev.opening_line, 'Expected start of element.')
        except StopIteration:
            return None, position


class End(Parser):
    """Match end of an element, consuming nothing."""

    def __call__(self, position: Element) \
            -> Tuple[None, Element]:  # noqa: D102
        try:
            # Element is really a Thunk[Element] so we need to cast it and
            # force it's evaluation to determine if a parse error is thrown,
            # indicating the end of the current level of elements.
            yzal.strict(cast(yzal.Thunk[Element], position))
        except LocalParseFailure:
            return None, position
        raise LocalParseFailure(
            position.file, position.closing_line, 'Expected end of element.')


class AnyElement(Parser):
    """Parser that matches any element."""

    def __call__(self, position: Element) \
            -> Tuple[Element, Element]:  # noqa: D102
        return yzal.strict(position), next_element(position)


class Tag(Parser):
    """Match an element by it's tag name.

    Parameters
    ----------
    name
        Tag name to match.

    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, position: Element) \
            -> Tuple[Element, Element]:  # noqa: D102
        if position.tag == self._name:
            return yzal.strict(position), next_element(position)
        raise LocalParseFailure(position.file, position.opening_line)


def lazy(parser_func: Callable[[], Parser]) -> Parser:
    """Delays construction of parser until evaluated.

    Parameters
    ----------
    parser_func
        A zero argument function that returns a parser when called.  This will
        be used to delay construction of the parser.

    Returns
    -------
    Parser
        A new parser that is equivalent to the parser returned by
        :paramref:`parser_func`.

    """
    return Lazy(parser_func)


def at(parser: Parser) -> Parser:
    """Succeeds if and only if the given parser succeeds, consumes nothing.

    Parameters
    ----------
    parser : Parser
        The parser that must succeed.

    Returns
    -------
    Parser
        A new parser that succeeds if and only if :paramref:`parser` succeeds,
        but does not consume input.

    """
    return At(parser)


def not_at(parser: Parser) -> Parser:
    """Succeeds if and only if the given parser fails, consumes nothing.

    Parameters
    ----------
    parser : Parser
        The parser that must fail.

    Returns
    -------
    Parser
        A new parser that succeeds if and only if :paramref:`parser` fails,
        but does not consume input.

    """
    return ~At(parser)


def opt(parser: Parser) -> Parser:
    """Parser that always succeeds, regardless of the given parser.

    Parameters
    ----------
    parser : Parser
        An optional parser that can succeed or fail.

    Returns
    -------
    Parser
        A new parser that optionally matches :paramref:`parser`.  If
        :paramref:`parser` succeeds this parser will be transparent, as if
        :paramref:`parser` was called directly. If :paramref:`parser` fails
        this :func:`opt` returns None as the result and does not consume
        anything.

    """
    return parser | Success()


def plus(parser: Parser) -> Parser:
    """Match the given parser as much as possible, must match at least once.

    Parameters
    ----------
    parser : Parser
        Parser to match one or more times (greedily).

    Returns
    -------
    Parser
        A new parser that matches :paramref:`parser` one or more times.
        Failing if no matches are made.

    """
    return parser + Repeat(parser)


def seq(*parsers: Parser) -> Parser:
    """Match sequence of parsers in order, succeeding iff all succeed.

    Parameters
    ----------
    *parsers :
        One or more parsers to match in order.

    Returns
    -------
    Parser
        A new parser that matches all the given :paramref:`parser`'s in order,
        failing if any one of the :paramref:`parser`'s fails.

    """
    return Sequence(*parsers)


def sor(*parsers: Parser) -> Parser:
    """Match the first of the given parsers, failing if all fail.

    Parameters
    ----------
    *parsers :
        One or more parsers to match.  The first parser that succeeds will
        take the place of this parser.  If all fail then this parser will
        also fail.

    Returns
    -------
    Parser
        A new parser that matches the first :paramref:`parser` that succeeds or
        fails if all :paramref:`parser`'s fail.

    """
    return Alternate(*parsers)


def star(parser: Parser) -> Parser:
    """Match the given parser as much as possible, can match zero times.

    Parameters
    ----------
    parser : Parser
        Parser to match zero or more times (greedily).

    Returns
    -------
    Parser
        A new parser that matches :paramref:`parser` one or more times.
        Failing if no matches are made.

    """
    return Repeat(parser)


def must(parser: Parser) -> Parser:
    """Raise a local parse failure to a global parse failure.

    Local parse failures (:class:`LocalParseFailure`) are typically caught
    by :class:`Alternate` or other such parsers that allow some parser's to
    fail.  In particular, local failures are an expected part of parser
    combinators and simply signal that a particular parser could not parse
    the given elements.  A global parse failure
    (:class:`GlobalParseFailure`) should only be caught at the top level and
    signals that the entire parse is a failure.

    Parameters
    ----------
    parser : Parser
        A parser that must match, else the entire parse is failed.

    Returns
    -------
        A parser that must succeed, if it fails a
        :class:`GlobalParserFailure` is raised.

    """
    return Must(parser)


def rep(parser: Parser, times: int) -> Parser:
    """Match the given parser a given number of times.

    Fails if the parser does not succeed the given number of times.

    Parameters
    ----------
    parser : Parser
        The parser to match :paramref:`times`.
    times : int
        Number of times the :paramref:`parser` must succeed.

    Returns
    -------
    Parser
        A parser that succeeds only if the given :paramref:`parser` matches the
        given number of times :paramref:`times`.

    """
    return Sequence(*([parser] * times))


def until(parser: Parser) -> Parser:
    """Match all elements until the given :paramref:`parser` matches.

    Does not consume the elements that the given :paramref:`parser` matches.

    Parameters
    ----------
    parser
        The parser to end matching with.

    Returns
    -------
    Parser
        A parser that will consume all elements until the given
        :paramref:`parser` matches.  It will not consume the elements that the
        given :paramref:`parser` matched.
    """
    def process(elements: typing.Sequence[Element]) -> Element:
        return elements[-1]

    def process2(elements: Tuple[MutableSequence[Element], Element]) \
            -> typing.Sequence[Element]:
        start_elements, last_element = elements
        start_elements.append(last_element)
        return start_elements

    return star(not_at(parser) + not_at(end()) + any() ^ process
                ) + at(parser) ^ process2


def failure() -> Parser:
    """Parser that always fails.

    Returns
    -------
    Parser
        A new parser that always fails, consuming nothing.

    """
    return Failure()


def success() -> Parser:
    """Parser that always succeeds.

    Returns
    -------
    Parser
        A new parser that always succeeds, consuming nothing.

    """
    return Success()


def start() -> Parser:
    """Match the beginning of an element.

    Returns
    -------
        A new parser that matches the beginning of an element, consuming
        nothing.

    """
    return Start()


def end() -> Parser:
    """Match the end of an element.

    Returns
    -------
        A new parser that matches the end of an element, consuming nothing.

    """
    return End()


def any() -> Parser:  # pylint: disable=redefined-builtin
    """Match any element.

    Returns
    -------
    Parser
        A new parser that matches any single element.

    """
    return AnyElement()


def tag(name: str) -> Parser:
    """Match an element by tag name.

    Parameters
    ----------
    name
        Tag name to match.

    Returns
    -------
    Parser
        Parser matching the given :paramref:`tag` name.

    """
    return Tag(name)
