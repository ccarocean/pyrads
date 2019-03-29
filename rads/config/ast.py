"""Abstract Syntax Tree elements for RADS configuration file."""

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from difflib import get_close_matches
from typing import (Any, Optional, Container, Sequence, Union, MutableMapping,
                    Mapping, Collection, Iterator, Callable, cast)

from dataclass_builder import UndefinedFieldError, MissingFieldError, MISSING

from ._builders import PhaseBuilder
from .._utility import xor

ActionType = Callable[[Any, str, Any], None]


def _get_mapping(environment: Any, attr: str):
    if (not hasattr(environment, attr) or
            getattr(environment, attr) == MISSING):
        setattr(environment, attr, {})
    return getattr(environment, attr)


def _suggest_field(dataclass: Any, attempt: str) -> Optional[str]:
    matches = get_close_matches(
        attempt, [f.name for f in fields(dataclass)], 1, 0.1)
    if matches:
        return matches[0]
    return None


def _has(o: Any, attr: str) -> bool:
    if isinstance(o, Mapping):
        return attr in o
    return hasattr(o, attr)


def _get(o: Any, attr: str) -> Any:
    if isinstance(o, Mapping):
        return o[attr]
    return getattr(o, attr)


def _set(o: Any, attr: str, value: Any) -> None:
    if isinstance(o, Mapping):
        o[attr] = value
    else:
        setattr(o, attr, value)


def replace(environment: Any, attr: str, value: Any) -> None:
    """Set value in the given environment.

    Sets :paramref:`attr` to the given :paramref:`value` in the given
    :paramref:`environment`.  If the :paramref:`attr` has already been set
    it will be overwritten.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of :paramref:`attr` in.
    attr
        Name of the value to change in the :paramref:`environment`.
    value
        New value to use for the action.

    """
    _set(environment, attr, value)


def keep(environment: Any, attr: str, value: Any) -> None:
    """Set value in the given environment.

    Sets :paramref:`attr` to the given :paramref:`value` in the given
    :paramref:`environment`.  If the :paramref:`attr` has already ben set it
    will be left intact and the new value discarded.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of :paramref:`attr` in.
    attr
        Name of the value to change in the :paramref:`environment`.
    value
        New value to use for the action.

    """
    if not _has(environment, attr) or _get(environment, attr) == MISSING:
        _set(environment, attr, value)


def append(
        environment: MutableMapping[str, Any], attr: str, value: Any) -> None:
    """Set key/value pair in the given environment.

    Sets :paramref:`attr` to the given :paramref:`value` in the given
    :paramref:`environment`.  If the :paramref:`attr` has already ben set the
    new value will be appended, placing the original value in a list if
    necessary.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of :paramref:`attr` in.
    attr
        Name of the value to change in the :paramref:`environment`.
    value
        New value to use for the action.

    """
    if not _has(environment, attr) or _get(environment, attr) == MISSING:
        _set(environment, attr, value)
    else:
        if not _has(_get(environment, attr), 'append'):
            _set(environment, attr, [_get(environment, attr)])
        try:
            _get(environment, attr).extend(value)
        except TypeError:
            _get(environment, attr).append(value)


class Condition(ABC):
    """Base class of AST node conditionals."""

    def __repr__(self) -> str:
        return f'{self.__class__.__qualname__}()'

    @abstractmethod
    def test(self, selectors: Mapping[str, Any]) -> bool:
        """Evaluate condition to determine match.

        This is used to determine whether or not a block should be executed.

        Parameters
        ----------
        selectors
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.

        Returns
        -------
        bool
            True if the condition is a match, otherwise False.

        """


class TrueCondition(Condition):
    """Condition that is always true."""

    def test(self, selectors: Mapping[str, Any]) -> bool:  # noqa: D102
        return True


class FalseCondition(Condition):
    """Condition that is always false."""

    def test(self, selectors: Mapping[str, Any]) -> bool:  # noqa: D102
        return False


class SatelliteCondition(Condition):
    """Condition that matches based on the satellite `id`.

    This makes use of the `id` key/value in the :paramref:`selectors` mapping
    if it exists.

    Parameters
    ----------
    satellites : Container[str]
        Set of satellite ID's to match.
    invert : bool
        Set to True to invert the match.
    """

    satellites: Container[str]
    invert: bool

    def __init__(self, satellites: Container[str], invert: bool = False):
        self.satellites = satellites
        self.invert = invert

    def __repr__(self):
        prefix = f'{self.__class__.__qualname__}({repr(self.satellites)}'
        if self.invert:
            return prefix + f', invert={repr(self.invert)})'
        return prefix + ')'

    def test(self, selectors: Mapping[str, Any]) -> bool:  # noqa: D102
        try:
            return xor(selectors['id'] in self.satellites, self.invert)
        except KeyError:
            return False


@dataclass(frozen=True)
class Source:
    line: int
    file: Optional[str] = None


class ASTEvaluationError(Exception):
    message: str
    line: Optional[int] = None
    file: Optional[str] = None

    def __init__(self, message: str = 'evaluation failed',
                 source: Optional[Source] = None):
        if source:
            self.line = source.line
            self.file = source.file
        file = self.file if self.file else ''
        line = self.line if self.line else ''
        super().__init__(f'{file}:{line}: {message}')


class Statement(ABC):
    """Base class of Abstract Syntax Tree nodes."""

    source: Optional[Source]

    def __init__(self, *, source: Optional[Source] = None) -> None:
        self.source = source

    def __repr__(self) -> str:
        return f'{self.__class__.__qualname__}()'

    @abstractmethod
    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate statement, adding to the environment dictionary.

        Parameters
        ----------
        environment
            Environment dictionary.  Additions from this statement will be
            added to this mapping.
        selectors
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.

        """


class NullStatement(Statement):

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        pass


class CompoundStatement(Sequence[Statement], Statement):
    """Ordered collection of statements.

    Parameters
    ----------
    *statements
        Statements to be stored in the :class:`CompoundStatement`.

    """

    def __init__(self, *statements: Statement,
                 source: Optional[Source] = None) -> None:
        super().__init__(source=source)
        self._statements = statements

    @typing.overload
    def __getitem__(self, key: int) -> Statement:
        pass

    @typing.overload
    def __getitem__(self, key: slice) -> 'CompoundStatement':
        pass

    def __getitem__(self, key: Union[int, slice]) -> \
            Union[Statement, 'CompoundStatement']:
        if isinstance(key, slice):
            return CompoundStatement(*self._statements[key])
        return self._statements[key]

    def __len__(self) -> int:
        return len(self._statements)

    def __repr__(self) -> str:
        return (f'{self.__class__.__qualname__}'
                f'({", ".join(repr(s) for s in self._statements)})')

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate compound statement, adding to the environment dictionary.

        This will evaluate each statement in this compound statement in order.

        Parameters
        ----------
        environment
            Environment dictionary.  Additions from this statement will be
            added to this mapping.
        selectors
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.

        """
        for statement in self:
            statement.eval(environment, selectors)


class If(Statement):
    """If/else statement AST node.

    Attributes
    ----------
    condition
        Condition that must be True for the :attr:`true_statement` branch
        to be executed.  Otherwise, the :attr:`false_statement` is
        executed.
    true_statement:
        Statement to be executed if :attr:`condition` evaluates to True.
    false_statement:
        Optional statement to be executed if :attr:`condition` evaluates
        to False.

    """

    condition: Condition
    true_statement: Statement
    false_statement: Optional[Statement]

    def __init__(self, condition: Condition, true_statement: Statement,
                 false_statement: Optional[Statement] = None,
                 *, source: Optional[Source] = None) -> None:
        super().__init__(source=source)
        self.condition = condition
        self.true_statement = true_statement
        self.false_statement = false_statement

    def __repr__(self) -> str:
        prefix = (f'{self.__class__.__qualname__}'
                  f'({repr(self.condition)}, {repr(self.true_statement)}')
        if self.false_statement is None:
            return prefix + ')'
        return prefix + f', {repr(self.false_statement)})'

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate if/else statement, adding to the environment dictionary.

        If the satellite matches the condition then the :attr:`true_statement`
        will be evaluated.  Otherwise the :attr:`false_statement` will be
        evaluated (if it exists).

        Parameters
        ----------
        environment
            Environment dictionary.  Additions from this statement will be
            added to this mapping.
        selectors
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.

        """
        if self.condition.test(selectors):
            self.true_statement.eval(environment, selectors)
        elif self.false_statement is not None:
            self.false_statement.eval(environment, selectors)


class Assignment(Statement):
    """Assignment statement (value to variable) AST node.

    Attributes
    ----------
    name
        Name of variable.
    value
        Value to assign to variable.
    condition
        Condition that must be true for this assignment to be executed.  This
        defaults to the :class:`TrueCondition`.
    action
        Action to take if this variable has already been set.

    """

    name: str
    value: Any
    condition: Condition = TrueCondition()
    # TODO: Remove Optional when https://github.com/python/mypy/issues/708 is
    #  fixed.
    action: Optional[ActionType] = staticmethod(replace)

    def __init__(self, name: Any, value: str,
                 condition: Optional[Condition] = None,
                 action: Optional[ActionType] = None,
                 *, source: Optional[Source] = None) -> None:
        super().__init__(source=source)
        self.name = name
        self.value = value
        if condition is not None:
            self.condition = condition
        if action is not None:
            self.action = action

    def __repr__(self):
        prefix = (f'{self.__class__.__qualname__}'
                  f'({repr(self.name)}, {repr(self.value)}')
        if not isinstance(self.condition, TrueCondition):
            prefix += f', {repr(self.condition)}'
        if self.action == replace:
            return prefix + ')'
        return prefix + f', {self.action.__qualname__})'

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate assignment, adding to the environment dictionary.

        Parameters
        ----------
        environment
            Environment dictionary.  The addition from this statement will be
            added to this mapping.
        selectors
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.

        """
        if self.condition.test(selectors):
            action = cast(ActionType, self.action)
            try:
                action(environment, self.name, self.value)
            except UndefinedFieldError as err:
                message = f"invalid assignment to '{err.field}'"
                suggested = _suggest_field(err.dataclass, err.field)
                if suggested:
                    message += f", did you mean '{suggested}'"
                raise ASTEvaluationError(message, source=self.source)


class Alias(Statement):
    alias: str
    variables: Sequence[str]
    condition: Condition = TrueCondition()
    # TODO: Remove Optional when https://github.com/python/mypy/issues/708 is
    #  fixed.
    action: Optional[ActionType] = staticmethod(replace)

    def __init__(self, alias: Any, variables: Sequence[str],
                 condition: Optional[Condition] = None,
                 action: Optional[ActionType] = None,
                 *, source: Optional[Source] = None) -> None:
        super().__init__(source=source)
        self.alias = alias
        self.variables = variables
        if condition is not None:
            self.condition = condition
        if action is not None:
            self.action = action

    def __repr__(self):
        prefix = (f'{self.__class__.__qualname__}'
                  f'({repr(self.alias)}, {repr(self.variables)}')
        if not isinstance(self.condition, TrueCondition):
            prefix += f', {repr(self.condition)}'
        if self.action == replace:
            return prefix + ')'
        return prefix + f', {self.action.__qualname__})'

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        if self.condition.test(selectors):
            action = cast(ActionType, self.action)
            aliases = _get_mapping(environment, 'aliases')
            action(aliases, self.alias, self.variables)


class SatelliteID(Statement):
    id: str
    id3: str
    names: Collection[str]

    def __init__(self, id: str, id3: str,
                 names: Optional[Collection[str]] = None,
                 *, source: Optional[Source] = None) -> None:
        super().__init__(source=source)
        self.id = id
        self.id3 = id3
        if names is None:
            self.names = set()
        else:
            self.names = names

    def __repr__(self) -> str:
        prefix = (f'{self.__class__.__qualname__}'
                  f'({repr(self.id)}, {repr(self.id3)}')
        if self.names:
            return prefix + f', {self.names})'
        return prefix + ')'

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        if selectors == self.id:
            setattr(environment, 'id', self.id)
            setattr(environment, 'id3', self.id3)
            setattr(environment, 'names', self.names)


class Satellites(Mapping[str, Statement], Statement):

    def __init__(self, *satellites: SatelliteID,
                 source: Optional[Source] = None) -> None:
        super().__init__(source=source)
        self._satellites: MutableMapping[str, SatelliteID] = {}
        for satellite in satellites:
            self._satellites[satellite.id] = satellite

    def __getitem__(self, key: str) -> SatelliteID:
        return self._satellites.__getitem__(key)

    def __iter__(self) -> Iterator[str]:
        return self._satellites.__iter__()

    def __len__(self) -> int:
        return self._satellites.__len__()

    def __repr__(self) -> str:
        return (f'{self.__class__.__qualname__}('
                f'{", ".join(repr(v) for v in self._satellites.values())})')

    def eval(self, environment: Any,
             selectors: Mapping[str, Any]) -> None:
        try:
            if selectors['id'] in self:
                self[selectors['id']].eval(environment, selectors['id'])
        except KeyError:
            pass


class Phase(Statement):
    name: str
    inner_statement: Statement
    condition: Condition = TrueCondition()

    def __init__(self, name: str, inner_statement,
                 condition: Optional[Condition] = None,
                 *, source: Optional[Source] = None):
        super().__init__(source=source)
        self.name = name
        self.inner_statement = inner_statement
        if condition is not None:
            self.condition = condition

    def __repr__(self):
        prefix = (f'{self.__class__.__qualname__}'
                  f'({repr(self.name)}, {repr(self.inner_statement)}')
        if isinstance(self.condition, TrueCondition):
            return prefix + ')'
        return prefix + f', {self.condition})'

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        # initialize and/or retrieve phases
        phases = _get_mapping(environment, 'phases')

        if self.name in phases:
            # update current phase structure
            self.inner_statement.eval(phases[self.name], selectors)
        else:
            # build in initial phase structure
            builder = PhaseBuilder(id=self.name)
            self.inner_statement.eval(builder, selectors)
            try:
                phases[self.name] = builder.build()
            except MissingFieldError as err:
                raise ASTEvaluationError(
                    f"missing required attribute '{err.field.name}'",
                    source=self.source)
