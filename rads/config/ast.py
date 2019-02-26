"""Abstract Syntax Tree elements for RADS configuration file."""

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Any, Optional, Container, Sequence, Union, MutableMapping,
                    Mapping, Collection, Iterator, Callable, cast)

from .._utility import xor

ActionType = Callable[[MutableMapping[str, Any], str, Any], None]


def replace_action(
        environment: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set key/value pair in the given environment.

    Sets :paramref:`key`/:paramref:`value` pair in the given
    :paramref:`environment`.  If the :paramref:`key` already exists it will
    be overwritten.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of key in.
    key
        Name of the value to change in the :paramref:`environment`.
    value
        New value to use for the action.

    """
    environment[key] = value


def noreplace_action(
        environment: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set key/value pair in the given environment.

    Sets :paramref:`key`/:paramref:`value` pair in the given
    :paramref:`environment`.  If the :paramref:`key` already exists then it
    will be left intact and the new value discarded.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of key in.
    key
        Name of the value to change in the :paramref:`environment`.
    value
        New value to use for the action.

    """
    if key not in environment:
        environment[key] = value


def append_action(
        environment: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set key/value pair in the given environment.

    Sets :paramref:`key`/:paramref:`value` pair in the given
    :paramref:`environment`.  If the :paramref:`key` already exists the new
    value will be appended, placing the original value in a list if
    necessary.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of key in.
    key
        Name of the value to change in the :paramref:`environment`.
    value
        New value to use for the action.

    """
    if key in environment:
        if not hasattr(environment[key], 'append'):
            environment[key] = [environment[key]]
        try:
            environment[key].extend(value)
        except TypeError:
            environment[key].append(value)
    else:
        environment[key] = [value]


class Condition(ABC):
    """Base class of AST node conditionals."""

    @abstractmethod
    def eval(self, selectors: Mapping[str, Any]) -> bool:
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

    def eval(self, selectors: Mapping[str, Any]) -> bool:  # noqa: D102
        return True

    def __repr__(self) -> str:
        """Get text representation 'TrueCondition()'."""
        return 'TrueCondition()'


class FalseCondition(Condition):
    """Condition that is always false."""

    def eval(self, selectors: Mapping[str, Any]) -> bool:  # noqa: D102
        return False

    def __repr__(self) -> str:
        """Get text representation 'FalseCondition()'."""
        return 'FalseCondition()'


@dataclass(frozen=True)
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
    invert: bool = False

    def eval(self, selectors: Mapping[str, Any]) -> bool:  # noqa: D102
        try:
            return xor(selectors['id'] in self.satellites, self.invert)
        except KeyError:
            return False


class Statement(ABC):
    """Base class of Abstract Syntax Tree nodes."""

    @abstractmethod
    def eval(self, environment: MutableMapping[str, Any],
             selectors: Mapping[str, Any]) -> None:
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

    def eval(self, environment: MutableMapping[str, Any],
             selectors: Mapping[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return 'NullStatement()'


class CompoundStatement(Sequence[Statement], Statement):
    """Ordered collection of statements.

    Parameters
    ----------
    *statements
        Statements to be stored in the :class:`CompoundStatement`.

    """

    def __init__(self, *statements: Statement) -> None:
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
        return 'CompoundStatement({:s})'.format(
            ', '.join(repr(s) for s in self._statements))

    def eval(self, environment: MutableMapping[str, Any],
             selectors: Mapping[str, Any]) -> None:
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


@dataclass(frozen=True)
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
    false_statement: Optional[Statement] = None

    def eval(self, environment: MutableMapping[str, Any],
             selectors: Mapping[str, Any]) -> None:
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
        if self.condition.eval(selectors):
            self.true_statement.eval(environment, selectors)
        elif self.false_statement is not None:
            self.false_statement.eval(environment, selectors)


@dataclass(frozen=True)
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
    action: Optional[ActionType] = replace_action

    def eval(self, environment: MutableMapping[str, Any],
             selectors: Mapping[str, Any]) -> None:
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
        if self.condition.eval(selectors):
            action = cast(ActionType, self.action)
            # TODO: Remove pylint override if it ever registers correctly.
            # pylint: disable=too-many-function-args
            action(environment, self.name, self.value)


@dataclass(frozen=True)
class VariableAlias(Statement):
    alias: str
    variables: Sequence[str]
    condition: Condition = TrueCondition()
    # TODO: Remove Optional when https://github.com/python/mypy/issues/708 is
    #  fixed.
    action: Optional[ActionType] = replace_action

    def eval(self, environment: MutableMapping[str, Any],
             selectors: Mapping[str, Any]) -> None:
        if self.condition.eval(selectors):
            if 'variable_aliases' not in environment:
                environment['variable_aliases'] = dict()
            action = cast(ActionType, self.action)
            # TODO: Remove pylint override if it ever registers correctly.
            # pylint: disable=too-many-function-args
            action(environment['variable_aliases'], self.alias, self.variables)


@dataclass(frozen=True)
class SatelliteID(Statement):
    id: str
    id3: str
    names: Collection[str] = field(default_factory=set)

    def eval(self, environment: MutableMapping[str, Any],
             selectors: Mapping[str, Any]) -> None:
        if selectors == self.id:
            environment['id'] = self.id
            environment['id3'] = self.id3
            environment['names'] = self.names


class Satellites(Mapping[str, Statement], Statement):

    def __init__(self, *satellites: SatelliteID):
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
        return 'Satellites({:s})'.format(
            ', '.join(repr(v) for v in self._satellites.values()))

    def eval(self, environment: MutableMapping[str, Any],
             selectors: Mapping[str, Any]) -> None:
        try:
            if selectors['id'] in self:
                self[selectors['id']].eval(environment, selectors['id'])
        except KeyError:
            pass


@dataclass(frozen=True)
class Phase(Statement):
    name: str
    inner_statement: Statement
    condition: Condition = TrueCondition()

    def eval(self, environment: MutableMapping[str, Any],
             selectors: Mapping[str, Any]) -> None:
        if self.condition.eval(selectors):
            if 'phases' not in environment:
                environment['phases'] = dict()
            if self.name not in environment['phases']:
                environment['phases'][self.name] = dict()
            self.inner_statement.eval(
                environment['phases'][self.name], selectors)
