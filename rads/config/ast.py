"""Abstract Syntax Tree elements for RADS configuration file."""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import (Any, Optional, Container, Sequence, Union, MutableMapping,
                    Mapping, Collection, Iterator)
import typing

from .._utility import xor


class Condition(ABC):
    """Base class of AST node conditionals."""

    @abstractmethod
    def eval(self, satellite: Optional[str] = None) -> bool:
        """Evaluate condition to determine match based on satellite.

        This is used to determine whether or not a block should be executed.

        Parameters
        ----------
        satellite
            Name of current satellite.

        Returns
        -------
        bool
            True if the condition is a match, otherwise False.

        """


class TrueCondition(Condition):
    """Condition that is always true."""

    def eval(self, satellite: Optional[str] = None) -> bool:  # noqa: D102
        return True

    def __repr__(self) -> str:
        """Get text representation 'TrueCondition()'."""
        return 'TrueCondition()'


class FalseCondition(Condition):
    """Condition that is always false."""

    def eval(self, satellite: Optional[str] = None) -> bool:  # noqa: D102
        return False

    def __repr__(self) -> str:
        """Get text representation 'FalseCondition()'."""
        return 'FalseCondition()'


@dataclass(frozen=True)
class SatelliteCondition(Condition):
    """Condition that matches based on the satellite.

    Parameters
    ----------
    satellites : Container[str]
        Set of satellites to match.
    invert : bool
        Set to True to invert the match.
    """

    satellites: Container[str]
    invert: bool = False

    def eval(self, satellite: Optional[str] = None) -> bool:  # noqa: D102
        return not satellite or xor(satellite in self.satellites, self.invert)


class Statement(ABC):
    """Base class of Abstract Syntax Tree nodes."""

    @abstractmethod
    def eval(self, environment: MutableMapping[str, Any],
             satellite: Optional[str] = None) -> None:
        """Evaluate statement, adding to the environment dictionary.

        Parameters
        ----------
        environment
            Environment dictionary.  Additions from this statement will be
            added to this mapping.
        satellite
            Current satellite.

        """


class NullStatement(Statement):

    def eval(self, environment: MutableMapping[str, Any],
             satellite: Optional[str] = None) -> None:
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
             satellite: Optional[str] = None) -> None:
        """Evaluate compound statement, adding to the environment dictionary.

        This will evaluate each statement in this compound statement in order.

        Parameters
        ----------
        environment
            Environment dictionary.  Additions from this statement will be
            added to this mapping.
        satellite
            Current satellite.

        """
        for statement in self:
            statement.eval(environment, satellite)


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
             satellite: Optional[str] = None) -> None:
        """Evaluate if/else statement, adding to the environment dictionary.

        If the satellite matches the condition then the :attr:`true_statement`
        will be evaluated.  Otherwise the :attr:`false_statement` will be
        evaluated (if it exists).

        Parameters
        ----------
        environment
            Environment dictionary.  Additions from this statement will be
            added to this mapping.
        satellite
            Current satellite.

        """
        if self.condition.eval(satellite):
            self.true_statement.eval(environment, satellite)
        elif self.false_statement is not None:
            self.false_statement.eval(environment, satellite)


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
        Condition that must be true for this assignment to be executed.
    action
        Action to take if this variable has already been set, the allowable
        actions are listed in the table below.

        ================= ==================================================
        Action            Description
        ================= ==================================================
        replace (default) Replace the existing value.
        noreplace         Keep the original value.
        append            Extend current value/s.
        ================= ==================================================

    """

    name: str
    value: Any
    condition: Condition
    action: str = 'replace'

    def eval(self, environment: MutableMapping[str, Any],
             satellite: Optional[str] = None) -> None:
        """Evaluate assignment, adding to the environment dictionary.

        Parameters
        ----------
        environment
            Environment dictionary.  The addition from this statement will be
            added to this mapping.
        satellite
            Current satellite.

        """
        if self.condition.eval(satellite):
            if (self.name not in environment) or self.action == 'replace':
                environment[self.name] = self.value
            elif self.action == 'append':
                if not hasattr(environment[self.name], 'append'):
                    environment[self.name] = [environment[self.name]]
                try:
                    environment[self.name].extend(self.value)
                except TypeError:
                    environment[self.name].append(self.value)
            elif self.action != 'noreplace':
                raise ValueError("Invalid action '{:s}'".format(self.action))


@dataclass(frozen=True)
class SatelliteID(Statement):
    id: str
    id3: str
    names: Collection[str] = field(default_factory=set)

    def eval(self, environment: MutableMapping[str, Any],
             satellite: Optional[str] = None) -> None:
        if satellite == self.id:
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
             satellite: Optional[str] = None) -> None:
        if satellite and satellite in self:
            self[satellite].eval(environment, satellite)
