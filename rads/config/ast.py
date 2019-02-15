"""Abstract Syntax Tree elements for RADS configuration file."""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Optional, Container, Sequence, Union
import typing

from .._utility import xor


class Condition(ABC):
    """Base class of AST node conditionals."""

    @abstractmethod
    def eval(self, satellite: Optional[str] = None,
             phase: Optional[str] = None) -> bool:
        """Evaluate condition to determine match based on satellite and phase.

        This is used to determine whether or not a block should be executed.

        Parameters
        ----------
        satellite
            Name of current satellite.
        phase
            Single character phase ID.

        Returns
        -------
        bool
            True if the condition is a match, otherwise False.

        """


class TrueCondition(Condition):
    """Condition that is always true."""

    def eval(self, satellite: Optional[str] = None,
             phase: Optional[str] = None) -> bool:  # noqa: D102
        return True

    def __repr__(self) -> str:
        """Get text representation 'TrueCondition()'."""
        return 'TrueCondition()'


class FalseCondition(Condition):
    """Condition that is always false."""

    def eval(self, satellite: Optional[str] = None,
             phase: Optional[str] = None) -> bool:  # noqa: D102
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

    def eval(self, satellite: Optional[str] = None,
             phase: Optional[str] = None) -> bool:  # noqa: D102
        return not satellite or xor(satellite in self.satellites, self.invert)


class Statement:
    """Base class of Abstract Syntax Tree nodes."""


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


@dataclass(frozen=True)
class If(Statement):
    """If/else statement AST node.

    Attributes
    ----------
    condition
        Condition that must be True for the :paramref:`true_statement` branch
        to be executed.  Otherwise, the :paramref:`false_statement` is
        executed.
    true_statement:
        Statement to be executed if :paramref:`condition` evaluates to True.
    false_statement:
        Optional statement to be executed if :paramref:`condition` evaluates
        to False.

    """

    condition: Condition
    true_statement: Statement
    false_statement: Optional[Statement] = None


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

        ================= =================================
        Action            Description
        ================= =================================
        replace (default) Replace the existing value.
        noreplace         Keep the original value.
        append            Append (with space separator).
        add               Add value (must be numeric).
        subtract          Subtract value (must be numeric).
        ================= =================================

    """

    name: str
    value: Any
    condition: Condition
    action: str = 'replace'
