"""Abstract Syntax Tree elements for RADS configuration file."""

import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields
from difflib import get_close_matches
from typing import (
    Any,
    Callable,
    Collection,
    Container,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
    cast,
)

from dataclass_builder import MISSING, MissingFieldError, UndefinedFieldError

from ..rpn import CompleteExpression, Expression
from ..utility import delete_sublist, merge_sublist, xor
from .builders import PhaseBuilder, VariableBuilder

__all__ = [
    "ActionType",
    "append",
    "delete",
    "edit_append",
    "merge",
    "replace",
    "Condition",
    "TrueCondition",
    "FalseCondition",
    "SatelliteCondition",
    "Source",
    "ASTEvaluationError",
    "Statement",
    "NullStatement",
    "CompoundStatement",
    "If",
    "NamedBlock",
    "UniqueNamedBlock",
    "Alias",
    "Assignment",
    "Phase",
    "SatelliteID",
    "Satellites",
    "Variable",
]

ActionType = Callable[[Any, str, Any], None]


def _get_mapping(
    environment: Any, attr: str, mapping: Callable[[], Mapping[str, Any]] = dict
) -> Any:
    if not hasattr(environment, attr) or getattr(environment, attr) == MISSING:
        setattr(environment, attr, mapping())
    return getattr(environment, attr)


def _suggest_field(dataclass: Any, attempt: str) -> Optional[str]:
    matches = get_close_matches(attempt, [f.name for f in fields(dataclass)], 1, 0.1)
    if matches:
        return cast(str, matches[0])
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
    if isinstance(o, MutableMapping):
        o[attr] = value
    else:
        setattr(o, attr, value)


def _del(o: Any, attr: str) -> None:
    if isinstance(o, MutableMapping):
        del o[attr]
    return delattr(o, attr)


def replace(environment: Any, attr: str, value: Any) -> None:
    """Set value in the given environment.

    Sets `attr` to the given `value` in the given `environment`.  If the `attr`
    has already been set it will be overwritten.

    :param environment:
        Environment to apply the action to the value of `attr` in. If this
        environment is a :class:`collections.abc.MutableMapping` then
        key/values will be used. Otherwise, object attributes will be used.
    :param attr:
        Name of the value to change in the `environment`.
    :param value:
        New value to use for the action.  If this is an
        :class:`rads.rpn.Expression` it will be converted to a
        :class:`rads.rpn.CompleteExpression` via the
        :func:`rads.rpn.Expression.complete` method.

    :raises ValueError:
        If a delayed value (currently only :class:`Expression`) fails it's
        parsing.
    """
    if isinstance(value, Expression):
        # force CompleteExpression's
        _set(environment, attr, value.complete())
    else:
        _set(environment, attr, value)


def append(
    environment: MutableMapping[str, Any],
    attr: str,
    value: Union[str, List[Any], Expression],
) -> None:
    """Append to value in the given environment.

    Sets `attr` to the given `value` in the given `environment`.  If the `attr`
    has already ben set the new value will be appended.  This behaves
    differently depending on the type of the current value.

    :class:`str`
        Appends the new string `value` to the current string with a single
        space between them.
    :class:`list`
        Extends the current list with the elements form the new list `value`.
    :class:`rads.rpn.CompleteExpression`
        Appends the tokens from the new :class:`rads.rpn.Expression` to the end
        of the existing :class:`rads.rpn.CompleteExpression`.

    :param environment:
        Environment to apply the action to the value of `attr` in. If this
        environment is a :class:`collections.abc.MutableMapping` then
        key/values will be used. Otherwise, object attributes will be used.
    :param attr:
        Name of the value to change in the `environment`.
    :param value:
        New value to use for the action.

    :raises TypeError:
        If the current value is not a :class:`str`, :class:`list`, or
        :class:`rads.rpn.Expression` or if the new `value` does not match
        the type of the current value.
    :raises ValueError:
        If a delayed value (currently only :class:`rads.rpn.Expression`) fails
        it's parsing.
    """
    # no current value
    if not _has(environment, attr) or _get(environment, attr) == MISSING:
        replace(environment, attr, value)
        return
    # has current value
    current_value = _get(environment, attr)
    if isinstance(current_value, str) and isinstance(value, str):
        _set(environment, attr, current_value + " " + value)
    elif isinstance(current_value, Expression) and isinstance(value, Expression):
        # force CompleteExpression's
        _set(environment, attr, (current_value + value).complete())
    elif isinstance(current_value, List) and isinstance(value, List):
        _set(environment, attr, current_value + value)
    else:
        raise TypeError(
            "current value and new value are of unsupported types"
            f"'{type(current_value)}' and '{type(value)}' for the 'append' action"
        )


def delete(
    environment: MutableMapping[str, Any],
    attr: str,
    value: Union[str, List[Any], Expression],
) -> None:
    """Delete part of value in the given environment.

    Removes matching `value` from the part of the existing `attr` in the given
    `environment`.  This behaves differently depending on the type of the
    current value.

    :class:`str`
        Removes the substring `value` from the current string.  No change if
        the current string does not contain the substring `value`.
    :class:`list`
        Removes the sublist `value` from the current list.  No change if the
        current list does not contain the sublist `value`.
    :class:`rads.rpn.CompleteExpression`
        Removes from the current expression the section that matches `value`.
        No change if the current expression does not contain the expression
        `value`.

    :param environment:
        Environment to apply the action to the value of `attr` in.  If this
        environment is a :class:`collections.abc.MutableMapping` then
        key/values will be used.  Otherwise, object attributes will be used.
    :param attr:
        Name of the value to change in the `environment`.
    :param value:
        New value to use for the action.

    :raises TypeError:
        If the current value is not a :class:`str`, :class:`list`, or
        :class:`rads.rpn.Expression` or if the new `value` does not match
        the type of the current value.
    :raises ValueError:
        If a delayed value (currently only :class:`rads.rpn.Expression`) fails
        it's parsing.
    """
    # no current value, do nothing
    if not _has(environment, attr) or _get(environment, attr) == MISSING:
        return
    # has current value
    current_value = _get(environment, attr)
    if isinstance(current_value, str) and isinstance(value, str):
        _set(environment, attr, current_value.replace(value, ""))
    elif isinstance(current_value, Expression) and isinstance(value, Expression):
        # force CompleteExpression's
        _set(
            environment,
            attr,
            CompleteExpression(delete_sublist(list(current_value), list(value))),
        )
    elif isinstance(current_value, List) and isinstance(value, List):
        _set(environment, attr, delete_sublist(current_value, value))
    else:
        raise TypeError(
            "current value is of unsupported type"
            f"'{type(current_value)}' for the 'append' action"
        )


def merge(
    environment: MutableMapping[str, Any],
    attr: str,
    value: Union[str, List[Any], Expression],
) -> None:
    """Merge with current value in the given environment.

    Sets `attr` to the given `value` in the given `environment`.  If the `attr`
    has already ben set the new value will be appended if the existing value
    does not already contain the new value.  This behaves differently depending
    on the type of the current value.

    :class:`str`
        Appends the new string `value` to the current string with a single
        space between them if the new string is not a substring of the original.
    :class:`list`
        Extends the current list with the elements form the new list `value`
        if the new list is not a sublist of the original.
    :class:`rads.rpn.CompleteExpression`
        Appends the tokens from the new :class:`rads.rpn.Expression` to the
        end of the existing :class:`rads.rpn.CompleteExpression` if the new
        :class:`rads.rpn.Expression` is not contained within the original
        expression.

    :param environment:
        Environment to apply the action to the value of `attr` in. If
        this environment is a :class:`collections.abc.MutableMapping` then
        key/values will be used.  Otherwise, object attributes will be used.
    :param attr:
        Name of the value to change in the `environment`.
    :param value:
        New value to use for the action.

    :raises TypeError:
        If the current value is not a :class:`str`, :class:`list`, or
        :class:`rads.rpn. Expression` or if the new `value` does not match
        the type of the current value.
    :raises ValueError:
        If a delayed value (currently only :class:`rads.rpn.Expression`) fails
        it's parsing.
    """
    # no current value, set value
    if not _has(environment, attr) or _get(environment, attr) == MISSING:
        replace(environment, attr, value)
        return
    # has current value
    current_value = _get(environment, attr)
    if isinstance(current_value, str) and isinstance(value, str):
        if value not in current_value:
            _set(environment, attr, current_value + " " + value)
        # do nothing if value in current value
    elif isinstance(current_value, Expression) and isinstance(value, Expression):
        # force CompleteExpression's
        _set(
            environment,
            attr,
            CompleteExpression(merge_sublist(list(current_value), list(value))),
        )
    elif isinstance(current_value, List) and isinstance(value, List):
        _set(environment, attr, merge_sublist(current_value, value))
    else:
        raise TypeError(
            "current value is of unsupported type"
            f"'{type(current_value)}' for the 'append' action"
        )


def edit_append(environment: MutableMapping[str, Any], attr: str, string: str) -> None:
    """Append sentence to string in the given environment.

    Sets `attr` to the given `string` in the given `environment`.  If the
    `attr` has already ben set the new string will be append after ". ".

    .. note::

        This action only works for strings.

    :param environment:
        Environment to apply the action to the value of `attr` in. If
        this environment is a :class:`collections.abc.MutableMapping` then
        key/values will be used.  Otherwise, object attributes will be used.
    :param attr:
        Name of the value to change in the `environment`.
    :param string:
        New string to use for the action.

    :raises TypeError:
        If not both the current and new values are strings.
    """
    if not _has(environment, attr) or _get(environment, attr) == MISSING:
        _set(environment, attr, string)
    else:
        _set(environment, attr, _get(environment, attr) + ". " + string)


class Condition(ABC):
    """Base class of AST node conditionals."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    @abstractmethod
    def test(self, selectors: Mapping[str, Any]) -> bool:
        """Evaluate condition to determine match.

        This is used to determine whether or not a block should be executed.

        :param selectors:
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.

        :return bool:
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

    This makes use of the `id` key/value in the `selectors` mapping if it
    exists.
    """

    satellites: Container[str]
    invert: bool

    def __init__(self, satellites: Container[str], invert: bool = False):
        """
        :param satellites:
            Set of satellite ID's to match.
        :param invert:
            Set to True to invert the match.
        """
        self.satellites = satellites
        self.invert = invert

    def __repr__(self) -> str:
        prefix = f"{self.__class__.__qualname__}({repr(self.satellites)}"
        if self.invert:
            return prefix + f", invert={repr(self.invert)})"
        return prefix + ")"

    def test(self, selectors: Mapping[str, Any]) -> bool:  # noqa: D102
        try:
            return xor(selectors["id"] in self.satellites, self.invert)
        except KeyError:
            return False


@dataclass(frozen=True)
class Source:
    """**dataclass**: Source file location."""

    line: Optional[int] = None
    """Line number."""
    file: Optional[str] = None
    """File name."""


class ASTEvaluationError(Exception):
    """Abstract syntax tree evaluation error.

    This is raised when evaluation of the AST leads to an unrecoverable error
    in an attempt to give the user some information on what happened and what
    part of the RADS XML file caused the problem.
    """

    message: str
    """Error message"""
    line: Optional[int] = None
    """Number of the line that caused the error."""
    file: Optional[str] = None
    """Path of the file that caused the error."""

    def __init__(
        self, message: str = "evaluation failed", source: Optional[Source] = None
    ):
        """
        :param message:
            The error message.
        :param source:
            The source file location of the error, if known.
        """
        self.message = message
        if source:
            self.line = source.line
            self.file = source.file
        file = self.file if self.file else ""
        line = self.line if self.line else ""
        super().__init__(f"{file}:{line}: {message}")


class Statement(ABC):
    """Base class of Abstract Syntax Tree nodes."""

    source: Optional[Source]
    """Location in the source configuration this statement came from."""

    def __init__(self, *, source: Optional[Source] = None):
        """
        :param source:
            Location in the source configuration this statement came from.
        """
        self.source = source

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    @abstractmethod
    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate statement, modifying the environment object.

        :param environment:
            Environment object to modify.  If this is a mapping it will be used
            as such, otherwise attribute assignment will be used instead.
        :param selectors:
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.
        """


class NullStatement(Statement):
    """A null statement that does nothing when evaluated."""

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Do nothing.

        :param environment:
            Environment object to modify.  If this is a mapping it will be used
            as such, otherwise attribute assignment will be used instead.
        :param selectors:
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.
        """


class CompoundStatement(Sequence[Statement], Statement):
    """A sequence of statements."""

    def __init__(self, *statements: Statement, source: Optional[Source] = None):
        r"""
        :param \*statements:
            Statements to be stored in the :class:`CompoundStatement`.
        :param source:
            Location in the source configuration this statement came from.
        """
        super().__init__(source=source)
        self._statements = statements

    @typing.overload
    def __getitem__(self, key: int) -> Statement:
        pass

    @typing.overload
    def __getitem__(self, key: slice) -> "CompoundStatement":
        pass

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[Statement, "CompoundStatement"]:
        if isinstance(key, slice):
            return CompoundStatement(*self._statements[key])
        return self._statements[key]

    def __len__(self) -> int:
        return len(self._statements)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}"
            f"({', '.join(repr(s) for s in self._statements)})"
        )

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate compound statement, modifying the environment object.

        This will evaluate each statement in this compound statement in order.

        :param environment:
            Environment object to modify.  If this is a mapping it will be used
            as such, otherwise attribute assignment will be used instead.
        :param selectors:
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.
        """
        for statement in self:
            statement.eval(environment, selectors)


class If(Statement):
    """If/else statement AST node."""

    condition: Condition
    """
    Condition that must be True for the :attr:`true_statement` branch
    to be executed.  Otherwise, the :attr:`false_statement` is
    executed.
    """
    true_statement: Statement
    """Statement to be executed if :attr:`condition` evaluates to True."""
    false_statement: Optional[Statement] = None
    """Optional statement to be executed if :attr:`condition` evaluates to False."""

    def __init__(
        self,
        condition: Condition,
        true_statement: Statement,
        false_statement: Optional[Statement] = None,
        *,
        source: Optional[Source] = None,
    ):
        """
        :param condition:
            Condition for the If statement.
        :param true_statement:
            Statement to be executed if the condition is true.
        :param false_statement:
            Optional statement to be executed if the condition is false.
        :param source:
            Location in the source configuration this statement came from.
        """
        super().__init__(source=source)
        self.condition = condition
        self.true_statement = true_statement
        self.false_statement = false_statement

    def __repr__(self) -> str:
        prefix = (
            f"{self.__class__.__qualname__}"
            f"({repr(self.condition)}, {repr(self.true_statement)}"
        )
        if self.false_statement is None:
            return prefix + ")"
        return prefix + f", {repr(self.false_statement)})"

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate if/else statement, modifying the environment object.

        If the satellite matches the condition then the :attr:`true_statement`
        will be evaluated.  Otherwise the :attr:`false_statement` will be
        evaluated (if it exists).

        :param environment:
            Environment object to modify.  If this is a mapping it will be used
            as such, otherwise attribute assignment will be used instead.
        :param selectors:
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.
        """
        if self.condition.test(selectors):
            self.true_statement.eval(environment, selectors)
        elif self.false_statement is not None:
            self.false_statement.eval(environment, selectors)


class Assignment(Statement):
    """Assignment statement (value to variable) AST node."""

    name: str
    """Name to assign a value to."""
    value: Any
    """Value to assign."""
    condition: Condition = TrueCondition()
    """Condition that must be true for this assignment to be executed."""
    # TODO: Remove Optional when https://github.com/python/mypy/issues/708 is
    #  fixed.
    action: Optional[ActionType] = cast(ActionType, staticmethod(replace))
    """Action to take with this assigment.

    *This is an attribute which contains a callable, not a static method.  The
    automatic documentation tools just don't understand this*

    .. seealso::

        :func:`replace`
            Set or replace current value.
        :func:`append`
            Append to value in the given environment.
        :func:`delete`
            Delete part of value in the given environment.
        :func:`merge`
            Merge with current value in the given environment.
        :func:`edit_append`
            Append sentence to string in the given environment.
    """

    def __init__(
        self,
        name: Any,
        value: Any,
        condition: Optional[Condition] = None,
        action: Optional[ActionType] = None,
        *,
        source: Optional[Source] = None,
    ):
        """
        :param name:
            Name to assign a value to.
        :param value:
            Value to assign.
        :param condition:
            Condition that must be true for this assignment to be executed.
        :param action:
            Action to take with this assignment.  See :attr:`action`.
        :param source:
            Location in the source configuration this statement came from.
        """
        super().__init__(source=source)
        self.name = name
        self.value = value
        if condition is not None:
            self.condition = condition
        if action is not None:
            self.action = action

    def __repr__(self) -> str:
        prefix = (
            f"{self.__class__.__qualname__}" f"({repr(self.name)}, {repr(self.value)}"
        )
        if not isinstance(self.condition, TrueCondition):
            prefix += f", {repr(self.condition)}"
        if self.action == replace:
            return prefix + ")"
        return prefix + f", {cast(ActionType, self.action).__qualname__})"

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate value assignment.

        Modifies the environment object if the :attr:`condition` is true.

        :param environment:
            Environment object to modify.  If this is a mapping it will be used
            as such, otherwise attribute assignment will be used instead.
        :param selectors:
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.
        """
        if self.condition.test(selectors):
            action = cast(ActionType, self.action)
            try:
                action(environment, self.name, deepcopy(self.value))
            except UndefinedFieldError as err:
                message = f"invalid assignment to '{err.field}'"
                suggested = _suggest_field(err.dataclass, err.field)
                if suggested:
                    message += f", did you mean '{suggested}'"
                raise ASTEvaluationError(message, source=self.source)
            except (TypeError, ValueError, KeyError) as err:
                raise ASTEvaluationError(str(err), source=self.source)


class Alias(Statement):
    """Variable alias statement."""

    alias: str
    """The name of the alias (pseudo variable)."""
    variables: Sequence[str]
    """RADS variables that the alias maps to."""
    condition: Condition = TrueCondition()
    """Condition that must be true for this alias to be added."""
    # TODO: Remove Optional when https://github.com/python/mypy/issues/708 is
    #  fixed.
    action: Optional[ActionType] = cast(ActionType, staticmethod(replace))
    """Action to take with this alias.

    *This is an attribute which contains a callable, not a static method.  The
    automatic documentation tools just don't understand this*

    .. seealso::

        :func:`replace`
            Set or replace current value.
        :func:`append`
            Append to value in the given environment.
        :func:`delete`
            Delete part of value in the given environment.
        :func:`merge`
            Merge with current value in the given environment.
        :func:`edit_append`
            Append sentence to string in the given environment.
    """

    def __init__(
        self,
        alias: Any,
        variables: Sequence[str],
        condition: Optional[Condition] = None,
        action: Optional[ActionType] = None,
        *,
        source: Optional[Source] = None,
    ):
        """
        :param alias:
            Alias name (pseudo variable).
        :param variables:
            RADS variables that the alias maps to.
        :param condition:
            Condition that must be true for this alias to be executed.
        :param action:
            Action to take with this alias.  See :attr:`action`.
        :param source:
            Location in the source configuration this statement came from.
        """
        super().__init__(source=source)
        self.alias = alias
        self.variables = variables
        if condition is not None:
            self.condition = condition
        if action is not None:
            self.action = action

    def __repr__(self) -> str:
        prefix = (
            f"{self.__class__.__qualname__}"
            f"({repr(self.alias)}, {repr(self.variables)}"
        )
        if not isinstance(self.condition, TrueCondition):
            prefix += f", {repr(self.condition)}"
        if self.action == replace:
            return prefix + ")"
        return prefix + f", {cast(ActionType, self.action).__qualname__})"

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate alias assignment.

        Modifies the environment object if the :attr:`condition` is true.

        :param environment:
            Environment object to modify.  If this is a mapping it will be used
            as such, otherwise attribute assignment will be used instead.
        :param selectors:
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.
        """
        if self.condition.test(selectors):
            action = cast(ActionType, self.action)
            aliases = _get_mapping(environment, "aliases")
            try:
                action(aliases, self.alias, deepcopy(self.variables))
            except (TypeError, ValueError, KeyError) as err:
                raise ASTEvaluationError(str(err), source=self.source)


class SatelliteID(Statement):
    """Satellite ID statement."""

    id: str
    """2 character ID of the satellite represented in the statement."""
    id3: str
    """3 character ID of the satellite represented in the statement."""
    names: Collection[str]
    """Other names for the satellite."""

    def __init__(
        self,
        id: str,
        id3: str,
        names: Optional[Collection[str]] = None,
        *,
        source: Optional[Source] = None,
    ):
        """
        :param id:
            2 character ID of the satellite.
        :param id3:
            3 character ID of the satellite.
        :param names:
            Other names for the satellite.
        :param source:
            Location in the source configuration this statement came from.
        """
        super().__init__(source=source)
        self.id = id
        self.id3 = id3
        if names is None:
            self.names = set()
        else:
            self.names = names

    def __repr__(self) -> str:
        prefix = f"{self.__class__.__qualname__}" f"({repr(self.id)}, {repr(self.id3)}"
        if self.names:
            return prefix + f", {self.names})"
        return prefix + ")"

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate satellite ID statement, setting ID's and names.

        :param environment:
            Environment object to modify.  If this is a mapping it will be used
            as such, otherwise attribute assignment will be used instead.
        :param selectors:
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.
        """
        if "id" in selectors and selectors["id"] == self.id:
            environment.id = self.id
            environment.id3 = self.id3
            environment.names = self.names


class Satellites(Mapping[str, Statement], Statement):
    """A collection of :class:`SatelliteID` statements.

    In particular this is a mapping from 2 character satellite ID's to
    :class:`SatelliteID` statements.
    """

    def __init__(self, *satellites: SatelliteID, source: Optional[Source] = None):
        r"""
        :param \*satellites:
            :class:`SatelliteID` statements.
        :param source:
            Location in the source configuration this statement came from.
        """
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
        return (
            f"{self.__class__.__qualname__}("
            f"{', '.join(repr(v) for v in self._satellites.values())})"
        )

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        """Evaluate contained satellite ID statements, setting ID's and names.

        :param environment:
            Environment object to modify.  If this is a mapping it will be used
            as such, otherwise attribute assignment will be used instead.
        :param selectors:
            Key/value pairs of things that can be used for conditional parsing
            of the configuration file.
        """
        try:
            if selectors["id"] in self:
                self[selectors["id"]].eval(environment, selectors)
        except KeyError:
            pass


class NamedBlock(Statement, ABC):
    """Abstract named block statement, non unique version.

    This named block need not be unique because each name maintains a list of
    blocks that match it in the environment.
    """

    name: str
    """Name of the block."""
    inner_statement: Statement
    """Inner statement stored in the block."""
    condition: Condition = TrueCondition()
    """Condition that must be true for the `inner_statement` to be executed."""

    def __init__(
        self,
        name: str,
        inner_statement: Statement,
        condition: Optional[Condition] = None,
        *,
        source: Optional[Source] = None,
    ):
        """
        :param name:
            Name of the block.  Usually the XML tag name.
        :param inner_statement:
            The inner statement stored in the block.  This is usually a
            :class:`CompoundStatement`.
        :param condition:
            Condition that must be true for the `inner_statement` to be executed.
        :param source:
            Location in the source configuration this statement came from.
        """
        super().__init__(source=source)
        self.name = name
        self.inner_statement = inner_statement
        if condition is not None:
            self.condition = condition

    def __repr__(self) -> str:
        prefix = (
            f"{self.__class__.__qualname__}"
            f"({repr(self.name)}, {repr(self.inner_statement)}"
        )
        if isinstance(self.condition, TrueCondition):
            return prefix + ")"
        return prefix + f", {self.condition})"

    def _eval_runner(
        self, mapping: str, builder: Any, environment: Any, selectors: Mapping[str, Any]
    ) -> None:
        if self.condition and not self.condition.test(selectors):
            return

        # initialize and/or retrieve mapping
        mapping_ = _get_mapping(environment, mapping)

        builder_ = builder(id=self.name)
        self.inner_statement.eval(builder_, selectors)
        try:
            if self.name not in mapping_:
                mapping_[self.name] = []
            mapping_[self.name].append(builder_.build())
        except MissingFieldError as err:
            raise ASTEvaluationError(
                f"missing required attribute '{err.field.name}'", source=self.source
            )


class UniqueNamedBlock(NamedBlock, ABC):
    """Abstract named block statement, unique version.

    This named block is unique.  Therefore, a delicately named block will
    update the original data in the environment.
    """

    def _eval_runner(
        self, mapping: str, builder: Any, environment: Any, selectors: Mapping[str, Any]
    ) -> None:
        if self.condition and not self.condition.test(selectors):
            return

        # initialize and/or retrieve mapping
        mapping_ = _get_mapping(environment, mapping)

        if self.name in mapping_:
            # update current structure
            self.inner_statement.eval(mapping_[self.name], selectors)
        else:
            # build initial structure
            builder_ = builder(id=self.name)
            self.inner_statement.eval(builder_, selectors)
            try:
                mapping_[self.name] = builder_.build()
            except MissingFieldError as err:
                raise ASTEvaluationError(
                    f"missing required attribute '{err.field.name}'", source=self.source
                )


class Phase(NamedBlock):
    """Phase statement, for all phase related information."""

    def eval(
        self, environment: Any, selectors: Mapping[str, Any]
    ) -> None:  # noqa: D102
        self._eval_runner("phases", PhaseBuilder, environment, selectors)


class Variable(UniqueNamedBlock):
    """Variable statement, for all phase related information."""

    def eval(
        self, environment: Any, selectors: Mapping[str, Any]
    ) -> None:  # noqa: D102
        self._eval_runner("variables", VariableBuilder, environment, selectors)
