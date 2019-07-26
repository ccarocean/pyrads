"""Abstract Syntax Tree elements for RADS configuration file."""

import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields
from difflib import get_close_matches
from typing import (Any, Callable, Collection, Container, Iterator, List,
                    Mapping, MutableMapping, Optional, Sequence, Union, cast)

from dataclass_builder import UndefinedFieldError, MissingFieldError, MISSING

from ._builders import PhaseBuilder, VariableBuilder
from .._utility import xor, delete_sublist, merge_sublist
from ..rpn import CompleteExpression, Expression

ActionType = Callable[[Any, str, Any], None]


def _get_mapping(environment: Any, attr: str,
                 mapping: Callable[[], Mapping] = dict):
    if (not hasattr(environment, attr) or
            getattr(environment, attr) == MISSING):
        setattr(environment, attr, mapping())
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

    Sets :paramref:`attr` to the given :paramref:`value` in the given
    :paramref:`environment`.  If the :paramref:`attr` has already been set
    it will be overwritten.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of :paramref:`attr` in.
        If this environment is a :class:`MutableMapping` then key/values will
        be used.  Otherwise, object attributes will be used.
    attr
        Name of the value to change in the :paramref:`environment`.
    value
        New value to use for the action.  If this is an :class:`Expression` it
        will be converted to a :class:`CompleteExpression` via the
        :func:`Expression.complete` method.

    Raises
    ------
    ValueError
        If a delayed value (currently only :class:`Expression`) fails it's
        parsing.

    """
    if isinstance(value, Expression):
        # force CompleteExpression's
        _set(environment, attr, value.complete())
    else:
        _set(environment, attr, value)


def append(environment: MutableMapping[str, Any], attr: str,
           value: Union[str, List[Any], Expression]) -> None:
    """Set/append key/value pair in the given environment.

    Sets :paramref:`attr` to the given :paramref:`value` in the given
    :paramref:`environment`.  If the :paramref:`attr` has already ben set the
    new value will be appended.  This behaves differently depending on the type
    of the current value.

    :class:`str`
        Appends the new string :paramref:`value` to the current string with a
        single space between them.
    :class:`list`
        Extends the current list with the elements form the new list
        :paramref:`value`.
    :class:`CompleteExpression`
        Appends the tokens from the new :class:`Expression` to the end of the
        existing :class:`CompleteExpression`.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of :paramref:`attr` in.
        If this environment is a :class:`MutableMapping` then key/values will
        be used.  Otherwise, object attributes will be used.
    attr
        Name of the value to change in the :paramref:`environment`.
    value
        New value to use for the action.

    Raises
    ------
    TypeError
        If the current value is not a :class:`str`, :class:`list`, or
        :class:`Expression` or if the new :paramref:`value` does not match
        the type of the current value.
    ValueError
        If a delayed value (currently only :class:`Expression`) fails it's
        parsing.

    """
    # no current value
    if not _has(environment, attr) or _get(environment, attr) == MISSING:
        replace(environment, attr, value)
        return
    # has current value
    current_value = _get(environment, attr)
    if isinstance(current_value, str):
        new_value = current_value + ' ' + value
    elif isinstance(current_value, Expression):
        # force CompleteExpression's
        new_value = (current_value + value).complete()
    elif isinstance(current_value, List):
        new_value = current_value + value
    else:
        raise TypeError("current value is of unsupported type"
                        f"'{type(current_value)}' for the 'append' action")
    _set(environment, attr, new_value)


def delete(environment: MutableMapping[str, Any], attr: str,
           value: Union[str, List[Any], Expression]) -> None:
    """Remove/edit key/value pair in the given environment.

    Removes matching :paramref:`value` from the part of the existing
    :paramref:`attr` in the given :paramref:`environement`.  This behaves
    differently depending on the type of the current value.

    :class:`str`
        Removes the substring :paramref:`value` from the current string.  No
        change if the current string does not contain the substring
        :paramref:`value`.
    :class:`list`
        Removes the sublist :paramref:`value` from the current list.  No change
        if the current list does not contain the sublist :paramref:`value`.
    :class:`CompleteExpression`
        Removes from the current expression the section that matches
        :paramref:`value`.  No change if the current expression does not
        contain the expression :paramref:`value`.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of :paramref:`attr` in.
        If this environment is a :class:`MutableMapping` then key/values will
        be used.  Otherwise, object attributes will be used.
    attr
        Name of the value to change in the :paramref:`environment`.
    value
        New value to use for the action.

    Raises
    ------
    TypeError
        If the current value is not a :class:`str`, :class:`list`, or
        :class:`Expression` or if the new :paramref:`value` does not match
        the type of the current value.
    ValueError
        If a delayed value (currently only :class:`Expression`) fails it's
        parsing.

    """
    # no current value, do nothing
    if not _has(environment, attr) or _get(environment, attr) == MISSING:
        return
    # has current value
    current_value = _get(environment, attr)
    if isinstance(current_value, str):
        new_value = current_value.replace(value, '')
    elif isinstance(current_value, Expression):
        # force CompleteExpression's
        new_value = CompleteExpression(
            delete_sublist(list(current_value), list(value)))
    elif isinstance(current_value, List):
        new_value = delete_sublist(current_value, value)
    else:
        raise TypeError("current value is of unsupported type"
                        f"'{type(current_value)}' for the 'append' action")
    _set(environment, attr, new_value)


def merge(environment: MutableMapping[str, Any], attr: str,
          value: Union[str, List[Any], Expression]) -> None:
    """Set/merge key/value pair in the given environment.

    Sets :paramref:`attr` to the given :paramref:`value` in the given
    :paramref:`environment`.  If the :paramref:`attr` has already ben set the
    new value will be appended if the existing value does not already contain
    the new value.  This behaves differently depending on the type of the
    current value.

    :class:`str`
        Appends the new string :paramref:`value` to the current string with a
        single space between them if the new string is not a substring of the
        original.
    :class:`list`
        Extends the current list with the elements form the new list
        :paramref:`value` if the new list is not a sublist of the original.
    :class:`CompleteExpression`
        Appends the tokens from the new :class:`Expression` to the end of the
        existing :class:`CompleteExpression` if the new :class:`Expression` is\
        not contained within the original expression.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of :paramref:`attr` in.
        If this environment is a :class:`MutableMapping` then key/values will
        be used.  Otherwise, object attributes will be used.
    attr
        Name of the value to change in the :paramref:`environment`.
    value
        New value to use for the action.

    Raises
    ------
    TypeError
        If the current value is not a :class:`str`, :class:`list`, or
        :class:`Expression` or if the new :paramref:`value` does not match
        the type of the current value.
    ValueError
        If a delayed value (currently only :class:`Expression`) fails it's
        parsing.

    """
    # no current value, set value
    if not _has(environment, attr) or _get(environment, attr) == MISSING:
        replace(environment, attr, value)
        return
    # has current value
    current_value = _get(environment, attr)
    if isinstance(current_value, str):
        if value in current_value:
            new_value = current_value
        else:
            new_value = current_value + ' ' + value
    elif isinstance(current_value, Expression):
        # force CompleteExpression's
        new_value = CompleteExpression(
            merge_sublist(list(current_value), list(value)))
    elif isinstance(current_value, List):
        new_value = merge_sublist(current_value, value)
    else:
        raise TypeError("current value is of unsupported type"
                        f"'{type(current_value)}' for the 'append' action")
    _set(environment, attr, new_value)


def edit_append(
        environment: MutableMapping[str, Any], attr: str, string: str) -> None:
    """Set key/value pair in the given environment.

    Sets :paramref:`attr` to the given :paramref:`string` in the given
    :paramref:`environment`.  If the :paramref:`attr` has already ben set the
    new string will be append after '. '.

    .. note::

        This action only works for strings.

    Parameters
    ----------
    environment
        Environment to apply the action to the value of :paramref:`attr` in.
        If this environment is a :class:`MutableMapping` then key/values will
        be used.  Otherwise, object attributes will be used.
    attr
        Name of the value to change in the :paramref:`environment`.
    string
        New string to use for the action.

    Raises
    ------
    TypeError
        If not both the current and new values are strings.

    """
    if not _has(environment, attr) or _get(environment, attr) == MISSING:
        _set(environment, attr, string)
    else:
        _set(environment, attr, _get(environment, attr) + '. ' + string)


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
        self.message = message
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

    def __init__(self, name: Any, value: Any,
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
            try:
                action(aliases, self.alias, deepcopy(self.variables))
            except (TypeError, ValueError, KeyError) as err:
                raise ASTEvaluationError(str(err), source=self.source)


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
        if 'id' in selectors and selectors['id'] == self.id:
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
                self[selectors['id']].eval(environment, selectors)
        except KeyError:
            pass


class NamedBlock(Statement, ABC):
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

    def _eval_runner(self, mapping: str, builder: Any,
                     environment: Any, selectors: Mapping[str, Any]):
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
                f"missing required attribute '{err.field.name}'",
                source=self.source)


class UniqueNamedBlock(NamedBlock, ABC):

    def _eval_runner(self, mapping: str, builder: Any,
                     environment: Any, selectors: Mapping[str, Any]):
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
                    f"missing required attribute '{err.field.name}'",
                    source=self.source)


class Phase(NamedBlock):

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        self._eval_runner('phases', PhaseBuilder, environment, selectors)


class Variable(UniqueNamedBlock):

    def eval(self, environment: Any, selectors: Mapping[str, Any]) -> None:
        self._eval_runner(
            'variables', VariableBuilder, environment, selectors)
