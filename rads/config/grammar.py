from typing import (Any, Optional, Callable, Mapping, Sequence, Tuple,
                    Iterable, TypeVar, List, cast)

import rads.config.parsers as p
from ..xml.base import Element
from .ast import (Assignment, SatelliteCondition, TrueCondition, Statement,
                  CompoundStatement, Condition, If, NullStatement, SatelliteID,
                  Satellites, VariableAlias, Phase, ActionType, replace_action,
                  noreplace_action, append_action)

T = TypeVar('T')


def filter_none(elements: Iterable[Any]) -> Iterable[Any]:
    return (x for x in elements if x is not None)


def ignore(tag: Optional[str] = None) -> p.Parser:
    def process(_: Element) -> NullStatement:
        return NullStatement()
    if tag:
        return p.tag(tag) ^ process
    return p.any() ^ process


def parse_condition(attr: Mapping[str, str]) -> Condition:
    # currently the only condition RADS uses is based on the satellite
    try:
        sat = attr['sat'].strip()
        return SatelliteCondition(
            satellites=set(sat.strip('!').split()), invert=sat.startswith('!'))
    except KeyError:
        return TrueCondition()


def parse_action(element: Element) -> ActionType:
    action = element.attributes.get('action', 'replace')
    if action == 'replace':
        return replace_action
    if action == 'noreplace':
        return noreplace_action
    if action == 'append':
        return append_action
    raise p.GlobalParseFailure(
        element.file, element.opening_line,
        'Invalid action="{:s}".'.format(action))


def list_of(parser: Callable[[str], T]) -> Callable[[str], List[T]]:
    def _parser(string: str) -> List[T]:
        return [parser(s) for s in string.split()]
    return _parser


def variable_alias() -> p.Parser:
    def process(element: Element) -> VariableAlias:
        def error(message: str) -> p.GlobalParseFailure:
            return p.GlobalParseFailure(
                element.file, element.opening_line, message)
        try:
            alias = element.attributes['name']
        except KeyError:
            raise error("'name' attribute missing from <alias>")
        variables = element.text.split() if element.text else []
        if not variables:
            raise error('<alias> cannot be empty')
        condition = parse_condition(element.attributes)
        action = parse_action(element)
        return VariableAlias(alias, variables, condition, action)
    return p.tag('alias') ^ process


def value(parser: Callable[[str], Any], tag: Optional[str] = None,
          var: Optional[str] = None) -> p.Parser:
    def process(element: Element) -> Assignment:
        var_ = var if var else element.tag
        condition = parse_condition(element.attributes)
        action = parse_action(element)
        text = element.text if element.text else ''
        try:
            return Assignment(condition=condition,
                              name=var_,
                              value=parser(text),
                              action=action)
        except ValueError as err:
            raise p.GlobalParseFailure(
                element.file, element.opening_line, str(err))
    if tag:
        return p.tag(tag) ^ process
    return p.any() ^ process


def if_statement(internal: p.Parser) -> p.Parser:
    def process(statements: Tuple[Element, Statement]) -> Statement:
        if_element, false_statement = statements
        condition = parse_condition(if_element.attributes)
        true_statement = internal(if_element.down())[0]
        return If(condition, true_statement, false_statement)
    return p.tag('if') + p.opt(
        elseif_statement(internal) | else_statement(internal)) ^ process


def elseif_statement(internal: p.Parser) -> p.Parser:
    def process(statements: Iterable[Any]) -> Statement:
        elseif_element, false_statement = statements
        condition = parse_condition(elseif_element.attributes)
        true_statement = internal(elseif_element.down())[0]
        return If(condition, true_statement, false_statement)
    return p.Apply(p.tag('elseif') + p.opt(
        p.lazy(lambda: elseif_statement(internal)) | else_statement(
            internal)), process)


def else_statement(internal: p.Parser) -> p.Parser:
    def process(element: Element) -> Any:
        return internal(element.down())[0]
    return p.tag('else') ^ process


def satellites() -> p.Parser:
    def process(element: Element) -> Satellites:
        if not element.text:
            return Satellites()
        satellites_ = []
        for line in element.text.strip().splitlines():
            try:
                id_, id3, *names = line.split()
            except ValueError:
                raise TypeError('TODO')
            satellites_.append(SatelliteID(id_, id3, set(names)))
        return Satellites(*satellites_)

    return p.tag('satellites') ^ process


def phase_statements() -> p.Parser:
    def process(statements: Sequence[Statement]) -> Statement:
        # flatten if only a single statement
        if len(statements) == 1:
            return statements[0]
        return CompoundStatement(*statements)

    statements = p.star(
        value(str, 'mission') |
        value(str, 'cycles') |
        value(str, 'repeat') |
        value(str, 'ref_pass') |
        value(str, 'start_time') |
        value(str, 'subcycles')
    )
    return (p.start() + (statements ^ process) + p.end()
            << 'Invalid configuration block or value.') ^ (lambda x: x[1])


def phase() -> p.Parser:
    def process(element: Element) -> Phase:
        try:
            name = element.attributes['name']
        except KeyError:
            raise p.GlobalParseFailure(
                element.file, element.opening_line,
                "<phase> has no 'name' attribute.")
        try:
            statement = cast(Statement, phase_statements()(element.down())[0])
        except StopIteration:
            statement = NullStatement()
        condition = parse_condition(element.attributes)
        return Phase(name, statement, condition)

    return p.tag('phase') ^ process


def root_statements() -> p.Parser:
    def process(statements: Sequence[Statement]) -> Statement:
        # flatten if only a single statement
        if len(statements) == 1:
            return statements[0]
        return CompoundStatement(*statements)

    statements = p.star(
        ignore('global_attributes') |
        satellites() |
        ignore('var') |
        variable_alias() |
        value(str, 'satellite', var='name') |
        value(int, 'satid') |
        value(float, 'dt1hz') |
        value(float, 'inclination') |
        value(list_of(float), 'frequency') |
        value(list_of(float), 'xover_params') |
        phase() |
        if_statement(p.lazy(root_statements)))
    return (p.start() + (statements ^ process) + p.end()
            << 'Invalid configuration block or value.') ^ (lambda x: x[1])


def parse(root: Element) -> Statement:
    return cast(Statement, root_statements()(root.down())[0])


def preparse(root: Element) -> Statement:
    def process(elements: Sequence[Element]) -> Element:
        return elements[-1]

    parser = p.until(p.tag('satellites')) + satellites() ^ process
    return cast(Statement, parser(root.down())[0])
