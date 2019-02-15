from typing import (Any, Optional, Callable, Mapping, Sequence, Tuple,
                    Iterable, TypeVar, List, cast)

import rads.config.parsers as p
from ..xml.base import Element
from .ast import (Assignment, SatelliteCondition, TrueCondition, Statement,
                  CompoundStatement, Condition, If, NullStatement, SatelliteID,
                  Satellites)

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


def list_of(parser: Callable[[str], T]) -> Callable[[str], List[T]]:
    def _parser(string: str) -> List[T]:
        return [parser(s) for s in string.split()]
    return _parser


def value(parser: Callable[[str], Any], tag: Optional[str] = None,
          var: Optional[str] = None) -> p.Parser:
    def process(element: Element) -> Assignment:
        var_ = var if var else element.tag
        try:
            condition = parse_condition(element.attributes)
            action = element.attributes.get('action', 'replace')
            text = element.text if element.text else ''
            if action not in ['replace', 'noreplace', 'append']:
                raise p.GlobalParseFailure(
                    element.file, element.opening_line,
                    'Invalid action="{:s}".'.format(action))
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


def satellites_statement() -> p.Parser:
    def process(element: Element) -> Satellites:
        if not element.text:
            return Satellites()
        satellites = []
        for line in element.text.strip().splitlines():
            try:
                id_, id3, *names = line.split()
            except ValueError:
                raise TypeError('TODO')
            satellites.append(SatelliteID(id_, id3, set(names)))
        return Satellites(*satellites)
    return p.tag('satellites') ^ process


def root_statements() -> p.Parser:
    def process(statements: Sequence[Statement]) -> Statement:
        # flatten if only a single statement
        if len(statements) == 1:
            return statements[0]
        return CompoundStatement(*statements)
    statements = p.star(ignore('global_attributes') |
                        satellites_statement() |
                        ignore('var') |
                        ignore('alias') |
                        ignore('phase') |
                        value(str, 'satellite', var='name') |
                        value(int, 'satid') |
                        value(float, 'dt1hz') |
                        value(float, 'inclination') |
                        value(list_of(float), 'frequency') |
                        value(list_of(float), 'xover_params') |
                        if_statement(p.lazy(root_statements)))
    return (p.start() + (statements ^ process) + p.end()
            << 'Invalid configuration block or value.') ^ (lambda x: x[1])


def parse(root: Element) -> Statement:
    return cast(Statement, root_statements()(root.down())[0])


def preparse(root: Element) -> Statement:
    def process(elements: Sequence[Element]) -> Element:
        return elements[-1]
    parser = p.until(p.tag('satellites')) + satellites_statement() ^ process
    return cast(Statement, parser(root.down())[0])
