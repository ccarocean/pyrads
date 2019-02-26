from typing import (Any, Optional, Callable, Mapping, Sequence, Tuple,
                    Iterable, TypeVar, List, cast)

import rads.config.ast as ast
import rads.config.parsers as p
from .elements import Cycles
from ..xml.base import Element

T = TypeVar('T')


def ignore(tag: Optional[str] = None) -> p.Parser:
    def process(_: Element) -> ast.NullStatement:
        return ast.NullStatement()

    if tag:
        return p.tag(tag) ^ process
    return p.any() ^ process


def parse_condition(attr: Mapping[str, str]) -> ast.Condition:
    # currently the only condition RADS uses is based on the satellite
    try:
        sat = attr['sat'].strip()
        return ast.SatelliteCondition(
            satellites=set(sat.strip('!').split()), invert=sat.startswith('!'))
    except KeyError:
        return ast.TrueCondition()


def parse_action(element: Element) -> ast.ActionType:
    action = element.attributes.get('action', 'replace')
    if action == 'replace':
        return ast.replace_action
    if action == 'noreplace':
        return ast.noreplace_action
    if action == 'append':
        return ast.append_action
    raise p.GlobalParseFailure(
        element.file, element.opening_line,
        'Invalid action="{:s}".'.format(action))


def list_of(parser: Callable[[str], T]) -> Callable[[str], List[T]]:
    def _parser(string: str) -> List[T]:
        return [parser(s) for s in string.split()]

    return _parser


def variable_alias() -> p.Parser:
    def process(element: Element) -> ast.VariableAlias:
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
        return ast.VariableAlias(alias, variables, condition, action)

    return p.tag('alias') ^ process


def value(parser: Callable[[str], Any], tag: Optional[str] = None,
          var: Optional[str] = None) -> p.Parser:
    def process(element: Element) -> ast.Assignment:
        var_ = var if var else element.tag
        condition = parse_condition(element.attributes)
        action = parse_action(element)
        text = element.text if element.text else ''
        try:
            return ast.Assignment(condition=condition,
                                  name=var_,
                                  value=parser(text),
                                  action=action)
        except (ValueError, TypeError) as err:
            raise p.GlobalParseFailure(
                element.file, element.opening_line, str(err))

    if tag:
        return p.tag(tag) ^ process
    return p.any() ^ process


def if_statement(internal: p.Parser) -> p.Parser:
    def process(statements: Tuple[Element, ast.Statement]) -> ast.Statement:
        if_element, false_statement = statements
        condition = parse_condition(if_element.attributes)
        true_statement = internal(if_element.down())[0]
        return ast.If(condition, true_statement, false_statement)

    return p.tag('if') + p.opt(
        elseif_statement(internal) | else_statement(internal)) ^ process


def elseif_statement(internal: p.Parser) -> p.Parser:
    def process(statements: Iterable[Any]) -> ast.Statement:
        elseif_element, false_statement = statements
        condition = parse_condition(elseif_element.attributes)
        true_statement = internal(elseif_element.down())[0]
        return ast.If(condition, true_statement, false_statement)

    return p.Apply(p.tag('elseif') + p.opt(
        p.lazy(lambda: elseif_statement(internal)) | else_statement(
            internal)), process)


def else_statement(internal: p.Parser) -> p.Parser:
    def process(element: Element) -> Any:
        return internal(element.down())[0]

    return p.tag('else') ^ process


def satellites() -> p.Parser:
    def process(element: Element) -> ast.Satellites:
        if not element.text:
            return ast.Satellites()
        satellites_ = []
        for line in element.text.strip().splitlines():
            try:
                id_, id3, *names = line.split()
            except ValueError:
                # TODO: Fix this.
                raise TypeError('TODO')
            satellites_.append(ast.SatelliteID(id_, id3, set(names)))
        return ast.Satellites(*satellites_)

    return p.tag('satellites') ^ process


def cycles(string: str) -> Cycles:
    try:
        return Cycles(*(int(s) for s in string.split()))
    except TypeError:
        num_cycles = len(string.split())
        if num_cycles == 0:
            raise TypeError("missing 'first' cycle")
        if num_cycles == 1:
            raise TypeError("missing 'last' cycle")
        raise TypeError(
            "too many cycles given, expected only 'first' and 'last'")


def phase_statements() -> p.Parser:
    def process(statements: Sequence[ast.Statement]) -> ast.Statement:
        # flatten if only a single statement
        if len(statements) == 1:
            return statements[0]
        return ast.CompoundStatement(*statements)

    statements = p.star(
        value(str, 'mission') |
        value(cycles, 'cycles') |
        value(str, 'repeat') |
        value(str, 'ref_pass') |
        value(str, 'start_time') |
        value(str, 'subcycles')
    )
    return (p.start() + (statements ^ process) + p.end()
            << 'Invalid configuration block or value.') ^ (lambda x: x[1])


def phase() -> p.Parser:
    def process(element: Element) -> ast.Phase:
        try:
            name = element.attributes['name']
        except KeyError:
            raise p.GlobalParseFailure(
                element.file, element.opening_line,
                "<phase> has no 'name' attribute.")
        try:
            statement = cast(ast.Statement, phase_statements()(
                element.down())[0])
        except StopIteration:
            statement = ast.NullStatement()
        condition = parse_condition(element.attributes)
        return ast.Phase(name, statement, condition)

    return p.tag('phase') ^ process


def root_statements() -> p.Parser:
    def process(statements: Sequence[ast.Statement]) -> ast.Statement:
        # flatten if only a single statement
        if len(statements) == 1:
            return statements[0]
        return ast.CompoundStatement(*statements)

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


def parse(root: Element) -> ast.Statement:
    return cast(ast.Statement, root_statements()(root.down())[0])


def preparse(root: Element) -> ast.Statement:
    def process(elements: Sequence[Element]) -> Element:
        return elements[-1]

    parser = p.until(p.tag('satellites')) + satellites() ^ process
    return cast(ast.Statement, parser(root.down())[0])
