from datetime import datetime
from typing import (Any, Optional, Callable, Mapping, Sequence, Tuple,
                    Iterable, TypeVar, List, cast)
from numbers import Number

import numpy as np

import rads.config.ast as ast
import rads.config.parsers as p
from .tree import (Cycles, Compress, Repeat, ReferencePass, SubCycles,
                   Unit, Range)
from ..xml.base import Element

T = TypeVar('T')


def nop(value: T) -> T:
    return value


def types(parsers: Sequence[Callable[[str], Any]]) \
        -> Callable[[str], Any]:
    parser_types = ', '.join(parser.__qualname__ for parser in parsers)

    def _parser(string: str) -> Any:
        for parser in parsers:
            try:
                return parser(string)
            except (TypeError, ValueError):
                pass

        raise TypeError(f"cannot convert '{string}' to any of the following "
                        f"types: {parser_types}")

    return _parser


def source_from_element(element: Element):
    return ast.Source(line=element.opening_line, file=element.file)


def error_at(element: Element) -> Callable[[str], p.GlobalParseFailure]:
    def error(message: str) -> p.GlobalParseFailure:
        return p.GlobalParseFailure(
            element.file, element.opening_line, message)

    return error


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
        return ast.replace
    if action == 'noreplace':
        return ast.keep
    if action == 'append':
        return ast.append
    raise error_at(element)('Invalid action="{:s}".'.format(action))


def list_of(parser: Callable[[str], T]) -> Callable[[str], List[T]]:
    def _parser(string: str) -> List[T]:
        return [parser(s) for s in string.split()]

    return _parser


def alias() -> p.Parser:
    def process(element: Element) -> ast.Alias:
        try:
            alias = element.attributes['name']
        except KeyError:
            raise error_at(element)("'name' attribute missing from <alias>")
        variables = element.text.split() if element.text else []
        if not variables:
            raise error_at(element)('<alias> cannot be empty')
        condition = parse_condition(element.attributes)
        action = parse_action(element)
        source = source_from_element(element)
        return ast.Alias(alias, variables, condition, action, source=source)

    return p.tag('alias') ^ process


def ignore(tag: Optional[str] = None) -> p.Parser:
    def process(element: Element) -> ast.NullStatement:
        return ast.NullStatement(source=source_from_element(element))

    if tag:
        return p.tag(tag) ^ process
    return p.any() ^ process


def value(parser: Callable[[str], Any] = nop, tag: Optional[str] = None,
          var: Optional[str] = None) -> p.Parser:
    def process(element: Element) -> ast.Assignment:
        var_ = var if var else element.tag
        condition = parse_condition(element.attributes)
        action = parse_action(element)
        text = element.text if element.text else ''
        source = source_from_element(element)
        try:
            return ast.Assignment(
                name=var_,
                value=parser(text),
                condition=condition,
                action=action,
                source=source)
        except (ValueError, TypeError) as err:
            raise error_at(element)(str(err))

    if tag:
        return p.tag(tag) ^ process
    return p.any() ^ process


def if_statement(internal: p.Parser) -> p.Parser:
    def process(statements: Tuple[Element, ast.Statement]) -> ast.Statement:
        if_element, false_statement = statements
        condition = parse_condition(if_element.attributes)
        true_statement = internal(if_element.down())[0]
        source = source_from_element(if_element)
        return ast.If(condition=condition,
                      true_statement=true_statement,
                      false_statement=false_statement,
                      source=source)

    return p.tag('if') + p.opt(
        elseif_statement(internal) | else_statement(internal)) ^ process


def elseif_statement(internal: p.Parser) -> p.Parser:
    def process(statements: Iterable[Any]) -> ast.Statement:
        elseif_element, false_statement = statements
        condition = parse_condition(elseif_element.attributes)
        true_statement = internal(elseif_element.down())[0]
        source = source_from_element(elseif_element)
        return ast.If(condition, true_statement, false_statement,
                      source=source)

    return p.Apply(p.tag('elseif') + p.opt(
        p.lazy(lambda: elseif_statement(internal)) | else_statement(
            internal)), process)


def else_statement(internal: p.Parser) -> p.Parser:
    def process(element: Element) -> Any:
        return internal(element.down())[0]

    return p.tag('else') ^ process


def satellites() -> p.Parser:
    def process(element: Element) -> ast.Satellites:
        source = source_from_element(element)
        if not element.text:
            return ast.Satellites(source=source)
        satellites_ = []
        for num, line in enumerate(element.text.strip().splitlines()):
            line = line.strip()
            if line:
                id_source = ast.Source(
                    line=element.opening_line + num + 1,
                    file=element.file)
                try:
                    id_, id3, *names = line.split()
                except ValueError:
                    raise p.GlobalParseFailure(
                        id_source.file, id_source.line,
                        f"missing 3 character ID for satellite '{id_}'")
                satellites_.append(
                    ast.SatelliteID(id_, id3, set(names), source=id_source))
        return ast.Satellites(*satellites_, source=source)

    return p.tag('satellites') ^ process


def cycles(cycles_string: str) -> Cycles:
    try:
        return Cycles(*(int(s) for s in cycles_string.split()))
    except TypeError:
        num_values = len(cycles_string.split())
        if num_values == 0:
            raise TypeError("missing 'first' cycle")
        if num_values == 1:
            raise TypeError("missing 'last' cycle")
        raise TypeError(
            "too many cycles given, expected only 'first' and 'last'")


def repeat(repeat_string: str) -> Repeat:
    parts = repeat_string.split()
    if len(parts) > 3:
        raise TypeError(
            "too many values given, expected only 'days', "
            "'passes', and 'unknown'")
    try:
        return Repeat(*(f(s) for f, s in zip((float, int, float), parts)))
    except TypeError:
        if parts:
            raise TypeError("missing length of repeat cycle in 'passes'")
        raise TypeError("missing length of repeat cycle in 'days'")


def time(time_string: str) -> datetime:
    try:
        return datetime.strptime(time_string, '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        try:
            return datetime.strptime(time_string, '%Y-%m-%dT%H:%M')
        except ValueError:
            try:
                return datetime.strptime(time_string, '%Y-%m-%dT%H')
            except ValueError:
                try:
                    return datetime.strptime(time_string, '%Y-%m-%dT')
                except ValueError:
                    try:
                        return datetime.strptime(time_string, '%Y-%m-%d')
                    except ValueError:
                        # required to avoid 'unconverted data' message from
                        # strptime
                        raise ValueError(
                            "time data '{:s}' does not match format "
                            "'%Y-%m-%dT%H:%M:%S'".format(time_string))


def ref_pass(ref_pass_string: str) -> ReferencePass:
    parts = ref_pass_string.split()
    if len(parts) > 5:
        raise TypeError("too many values given, expected only 'time', "
                        "'longitude', 'cycle number', 'pass number', and "
                        "optionally 'absolute orbit number'")
    try:
        funcs: Sequence[Callable[[str], Any]] = (time, float, int, int, int)
        return ReferencePass(*(f(s) for f, s in zip(funcs, parts)))
    except TypeError:
        if not parts:
            raise TypeError("missing 'time' of reference pass")
        if len(parts) == 1:
            raise TypeError("missing 'longitude' of reference pass")
        if len(parts) == 2:
            raise TypeError("missing 'cycle number' of reference pass")
        # len(parts) == 3
        raise TypeError("missing 'pass number' of reference pass")
        # absolute orbit number is defaulted in ReferencePass


def unit(unit_string) -> Unit:
    try:
        return Unit(unit_string)
    except ValueError:
        # TODO: Need better handling for dB and yymmddhhmmss units.
        return unit_string.strip()


def range_of(parser: Callable[[str], Number]) -> Callable[[str], Range]:
    def _parser(string: str) -> Range:
        min, max = [parser(s) for s in string.split()]
        return Range(min, max)
    return _parser


def subcycles() -> p.Parser:
    def process(element: Element) -> ast.Statement:
        start: Optional[int]
        condition = parse_condition(element.attributes)
        action = parse_action(element)
        try:
            start = int(element.attributes['start'])
        except KeyError:
            start = None
        except ValueError as err:
            raise error_at(element)(str(err))
        text = element.text if element.text else ''
        lengths = [int(s) for s in text.split()]
        source = source_from_element(element)
        return ast.Assignment(
            name='subcycles',
            value=SubCycles(lengths, start=start),
            condition=condition,
            action=action,
            source=source)

    return p.tag('subcycles') ^ process


def rads_type(type_string: str) -> type:
    switch = {
        'int1': np.int8,
        'int2': np.int16,
        'int4': np.int32,
        'real': np.float32,
        'dble': np.float64
    }
    try:
        return switch[type_string.lower()]
    except KeyError:
        raise TypeError(f"invalid type string '{type_string}'")


def compress(compress_string: str) -> Compress:
    parts = compress_string.split()
    if len(parts) > 3:
        raise TypeError(
            "too many values given, expected only 'type', "
            "'scale_factor', and 'add_offset'")
    try:
        return Compress(
            *(f(s)for f, s in zip((rads_type, float, float), parts)))
    except TypeError:
        raise TypeError("'missing 'type'")


def block(
        parser: p.Parser,
        error_msg: str = 'Invalid configuration block or value.') -> p.Parser:
    def process(statements: Sequence[ast.Statement]) -> ast.Statement:
        # flatten if only a single statement
        if len(statements) == 1:
            return statements[0]
        return ast.CompoundStatement(*statements)

    def recursive_parser() -> p.Parser:
        return block(parser, error_msg)

    block_parser = p.star(
        parser | if_statement(p.lazy(recursive_parser)) | value())
    return (p.start() + (block_parser ^ process) + p.end()
            << error_msg) ^ (lambda x: x[1])


def named_block_processor(tag: str, parser: p.Parser, node: ast.NamedBlock) \
        -> Callable[[Element], ast.NamedBlock]:
    def process(element: Element) -> ast.NamedBlock:
        try:
            name = element.attributes['name']
        except KeyError:
            raise error_at(element)(f"<{tag}> is missing 'name' attribute.")
        try:
            statement = cast(
                ast.Statement, parser(element.down())[0])
        except StopIteration:
            statement = ast.NullStatement()
        condition = parse_condition(element.attributes)
        source = source_from_element(element)
        return node(name, statement, condition, source=source)

    return process

def phase() -> p.Parser:
    phase_block = block(
        value(str, 'mission') |
        value(cycles, 'cycles') |
        value(repeat, 'repeat') |
        value(ref_pass, 'ref_pass', var='reference_pass') |
        value(time, 'start_time') |
        value(time, 'end_time') |
        subcycles()
    )
    process = named_block_processor('phase', phase_block, ast.Phase)
    return p.tag('phase') ^ process


def variable() -> p.Parser:
    variable_block = block(
        value(str, 'long_name', var='name') |
        value(str, 'standard_name') |
        value(str, 'source') |
        value(str, 'comment') |
        value(unit, 'units') |
        value(list_of(str), 'flag_values') |
        value(list_of(str), 'flag_masks') |
        value(range_of(types((int, float))), 'limits') |
        value(range_of(types((int, float))), 'plot_range') |
        # used by rads for database generation, has no effect on end users
        ignore('parameters') |
        ignore('data') |  # TODO: Complex field.
        value(list_of(str), 'quality_flag') |
        # not currently used
        value(int, 'dimensions') |
        ignore('format') |  # TODO: Complex field.
        value(compress, 'compress') |
        value(types((int, float)), 'default')
    )
    process = named_block_processor('var', variable_block, ast.Variable)
    return p.tag('var') ^ process


def parse(root: Element) -> ast.Statement:
    root_block = block(
        # ignore the global attributes
        ignore('global_attributes') |

        # satellite id/names table
        satellites() |

        # top level satellite parameters
        value(str, 'satellite', var='name') |
        ignore('satid') |
        value(float, 'dt1hz') |
        value(float, 'inclination') |
        value(list_of(float), 'frequency') |
        ignore('xover_params') |

        # satellite phase
        phase() |

        # variable aliases
        alias() |

        # variables
        variable()
    )
    return cast(ast.Statement, root_block(root.down())[0])


def preparse(root: Element) -> ast.Statement:
    def process(elements: Sequence[Element]) -> Element:
        return elements[-1]

    parser = p.until(p.tag('satellites')) + satellites() ^ process
    return cast(ast.Statement, parser(root.down())[0])
