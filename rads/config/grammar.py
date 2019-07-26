from typing import (Any, Callable, Iterable, Mapping, Optional, Sequence,
                    Tuple)

import fortran_format_converter as ffc

from .ast import (Alias, Assignment, CompoundStatement, If, NullStatement,
                  Phase, SatelliteID, Satellites, Source, Statement, Variable)
from .text_parsers import (TextParseError, lift, list_of, one_of, range_of,
                           compress, cycles, data, nop, ref_pass, repeat, time,
                           unit)
from .tree import SubCycles
from .utility import (error_at, source_from_element, parse_action,
                      parse_condition, named_block_processor)
from .xml_parsers import (GlobalParseFailure, Apply, Parser, any, end, lazy,
                          opt, star, start, tag, until)
from ..xml.base import Element


def alias() -> Parser:
    def process(element: Element) -> Alias:
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
        return Alias(alias, variables, condition, action, source=source)

    return tag('alias') ^ process


def ignore(tag_: Optional[str] = None) -> Parser:
    def process(element: Element) -> NullStatement:
        return NullStatement(source=source_from_element(element))

    if tag_:
        return tag(tag_) ^ process
    return any() ^ process


def value(parser: Callable[[str, Mapping[str, str]], Any] = nop,
          tag_: Optional[str] = None,
          var: Optional[str] = None) -> Parser:
    def process(element: Element) -> Assignment:
        var_ = var if var else element.tag
        condition = parse_condition(element.attributes)
        action = parse_action(element)
        text = element.text if element.text else ''
        source = source_from_element(element)
        try:
            value = parser(text, element.attributes)
        except TextParseError as err:
            raise error_at(element)(str(err)) from err
        return Assignment(name=var_, value=value, condition=condition,
                          action=action, source=source)

    if tag_:
        return tag(tag_) ^ process
    return any() ^ process


def if_statement(internal: Parser) -> Parser:
    def process(statements: Tuple[Element, Statement]) -> Statement:
        if_element, false_statement = statements
        condition = parse_condition(if_element.attributes)
        true_statement = internal(if_element.down())[0]
        source = source_from_element(if_element)
        return If(condition=condition,
                  true_statement=true_statement,
                  false_statement=false_statement,
                  source=source)

    return tag('if') + opt(
        elseif_statement(internal) | else_statement(internal)) ^ process


def elseif_statement(internal: Parser) -> Parser:
    def process(statements: Iterable[Any]) -> Statement:
        elseif_element, false_statement = statements
        condition = parse_condition(elseif_element.attributes)
        true_statement = internal(elseif_element.down())[0]
        source = source_from_element(elseif_element)
        return If(condition, true_statement, false_statement, source=source)

    return Apply(tag('elseif') + opt(
        lazy(lambda: elseif_statement(internal)) | else_statement(
            internal)), process)


def else_statement(internal: Parser) -> Parser:
    def process(element: Element) -> Any:
        return internal(element.down())[0]

    return tag('else') ^ process


def satellites() -> Parser:
    def process(element: Element) -> Satellites:
        source = source_from_element(element)
        if not element.text:
            return Satellites(source=source)
        satellites_ = []
        for num, line in enumerate(element.text.strip().splitlines()):
            line = line.strip()
            if line:
                id_source = Source(
                    line=element.opening_line + num + 1,
                    file=element.file)
                try:
                    id_, id3, *names = line.split()
                except ValueError:
                    raise GlobalParseFailure(
                        id_source.file, id_source.line,
                        f"missing 3 character ID for satellite '{id_}'")
                satellites_.append(
                    SatelliteID(id_, id3, set(names), source=id_source))
        return Satellites(*satellites_, source=source)

    return tag('satellites') ^ process


def satellite_ids() -> Parser:
    def process(element: Element) -> Assignment:
        source = source_from_element(element)
        if not element.text:
            return Assignment(name='satellites', value=[], source=source)
        satellites_ = []
        for num, line in enumerate(element.text.strip().splitlines()):
            line = line.strip()
            if line:
                id_source = Source(
                    line=element.opening_line + num + 1,
                    file=element.file)
                id_, *_ = line.split()
                if len(id_) != 2:
                    raise GlobalParseFailure(
                        id_source.file, id_source.line,
                        'satellite id must be exactly 2 characters, found '
                        f"'{id_}'")
                satellites_.append(id_)
        return Assignment(name='satellites', value=satellites_, source=source)

    return tag('satellites') ^ process


def subcycles() -> Parser:
    def process(element: Element) -> Statement:
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
        value = SubCycles(lengths, start=start)
        return Assignment(name='subcycles', value=value, condition=condition,
                          action=action, source=source)

    return tag('subcycles') ^ process


def block(parser: Parser,
          error_msg: str = 'Invalid configuration block or value.') -> Parser:
    def process(statements: Sequence[Statement]) -> Statement:
        # flatten if only a single statement
        if len(statements) == 1:
            return statements[0]
        return CompoundStatement(*statements)

    def recursive_parser() -> Parser:
        return block(parser, error_msg)

    block_parser = star(
        parser | if_statement(lazy(recursive_parser)) | value())
    return (start() + (block_parser ^ process) + end()
            << error_msg) ^ (lambda x: x[1])


def phase() -> Parser:
    phase_block = block(
        value(lift(str), 'mission') |
        value(cycles, 'cycles') |
        value(repeat, 'repeat') |
        value(ref_pass, 'ref_pass', var='reference_pass') |
        value(time, 'start_time') |
        value(time, 'end_time') |
        subcycles()
    )
    process = named_block_processor('phase', phase_block, Phase)
    return tag('phase') ^ process


def variable() -> Parser:
    # NOTE: These must be duplicated in the variable_overrides below.
    variable_block = block(
        value(lift(str), 'long_name', var='name') |
        value(lift(str), 'standard_name') |
        value(lift(str), 'source') |
        value(lift(str), 'comment') |
        value(unit, 'units') |
        value(list_of(lift(str)), 'flag_values') |
        value(list_of(lift(str)), 'flag_masks') |
        value(range_of(one_of((lift(int), lift(float)))), 'limits') |
        value(range_of(one_of((lift(int), lift(float)))), 'plot_range') |
        # used by rads for database generation, has no effect on end users
        ignore('parameters') |
        value(data, 'data') |
        value(list_of(lift(str)), 'quality_flag') |
        # not currently used
        value(lift(int), 'dimensions') |
        value(lift(ffc.convert), 'format') |
        value(compress, 'compress') |
        value(one_of((lift(int), lift(float))), 'default')
    )
    process = named_block_processor('var', variable_block, Variable)
    return tag('var') ^ process


def variable_override(parser: Callable[[str, Mapping[str, str]], Any],
                      tag_: str,
                      var: Optional[str] = None) -> Parser:
    def process(element: Element) -> Variable:
        try:
            name = element.attributes['var']
        except KeyError:
            raise error_at(element)(f"<{tag_}> is missing 'var' attribute.")
        var_ = var if var else element.tag
        condition = parse_condition(element.attributes)
        action = parse_action(element)
        text = element.text if element.text else ''
        source = source_from_element(element)
        try:
            value = parser(text, element.attributes)
        except TextParseError as err:
            raise error_at(element)(str(err)) from err
        statement = Assignment(name=var_, value=value, action=action,
                               source=source)
        return Variable(name, statement, condition, source=source)

    return tag(tag_) ^ process


def variable_overrides() -> Parser:
    overrides = (
            variable_override(lift(str), 'long_name', var='name') |
            variable_override(lift(str), 'standard_name') |
            variable_override(lift(str), 'source') |
            variable_override(lift(str), 'comment') |
            variable_override(unit, 'units') |
            variable_override(list_of(lift(str)), 'flag_variable_overrides') |
            variable_override(list_of(lift(str)), 'flag_masks') |
            variable_override(
                range_of(one_of((lift(int), lift(float)))), 'limits') |
            variable_override(
                range_of(one_of((lift(int), lift(float)))), 'plot_range') |
            # used by rads for database generation, has no effect on end users
            ignore('parameters') |
            variable_override(data, 'data') |
            variable_override(list_of(lift(str)), 'quality_flag') |
            # not currently used
            variable_override(lift(int), 'dimensions') |
            variable_override(lift(ffc.convert), 'format') |
            variable_override(compress, 'compress') |
            variable_override(one_of((lift(int), lift(float))), 'default'))
    return overrides


def satellite_grammar() -> Parser:
    root_block = block(
        # ignore the global attributes
        ignore('global_attributes') |

        # satellite id/names table
        satellites() |

        # top level satellite parameters
        value(lift(str), 'satellite', var='name') |
        ignore('satid') |
        value(lift(float), 'dt1hz') |
        value(lift(float), 'inclination') |
        value(list_of(lift(float)), 'frequency') |
        ignore('xover_params') |

        # satellite phase
        phase() |

        # variable aliases
        alias() |

        # variables
        variable() |
        variable_overrides() |

        # pyrads specific tags
        ignore('dataroot') |
        ignore('blacklist')
    )
    return root_block


def pre_config_grammar() -> Parser:
    root_block = block(
        satellite_ids() |
        value(list_of(lift(str)), 'blacklist') |
        ignore()  # ignore everything else
    )
    return root_block


def dataroot_grammar() -> Parser:
    root_block = block(
        value(lift(str), 'dataroot') |
        ignore()  # ignore everything else
    )
    return root_block
