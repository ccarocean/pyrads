"""Utility functions for configuration parsing."""

from typing import Callable, Mapping, cast

from .ast import (
    Condition,
    NamedBlock,
    NullStatement,
    SatelliteCondition,
    Source,
    Statement,
    TrueCondition,
    ActionType,
    edit_append,
    replace,
    append,
    delete,
    merge,
)
from .xml_parsers import GlobalParseFailure, LocalParseFailure, Parser
from ..xml.base import Element

__all__ = [
    "error_at",
    "continue_from",
    "source_from_element",
    "parse_action",
    "parse_condition",
    "named_block_processor",
]


def error_at(element: Element) -> Callable[[str], GlobalParseFailure]:
    def error(message: str) -> GlobalParseFailure:
        return GlobalParseFailure(element.file, element.opening_line, message)

    return error


def continue_from(element: Element) -> Callable[[str], LocalParseFailure]:
    def error(message: str) -> LocalParseFailure:
        return LocalParseFailure(element.file, element.opening_line, message)

    return error


def source_from_element(element: Element):
    return Source(line=element.opening_line, file=element.file)


def parse_action(element: Element) -> ActionType:
    # edit is a special type of action for strings
    if "edit" in element.attributes:
        if element.attributes["edit"] == "append":
            return edit_append
        raise error_at(element)(f'invalid edit="{element.attributes["edit"]}"')
    # default action is replace
    action = element.attributes.get("action", "replace")
    if action == "replace":
        return replace
    if action == "append":
        return append
    if action == "delete":
        return delete
    if action == "merge":
        return merge
    raise error_at(element)('invalid action="{:s}".'.format(action))


def parse_condition(attr: Mapping[str, str]) -> Condition:
    # currently the only condition RADS uses is based on the satellite
    try:
        sat = attr["sat"].strip()
        return SatelliteCondition(
            satellites=set(sat.strip("!").split()), invert=sat.startswith("!")
        )
    except KeyError:
        return TrueCondition()


def named_block_processor(
    tag: str, parser: Parser, node: NamedBlock
) -> Callable[[Element], NamedBlock]:
    def process(element: Element) -> NamedBlock:
        try:
            name = element.attributes["name"]
        except KeyError:
            raise error_at(element)(f"<{tag}> is missing 'name' attribute.")
        try:
            statement = cast(Statement, parser(element.down())[0])
        except StopIteration:
            statement = NullStatement()
        condition = parse_condition(element.attributes)
        source = source_from_element(element)
        return node(name, statement, condition, source=source)

    return process
