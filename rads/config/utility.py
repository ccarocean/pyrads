"""Utility functions for configuration parsing."""

from typing import Callable, Mapping, Type, cast

from ..xml.base import Element
from .ast import (
    ActionType,
    Condition,
    NamedBlock,
    NullStatement,
    SatelliteCondition,
    Source,
    Statement,
    TrueCondition,
    append,
    delete,
    edit_append,
    merge,
    replace,
)
from .xml_parsers import Parser, TerminalXMLParseError, XMLParseError

__all__ = [
    "error_at",
    "continue_from",
    "source_from_element",
    "parse_action",
    "parse_condition",
    "named_block_processor",
]


def error_at(element: Element) -> Callable[[str], TerminalXMLParseError]:
    """Make function to generate `TerminalXMLParseError` from XML element.

    :param element:
        The XML element to generate the failure at.

    :return:
        A function that takes an error message and returns a
        :class:`rads.config.xml_parsers.TerminalXMLParseError` at the given XML
        `element`.
    """

    def error(message: str) -> TerminalXMLParseError:
        return TerminalXMLParseError(element.file, element.opening_line, message)

    return error


def continue_from(element: Element) -> Callable[[str], XMLParseError]:
    """Make function to generate `XMLParseError` from XML element.

    :param element:
        The XML element to generate the failure at.

    :return:
        A function that takes an error message and returns a
        :class:`rads.config.xml_parsers.XMLParseError` at the given XML
        `element`.
    """

    def error(message: str) -> XMLParseError:
        return XMLParseError(element.file, element.opening_line, message)

    return error


def source_from_element(element: Element) -> Source:
    """Get :class:`rads.config.ast.Source` from an XML element.

    :param element:
        The XML element to get the :class:`rads.config.ast.Source` from.

    :return:
        The source object giving the file and opening line of the element.
    """
    return Source(line=element.opening_line, file=element.file)


def parse_action(element: Element) -> ActionType:
    """Parse out the "action" attribute from the given XML element.

    This will return a function that when given the environment, the name of
    the value in the environment and the new value will handle the appropriate
    action.  The valid actions are:

        * "replace" - replace the new value with the old
        * "append" - append the new value to the old (only for :class:`str`,
          :class:`list`, and :class:`rads.rpn.Expression`).
        * "delete" - remove any portion of the old value that matches the new
          (only for :class:`str`, :class:`list` and
          :class:`rads.rpn.Expression`)
        * "merge" - append the new value to the old, only if it is not
          contained within the original value, (only for :class:`str`,
          :class:`list`, and :class:`rads.rpn.Expression`)

    The action is usually determined by the "action" attribute but for strings
    can be determined by the "edit" attribute which supports a single action
    of "append".  This is similar to the "append" above but with punctuation.

    If the given `element` does not contain an "action" or "edit" attribute
    "replace" is assumed.

    :param element:
        The XML element to parse the "action" attribute from.

    :return:
        A function to be called with an environment object (using attrs or a
        mapping), the name of the variable to set, and finally the new value
        that will perform one of the actions above.
    """
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
    """Parse out any conditions on element given it's attributes.

    .. note::

        Currently the only condition is "sat" which is a list of 2 character
        satellite identifiers which the element is to be used for (or not used
        for if the list begins with '!').  If the "sat" attribute is found a
        :class:`rads.config.ast.SatelliteCondition` will be returned.

    :param attr:
        The attributes of the XML element.

    :return:
        A condition that can be tested to see if the element should be used.
        If no conditions were found in the given attributes
        :class:`rads.config.ast.TrueCondition` is returned.
    """
    # currently the only condition RADS uses is based on the satellite
    try:
        sat = attr["sat"].strip()
        return SatelliteCondition(
            satellites=set(sat.strip("!").split()), invert=sat.startswith("!")
        )
    except KeyError:
        return TrueCondition()


def named_block_processor(
    parser: Parser, node: Type[NamedBlock]
) -> Callable[[Element], NamedBlock]:
    """Create a processor for a named XML block.

    This function will create a function that takes an XML element and returns
    a :class:`rads.config.ast.NamedBlock`.

    :param parser:
        The :class:`rads.config.xml_parsers.Parser` to use for parsing the
        tags within the named block.
    :param node:
        A subclass of :class:`rads.config.ast.NamedBlock` to use as the AST
        node generated by the parser.  The results of the `parser` will be
        stored in it's `inner_statement` attribute.

    :return:
        A function that when called on an XML element representing a named
        block will return the AST node for that block, parsed with the given
        `parser`.
    """

    def process(element: Element) -> NamedBlock:
        try:
            name = element.attributes["name"]
        except KeyError:
            raise error_at(element)(f"<{element.tag}> is missing 'name' attribute.")
        try:
            statement = cast(Statement, parser(element.down())[0])
        except StopIteration:
            statement = NullStatement()
        condition = parse_condition(element.attributes)
        source = source_from_element(element)
        return node(name, statement, condition, source=source)

    return process
