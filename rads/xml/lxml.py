"""XML tools using the lxml_ library.

.. _lxml: https://lxml.de/
"""

from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Text,
    Union,
    cast,
)

from lxml import etree  # type: ignore
from lxml.etree import ETCompatXMLParser, ParseError, XMLParser  # type: ignore

from ..xml import base

# TODO: Change to functools.cached_property when dropping support for
#       Python 3.7
if TYPE_CHECKING:
    # property behaves properly with Mypy but cached_property does not, even
    # with the same type stub.
    cached_property = property
else:
    from cached_property import cached_property

__all__ = [
    "ParseError",
    "Element",
    "XMLParser",
    "parse",
    "fromstring",
    "fromstringlist",
    "error_with_file",
]


class Element(base.Element):
    """XML element that encapsulates an element from lxml_.

    Supports line number examination.

    .. _lxml: https://lxml.de/
    """

    def __init__(self, element: etree._Element, file: Optional[str] = None):
        """
        :param:
            XML element from the lxml_ library.
        :param file:
            Optional filename/path the element is from.
        """
        self._element = element
        self._file = file

    def __len__(self) -> int:
        return len(self._element)

    def __iter__(self) -> Iterator["Element"]:
        return (Element(e, file=self._file) for e in self._element)

    def next(self) -> "Element":  # noqa: D102
        element = self._element.getnext()
        if element is None:
            raise StopIteration()
        return Element(element, file=self._file)

    def prev(self) -> "Element":  # noqa: D102
        element = self._element.getprevious()
        if element is None:
            raise StopIteration()
        return Element(element, file=self._file)

    def up(self) -> "Element":  # noqa: D102
        element = self._element.getparent()
        if element is None:
            raise StopIteration()
        return Element(element, file=self._file)

    def down(self) -> "Element":  # noqa: D102
        # throws StopIteration if there are no children
        return Element(next(self._element.iterchildren()), file=self._file)

    @property
    def file(self) -> str:
        if self._file:
            return self._file
        return cast(str, self._element.base)

    @property
    def opening_line(self) -> int:
        return cast(int, self._element.sourceline)

    @cached_property
    def num_lines(self) -> int:
        return len(etree.tostring(self._element).strip().split(b"\n"))

    @cached_property
    def closing_line(self) -> int:
        return self.opening_line + self.num_lines - 1

    @property
    def tag(self) -> str:
        return cast(str, self._element.tag)

    @property
    def text(self) -> Optional[str]:
        return cast(str, self._element.text)

    @property
    def attributes(self) -> Mapping[str, str]:
        return cast(Mapping[str, str], self._element.attrib)


_ParserInputType = Union[bytes, Text]
_FileOrFilename = Union[str, bytes, int, IO[Any]]


# The following functions are here to make lxml more compatible with etree.


def parse(
    source: _FileOrFilename, parser: Optional[XMLParser] = None
) -> etree._ElementTree:
    """Parse XML document into element tree.

    This is wrapper around :func:`lxml.etree.parse` to make it behave like
    :func:`xml.etree.ElementTree.parse`.

    :param source:
        Filename or file object containing XML data.
    :param parser:
        Optional parser instance, defaulting to
        :class:`lxml.etree.ETCompatXMLParser`.

    :return:
        An ElementTree instance.
    """
    if parser is None:
        parser = ETCompatXMLParser()
    return etree.parse(source, parser)


def fromstring(
    text: _ParserInputType, parser: Optional[XMLParser] = None
) -> etree._Element:
    """Parse XML document from string constant.

    This function can be used to embed 'XML Literals' in Python code.

    This is wrapper around :func:`lxml.etree.fromstring` to make it behave like
    :func:`xml.etree.ElementTree.fromtstring`.

    :param text:
        A string containing XML data.
    :param parser:
        Optional parser instance, defaulting to
        :class:`lxml.etree.ETCompatXMLParser`.

    :return:
        An Element instance.
    """
    if parser is None:
        parser = ETCompatXMLParser()
    return etree.fromstring(text, parser)


def fromstringlist(
    sequence: Sequence[_ParserInputType], parser: Optional[XMLParser] = None
) -> etree._Element:
    """Parse XML document from sequence of string fragments.

    :param sequence:
        A list or other sequence of strings containing XML data.
    :param parser:
        Optional parser instance, defaulting to
        :class:`lxml.etree.ETCompatXMLParser`.

    :return:
        An Element instance.
    """
    if parser is None:
        parser = ETCompatXMLParser()
    return etree.fromstringlist(sequence, parser)


def error_with_file(error: ParseError, file: str) -> ParseError:
    """Add filename to an XML parse error.

    :param error:
        Original XML parse error.
    :param file:
        Filename to add.

    :return:
        A new parse error (of the same type as `error`) with the `filename`
        added.
    """
    error.filename = file
    return type(error)(
        error.msg, error.code, error.position[0], error.position[1], file
    )
