"""XML tools using the :mod:`lxml` library."""

from typing import (Mapping, Optional, Iterator, Union, IO, Any, Text,
                    Sequence, cast)
from lxml import etree  # type: ignore
from lxml.etree import XMLParser, ETCompatXMLParser   # type: ignore
from cached_property import cached_property  # type: ignore
from ..xml import base


__all__ = ['Element', 'XMLParser', 'parse', 'fromstring', 'fromstringlist']


class Element(base.Element):
    """XML element that encapsulates an element from :mod:`lxml`.

    Supports line number examination.

    Parameters
    ----------
    element
        XML element from the :mod:`lxml` library.

    """

    def __init__(self, element: etree._Element, file: Optional[str] = None) \
            -> None:
        self._element = element
        self._file = file

    def __len__(self) -> int:  # noqa: D105
        return len(self._element)

    def __iter__(self) -> Iterator['Element']:  # noqa: D105
        return (Element(e, file=self._file) for e in self._element)

    def next(self) -> 'Element':  # noqa: D102
        element = self._element.getnext()
        if element is None:
            raise StopIteration()
        return Element(element, file=self._file)

    def prev(self) -> 'Element':  # noqa: D102
        element = self._element.getprevious()
        if element is None:
            raise StopIteration()
        return Element(element, file=self._file)

    def up(self) -> 'Element':  # noqa: D102
        element = self._element.getparent()
        if element is None:
            raise StopIteration()
        return Element(element, file=self._file)

    def down(self) -> 'Element':  # noqa: D102
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

    @cached_property  # type: ignore
    def num_lines(self) -> int:
        return len(etree.tostring(self._element).strip().split())

    @cached_property  # type: ignore
    def closing_line(self) -> int:
        return cast(int, self.opening_line + self.num_lines - 1)

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


def parse(source: _FileOrFilename, parser: Optional[XMLParser] = None) \
        -> etree._ElementTree:
    """Parse XML document into element tree.

    This is wrapper around :func:`lxml.etree.parse` to make it behave like
    :func:`xml.etree.ElementTree.parse`.

    Parameters
    ----------
    source
        Filename or file object containing XML data.
    parser
        Optional parser instance, defaulting to
        :class:`lxml.etree.ETCompatXMLParser`.

    Returns
    -------
    _ElementTree
        An ElementTree instance.

    """
    if parser is None:
        parser = ETCompatXMLParser()
    return etree.parse(source, parser)


def fromstring(text: _ParserInputType, parser: Optional[XMLParser] = None) \
        -> etree._Element:
    """Parse XML document from string constant.

    This function can be used to embed 'XML Literals' in Python code.

    This is wrapper around :func:`lxml.etree.fromstring` to make it behave like
    :func:`xml.etree.ElementTree.fromtstring`.

    Parameters
    ----------
    text
        A string containing XML data.
    parser
        Optional parser instance, defaulting to
        :class:`lxml.etree.ETCompatXMLParser`.

    Returns
    -------
    _Element
        An Element instance.

    """
    if parser is None:
        parser = ETCompatXMLParser()
    return etree.fromstring(text, parser)


def fromstringlist(sequence: Sequence[_ParserInputType],
                   parser: Optional[XMLParser] = ...) -> etree._Element:
    """Parse XML document from sequence of string fragments.

    Parameters
    ----------
    sequence
        A list or other sequence of strings containing XML data.
    parser
        Optional parser instance, defaulting to
        :class:`lxml.etree.ETCompatXMLParser`.

    Returns
    -------
    _Element
        An Element instance.

    """
    if parser is None:
        parser = ETCompatXMLParser()
    return etree.fromstringlist(sequence, parser)
