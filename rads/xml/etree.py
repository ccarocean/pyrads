"""XML tools using :mod:`xml.etree.ElementTree`."""

import xml.etree.ElementTree as etree
from typing import Iterator, Mapping, Optional
from xml.etree.ElementTree import (
    ParseError,
    XMLParser,
    fromstring,
    fromstringlist,
    parse,
)

from ..xml import base

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
    """XML element that encapsulates an element from the ElementTree module.

    Does not support line number examination.

    .. note::

        It is recommended to use :class:`rads.xml.lxml.Element` if libxml is
        available on your system as the etree version does not support line
        numbers which can make debugging XML files for syntax errors more
        difficult.
    """

    def __init__(
        self,
        element: etree.Element,
        *,
        index: Optional[int] = None,
        parent: Optional["Element"] = None,
        file: Optional[str] = None,
    ):
        """
        :param element:
            XML element from the standard :mod:`xml.etree.ElementTree`
            package.
        :param index:
            Index of element at current level, among it's siblings. Not
            required if this element does not have any siblings.
        :param parent:
            The parent of this element.
        :param file:
            Filename of the XML document.
        """
        assert parent is None or isinstance(parent, Element)
        self._element = element
        self._index = index
        self._parent = parent
        self._file = file

    def __len__(self) -> int:
        return len(self._element)

    def __iter__(self) -> Iterator["Element"]:
        for i, e in enumerate(self._element):
            yield Element(e, index=i, parent=self, file=self._file)

    def next(self) -> "Element":  # noqa: D102
        if self._parent is None or self._index is None:
            raise StopIteration()
        siblings = list(self._parent._element)
        new_index = self._index + 1
        if new_index >= len(siblings):
            raise StopIteration()
        return Element(
            siblings[new_index], index=new_index, parent=self._parent, file=self._file
        )

    def prev(self) -> "Element":  # noqa: D102
        if self._parent is None or self._index is None:
            raise StopIteration()
        siblings = list(self._parent._element)
        new_index = self._index - 1
        if new_index < 0:
            raise StopIteration()
        return Element(
            siblings[new_index], index=new_index, parent=self._parent, file=self._file
        )

    def up(self) -> "Element":  # noqa: D102
        if self._parent is None:
            raise StopIteration()
        return self._parent

    def down(self) -> "Element":  # noqa: D102
        try:
            element = list(self._element)[0]
            return Element(element, index=0, parent=self, file=self.file)
        except IndexError:
            raise StopIteration()

    @property
    def file(self) -> Optional[str]:
        return self._file

    @property
    def tag(self) -> str:
        return self._element.tag

    @property
    def text(self) -> Optional[str]:
        return self._element.text

    @property
    def attributes(self) -> Mapping[str, str]:
        return self._element.attrib


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
    new_error = type(error)(
        error.msg, (file, error.position[0], error.position[1], error.text)
    )
    new_error.code = error.code
    new_error.position = error.position
    return new_error
