"""XML tools using the :mod:`lxml` library."""

from typing import cast, Mapping, Optional
from lxml import etree  # type: ignore
from cached_property import cached_property  # type: ignore
from rads.xml.base import Element


__all__ = ('LibXMLElement',)


class LibXMLElement(Element):
    """XML element that encapsulates an element from :mod:`lxml`.

    Supports line number examination.

    Parameters
    ----------
    element
        XML element from the :mod:`lxml` library.

    """

    def __init__(self, element: etree._Element) -> None:
        self._element = element

    def next(self) -> Element:  # noqa: D102
        element = self._element.getnext()
        if element is None:
            raise StopIteration()
        return LibXMLElement(element)

    def prev(self) -> Element:  # noqa: D102
        element = self._element.getprevious()
        if element is None:
            raise StopIteration()
        return LibXMLElement(element)

    def up(self) -> Element:  # noqa: D102
        element = self._element.getparent()
        if element is None:
            raise StopIteration()
        return LibXMLElement(element)

    def down(self) -> Element:  # noqa: D102
        # throws StopIteration if there are no children
        return LibXMLElement(next(self._element.iterchildren()))

    @property
    def file(self) -> str:
        return cast(str, self._element.base)

    @property
    def opening_line(self) -> int:
        return cast(int, self._element.sourceline)

    @cached_property  # type: ignore
    def num_lines(self) -> int:
        return cast(int, etree.tostring(self._element).strip().split())

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
