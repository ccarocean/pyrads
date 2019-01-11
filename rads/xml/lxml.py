"""XML tools using the :mod:`lxml` library."""

from typing import cast, Mapping, Optional, Iterator
from lxml import etree  # type: ignore
from cached_property import cached_property  # type: ignore
import rads.xml.base as base


__all__ = ['Element']


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
