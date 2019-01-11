"""XML tools using :mod:`xml.etree.ElementTree`."""

from typing import Optional, Mapping, Iterator
import xml.etree.ElementTree as etree
import rads.xml.base as base

__all__ = ['Element']


class Element(base.Element):
    """XML element that encapsulates an element from the ElementTree module.

    Does not support line number examination.


    .. note::

        It is recommended to use :class:`rads.xml.lxml.Element` if libxml is
        available on your system as this version does not support line numbers
        which can make debugging XML files for syntax errors more difficult.

    Parameters
    ----------
    element
        XML element from the standard :mod:`xml.etree.ElementTree`
        package.
    index
        Index of element at current level, among it's siblings. Not
        required if this element does not have any siblings.
    parent
        The parent of this element.
    file
        Filename of the XML document.

    """

    def __init__(self, element: etree.Element, index: Optional[int] = None,
                 parent: Optional['Element'] = None,
                 file: Optional[str] = None) -> None:
        assert parent is None or isinstance(parent, Element)
        self._element = element
        self._index = index
        self._parent = parent
        self._file = file

    def __iter__(self) -> Iterator['Element']:  # noqa: D105
        return (Element(e, i, self, self._file)
                for i, e in enumerate(self._element))

    def __len__(self) -> int:  # noqa: D105
        return len(self._element)

    def next(self) -> 'Element':  # noqa: D102
        if self._parent is None or self._index is None:
            raise StopIteration()
        siblings = list(self._parent._element)
        new_index = self._index + 1
        if new_index >= len(siblings):
            raise StopIteration()
        return Element(siblings[new_index], new_index,
                       self._parent, self._file)

    def prev(self) -> 'Element':  # noqa: D102
        if self._parent is None or self._index is None:
            raise StopIteration()
        siblings = list(self._parent._element)
        new_index = self._index - 1
        if new_index < 0:
            raise StopIteration()
        return Element(siblings[new_index], new_index,
                       self._parent, self._file)

    def up(self) -> 'Element':  # noqa: D102
        if self._parent is None:
            raise StopIteration()
        return self._parent

    def down(self) -> 'Element':  # noqa: D102
        try:
            element = list(self._element)[0]
            return Element(element, 0, self, self.file)
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
