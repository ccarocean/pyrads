"""Generic XML tools, not relating to a specific backend."""

from abc import ABC, abstractmethod
from collections.abc import Sized
from itertools import chain
from typing import Iterable, Iterator, Mapping, Optional, Union

__all__ = ["Element"]


class Element(Iterable["Element"], Sized, ABC):
    """A generic XML element.

    Base class of XML elements.
    """

    def __repr__(self) -> str:
        """Get text representation of the element.

        :return:
            Opening tag of the element.
        """
        attributes = " ".join(
            '{:s}="{}"'.format(k, v) for k, v in self.attributes.items()
        )
        if attributes:
            attributes = " " + attributes
        return "<{:s}{:s}>".format(self.tag, attributes)

    @abstractmethod
    def __iter__(self) -> Iterator["Element"]:
        """Get the children of this element.

        :return:
            An iterable to the children of this element in the same order as
            they are in the XML file.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Get number of children.

        :return:
            Number of children.
        """

    def dumps(
        self, *, indent: Optional[Union[int, str]] = None, _current_indent: str = ""
    ) -> str:
        """Get string representation of this element and all child elements.

        :param indent:
            Amount to indent each level.  Can be given as an int or a string.
            Defaults to 4 spaces.

        :return:
            String representation of this and all child elements.
        """
        attributes = ""
        text = ""
        children = ""
        closing_indent = ""
        multiline = False

        # compute next indent
        if isinstance(indent, int):
            next_indent = _current_indent + " " * indent
        elif isinstance(indent, str):
            next_indent = _current_indent + indent
        else:
            next_indent = _current_indent + "    "

        if self.attributes:
            attributes = " " + " ".join(
                '{:s}="{}"'.format(k, v) for k, v in self.attributes.items()
            )
        if self:  # has children
            children_ = (
                c.dumps(indent=indent, _current_indent=next_indent) for c in self
            )
            children = "\n".join(chain([""], children_, [""]))
            multiline = True
        if self.text and self.text.strip():
            text = self.text.rstrip()
            if "\n" in text:
                multiline = True
            if multiline:
                text = text + "\n"
        if multiline:
            closing_indent = _current_indent

        format_str = (
            "{_current_indent:s}<{tag:s}{attributes:s}>"
            "{text:s}{children:s}{closing_indent:s}</{tag:s}>"
        )
        text = format_str.format(
            _current_indent=_current_indent,
            tag=self.tag,
            attributes=attributes,
            text=text,
            children=children,
            closing_indent=closing_indent,
        )
        return text

    @abstractmethod
    def next(self) -> "Element":
        """Get the next sibling element.

        :return:
            Next XML sibling element.

        :raises StopIteration:

            If there is no next sibling element.
        """

    @abstractmethod
    def prev(self) -> "Element":
        """Get the previous sibling element.

        :return:
            Previous XML sibling element.

        :raises StopIteration:
            If there is no previous sibling element.
        """

    @abstractmethod
    def up(self) -> "Element":
        """Get the parent of this element.

        :return:
            Parent XML element.

        :raises StopIteration:
            If there is no parent element.
        """

    @abstractmethod
    def down(self) -> "Element":
        """Get the first child of this element.

        :return:
            First child XML element.

        :raises StopIteration:
            If this element does not have any children.
        """

    @property
    def file(self) -> Optional[str]:
        """Get the name of the XML file containing this element.

        :return:
            Name of the file containing this element, or None.
        """
        return None

    @property
    def opening_line(self) -> Optional[int]:
        """Get the opening line of the XML element.

        :return:
            Opening line number, or None.
        """
        return None

    @property
    def num_lines(self) -> Optional[int]:
        """Get the number of lines making up the XML element.

        :return:
            Number of lines in XML element, or None.
        """
        return None

    @property
    def closing_line(self) -> Optional[int]:
        """Get the closing line of the XML element.

        :return:
            Closing line number, or None.
        """
        return None

    @property
    @abstractmethod
    def tag(self) -> str:
        """Tag name of the element."""

    @property
    @abstractmethod
    def text(self) -> Optional[str]:
        """Internal text of the element."""

    @property
    @abstractmethod
    def attributes(self) -> Mapping[str, str]:
        """The attributes of the element, as a dictionary."""
