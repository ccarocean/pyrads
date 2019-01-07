"""Generic XML tools, not relating to a specific backend."""

from abc import abstractmethod, ABC
from typing import Optional, Mapping


class Element(ABC):
    """A generic XML element.

    Base class of XML elements.
    """

    def __repr__(self) -> str:
        """Get text representation of the element.

        Returns
        -------
        str
            Opening tag of the element.

        """
        attributes = ' '.join('{:s}="{}"'.format(k, v)
                              for k, v in self.attributes.items())
        if attributes:
            attributes = ' ' + attributes
        return '<{:s}{:s}>'.format(self.tag, attributes)

    @abstractmethod
    def next(self) -> 'Element':
        """Get the next sibling element.

        Returns
        -------
        Element
            Next XML sibling element.

        Raises
        ------
        StopIteration
            If there is no next sibling element.

        """

    @abstractmethod
    def prev(self) -> 'Element':
        """Get the previous sibling element.

        Returns
        -------
        Element
            Previous XML sibling element.

        Raises
        ------
        StopIteration
            If there is no previous sibling element.

        """

    @abstractmethod
    def up(self) -> 'Element':
        """Get the parent of this element.

        Returns
        -------
        Element
            Parent XML element.

        Raises
        ------
        StopIteration
            If there is no parent element.

        """

    @abstractmethod
    def down(self) -> 'Element':
        """Get the first child of this element.

        Returns
        -------
        Element
            First child XML element.

        Raises
        ------
        StopIteration
            If this element does not have any children.

        """

    @property
    def file(self) -> Optional[str]:
        """Get the name of the XML file containing this element.

        Returns
        -------
        str or None
            Name of the file containing this element, or None.

        """
        return None

    @property
    def opening_line(self) -> Optional[int]:
        """Get the opening line of the XML element.

        Returns
        -------
        int or None
            Opening line number, or None.

        """
        return None

    @property
    def num_lines(self) -> Optional[int]:
        """Get the number of lines making up the XML element.

        Returns
        -------
        int or None
            Number of lines in XML element, or None.

        """
        return None

    @property
    def closing_line(self) -> Optional[int]:
        """Get the closing line of the XML element.

        Returns
        -------
        int or None
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
