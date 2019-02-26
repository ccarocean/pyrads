"""XML library, specifically for reading the RADS's configuration files.

This includes :class:`Element` which allows easy traversal of the XML tree new
versions of the :func:`parse`, :func:`fromstring` and :func:`fromstringlist`
functions for parsing and XML document that return :class:`Element`.  These
functions also and support XML documents without a root element, such as the
RADS v4 configuration file.
"""

try:
    from .lxml import Element
except ImportError:
    # TODO: Remove 'ignore' when https://github.com/python/mypy/issues/1153 is
    #  fixed.
    from .etree import Element  # type: ignore

from .utility import parse, fromstring, fromstringlist

__all__ = ['Element', 'parse', 'fromstring', 'fromstringlist']
