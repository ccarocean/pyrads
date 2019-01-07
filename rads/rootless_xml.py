"""Special XML library that can handle rootless XML files."""

import os  # pylint: disable=unused-import
from typing import Any  # pylint: disable=unused-import
from typing import Iterable, BinaryIO, Union, cast
from itertools import tee, takewhile, dropwhile, chain
import xml.etree.ElementTree as ET


__all__ = ['wrap_with_root', 'parse', 'fromstring', 'fromstringlist']

Path = Union[str, bytes, int, 'os.PathLike[Any]']


def wrap_with_root(lines: Iterable[bytes]) -> Iterable[bytes]:
    """Wrap XML document with <ROOT> tags.

    Parameters
    ----------
    lines : [bytes]
        Lines of XML document without a root element.

    Returns
    -------
    : iterable
        XML document with root element.

    """
    def is_prolog(text: bytes) -> bool:
        return text.startswith(b'<?xml version')
    it1, it2 = tee(lines)
    prolog = takewhile(is_prolog, it1)
    body = dropwhile(is_prolog, it2)
    return chain(prolog, [b'<ROOT>'], body, [b'</ROOT>'])


def parse(source: Union[Path, BinaryIO]) -> ET.Element:
    """Parse a rootless XML document into an element tree.

    Parameters
    ----------
    source : file-like or path-like
        Rootless XML file to parse.

    Returns
    -------
    : Element
        Root of parsed XML tree.

    """
    close_file = False
    data: BinaryIO
    if not hasattr(source, 'read'):
        data = open(cast(Path, source), 'rb')
        close_file = True
    else:
        data = cast(BinaryIO, source)
    try:
        return ET.fromstringlist(list(wrap_with_root(data.readlines())))
    finally:
        if close_file:
            data.close()


def fromstring(text: bytes) -> ET.Element:
    """Parse a rootless XML document from a string.

    Parameters
    ----------
    text : bytes
        XML data

    Returns
    -------
    : Element
        Root of parsed XML tree.

    """
    return ET.fromstringlist(list(wrap_with_root(text.splitlines())))


def fromstringlist(sequence: Iterable[bytes]) -> ET.Element:
    """Parse a rootless XML document from a list of strings.

    Parameters
    ----------
    sequence : [bytes]
        A sequence of byte strings to parse.

    Returns
    -------
    : Element
        Root of parsed XML tree.

    """
    return ET.fromstringlist(list(wrap_with_root(sequence)))
