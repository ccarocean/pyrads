"""Special XML library that can handle rootless XML files."""

from itertools import tee, takewhile, dropwhile, chain
import xml.etree.ElementTree as ET


__all__ = ['wrap_with_root', 'parse', 'fromstring', 'fromstringlist']


def wrap_with_root(lines):
    """Wraps XML document with <ROOT> tags.

    Parameters
    ----------
    lines : [bytes]
        Lines of XML document without a root element.

    Returns
    -------
    : iterable
        XML document with root element.
    """
    def is_prolog(text):
        return text.startswith(b'<?xml version')
    it1, it2 = tee(lines)
    prolog = takewhile(is_prolog, it1)
    body = dropwhile(is_prolog, it2)
    return chain(prolog, [b'<ROOT>'], body, [b'</ROOT>'])


def parse(source, parser=None):
    """Parse a rootless XML document into an element tree.

    Parameters
    ----------
    source : file-like or path-like
        Rootless XML file to parse.
    parser :
        Optional parser instance, defaulting to XMLParser.

    Returns
    -------
    : Element
        Root of parsed XML tree.
    """
    close_file = False
    if not hasattr(source, 'read'):
        source = open(source, 'rb')
        close_file = True
    try:
        return ET.fromstringlist(wrap_with_root(source), parser)
    finally:
        if close_file:
            source.close()


def fromstring(text, parser=None):
    """

    Parameters
    ----------
    text : bytes
        XML data
    parser :
        Optional parser instance, defaulting to XMLParser.

    Returns
    -------
    : Element
        Root of parsed XML tree.
    """
    return ET.fromstringlist(wrap_with_root(text.splitlines()), parser)


def fromstringlist(sequence, parser=None):
    """

    Parameters
    ----------
    sequence : [bytes]
        A sequence of byte strings to parse.
    parser :
        Optional parser instance, defaulting to XMLParser.

    Returns
    -------
    : Element
        Root of parsed XML tree.
    """
    return ET.fromstringlist(wrap_with_root(sequence), parser)
