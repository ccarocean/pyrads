"""XML tools for reading the RADS's configuration files."""

import os
from typing import Optional, AnyStr, Sequence, cast, Any
from itertools import chain, tee, takewhile, dropwhile
from .._typing import PathOrFile, PathLike
from .._utility import ensure_open, filestring

try:
    from ..xml import lxml as xml
except ImportError:
    # TODO: Remove 'ignore' when https://github.com/python/mypy/issues/1153 is
    #  fixed.
    from ..xml import etree as xml  # type: ignore

__all__ = ['parse', 'fromstring', 'fromstringlist']


# TODO: Remove when ElementTree.parse accepts PathLike objects.
def _fix_source(source: PathOrFile) -> Any:
    if isinstance(source, int):
        return source
    if hasattr(source, 'read'):
        return source
    return os.fspath(cast(PathLike, source))


def _wrap_with_root_helper(
        sequence: Sequence[AnyStr],
        start_tag: AnyStr,
        end_tag: AnyStr,
        processing_instruction: AnyStr) -> Sequence[AnyStr]:
    def is_prolog(text: AnyStr) -> bool:
        return text.lstrip().startswith(processing_instruction)

    it1, it2 = tee(sequence)
    prolog = takewhile(is_prolog, it1)
    body = dropwhile(is_prolog, it2)
    return list(chain(prolog, [start_tag], body, [end_tag]))


def _wrap_with_root(sequence: Sequence[AnyStr]) -> Sequence[AnyStr]:
    if sequence and isinstance(sequence[0], bytes):
        return _wrap_with_root_helper(
            sequence, b'<rootless>', b'</rootless>', b'<?')
    if sequence and isinstance(sequence[0], str):
        return _wrap_with_root_helper(
            sequence, '<rootless>', '</rootless>', '<?')
    raise TypeError("'sequence' bust be a sequence of 'bytes' or 'str'")


def parse(source: PathOrFile,
          parser: Optional[xml.XMLParser] = None,
          rootless: bool = False) -> xml.Element:
    """Parse an XML document from a file or file-like object.

    Parameters
    ----------
    source
        File or file-like object containing the XML data.
    parser
        XML parser to use, defaults to the standard XMLParser, which is
        ElementTree compatible regardless of backend.
    rootless
        Set to True to parse an XML document that does not have a root.

        .. note::

            This is done by adding `<rootless>` tags around the document before
            parsing it.

    Returns
    -------
    xml.Element
        The root XML element.  If :paramref:`rootless` is True this will be the
        added `<rootless>` element

    """
    if rootless:
        with ensure_open(source) as file:
            return fromstringlist(
                file.readlines(), parser, rootless=True, file=filestring(file))
    return xml.Element(xml.parse(
        _fix_source(source), parser).getroot(), file=filestring(source))


def fromstring(text: AnyStr,
               parser: Optional[xml.XMLParser] = None,
               rootless: bool = False,
               file: Optional[str] = None) -> xml.Element:
    """Parse an XML document or section from a string constant.

    Parameters
    ----------
    text
        XML text to parse.
    parser
        XML parser to use, defaults to the standard XMLParser, which is
        ElementTree compatible regardless of backend.
    rootless
        Set to True to parse an XML document that does not have a root.

        .. note::

            This is done by adding `<rootless>` tags around the document before
            parsing it.
    file
        Optional filename to associate with the returned :class:`xml.Element`.

    Returns
    -------
    xml.Element
        The root XML element (of the section given in :paramref:`text`).  If
        :paramref:`rootless` is True this will be the added `<rootless>`
        element.

    """
    if rootless:
        return fromstringlist(text.splitlines(), parser, rootless=True)
    return xml.Element(xml.fromstring(text, parser), file=file)


def fromstringlist(sequence: Sequence[AnyStr],
                   parser: Optional[xml.XMLParser] = None,
                   rootless: bool = False,
                   file: Optional[str] = None) -> xml.Element:
    """Parse an XML document or section from a sequence of string fragments.

    Parameters
    ----------
    sequence
        String fragments containing the XML text to parse.
    parser
        XML parser to use, defaults to the standard XMLParser, which is
        ElementTree compatible regardless of backend.
    rootless
        Set to True to parse an XML document that does not have a root.

        .. note::

            This is done by adding `<rootless>` tags around the document before
            parsing it.
    file
        Optional filename to associate with the returned :class:`xml.Element`.

    Returns
    -------
    xml.Element
        The root XML element (of the section given in :paramref:`text`).  If
        :paramref:`rootless` is True this will be the added `<rootless>`
        element.

    """
    if rootless:
        sequence = _wrap_with_root(sequence)
    return xml.Element(xml.fromstringlist(sequence, parser), file=file)
