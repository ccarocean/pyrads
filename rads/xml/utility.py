"""XML tools for reading the RADS's configuration files."""

import os
import re
from itertools import chain, dropwhile, takewhile, tee
from typing import Any, Callable, Optional, Sequence, cast

from ..typing import PathLike, PathOrFile
from ..utility import ensure_open, filestring

try:
    from ..xml import lxml as xml
except ImportError:
    # TODO: Remove 'ignore' when https://github.com/python/mypy/issues/1153 is
    #  fixed.
    from ..xml import etree as xml  # type: ignore

__all__ = [
    "parse",
    "fromstring",
    "fromstringlist",
    "rads_fixer",
    "rootless_fixer",
    "is_empty",
    "strip_blanklines",
    "strip_comments",
    "strip_processing_instructions",
]


# TODO: Remove when ElementTree.parse accepts PathLike objects.
def _fix_source(source: PathOrFile) -> Any:
    if isinstance(source, int):
        return source
    if hasattr(source, "read"):
        return source
    return os.fspath(cast(PathLike, source))


def parse(
    source: PathOrFile,
    parser: Optional[xml.XMLParser] = None,
    fixer: Optional[Callable[[str], str]] = None,
) -> xml.Element:
    """Parse an XML document from a file or file-like object.

    :param source:
        File or file-like object containing the XML data.
    :param parser:
        XML parser to use, defaults to the standard XMLParser, which is
        ElementTree compatible regardless of backend.
    :param fixer:
        A function to pre-process the XML string.  This can be used to fix
        files during load.

    :return:
        The root XML element.  If `rootless` is True this will be the added
        `<rootless>` element
    """
    filename = filestring(source)
    if fixer:
        with ensure_open(source) as file:
            return fromstring(file.read(), parser=parser, fixer=fixer, file=filename)
    try:
        return xml.Element(
            xml.parse(_fix_source(source), parser).getroot(), file=filename
        )
    except xml.ParseError as err:
        if filename:
            raise xml.error_with_file(err, filename) from err
        raise


def fromstring(
    text: str,
    *,
    parser: Optional[xml.XMLParser] = None,
    fixer: Optional[Callable[[str], str]] = None,
    file: Optional[str] = None,
) -> xml.Element:
    """Parse an XML document or section from a string constant.

    :param text:
        XML text to parse.
    :param parser:
        XML parser to use, defaults to the standard XMLParser, which is
        ElementTree compatible regardless of backend.
    :param fixer:
        A function to pre-process the XML string.  This can be used to fix
        files during load.
    :param file:
        Optional filename to associate with the returned :class:`xml.Element`.

    :return:
        The root XML element (of the section given in `text`).  If `rootless`
        is True this will be the added `<rootless>` element.
    """
    if fixer is not None:
        text = fixer(text)
    try:
        return xml.Element(xml.fromstring(text, parser), file=file)
    # add file to error if known
    except xml.ParseError as err:
        if file:
            raise xml.error_with_file(err, file) from err
        raise


def fromstringlist(
    sequence: Sequence[str],
    parser: Optional[xml.XMLParser] = None,
    fixer: Optional[Callable[[str], str]] = None,
    file: Optional[str] = None,
) -> xml.Element:
    """Parse an XML document or section from a sequence of string fragments.

    :param sequence:
        String fragments containing the XML text to parse.
    :param parser:
        XML parser to use, defaults to the standard XMLParser, which is
        ElementTree compatible regardless of backend.
    :param fixer:
        A function to pre-process the XML string.  This can be used to fix
        files during load.  This will not be a string list but the full string
        with newlines.
    :param file:
        Optional filename to associate with the returned :class:`xml.Element`.

    :return:
        The root XML element (of the section given in `text`).  If `rootless`
        is True this will be the added `<rootless>` element.
    """
    if fixer is not None:
        return fromstring("\n".join(sequence), parser=parser, fixer=fixer, file=file)
    try:
        return xml.Element(xml.fromstringlist(sequence, parser), file=file)
    # add file to error if known
    except xml.ParseError as err:
        if file:
            raise xml.error_with_file(err, file) from err
        raise


def rads_fixer(text: str) -> str:
    """Fix XML problems with the upstream RADS XML configuration.

    This fixer is for problems that will not be fixed upstream or for which the
    fix has been delayed.  It is primary for making up the difference between
    the official RADS parser which is very lenient and the PyRADS parser which
    is very strict.

    Currently, this fixes the following bugs with the RADS config.

    * The RADS XML file does not have a root as dictated by the XML 1.0
      standard.  This is fixed by adding <__ROOTLESS__> tags around the entire
      file.  This is the only fix that is considered part of the RADS standard
      (that is RADS lies about it being XML 1.0).
    * The RADS MXL file has some instances of `int3` used in the <compress>
      tag.  This is an invalid type (there is no 3 byte integer) and in the
      official RADS implementation all invalid types default to `dble`
      (double).  However, The intended type here is `int4`.  This fix corrects
      this.

    :param text:
        RADS XML string to fix.
    :return:
        Repaired RADS XML string.
    """
    return rootless_fixer(text, preserve_empty=False).replace("int3", "int4")


def rootless_fixer(text: str, preserve_empty: bool = False) -> str:
    """Fix rootless XML files.

    Give this as the `fixer` argument in :func:`parse`, :func:`fromstring`, or
    :func:`fromstringlist` to load XML files that do not have a root tag.  This
    is done by adding a <__ROOTLESS__> block around the entire document.

    .. note:

        When `preserve_empty` is False this can also be used to detect empty
        XML files by first loading the file with this fixer and then using the
        :func:`rads.xml.Element.down()` method. If the original file was empty
        this will raise :class:`StopIteration`.

    :param text:
        XML text to wrap <__ROOTLESS__> tags around.
    :param preserve_empty:
        Set to False to skip adding <__ROOTLESS__> tags to an empty XML file.
        See :func:`is_empty` for the definition of *empty*.  In order to set
        this :func:`functools.partial` should be used.

    :return:
        The given `text` with <__ROOTLESS__> tags added (after beginning
        processing instructions).
    """
    if preserve_empty and is_empty(text):
        return text

    def is_prolog(text: str) -> bool:
        return text.lstrip().startswith("<?")

    it1, it2 = tee(text.splitlines())
    prolog = takewhile(is_prolog, it1)
    body = dropwhile(is_prolog, it2)
    return "\n".join(chain(prolog, ["<__ROOTLESS__>"], body, ["</__ROOTLESS__>"]))


def is_empty(text: str) -> bool:
    """Determine if XML string is empty.
    The XML string is considered empty if it only contains processing
    instructions and comments.
    :param text:
        XML text to check for being empty.

    :return:
        True if the given XML `text` is empty.
    """
    return (
        strip_blanklines(strip_comments(strip_processing_instructions(text))).strip()
        == ""
    )


def strip_comments(text: str) -> str:
    """Remove XML comments from a string.

    .. note::

        This will not remove lines that had comments, it only removes the text
        from "<!--" to "-->".

    :param text:
        XML text to strip comments from.

    :return:
        The given `text` without XML comments.
    """
    # thanks: https://stackoverflow.com/a/6806096
    return re.sub(r"(?s)<!--.*?-->", "", text)


def strip_processing_instructions(text: str) -> str:
    """Remove XML processing instructions from a string.

    .. note::

        This will not remove lines that had processing instructions, it only
        removes the text from "<?" to "?>".

    :param text:
        XML text to strip processing instructions from.

    :return:
        The given `text` without XML processing instructions.
    """
    return re.sub(r"(?s)<\?.*?\?>", "", text)


def strip_blanklines(text: str) -> str:
    """Remove blank lines from a string.

    Lines containing only whitespace characters are considered blank.

    :param text:
        String to remove blank lines from.

    :return:
        String without blank lines.
    """
    return "\n".join(line for line in text.splitlines() if line.strip())
