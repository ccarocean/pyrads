"""XML tools for reading the RADS's configuration files."""

import os
import sys
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


# TODO: Remove when Python 3.5 support is dropped.
if sys.version_info < (3, 6):
    def _fix_source(source: PathOrFile) -> Any:
        return source
else:
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
        return _wrap_with_root_helper(sequence, b'<ROOT>', b'</ROOT>', b'<?')
    if sequence and isinstance(sequence[0], str):
        return _wrap_with_root_helper(sequence, '<ROOT>', '</ROOT>', '<?')
    raise TypeError("'sequence' bust be a sequence of 'bytes' or 'str'")


def parse(source: PathOrFile,
          parser: Optional[xml.XMLParser] = None,
          rootless: bool = False) -> xml.Element:
    """TODO: Fill this in."""
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
    """TODO: Fill this in."""
    if rootless:
        return fromstringlist(text.splitlines(), parser, rootless=True)
    return xml.Element(xml.fromstring(text, parser), file=file)


def fromstringlist(sequence: Sequence[AnyStr],
                   parser: Optional[xml.XMLParser] = None,
                   rootless: bool = False,
                   file: Optional[str] = None) -> xml.Element:
    """TODO: Fill this in."""
    if rootless:
        sequence = _wrap_with_root(sequence)
    return xml.Element(xml.fromstringlist(sequence, parser), file=file)
