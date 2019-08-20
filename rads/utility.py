"""Utility functions."""

import os
from typing import IO, Any, List, Optional, Union, cast

from wrapt import ObjectProxy  # type: ignore

from .typing import PathLike, PathLikeOrFile

__all__ = [
    "ensure_open",
    "filestring",
    "xor",
    "contains_sublist",
    "merge_sublist",
    "delete_sublist",
    "fortran_float",
]


class _NoCloseIOWrapper(ObjectProxy):  # type: ignore
    def __exit__(self, *args: object, **kwargs: object) -> None:
        pass

    def close(self) -> None:
        pass


def ensure_open(
    file: PathLikeOrFile,
    mode: str = "r",
    buffering: int = -1,
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
    closefd: bool = True,
    closeio: bool = False,
) -> IO[Any]:
    """Open file or leave file-like object open.

    This function behaves identically to :func:`open` but can also accept a
    file-like object in the `file` parameter.

    :param file:
        A path-like object giving the pathname (absolute or relative to the
        current working directory) of the file to be opened or an integer file
        descriptor of the file to be wrapped or a file-like object.

        .. note::

            If a file descriptor is given, it is closed when the
            returned I/O object is closed, unless `closefd` is set to False.

        .. note::

            If a file-like object is given closing the returned I/O object will
            not close the given file unless `closeio` is set to True.
    :param mode:
        See :func:`open`
    :param buffering:
        See :func:`open`
    :param encoding:
        See :func:`open`
    :param errors:
        See :func:`open`
    :param newline:
        See :func:`open`
    :param closefd:
        See :func:`open`
    :param closeio:
        If set to True then if `file` is a file like object it will be closed
        when either the __exit__ or close methods are called on the returned
        I/O object.  By default these methods will be ignored when `file` is
        a file-like object.

    :return:
        An I/O object or the original file-like object if `file` is a file-like
        object.  If this is the original file-like object and `closeio` is set
        to False (the default) then it's close and __exit__ methods will be
        no-ops.

    .. seealso:: :func:`open`
    """
    if hasattr(file, "read"):
        if not closeio:
            return cast(IO[Any], _NoCloseIOWrapper(file))
        return cast(IO[Any], file)
    return open(
        cast(Union[PathLike, int], file),
        mode,
        buffering,
        encoding,
        errors,
        newline,
        closefd,
    )


def filestring(file: PathLikeOrFile) -> Optional[str]:
    """Convert a PathLikeOrFile to a string.

    :param file:
        file or file-like object to get the string for.

    :return:
        The string representation of the filename or path.  If it cannot get
        the name/path of the given file or file-like object or cannot convert
        it to a str, None will be returned.
    """
    if isinstance(file, int):
        return None
    if hasattr(file, "read"):
        try:
            return cast(IO[Any], file).name
        except AttributeError:
            return None
    if not isinstance(file, (str, bytes)):
        file = os.fspath(cast(PathLike, file))
    if isinstance(file, str):
        return file
    if isinstance(file, bytes):
        try:
            return cast(bytes, file).decode("utf-8")
        except UnicodeDecodeError:
            return None
    raise TypeError(f"'{type(file)}' is not a file like object")


def xor(a: bool, b: bool) -> bool:
    """Boolean XOR operator.

    This implements the XOR boolean operator and has the following truth table:

    ===== ===== =======
    a       b   a XOR b
    ===== ===== =======
    True  True  False
    True  False True
    False True  True
    False False False
    ===== ===== =======

    :param a:
        First boolean value.
    :param b:
        Second boolean value.

    :return:
        The result of `a` XOR `b` from the truth table above.
    """
    return (a and not b) or (not a and b)


def contains_sublist(list_: List[Any], sublist: List[Any]) -> bool:
    """Determine if a `list` contains a `sublist`.

    :param list_:
        list to search for the `sublist` in.
    :param sublist:
        Sub list to search for.

    :return:
        True if `list` contains `sublist`.

    """
    # Adapted from: https://stackoverflow.com/a/12576755
    if not sublist:
        return False
    for i in range(len(list_)):
        if list_[i] == sublist[0] and list_[i : i + len(sublist)] == sublist:
            return True
    return False


def merge_sublist(list_: List[Any], sublist: List[Any]) -> List[Any]:
    """Merge a `sublist` into a given `list_`.

    :param list_:
        List to merge `sublist` into.
    :param sublist:
        Sublist to merge into `list_`

    :return:
        A copy of `list_` with `sublist` at the end if `sublist` is not a
        sublist of `list_`.  Otherwise, a copy of `list_` is returned
        unchanged.
    """
    if contains_sublist(list_, sublist):
        return list_[:]
    return list_ + sublist


def delete_sublist(list_: List[Any], sublist: List[Any]) -> List[Any]:
    """Remove a `sublist` from the given `list_`.

    :param list_:
        List to remove the `sublist` from.
    :param sublist:
        Sublist to remove from `list_`.

    :return:
        A copy of `list_` with the `sublist` removed.
    """
    if not sublist:
        return list_[:]
    for i in range(len(list_)):
        if list_[i] == sublist[0] and list_[i : i + len(sublist)] == sublist:
            return list_[:i] + list_[i + len(sublist) :]
    return list_[:]


def fortran_float(string: str) -> float:
    """Construct :class:`float` from Fortran style float strings.

    This function can convert strings to floats in all of the formats below:

        * ``3.14e10``  (also parsable with :class:`float`)
        * ``3.14E10``  (also parsable with :class:`float`)
        * ``3.14d10``
        * ``3.14D10``
        * ``3.14e+10``  (also parsable with :class:`float`)
        * ``3.14E+10``  (also parsable with :class:`float`)
        * ``3.14d+10``
        * ``3.14D+10``
        * ``3.14e-10``  (also parsable with :class:`float`)
        * ``3.14E-10``  (also parsable with :class:`float`)
        * ``3.14d-10``
        * ``3.14D-10``
        * ``3.14+100``
        * ``3.14-100``

    .. note::

        Because RADS was written in Fortran, exponent characters in
        configuration and passindex files sometimes use 'D' or 'd' as
        the exponent separator instead of 'E' or 'e'.

    .. warning::

        If you are Fortran developer stop using 'Ew.d' and 'Ew.dDe' formats
        and use 'Ew.dEe' instead.  The first two are not commonly supported
        by other languages while the last version is the standard for nearly
        all languages.  Ok, rant over.

    :param string:
        String to attempt to convert to a float.

    :return:
        The float parsed from the given `string`.

    :raises ValueError:
        If `string` does not represent a valid float.
    """
    try:
        return float(string)
    except ValueError as err:
        try:
            return float(string.replace("d", "e").replace("D", "E"))
        except ValueError:
            try:
                return float(string.replace("+", "e+").replace("-", "e-"))
            except ValueError:
                raise err
