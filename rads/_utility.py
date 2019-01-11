"""Utility functions."""

from typing import cast, IO, Optional, Any, Union
from wrapt import ObjectProxy  # type: ignore
from ._typing import PathOrFile, PathLike


class _NoCloseIOWrapper(ObjectProxy):  # type: ignore

    def __exit__(self, *args: object, **kwargs: object) -> None:
        pass

    def close(self) -> None:
        pass


def ensure_open(file: PathOrFile,
                mode: str = 'r',
                buffering: int = -1,
                encoding: Optional[str] = None,
                errors: Optional[str] = None,
                newline: Optional[str] = None,
                closefd: bool = True,
                closeio: bool = False) -> IO[Any]:
    """Open file or leave file-like object open.

    This function behaves identically to :func:`open` but can also accept a
    file-like object in the :paramref:`file` parameter.


    Parameters
    ----------
    file
        A path-like object giving the pathname (absolute or relative to the
        current working directory) of the file to be opened or an integer file
        descriptor of the file to be wrapped or a file-like object.

        .. note::

            If a file descriptor is given, it is closed when the
            returned I/O object is closed, unless :paramref:`closefd` is set to
            False.

        .. note::

            If a file-like object is given closing the returned I/O object will
            not close the given file unless :paramref:`closeio` is set to True.

    mode
        See :func:`open`
    buffering
        See :func:`open`
    encoding
        See :func:`open`
    errors
        See :func:`open`
    newline
        See :func:`open`
    closefd
        See :func:`open`
    closeio
        If set to True then if :paramref:`file` is a file like object it will
        be closed when either the __exit__ or close methods are called on the
        returned I/O object.  By default these methods will be ignored
        when :paramref:`file` is a file-like object.

    Returns
    -------
    I/O object
        An I/O object or the original file-like object if :paramref:`file` is a
        file-like object.  If this is the original file-like object and
        :paramref:`closeio` is set to False (the default) then it's close and
        __exit__ methods will be no-ops.

    .. seealso:: :func:`open`

    """
    if hasattr(file, 'read'):
        if closeio:
            return cast(IO[Any], _NoCloseIOWrapper(file))
        return cast(IO[Any], file)
    return open(cast(Union[PathLike, int], file), mode, buffering,
                encoding, errors, newline, closefd)
