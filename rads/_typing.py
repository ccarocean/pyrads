"""Type aliases."""

import sys
from typing import Union, IO, Any, TYPE_CHECKING


# TODO: Remove when dropping Python 3.5 support.
if sys.version_info < (3, 6):
    PathLike = Union[str, bytes, int]
else:
    import os
    if TYPE_CHECKING:
        # pylint: disable=unsubscriptable-object
        PathLike = Union[str, bytes, os.PathLike[Any]]
    else:
        PathLike = Union[str, bytes, os.PathLike]

PathOrFile = Union[PathLike, IO[Any], int]
