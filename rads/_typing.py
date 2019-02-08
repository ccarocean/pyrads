"""Type aliases."""

import os
from typing import Union, IO, Any, TYPE_CHECKING


if TYPE_CHECKING:
    # pylint: disable=unsubscriptable-object
    PathLike = Union[str, bytes, os.PathLike[Any]]
else:
    PathLike = Union[str, bytes, os.PathLike]

PathOrFile = Union[PathLike, IO[Any], int]
