"""Type aliases."""

import os
import numbers
from typing import Union, IO, Any, TYPE_CHECKING

__all__ = ['PathLike', 'PathOrFile', 'Real']


if TYPE_CHECKING:
    # pylint: disable=unsubscriptable-object
    PathLike = Union[str, bytes, os.PathLike[Any]]
else:
    PathLike = Union[str, bytes, os.PathLike]

PathOrFile = Union[PathLike, IO[Any], int]


Real = Union[int, float, numbers.Real]
