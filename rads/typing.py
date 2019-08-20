"""Type aliases."""

import os
from typing import IO, TYPE_CHECKING, Any, Union

import numpy as np  # type: ignore

__all__ = [
    "PathLike",
    "PathLikeOrFile",
    "PathOrFile",
    "Number",
    "IntOrArray",
    "NumberOrArray",
]

if TYPE_CHECKING:
    PathLike = Union[str, os.PathLike[str]]
    PathOrFile = Union[os.PathLike[str], IO[Any]]
else:
    PathLike = Union[str, os.PathLike]
    PathOrFile = Union[os.PathLike, IO[Any]]

PathLikeOrFile = Union[PathLike, IO[Any]]

# for the purpose of PyRADS bool as a number since it will act
# as 0 or 1 when used as a number
Number = Union[int, float, bool]

IntOrArray = Union[int, np.generic, np.ndarray]
NumberOrArray = Union[Number, np.generic, np.ndarray]
