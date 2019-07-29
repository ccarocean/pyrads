"""Type aliases."""

import os
from pathlib import Path
from typing import Union, IO, Any, TYPE_CHECKING

import numpy as np  # type: ignore

__all__ = ["PathLike", "PathOrFile", "Number", "IntOrArray", "NumberOrArray"]

if TYPE_CHECKING:
    # pylint: disable=unsubscriptable-object
    PathLike = Union[str, bytes, Path, os.PathLike[Any]]
else:
    PathLike = Union[str, bytes, Path, os.PathLike]

PathOrFile = Union[PathLike, IO[Any], int]

# for the purpose of PyRADS bool as a number since it will act
# as 0 or 1 when used as a number
Number = Union[int, float, bool]

IntOrArray = Union[int, np.generic, np.ndarray]
NumberOrArray = Union[Number, np.generic, np.ndarray]
