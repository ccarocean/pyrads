"""Type aliases."""

import numbers
import os
from typing import Union, IO, Any, TYPE_CHECKING

import numpy as np  # type: ignore

__all__ = ['PathLike', 'PathOrFile', 'Real', 'Number', 'NumberOrArray']

if TYPE_CHECKING:
    # pylint: disable=unsubscriptable-object
    PathLike = Union[str, bytes, os.PathLike[Any]]
else:
    PathLike = Union[str, bytes, os.PathLike]

PathOrFile = Union[PathLike, IO[Any], int]

Real = Union[int, float, numbers.Real]

# for the purpose of PyRADS bool as a number since it will act
# as 0 or 1 when used as a number
Number = Union[Real, bool]

NumberOrArray = Union[Number, np.generic, np.ndarray]
