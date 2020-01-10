"""Type aliases."""

import os
from typing import IO, TYPE_CHECKING, Any, TypeVar, Union

import numpy as np  # type: ignore
from typing_extensions import Protocol

__all__ = [
    "PathLike",
    "PathLikeOrFile",
    "PathOrFile",
    "IntOrArray",
    "FloatOrArray",
    "SupportsGetItem",
]

if TYPE_CHECKING:
    PathLike = Union[str, os.PathLike[str]]
    PathOrFile = Union[os.PathLike[str], IO[Any]]
else:
    PathLike = Union[str, os.PathLike]
    PathOrFile = Union[os.PathLike, IO[Any]]

PathLikeOrFile = Union[PathLike, IO[Any]]

IntOrArray = Union[int, np.generic, np.ndarray]
FloatOrArray = Union[float, np.generic, np.ndarray]

_K = TypeVar("_K", contravariant=True)
_V = TypeVar("_V", covariant=True)


class SupportsGetItem(Protocol[_K, _V]):
    def __getitem__(self, item: _K) -> _V:
        pass
