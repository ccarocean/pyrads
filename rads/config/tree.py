from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Integral, Real
from typing import Sequence, Mapping, Optional, Union, cast

import numpy as np  # type: ignore
from cf_units import Unit  # type: ignore

from .._typing import PathLike, Number, IntOrArray, NumberOrArray
from ..rpn import CompleteExpression

__all__ = ['PreConfig', 'Cycles', 'ReferencePass', 'Repeat', 'SubCycles',
           'Phase', 'Compress', 'Constant', 'Flags', 'MultiBitFlag',
           'SingleBitFlag', 'SurfaceType', 'Grid', 'NetCDFAttribute',
           'NetCDFVariable', 'Range', 'Variable', 'Satellite', 'Config']




@dataclass
class PreConfig:
    dataroot: PathLike
    config_files: Sequence[PathLike]
    satellites: Sequence[str]
    blacklist: Sequence[str] = field(default_factory=list)


@dataclass
class Cycles:
    """Cycle range 'inclusive'."""
    first: int
    last: int


@dataclass
class ReferencePass:
    time: datetime
    longitude: float
    cycle_number: int
    pass_number: int
    absolute_orbit_number: int = 1


@dataclass
class Repeat:
    """Length of the repeat cycle."""
    days: float
    passes: int
    longitude_drift: Optional[float] = None


@dataclass
class SubCycles:
    lengths: Sequence[int]
    start: Optional[int] = None


@dataclass
class Phase:
    id: str
    mission: str
    cycles: Cycles
    repeat: Repeat
    reference_pass: ReferencePass
    start_time: datetime
    end_time: Optional[datetime] = None
    subcycles: Optional[SubCycles] = None


@dataclass
class Compress:
    type: np.dtype
    scale_factor: Union[int, float] = 1
    add_offset: Union[int, float] = 0


@dataclass
class Constant:
    value: Union[int, float]


class Flags(ABC):

    @abstractmethod
    def extract(self, flags: NumberOrArray) -> NumberOrArray:
        pass


@dataclass
class MultiBitFlag(Flags):
    bit: int
    length: int

    def __post_init__(self) -> None:
        if not isinstance(self.bit, Integral):
            raise TypeError("'bit' must be an integer")
        if not isinstance(self.length, Integral):
            raise TypeError("'length' must be an integer")
        if self.bit < 0:
            raise ValueError("'bit' must be non-negative")
        if self.length < 2:
            raise ValueError("'length' must be 2 or greater")

    def extract(self, flags: IntOrArray) -> IntOrArray:
        result = (flags & ~(~0 << self.length) << self.bit) >> self.bit

        # if NumPy array cast down to smallest type
        if hasattr(result, 'astype'):
            if self.length <= 8:
                return cast(np.generic, result).astype(np.uint8)
            if self.length <= 16:
                return cast(np.generic, result).astype(np.uint16)
            if self.length <= 32:
                return cast(np.generic, result).astype(np.uint32)
            if self.length <= 64:  # pragma: no cover
                return cast(np.generic, result).astype(np.uint64)
        return result
        # can't reach this unless a larger integer is added to NumPy


@dataclass
class SingleBitFlag(Flags):
    bit: int

    def __post_init__(self) -> None:
        if not isinstance(self.bit, Integral):
            raise TypeError("'bit' must be an integer")
        if self.bit < 0:
            raise ValueError("'bit' must be non-negative")

    def extract(self, flags: IntOrArray) -> IntOrArray:
        return flags & (1 << self.bit) != 0


@dataclass
class SurfaceType(Flags):

    def extract(self, flags: IntOrArray) -> IntOrArray:
        # NOTE: Enum not used because meanings are defined in XML config file
        if isinstance(flags, np.ndarray):
            ice = (((flags & 0b100) >> 2) * 4).astype(np.uint8)
            land = (((flags & 0b10000) >> 4) * 3).astype(np.uint8)
            lake = (((flags & 0b100000) >> 5) * 2).astype(np.uint8)
            ocean = np.zeros(ice.shape, dtype=np.uint8)
            return np.max(np.stack((ice, land, lake, ocean)), axis=0)
        if (flags & 0b100) != 0:
            return 4  # continental ice
        if (flags & 0b10000) != 0:
            return 3  # land
        if (flags & 0b100000) != 0:
            return 2  # enclosed sea or lake
        return 0  # ocean


@dataclass
class Grid:
    file: str
    x: str = 'lon'
    y: str = 'lat'
    method: str = 'linear'


@dataclass
class NetCDFAttribute:
    name: str
    variable: Optional[str] = None
    branch: Optional[str] = None


@dataclass
class NetCDFVariable:
    name: str
    branch: Optional[str] = None


@dataclass
class Range:
    min: Real
    max: Real


@dataclass
class Variable:
    id: str
    name: str
    data: Union[Constant, CompleteExpression, Flags, Grid,
                NetCDFAttribute, NetCDFVariable]
    units: Union[Unit, str] = Unit('-')
    standard_name: Optional[str] = None
    source: str = ''
    comment: str = ''
    flag_values: Optional[Sequence[str]] = None
    flag_masks: Optional[Sequence[str]] = None
    limits: Optional[Range] = None
    plot_range: Optional[Range] = None
    quality_flag: Optional[Sequence[str]] = None
    dimensions: int = 1  # not currently used
    format: Optional[str] = None
    compress: Optional[Compress] = None
    default: Optional[Number] = None


@dataclass
class Satellite:
    id: str
    id3: str
    name: str
    names: Sequence[str]
    dt1hz: float
    inclination: float
    frequency: Sequence[float]
    phases: Mapping[str, Phase] = field(default_factory=dict)
    aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    variables: Mapping[str, Variable] = field(default_factory=dict)


@dataclass
class Config:
    dataroot: PathLike
    config_files: Sequence[PathLike]
    satellites: Mapping[str, Satellite]

    def __init__(self, pre_config: PreConfig):
        self.dataroot = pre_config.dataroot
        self.config_files = pre_config.config_files[:]
        self.satellites = dict()
