from typing import Sequence, Mapping, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from cf_units import Unit  # type: ignore
import numpy as np  # type: ignore
from .._typing import Real, PathLike


@dataclass
class DataExpression:
    expr: str
    branch: Optional[str] = None


@dataclass
class GridData:
    file: str
    x: Optional[str] = None
    y: Optional[str] = None
    method: str = 'linear'


@dataclass
class ConstantData:
    value: object


@dataclass
class Limits:
    lower: Real
    upper: Real


@dataclass
class Range:
    min: Real
    max: Real


@dataclass
class Compress:
    type: np.dtype
    scale_factor: Optional[float] = None
    add_offset: Optional[float] = None


@dataclass
class Variable:
    name: str
    long_name: str
    standard_name: str
    source: str
    comment: str
    units: Union[Unit, str]
    data: Union[DataExpression, GridData, ConstantData]
    flag_values: Sequence[str] = field(default_factory=list)
    flag_mask: Sequence[str] = field(default_factory=list)
    limits: Optional[Limits] = None
    plot_range: Optional[Range] = None
    parameters: Optional[str] = None
    quality_flag: Sequence[str] = field(default_factory=list)
    dimension: int = 1  # not currently used
    format: Optional[str] = None
    compress: Optional[Compress] = None


@dataclass
class Cycles:
    """Cycle range 'inclusive'."""
    first: int
    last: int


@dataclass
class Repeat:
    """Length of the repeat cycle."""
    days: float
    passes: int


@dataclass
class ReferencePass:
    time: datetime
    longitude: float
    cycle_number: int
    pass_number: int
    absolute_orbit_number: int = 1


@dataclass
class Phase:
    id: str
    mission: str
    cycles: Cycles
    repeat: Repeat
    reference_pass: ReferencePass
    start_time: datetime


@dataclass
class Satellite:
    id: str
    id3: str
    name: str
    names: Sequence[str]
    dt1hz: float
    inclination: float
    frequency: Sequence[float]
    aliases: Mapping[str, Sequence[str]]
    phases: Mapping[str, Phase]
    variables: Mapping[str, Variable]


@dataclass
class RadsConfig:
    satellites: Mapping[str, Satellite]
    config_file: PathLike
    data_path: PathLike
