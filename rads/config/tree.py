from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence, Mapping, Optional, Union

import numpy as np  # type: ignore
from cf_units import Unit  # type: ignore

from .._typing import Real, PathLike

__all__ = ['DataExpression', 'GridData', 'ConstantData', 'Limits', 'Range',
           'Compress', 'Variable', 'Cycles', 'Repeat', 'ReferencePass',
           'Phase', 'Satellite', 'RadsConfig']


# TODO: Change the AST module to directly return these.

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
    id: str
    name: str
    standard_name: Optional[str] = None
    # source: str
    # comment: str
    # units: Union[Unit, str]
    # data: Union[DataExpression, GridData, ConstantData]
    # flag_values: Sequence[str] = field(default_factory=list)
    # flag_mask: Sequence[str] = field(default_factory=list)
    # limits: Optional[Limits] = None
    # plot_range: Optional[Range] = None
    # parameters: Optional[str] = None
    # quality_flag: Sequence[str] = field(default_factory=list)
    # dimension: int = 1  # not currently used
    # format: Optional[str] = None
    # compress: Optional[Compress] = None


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
    unknown: Optional[float] = field(default=None, repr=False)


@dataclass
class ReferencePass:
    time: datetime
    longitude: float
    cycle_number: int
    pass_number: int
    absolute_orbit_number: int = 1


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
class RadsConfig:
    satellites: Mapping[str, Satellite]
    config_file: PathLike
    data_path: PathLike
