"""Configuration tree classes.

This module contains the classes that make up the resulting PyRADS
configuration object.  In particular the :class:`rads.config.tree.Config`
class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Integral
from typing import (
    Any,
    Collection,
    Generic,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import numpy as np  # type: ignore
from cf_units import Unit  # type: ignore

from ..rpn import CompleteExpression
from ..typing import IntOrArray, Number, NumberOrArray, PathLike, PathLikeOrFile

__all__ = [
    "PreConfig",
    "Cycles",
    "ReferencePass",
    "Repeat",
    "SubCycles",
    "Phase",
    "Compress",
    "Constant",
    "Flags",
    "MultiBitFlag",
    "SingleBitFlag",
    "SurfaceType",
    "Grid",
    "NetCDFAttribute",
    "NetCDFVariable",
    "N",
    "Range",
    "Variable",
    "Satellite",
    "Config",
]


@dataclass
class PreConfig:
    """**dataclass**: Pre configuration settings.

    This is used for configuration before the individual satellite
    configurations are loaded.
    """

    dataroot: PathLike
    """The location of the RADS data root."""
    config_files: Sequence[PathLikeOrFile]
    """
    XML configuration files used to load this pre-config. Also the XML files to
    use when loading the main PyRADS configuration.
    """
    satellites: Collection[str]
    """
    A collection of 2 character satellite ID strings giving the satellites
    that are to be loaded.  This is usually all the satellites available.
    """
    blacklist: Collection[str] = field(default_factory=set)
    """
    A collection of 2 character satellite ID strings giving the satellites
    that should not be loaded regardless of the value of `satellites`.
    """


@dataclass
class Cycles:
    """**dataclass**: Cycle range 'inclusive'."""

    first: int
    """First cycle of the range."""
    last: int
    """Last cycle of the range."""


@dataclass
class ReferencePass:
    """**dataclass**: Reference equator crossing.

    This stores information related to a reference equator crossing used to
    fix the satellite in time and space.
    """

    time: datetime
    """Equator crossing time of the reference pass in UTC."""
    longitude: float
    """Longitude of the equator crossing of the reference pass."""
    cycle_number: int
    """Cycle number of the reference pass."""
    pass_number: int
    """Pass number of the reference pass."""
    absolute_orbit_number: int = 1
    """Absolute orbit number of reference pass."""


@dataclass
class Repeat:
    """**dataclass**: Length of the repeat cycle.

    .. note::

        With many satellites now using non exact repeats this is of
        questionable use since it is frequently disconnected from numbered
        cycles (which are actually sub cycles).
    """

    days: float
    """Number of days in a repeat cycle."""
    passes: int
    """Number of passes in a repeat cycle."""
    longitude_drift: Optional[float] = None
    """Longitudinal drift per repeat cycle."""


@dataclass
class SubCycles:
    """**dataclass**: Lengths of sub cycles."""

    lengths: Sequence[int]
    """List of the number of passes for each sub cycle."""
    start: Optional[int] = None
    """
    Start cycle of the sub cycle sequence.  Can be None, in which case the sub
    cycle sequence starts with the first cycle of the phase.
    """


@dataclass
class Phase:
    """**dataclass**: Mission phase."""

    id: str
    """Single letter ID of the mission phase."""
    mission: str
    """Descriptive name of the mission phase."""
    cycles: Cycles
    """Cycle range.

    See :class:`Cycles`.
    """
    repeat: Repeat
    """Repeat cycle (not sub cycle) information.

    See :class:`Repeat`.
    """
    reference_pass: ReferencePass
    """Equator crossing reference pass.

    See :class:`ReferencePass`.
    """
    start_time: datetime
    """Date and time the mission phase began."""
    end_time: Optional[datetime] = None
    """
    Date and time the mission phase ended.  This is only provided for the last
    mission phase of a given satellite (if that satellite has been
    decommissioned).  In all other instances it is None.
    """
    subcycles: Optional[SubCycles] = None
    """Sub cycle information for satellites with sub cycles, None otherwise.

    See :class:`SubCycles`.
    """

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Phase):
            return NotImplemented
        return self.start_time < other.start_time

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Phase):
            return NotImplemented
        return self.start_time > other.start_time


@dataclass
class Compress:
    """**dataclass**: Variable compression.

    This can usally be ignored by the end user, but may prove useful if
    extracting and saving data into another file.

    To store the variable `x`:

    .. code-block:: python

        x_store = ((x - add_offset) * scale_factor).astype(type)

    To unpack the variable `x`:

    .. code-block:: python

        x = (x_store/scale_factor + add_offset).astype(np.float64)
    """

    type: np.dtype
    """Type of stored data as a Numpy type."""
    scale_factor: Union[int, float] = 1
    """Scale factor of stored data."""
    add_offset: Union[int, float] = 0
    """Add offset of stored data."""


@dataclass
class Constant:
    """**dataclass**: Numerical constant for the data field."""

    value: Union[int, float]
    """The constant numerical value."""


class Flags(ABC):
    """Base class of all data fields of type flags."""

    @abstractmethod
    def extract(self, flags: NumberOrArray) -> NumberOrArray:
        """Extract the flag value from a number or array.

        See the concrete implementations for further information:

        * :class:`SurfaceType`
        * :class:`SingleBitFlag`
        * :class:`MultiBitFlag`

        :param flags:
            Integer or array of integers to extract flag value from.

        :return:
            Integer, bool, or array of integers or booleans depending on the
            type of flag.
        """


@dataclass
class MultiBitFlag(Flags):
    """**dataclass**: A single bit flag.

    This type of flag is used for extracting true/false from a given bit.

    This indicates that 2 or more continuous bits in the "flags" RADS variable
    are to be used as the data for the RADS variable.

    :raises TypeError:
        If `bit` or `length` are not integers.
    :raises ValueError:
        If `bit` is negative or `length` is less than 2.
    """

    bit: int
    """Bit index (starting at 0) where the flag is located."""
    length: int
    """Length of the flag in bits."""

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
        """Extract the flag value from a number or array.

        :param flags:
            Integer or array of integers to extract flag value from.

        :return:
            An integer or an array of integers which is the value of the
            extracted flag.
        """
        result = (flags & ~(~0 << self.length) << self.bit) >> self.bit

        # if NumPy array cast down to smallest type
        if hasattr(result, "astype"):
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
    """**dataclass**: A single bit flag.

    This type of flag is used for extracting true/false from a given bit.

    This indicates that a single bit in the "flags" RADS variable is to be
    used as the data for the RADS variable.

    :raises TypeError:
        If `bit` is not an integer.
    :raises ValueError:
        If `bit` is negative.
    """

    bit: int
    """Bit index (starting at 0) where the flag is located."""

    def __post_init__(self) -> None:
        if not isinstance(self.bit, Integral):
            raise TypeError("'bit' must be an integer")
        if self.bit < 0:
            raise ValueError("'bit' must be non-negative")

    def extract(self, flags: IntOrArray) -> IntOrArray:
        """Extract the flag value from a number or array.

        :param flags:
            Integer or array of integers to extract flag value from.

        :return:
            A bool or an array of booleans which is the value of the extracted
            flag.
        """
        return flags & (1 << self.bit) != 0


@dataclass
class SurfaceType(Flags):
    """**dataclass**: Surface type flag.

    This is special flag that is based on the 3, 4, and 5 bits (zero indexed)
    of the underlying data and results in one of the following numerical
    values:

    * 0 - ocean
    * 2 - enclosed sea or lake
    * 3 - land
    * 4 - continental ice

    This indicates that the surface type integer (above) is to be extracted
    from the "flags" RADS variable and used as the data for the RADS variable.
    """

    def extract(self, flags: IntOrArray) -> IntOrArray:
        """Extract the flag value from a number or array.

        :param flags:
            Integer or array of integers to extract flag value from.

        :return:
            The surface type integer or an array of surface type integers.
        """
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
    """**dataclass**: Grid file for the data field.

    This indicates that the value of the grid in the NetCDF file is to be
    interpolated to provide data for the RADS variable.
    """

    file: str
    """
    NetCDF file containing the grid.  This file can only contain one
    2-dimensional variable.
    """
    x: str = "lon"
    """Name of the RADS variable giving the x-coordinate for interpolation."""
    y: str = "lat"
    """Name of the RADS variable giving the y-coordinate for interpolation."""
    method: str = "linear"
    """Interpolation method to lookup values in the grid.

    The options are:

        * "linear" - bilinear interpolation
        * "spline" - cubic spline interpolation
        * "nearest" - nearest neighbor lookup
    """


@dataclass
class NetCDFAttribute:
    """**dataclass**: NetCDF attribute for the data field.

    This indicates that the value of the NetCDF attribute from the pass file
    is to be used as the data for the RADS variable.
    """

    name: str
    """Name of the NetCDF attribute."""
    variable: Optional[str] = None
    """Variable that the attribute is under.  None for global."""
    branch: Optional[str] = None
    """Postfix to append to 2 character mission folder when loading the file.

    .. note::

        PyRADS supports an unlimited number of branches.  However, to maintain
        compatibility with RADS no more than 4 should be used.
    """


@dataclass
class NetCDFVariable:
    """**dataclass**: NetCDF variable for the data field.

    This indicates that the value of the NetCDF variable from the pass file is
    to be used as the data for the RADS variable.
    """

    name: str
    """Name of hte NetCDF variable."""
    branch: Optional[str] = None
    """Postfix to append to 2 character mission folder when loading the file.

    .. note::

        PyRADS supports an unlimited number of branches.  However, to maintain
        compatibility with RADS no more than 4 should be used.
    """


N = TypeVar("N", int, float)


@dataclass
class Range(Generic[N]):
    """**dataclass**: Numerical range (inclusive)."""

    min: N
    """Minimum value in range."""
    max: N
    """Maximum value in range."""


@dataclass
class Variable(Generic[N]):
    """**dataclass**: A RADS variable descriptor."""

    id: str
    """Name identifier of the variable."""
    name: str
    """Descriptive name of the variable"""
    data: Union[
        Constant, CompleteExpression, Flags, Grid, NetCDFAttribute, NetCDFVariable
    ]
    """What data backs the variable.

    This can be any of the following:

    * :class:`Constant` - a numeric constant
    * :class:`CompleteExpression` - a mathematical combination of other RADS variables.
    * :class:`Flags` - an integer or boolean extracted from the "flags" RADS variable.
    * :class:`Grid` - an interpolated grid (provided by an external NetCDF file)
    * :class:`NetCDFAttribute` - a NetCDF attribute in the pass file
    * :class:`NetCDFVariable` - a NetCDF variable in the pass file
    """
    units: Union[Unit, str] = Unit("-")
    """The variable's units.

    There are three units used by RADS that are not supported by
    :class:`cf_units.Unit`.  The following table gives the mapping:

    ============  =======================
    Unit String   :class:`cf_units.Unit`
    ============  =======================
    db            :code:`Unit("no_unit")`
    decibel       :code:`Unit("no_unit")`
    yymmddhhmmss  :code:`Unit("unknown")`
    ============  =======================

    See :class:`cf_units.Unit`.
    """
    standard_name: Optional[str] = None
    """CF-1.7 compliant "standard_name"."""
    source: str = ""
    """Documentation of the source of the variable."""
    comment: str = ""
    """Comment string for the variable."""
    flag_values: Optional[Sequence[str]] = None
    """List of the meanings of the integers of a enumerated flag variable.

    This is mutually exclusive with `flag_masks`.
    """
    flag_masks: Optional[Sequence[str]] = None
    """List of the meanings of the bits (LSB to MSB) for a bit flag variable.

    This is mutually exclusive with `flag_values`.
    """
    limits: Optional[Range[N]] = None
    """Valid range of the variable.

    If outside this range the variable's data is considered bad and should be
    masked out.

    See :class:`Range`.
    """
    plot_range: Optional[Range[N]] = None
    """Recommended plotting range for the variable.

    See :class:`Range`.
    """
    quality_flag: Optional[Sequence[str]] = None
    """List of RADS variables that when bad make this variable bad as well."""
    dimensions: int = 1  # not currently used
    """Dimensionality of the variable."""
    format: Optional[str] = None
    """Recommended format string to use when printing the variable's value."""
    compress: Optional[Compress] = None
    """Compression scheme used for the variable.

    See :class:`Compress`.
    """
    default: Optional[Number] = None
    """Default numerical or boolean value to use when data sources is unavailable."""


@dataclass
class Satellite:
    """**dataclass**: Satellite descriptor."""

    id: str
    """2 character satellite ID."""
    id3: str
    """3 character satellite ID."""
    name: str
    """Satellite name.

    .. note::

        While PyRADS places no restrictions on the length of this field to
        maintain compatibility with RADS it should be no longer than 8
        characters.
    """
    names: Sequence[str]
    """Alternate satellite names."""
    dt1hz: float
    """Time step of 1-Hz data (in seconds)."""
    inclination: float
    """Orbital inclination in degrees."""
    frequency: Sequence[float]
    """List of altimeter frequencies."""
    phases: Mapping[str, Sequence[Phase]] = field(default_factory=dict)
    """Mapping from 1 character phase ID's to lists of mission phases.

    .. note::

        This being a mapping to a list of mission phases is a necessary evil
        brought about by satellites such as Sentinel-3B which change orbit
        during a mission phase.

    See :class:`Phase`.
    """
    aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    """Mapping from pseudo variables to a list of RADS variables.

    When the pseudo variable is accessed any of the RADS variables listed here
    can be used.  In particular, the first one available will be used.
    """
    variables: Mapping[str, Variable[Number]] = field(default_factory=dict)
    """Mapping from variable name identifiers to variable descriptors.

    These are all the variables supported by the satellite.

    See :class:`Variable`.
    """


@dataclass
class Config:
    """**dataclass**: PyRADS configuration."""

    dataroot: PathLike
    """Path to the RADS data root."""
    config_files: Sequence[PathLikeOrFile]
    """Paths to the XML configuration files used to load this configuration.

    *The order is the same as they were loaded.*
    """
    satellites: Mapping[str, Satellite]
    """Mapping from 2 character satellite ID's to satellite descriptors.

    See :class:`Satellite`.
    """

    def __init__(self, pre_config: PreConfig, satellites: Mapping[str, Satellite]):
        """
        :param pre_config:
            The pre-configuration object to use when loading this configuration
            object.
        :param satellites:
            A mapping of 2 character satellite names to satellite
            descriptor objects.
        """
        self.dataroot = pre_config.dataroot
        self.config_files = pre_config.config_files[:]
        self.satellites = satellites
