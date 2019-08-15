"""Python library for the Radar Altimeter Database System (RADS)."""

from .__version__ import __version__
from .config.loader import config_files, get_dataroot, load_config
from .exceptions import RADSError, ConfigError, InvalidDataroot
from .constants import EPOCH

__all__ = [
    "__version__",
    "EPOCH",
    "ConfigError",
    "InvalidDataroot",
    "RADSError",
    "config_files",
    "get_dataroot",
    "load_config",
]
