"""Python library for the Radar Altimeter Database System (RADS)."""

# NOTE: The major version should match the current major version of RADS.
__version__ = '0.1.0rc'

from .config.loader import ConfigError, config_files, get_dataroot, load_config
from .constants import EPOCH

__all__ = ['ConfigError',
           'config_files', 'get_dataroot', 'load_config',
           'EPOCH']
