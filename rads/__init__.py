"""Python library for the Radar Altimeter Database System (RADS)."""

# NOTE: The major version should match the current major version of RADS.
__version__ = '4.0.1a1'

from datetime import datetime


# Unless otherwise specified RADS uses an epoch of 1985-01-01 00:00:00 UTC
EPOCH = datetime(1985, 1, 1, 0, 0, 0)