"""Constants for all of PyRADS."""
from datetime import datetime

__all__ = ["EPOCH"]


# Unless otherwise specified RADS uses an epoch of 1985-01-01 00:00:00 UTC
EPOCH = datetime(1985, 1, 1, 0, 0, 0)
"""RADS epoch, 1985-01-01 00:00:00 UTC"""
