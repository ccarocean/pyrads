from dataclass_builder import dataclass_builder

from .tree import Satellite, Phase

__all__ = ['SatelliteBuilder', 'PhaseBuilder']

SatelliteBuilder = dataclass_builder(Satellite)

PhaseBuilder = dataclass_builder(Phase)
