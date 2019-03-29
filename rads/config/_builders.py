from dataclass_builder import dataclass_builder

from .tree import Satellite, Phase, Variable

__all__ = ['SatelliteBuilder', 'PhaseBuilder', 'VariableBuilder']

SatelliteBuilder = dataclass_builder(Satellite)

PhaseBuilder = dataclass_builder(Phase)

VariableBuilder = dataclass_builder(Variable)
