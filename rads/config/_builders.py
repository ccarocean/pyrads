from dataclass_builder import dataclass_builder

from .tree import Satellite, Phase, Variable, PreConfig

__all__ = ["SatelliteBuilder", "PhaseBuilder", "VariableBuilder", "PreConfigBuilder"]

SatelliteBuilder = dataclass_builder(Satellite)

PhaseBuilder = dataclass_builder(Phase)

VariableBuilder = dataclass_builder(Variable)

PreConfigBuilder = dataclass_builder(PreConfig)
