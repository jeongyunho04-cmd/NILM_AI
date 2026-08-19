"""
NILM Synthesis Package
Provides appliance segment pooling, physical data augmentations,
vector harmonic synthesis, grid voltage drop simulation, and PyTorch dataset generators.
"""
from .augmentor import DataAugmentor
from .dataset import DEFAULT_RECIPE_MIX, NILMBatchGenerator, NILMPyTorchDataset
from .grid_simulator import GridSimulator, VoltageEnvironment
from .scenario_generator import ScenarioGenerator
from .segment_pool import ApplianceActivation, SegmentPool, StandbyProfile
from .synthesizer import ApplianceSchedule, LoadSynthesizer, SyntheticLoadSample

__all__ = [
    "ApplianceActivation",
    "ApplianceSchedule",
    "DEFAULT_RECIPE_MIX",
    "DataAugmentor",
    "GridSimulator",
    "LoadSynthesizer",
    "NILMBatchGenerator",
    "NILMPyTorchDataset",
    "ScenarioGenerator",
    "SegmentPool",
    "StandbyProfile",
    "SyntheticLoadSample",
    "VoltageEnvironment",
]
