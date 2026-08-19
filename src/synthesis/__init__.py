"""
NILM Synthesis Package
Provides appliance segment pooling, physical data augmentations,
vector harmonic synthesis, grid voltage drop simulation, and PyTorch dataset generators.
"""
from .augmentor import DataAugmentor
from .dataset import NILMBatchGenerator, NILMPyTorchDataset
from .grid_simulator import GridSimulator
from .scenario_generator import ScenarioGenerator
from .segment_pool import ApplianceActivation, SegmentPool
from .synthesizer import ApplianceSchedule, LoadSynthesizer, SyntheticLoadSample

__all__ = [
    "ApplianceActivation",
    "ApplianceSchedule",
    "DataAugmentor",
    "GridSimulator",
    "LoadSynthesizer",
    "NILMBatchGenerator",
    "NILMPyTorchDataset",
    "ScenarioGenerator",
    "SegmentPool",
    "SyntheticLoadSample",
]
