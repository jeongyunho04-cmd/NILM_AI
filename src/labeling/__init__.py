"""
NILM Labeling Package
Provides appliance state configurations, hysteresis state classification, and 4-tier annotation.
"""
from .annotator import DataAnnotator
from .state_classifier import StateClassifier, TransitionEvent
from .state_definitions import (
    ApplianceStateConfig,
    STATE_CONFIGURATIONS,
    StateRule,
    get_appliance_config,
)

__all__ = [
    "ApplianceStateConfig",
    "DataAnnotator",
    "STATE_CONFIGURATIONS",
    "StateClassifier",
    "StateRule",
    "TransitionEvent",
    "get_appliance_config",
]
