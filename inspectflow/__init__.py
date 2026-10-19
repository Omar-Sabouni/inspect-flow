"""Conveyor defect inspection with classical rules and a small CNN."""

from .config import InspectionConfig
from .detectors import CnnPartDetector
from .generator import SyntheticConveyor
from .inspection import PartInspector
from .pipeline import InspectionPipeline
from .tracking import CentroidTracker

__all__ = [
    "CentroidTracker",
    "CnnPartDetector",
    "InspectionConfig",
    "InspectionPipeline",
    "PartInspector",
    "SyntheticConveyor",
]
