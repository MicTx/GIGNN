"""GeoDetector components."""

from gignn.geodetector.discretization import (
    DiscretizationMetadata,
    spatial_discretization,
)
from gignn.geodetector.factor import factor_detector, geodetector_q
from gignn.geodetector.interaction import interaction_detector
from gignn.geodetector.selector import GeoDetectorSelector

__all__ = [
    "DiscretizationMetadata",
    "GeoDetectorSelector",
    "factor_detector",
    "geodetector_q",
    "interaction_detector",
    "spatial_discretization",
]
