"""Geographically Informed Graph Neural Networks (Ma, Zhang & Song, 2025)."""

from typing import Any

from gignn.artifacts import ResearchArtifactBundle, export_research_artifacts
from gignn.auto import GIGNNAutoClassifier
from gignn.estimators import GIGNNClassifier
from gignn.exceptions import (
    EdgeWeightWarning,
    GIGNNError,
    NotFittedError,
    ValidationError,
)
from gignn.generalization import (
    CrossValidationReport,
    GeneralizationDiagnostics,
    MoranResult,
    SpatialDatasetReport,
    StandardizedSpatialDataset,
    cross_validate,
    generalization_diagnostics,
    model_accuracy_table,
    morans_i,
    standardize_spatial_dataset,
)
from gignn.models import GIGNN, multihop_sampling

_PLOTTING_EXPORTS = {
    "NATURE_COLOR_CYCLE",
    "NATURE_PALETTE",
    "add_panel_label",
    "apply_publication_style",
    "infer_publication_layout",
    "plot_accuracy_comparison",
    "plot_classification_map",
    "plot_discretization_histogram",
    "plot_observed_prediction_bias",
    "plot_interaction_network",
    "plot_metric_boxplot",
    "plot_prediction_comparison",
    "plot_training_curves",
    "save_publication_figure",
}

__all__ = [
    "CrossValidationReport",
    "GeneralizationDiagnostics",
    "GIGNNAutoClassifier",
    "GIGNNClassifier",
    "GIGNN",
    "MoranResult",
    "ResearchArtifactBundle",
    "SpatialDatasetReport",
    "StandardizedSpatialDataset",
    "NATURE_COLOR_CYCLE",
    "NATURE_PALETTE",
    "add_panel_label",
    "apply_publication_style",
    "infer_publication_layout",
    "cross_validate",
    "export_research_artifacts",
    "generalization_diagnostics",
    "model_accuracy_table",
    "morans_i",
    "multihop_sampling",
    "standardize_spatial_dataset",
    "plot_accuracy_comparison",
    "plot_classification_map",
    "plot_discretization_histogram",
    "plot_observed_prediction_bias",
    "plot_interaction_network",
    "plot_metric_boxplot",
    "plot_prediction_comparison",
    "plot_training_curves",
    "save_publication_figure",
    "EdgeWeightWarning",
    "GIGNNError",
    "NotFittedError",
    "ValidationError",
]


def __getattr__(name: str) -> Any:
    if name in _PLOTTING_EXPORTS:
        from gignn import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module 'gignn' has no attribute {name!r}")
