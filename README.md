# GIGNN

Python implementation of the geographically informed graph neural network (GIGNN) for spatial simulation and prediction.

**English** | [简体中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.4%2B-3c2179.svg)](https://pyg.org)
[![Version](https://img.shields.io/badge/version-0.1.0--beta-orange.svg)](docs/RELEASE.md)

## Why GIGNN

Spatial data usually arrives as a table: one row per region, grid cell, or sample, with numeric indicator columns such as NDVI, DEM, land-cover statistics, population, or road density. You probably already know how to classify such a table with a random forest or gradient boosting. Those models treat each row as independent — and that is exactly the limitation this library addresses.

Geographical entities are not independent. Their connections matter, and those connections form a graph: each spatial unit becomes a node, and neighbouring units are joined by edges. Graph neural networks (GNNs) can simulate spatial interactions over such structures. But a plain GNN has one flaw for spatial work: it treats a neighbour's influence as if proximity alone determined it. In real landscapes, adjacent units can be very different — a park next to a downtown core — and distant units with similar features may still interact.

GIGNN, introduced by Ma, Zhang & Song (2025), adds two geographical corrections to the GNN recipe:

1. **Edge weights that respond to feature differences.** Edge weights follow a Gaussian kernel of feature distances, so physically adjacent units with dissimilar features receive suppressed weights, while units with matching features keep influence across hops.
2. **Feature discovery before training.** GeoDetector-based discretization quantifies attribute differences by *spatial stratified heterogeneity* — the measurable divergence of features across spatial zones — and selects the factors that matter before any message passing starts.

On the paper's urbanization analysis of the Greater Perth Area (Australia), GIGNN reached 84.1% simulation accuracy (2016) and 81.0% generalization accuracy (2021), outperforming the compared machine learning and graph-based methods by 16.5–42.7 percentage points on those tasks. Results depend on data and featurization; your task may differ.

![Fig. 1 from the GIGNN paper: architecture of GIGNN for spatial modelling](docs/img/gignn-architecture.png)
*Figure: architecture of GIGNN for spatial modelling (Fig. 1 in [the paper](https://doi.org/10.1016/j.spasta.2025.100920)).*

This repository packages the notebook-aligned GIGNN core: feature discovery, graph construction, training, prediction, scenario simulation, and generalization diagnostics for spatial-unit tables. It implements the core algorithm path, not every paper experiment or external dataset — see [Implementation Status](docs/IMPLEMENTATION_STATUS.md) for the exact boundary. Raw remote-sensing image download, mosaicking, reprojection, raster sampling, and hosted service deployment are outside the core package.

> Ma, X., Zhang, Z., & Song, Y. (2025). Geographically informed graph neural networks. *Spatial Statistics*, 69, 100920. [https://doi.org/10.1016/j.spasta.2025.100920](https://doi.org/10.1016/j.spasta.2025.100920)

To cite this software, use [CITATION.cff](CITATION.cff) (GitHub and Gitea render a "Cite this repository" button from it).

If you have used scikit-style estimators, the API shape will feel familiar: `fit`, `predict`, `predict_proba`, `score`. The Quickstart below is the shortest path; the [Documentation](#documentation) table covers the rest.

## Release Status

`gignn==0.1.0` is a beta release candidate. The public API is usable for local research and package-grade validation, but minor API refinements may still occur before a stable `1.0` release.

This repository does not automatically publish to PyPI. Build and upload actions must be explicit.

## Documentation

| Document | Purpose |
| --- | --- |
| [LLM Short Context](llms.txt) | Short machine-readable project context for LLMs and crawlers. |
| [LLM Full Context](llms-full.txt) | Full machine-readable project context for long-context LLMs and agents. |
| [Implementation Status](docs/IMPLEMENTATION_STATUS.md) | Current audit of notebook GIGNN core implementation versus out-of-scope paper reproduction. |
| [Feature Overview](docs/FEATURE_OVERVIEW.md) | Layered map of all code-backed capabilities. |
| [User Guide](docs/USER_GUIDE.md) | End-to-end workflow from data preparation to prediction and simulation. |
| [API Reference](docs/API.md) | Public classes, functions, reports, inputs, returns, and exceptions. |
| [Architecture](docs/ARCHITECTURE.md) | Module boundaries and internal data flow. |
| [Reports](docs/REPORTS.md) | Report dataclasses and estimator attributes. |
| [Plotting](docs/PLOTTING.md) | Map, diagnostic, metric, and interaction visualizations. |
| [Data Contract](docs/DATA_CONTRACT.md) | Input table, label, geometry, edge, and resource-limit requirements. |
| [Examples](docs/EXAMPLES.md) | Bundled example index and expected behavior. |
| [Commands](docs/COMMANDS.md) | Install, test, build, and package validation commands. |
| [Environment](docs/ENV.md) | Runtime environment and dependency context. |
| [Contributing](docs/CONTRIBUTING.md) | Development workflow and PR checks. |
| [Release](docs/RELEASE.md) | Release checklist and artifact validation. |
| [Runbook](docs/RUNBOOK.md) | Operational checks and troubleshooting. |
| [Security](docs/SECURITY.md) | Checkpoint, file path, and untrusted input boundaries. |
| [Agent Guide](docs/AGENTS.md) | Task routing and verification guidance for coding agents. |
| [RAG Index](docs/RAG_INDEX.md) | Retrieval routing, chunking, and anti-hallucination guidance. |

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Why GIGNN](#why-gignn)
- [Release Status](#release-status)
- [Documentation](#documentation)
- [Repository Layout](#repository-layout)
- [Install](#install)
- [Quickstart](#quickstart)
- [Core Capabilities](#core-capabilities)
- [Verify](#verify)
- [License](#license)

## Repository Layout

```text
gignn/              runtime library package
tests/              regression and contract tests
examples/           runnable smoke examples
docs/               user, developer, release, and agent documentation
research/           source paper and notebook evidence for alignment audits
.spec/              task packages, archived records, and reusable project notes
```

## Install

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For geometry-heavy workflows:

```bash
python -m pip install -e '.[gis,dev]'
```

PyTorch and PyTorch Geometric are core dependencies. If dependency resolution fails on your platform, install the official PyTorch/PyG wheel set first, then reinstall GIGNN.

## Quickstart

`X` is a numeric indicator table (one row per spatial unit), `y` an ordered categorical target such as `["low", "medium", "high"]`, and `geometry` an optional geometry column; see the [Data Contract](docs/DATA_CONTRACT.md) for exact requirements.

```python
from gignn import GIGNNAutoClassifier

model = GIGNNAutoClassifier(
    detector_grid={"discretizer": ["quantiles"], "max_bins": [3]},
    graph_grid={"strategy": ["knn"], "k": [4]},
    model_grid={"hidden_channels": [(64,)], "learning_rate": [0.01], "max_epochs": [100]},
    random_state=42,
)
model.fit(X, y, geometry=geometry, class_order=["low", "medium", "high"])

predictions = model.predict(X_new, geometry=geometry_new)        # class labels
print(model.discovery_report_.selected_features)                 # GeoDetector selection
print(model.graph_report_.selected_strategy)                     # e.g. knn(k=4)
print(model.model_selection_report_.best_params)                 # tuned GNN params
print(model.generalization_report_.accuracy)                     # holdout accuracy
```

Simulation, probability output, and research-artifact export continue from the same fitted model — see the [User Guide](docs/USER_GUIDE.md) and [Examples](docs/EXAMPLES.md).

For a runnable end-to-end script (data included), run the bundled smoke example and expect a closing line like `plotting: generated histogram, training-curve, and accuracy-comparison figures`:

```bash
.venv/bin/python examples/quickstart.py
```

## Core Capabilities

GIGNN is organized into nine capability layers:

| Layer | Main API | What it does |
| --- | --- | --- |
| Data standardization | `standardize_spatial_dataset` | Builds a numeric feature matrix, target series, optional IDs, optional geometry, and a dataset report. |
| GeoDetector discovery | `spatial_discretization`, `factor_detector`, `interaction_detector`, `GeoDetectorSelector` | Discretizes continuous variables, scores factor q statistics, detects pairwise interactions, and selects features. |
| Graph construction | `build_spatial_edges`, `build_adjacency_dict`, `calculate_gaussian_edge_attributes`, `calculate_edge_weights`, `filter_isolated_nodes` | Validates explicit `edge_index` input or builds geometry graphs by `knn`, `distance`, or `contiguity`, then derives Gaussian edge attributes and consumable edge weights. |
| PyG conversion | `build_pyg_data` | Converts numeric features, labels, graph edges, optional `edge_attr`, and optional `edge_weight` into `torch_geometric.data.Data`. |
| GNN training | `GIGNNClassifier`, `GIGNN`, `GraphSAGEClassifier`, `train_node_classifier` | Trains notebook-aligned weighted multi-hop GIGNN node classifiers with class weights, deterministic seeds, and validation checks; `GraphSAGEClassifier` remains available as a lower-level model. |
| Automatic workflow | `GIGNNAutoClassifier` | Searches discovery, graph, and model grids, then exposes traceable reports. |
| Research artifact export | `export_research_artifacts`, `GIGNNAutoClassifier.export_research_artifacts` | Writes raw/preprocessing/process/results evidence bundles for scientific reproducibility. |
| Diagnostics and simulation | `generalization_diagnostics`, `morans_i`, `cross_validate`, `model_accuracy_table`, `simulate` | Measures prediction accuracy, bias, residual spatial autocorrelation, transductive k-fold metrics, class reports, and scenario changes. |
| Plotting | `plot_*` functions | Produces Matplotlib figures for maps, comparisons, histograms, training curves, metrics, and interaction networks. |

See [Feature Overview](docs/FEATURE_OVERVIEW.md) for the full hierarchy.

For the exact answer to whether the notebook GIGNN core is implemented, see [Implementation Status](docs/IMPLEMENTATION_STATUS.md).

## Verify

```bash
.venv/bin/python -m pytest
.venv/bin/python -m build
.venv/bin/python examples/quickstart.py
```

Use `.venv/bin/python` or another Python `>=3.10` environment with project dependencies installed. macOS system `python3` may be Python 3.9 and is not sufficient for this project.

## License

[MIT](LICENSE)
