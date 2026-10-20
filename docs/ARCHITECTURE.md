# Architecture

GIGNN is a Python library package, not a hosted service. The codebase is organized around a spatial classification pipeline: tabular spatial-unit data enters at the estimator or utility layer, graph structure is validated or built, Gaussian edge attributes/weights are derived, notebook-aligned weighted multi-hop GIGNN is trained, and reports/plots expose decisions and diagnostics.

## Repository Layout

```text
gignn/       runtime library package
tests/       regression and contract tests
examples/    runnable smoke examples
docs/        user, developer, release, and agent documentation
research/    source paper and notebook evidence for alignment audits
.spec/       task packages, archived records, and reusable project notes
```

Only `gignn/` is runtime package code. `research/` is evidence for implementation-status audits and should not be imported by the library.

## Package Layout

```text
gignn/
├── __init__.py              # top-level public exports and plotting lazy loading
├── auto.py                  # high-level automatic workflow estimator
├── auto_archive.py          # safe ZIP/JSON/NPY automatic-checkpoint container
├── auto_persistence.py      # automatic-workflow payload schema and reconstruction
├── splitting.py             # train/validation/test and spatial-group splits
├── ordinal.py               # explicit ordinal target encoding
├── estimators.py            # direct sklearn-style classifier
├── validation.py            # input and edge validation, label encoding
├── generalization.py        # dataset standardization and prediction diagnostics
├── reports.py               # report dataclasses
├── plotting.py              # Matplotlib/NetworkX plotting layer
├── limits.py                # resource limits
├── geodetector/             # discretization, q statistics, interactions, selector
├── graph/                   # graph builders and PyG data conversion
├── models/                  # notebook-aligned GIGNN and GraphSAGE compatibility model
├── training/                # training loop, class weights, seeds
└── io/                      # checkpoint persistence
```

## Public Entry Points

Top-level `gignn` exports are the main stable API:

- `GIGNNAutoClassifier`
- `GIGNNClassifier`
- generalization helpers
- plotting helpers
- package exceptions

`gignn.geodetector` and `gignn.graph` also expose documented utility APIs. `gignn.training`, `gignn.models`, `gignn.io`, and `gignn.validation` are useful for advanced users but should be treated as lower-level integration surfaces before `1.0`.

## Data Flow

```text
pandas DataFrame + labels + geometry/edge_index
        |
        v
validation and optional spatial dataset standardization
        |
        v
GeoDetector discretization -> q scoring -> feature selection
        |
        v
graph candidate construction or explicit edge validation
        |
        v
PyG Data conversion with encoded labels, edge_attr, and edge_weight
        |
        v
Weighted multi-hop GIGNN training
        |
        v
prediction, probability prediction, reports, simulation, plotting
```

## Automatic Workflow

`GIGNNAutoClassifier.fit` coordinates these stages:

1. Require either `geometry` or `edge_index`.
2. Validate search budget.
3. Reset feature/label indices and validate the explicit ordinal target contract.
4. Resolve disjoint train/validation/test partitions using random, external, or spatial-group assignment.
5. Fit GeoDetector selectors on training rows only and freeze their discretization metadata.
6. Transform all partitions with the frozen selector.
7. Build geometry graph candidates independently inside each partition, or filter and locally reindex explicit edges.
8. Train graph/model candidates on train and select only by validation accuracy; failed no-edge candidates remain auditable and do not abort viable candidates.
9. Evaluate the selected candidate once on the independent test partition.
10. Refit the deployment estimator on the full graph and expose validation/test reports.

Key design point: feature discovery uses only training rows, then applies stored discretization metadata to validation, test, full-data, and future prediction frames. Geometry graphs are also built independently per partition, so non-training geometry cannot change training topology. Validation metrics select candidates; independent test metrics never feed back into selection.

## Direct Estimator Workflow

`GIGNNClassifier.fit` performs a narrower workflow:

1. Validate training parameters and resource limits.
2. Resolve graph edges from explicit `edge_index` or geometry.
3. Compute Gaussian edge attributes from source/target feature differences and aggregate them into scalar edge weights.
4. Convert `X`, `y`, edges, `edge_attr`, and `edge_weight` into PyG `Data`.
5. Build notebook-aligned `GIGNN` with configured hidden layers and `num_neighbors_list`.
6. Train with class-weighted cross entropy and Adam.
7. Store model, label encoder, feature metadata, classes, and training losses.

`predict_proba` validates that feature columns exactly match training columns, resolves prediction graph edges, runs the model, and returns softmax probabilities.

## Module Boundaries

| Layer | Module | Boundary |
| --- | --- | --- |
| Validation | `gignn/validation.py` | Pure input validation and label encoding. |
| Feature discovery | `geodetector/*` | Discretization and GeoDetector statistics. |
| Graph construction | `gignn/graph/builders.py` | Edge validation and geometry-to-edge conversion. |
| PyG bridge | `gignn/graph/data.py` | Dependency boundary for `torch` and `torch_geometric.data.Data`. |
| Model | `gignn/models/gignn.py`, `gignn/models/graphsage.py` | Notebook-aligned weighted multi-hop GIGNN and compatibility GraphSAGE definitions only. |
| Training | `training/*` | Optimizer loop, class weights, seed control. |
| Estimation | `gignn/estimators.py` | Direct user-facing classifier lifecycle. |
| Orchestration | `gignn/auto.py`, `gignn/splitting.py`, `gignn/ordinal.py` | Three-way partitioning, explicit ordinal semantics, graph/model search, and reports. |
| Diagnostics | `gignn/generalization.py` | Dataset standardization and prediction analysis. |
| Output | `gignn/plotting.py`, `gignn/reports.py` | Human-facing tables, reports, and figures. |
| Persistence | `gignn/io/persistence.py`, `gignn/auto_archive.py`, `gignn/auto_persistence.py` | Direct-estimator weights-only checkpoints plus bounded ZIP/JSON/NPY automatic-workflow archives, schema/version guards, and state reconstruction. |

## Dependency Boundaries

Core dependencies are declared in `pyproject.toml`:

- NumPy and pandas for tables/arrays.
- SciPy and scikit-learn for GeoDetector p-values, metrics, and splitting.
- PyTorch and PyTorch Geometric for GIGNN training and PyG data.
- Matplotlib and NetworkX for plotting.

Optional GIS dependencies are separate:

- GeoPandas
- Shapely

The library does not require environment variables, a database, an HTTP server, Docker, or a background worker.

## Persistence Boundary

`GIGNNClassifier.save/load` persists the lower-level model through the existing PyTorch checkpoint channel and loads it with `torch.load(..., weights_only=True)` on CPU. `GIGNNAutoClassifier.save/load` uses a separate `.gignn` ZIP container: metadata and table/report values are JSON, numeric arrays and model tensors are NPY members loaded with `allow_pickle=False`, and each array is bound to declared dtype, shape, element count, byte length, and SHA-256. The automatic archive also restores the frozen selector, ordinal class order, split indices, selected graph configuration, validation/test reports, training losses, and fitted lower-level weights.

Both paths enforce file-size, schema, model-resource, and version guards before reconstruction. The automatic path additionally rejects symlink files, duplicate or unexpected ZIP members, unsupported compression, excessive expanded size/member count, invalid NPY headers, unreferenced arrays, and package major/minor incompatibility.

Checkpoint paths are local filesystem inputs and should be constrained by applications that expose model load/save through services.
