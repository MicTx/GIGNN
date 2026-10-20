# Contributing

Use this guide when developing, testing, and preparing GIGNN changes.

## Prerequisites

| Requirement | Version / Notes |
| --- | --- |
| Python | `>=3.10`; local validation commonly uses Python 3.11. |
| Package manager | `pip` or `uv` may be used for local setup. |
| Core ML dependencies | `torch>=2.0` and `torch-geometric>=2.4`. |
| Optional GIS dependencies | Install `.[gis]` for GeoPandas and Shapely workflows. |

## Setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For optional GIS helpers:

```bash
python -m pip install -e '.[gis,dev]'
```

If PyTorch or PyTorch Geometric fails to install automatically, install the official wheel set for your platform first, then rerun the GIGNN install command.

## Development Workflow

1. Keep changes scoped to one feature, bug fix, or documentation update.
2. Preserve existing public API behavior unless the change explicitly updates API documentation and tests.
3. Add or update tests for behavior changes.
4. Update documentation when user-facing behavior, inputs, outputs, reports, or commands change.
5. Run the verification commands before committing.

## Verification Commands

| Command | Purpose |
| --- | --- |
| `.venv/bin/python -m pytest` | Run unit and integration tests. |
| `.venv/bin/python -m black --check gignn tests examples` | Check formatting. |
| `.venv/bin/python examples/quickstart.py` | Exercise the end-to-end example. |
| `.venv/bin/python -m build` | Verify package build for release-facing changes. |

## Test Map

| Area | Tests |
| --- | --- |
| Imports and public exports | `tests/test_import.py` |
| Validation and labels | `tests/test_validation.py` |
| GeoDetector | `tests/test_geodetector.py` |
| Graph builders and PyG data | `tests/test_graph_builders.py` |
| Model and training utilities | `tests/test_model_training.py`, `tests/test_training.py` |
| Direct estimator | `tests/test_estimator.py` |
| Automatic workflow | `tests/test_auto.py` |
| Generalization diagnostics | `tests/test_generalization_standard.py` |
| Plotting | `tests/test_plotting.py` |

## Documentation Maintenance

| Document | Update When |
| --- | --- |
| `README.md` | Project positioning, install path, quickstart, or navigation changes. |
| `docs/FEATURE_OVERVIEW.md` | A capability layer or user-visible behavior changes. |
| `docs/API.md` | Public classes, functions, signatures, returns, exceptions, or reports change. |
| `docs/DATA_CONTRACT.md` | Input shapes, validation rules, labels, graph semantics, or resource limits change. |
| `docs/REPORTS.md` | Dataclass fields or report table semantics change. |
| `docs/PLOTTING.md` | Plotting inputs, outputs, or save behavior changes. |
| `docs/COMMANDS.md` | Build, test, install, or package commands change. |
| `docs/RELEASE.md` | Release checklist or artifact policy changes. |
| `docs/SECURITY.md` | Trust boundaries, checkpoint loading, or untrusted input handling changes. |

## Code Style

- Follow PEP 8.
- Use type annotations on public functions and methods.
- Prefer immutable dataclasses for reports and DTOs.
- Validate inputs at module boundaries.
- Avoid mutating caller-provided DataFrames or arrays.
- Keep public errors actionable by raising `ValidationError` or `NotFittedError` where appropriate.

## PR Checklist

- The change scope is clear.
- Tests cover new or changed behavior.
- Public API changes are documented.
- `.venv/bin/python -m pytest` passes.
- Formatting check passes or an explicit reason is documented.
- Package build passes for packaging or release-facing changes.
- No local artifacts, private paths, notebooks, or task files are unintentionally staged.
- PyPI publishing is not performed unless explicitly requested.
