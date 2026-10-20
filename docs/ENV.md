# Environment Variables

GIGNN is a Python library package, not a deployed service. The current codebase does not require runtime environment variables for normal library use.

## Required Variables

| Variable | Required | Description |
| --- | --- | --- |
| None | No | No package runtime environment variables are required. |

## Environment Sources Checked

| Source | Status |
| --- | --- |
| `.env.example` | Not present. |
| `.env.template` | Not present. |
| `.env.sample` | Not present. |
| Docker environment files | Not present. |

## Python Environment

Use Python `>=3.10`. The project is commonly validated with a local `.venv` using Python 3.11.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

macOS system `python3` may be Python 3.9 and is not sufficient for this project.

## Dependency Context

PyTorch and PyTorch Geometric are core dependencies. If automatic dependency resolution fails, install the official wheel set for the platform first, then reinstall GIGNN.

Optional GIS helpers are installed with:

```bash
python -m pip install -e '.[gis,dev]'
```

## Service Wrappers

If another application wraps GIGNN behind an API or scheduled job, that wrapper may need its own environment variables for file paths, storage, authentication, or resource limits. Those variables are outside the current GIGNN package contract.
