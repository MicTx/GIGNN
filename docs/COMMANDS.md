# Command Reference

Use these commands from the repository root. Prefer the project virtual environment when it exists.

## Environment Setup

| Command | Purpose |
| --- | --- |
| `python3.11 -m venv .venv` | Create a local development virtual environment. |
| `. .venv/bin/activate` | Activate the virtual environment in POSIX shells. |
| `python -m pip install --upgrade pip` | Upgrade pip inside the active environment. |
| `python -m pip install -e '.[dev]'` | Install GIGNN in editable mode with development tools. |
| `python -m pip install -e '.[gis,dev]'` | Install development tools plus optional GeoPandas/Shapely helpers. |

## Development Checks

| Command | Purpose | Source |
| --- | --- | --- |
| `.venv/bin/python -m pytest` | Run the full test suite under `tests/`. | `pyproject.toml` pytest config |
| `.venv/bin/python -m black --check gignn tests examples` | Check formatting without modifying files. | `.[dev]` extra includes Black |
| `.venv/bin/python examples/quickstart.py` | Run the bundled workflow smoke example. | `examples/quickstart.py` |

## Documentation Checks

Use this check after documentation-only changes and before release preparation.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import re

root = Path('.')
docs = [
    root / 'README.md',
    root / 'llms.txt',
    root / 'llms-full.txt',
    *sorted((root / 'docs').glob('*.md')),
]
missing = []
for path in docs:
    text = path.read_text()
    for match in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', text):
        target = match.group(1)
        if '://' in target or target.startswith('#'):
            continue
        rel = target.split('#', 1)[0]
        if rel and not (path.parent / rel).resolve().exists():
            missing.append(f'{path} -> {target}')
if missing:
    raise SystemExit('\n'.join(missing))
print('All local markdown links resolve')
PY
```

Also check that machine-readable entry points stay aligned with the public docs:

| File | Must Stay Aligned With |
| --- | --- |
| `llms.txt` | `README.md`, `docs/FEATURE_OVERVIEW.md`, `docs/API.md`, `docs/DATA_CONTRACT.md` |
| `llms-full.txt` | `docs/API.md`, `docs/DATA_CONTRACT.md`, `docs/REPORTS.md`, `docs/PLOTTING.md`, `docs/SECURITY.md`, `docs/COMMANDS.md` |
| `docs/AGENTS.md` | `docs/COMMANDS.md`, `docs/SECURITY.md`, current test map |
| `docs/RAG_INDEX.md` | All files listed in the README documentation table |

## Build Commands

| Command | Purpose | Source |
| --- | --- | --- |
| `.venv/bin/python -m build` | Build wheel and sdist into `dist/`. | `pyproject.toml` build backend |
| `.venv/bin/python -m build --sdist --wheel --outdir /tmp/gignn-build` | Build artifacts outside the repository for validation. | Python build frontend |

## Clean Artifact Validation

Use a repository-external virtual environment for clean wheel validation.

```bash
python3.11 -m venv /tmp/gignn-clean-venv
/tmp/gignn-clean-venv/bin/python -m pip install --upgrade pip
/tmp/gignn-clean-venv/bin/python -m pip install dist/gignn-*.whl
cd /tmp
/tmp/gignn-clean-venv/bin/python - <<'PY'
from gignn import GIGNNAutoClassifier, GIGNNClassifier
print(GIGNNClassifier.__name__, GIGNNAutoClassifier.__name__)
PY
```

## Package Facts

| Item | Value |
| --- | --- |
| Package name | `gignn` |
| Version | `0.1.0` |
| Python requirement | `>=3.10` |
| License | `MIT` |
| Build backend | `setuptools.build_meta` |
| Test path | `tests` |
| Pytest addopts | `-q` |
| Runtime environment variables | None required |

## Notes

- macOS system `python3` may be Python 3.9 and is not sufficient for this project.
- PyTorch and PyTorch Geometric are core dependencies; install official platform wheels first if dependency resolution fails.
- Do not commit local artifacts such as `.venv/`, `dist/`, `build/`, `*.egg-info/`, `__pycache__/`, or `.pytest_cache/` unless a release process explicitly requires a generated artifact.
