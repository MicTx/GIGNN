# Agent Guide

This guide is for coding agents, documentation agents, and automated reviewers working on GIGNN.

## Start Here

Read in this order for most tasks:

1. `llms.txt` for a short project context.
2. `README.md` for human-facing positioning and navigation.
3. `docs/IMPLEMENTATION_STATUS.md` when the task asks whether GIGNN is fully implemented or paper/notebook aligned.
4. `docs/FEATURE_OVERVIEW.md` for the capability hierarchy.
5. `docs/API.md` for public APIs and return types.
6. `docs/DATA_CONTRACT.md` for validation, graph, prediction, and resource limits.
7. Task-specific documents from the routing table below.

Use source code and tests as the final authority when documentation and implementation disagree.

## Task Routing

| User Task | Read First | Usually Touch |
| --- | --- | --- |
| Explain what GIGNN does | `llms.txt`, `docs/FEATURE_OVERVIEW.md`, `README.md` | Usually no edits. |
| Audit whether GIGNN is fully implemented | `docs/IMPLEMENTATION_STATUS.md`, `research/notebooks/space-prediction-generalization.ipynb`, `tests/test_model_training.py` | Docs, tests, SPEC package; code only if evidence shows a gap. |
| Add or change public API | `docs/API.md`, `docs/DATA_CONTRACT.md`, relevant tests | Code, tests, `docs/API.md`, `docs/FEATURE_OVERVIEW.md`, `llms*.txt`. |
| Change automatic workflow | `gignn/auto.py`, `tests/test_auto.py`, `docs/API.md`, `docs/REPORTS.md` | Auto code, tests, reports/data contract docs. |
| Change direct classifier | `gignn/estimators.py`, `tests/test_estimator.py`, `docs/API.md` | Estimator code, checkpoint docs, tests. |
| Change GeoDetector behavior | `gignn/geodetector/*.py`, `tests/test_geodetector.py`, `docs/FEATURE_OVERVIEW.md` | GeoDetector docs and tests. |
| Change graph semantics | `gignn/graph/*.py`, `tests/test_graph_builders.py`, `docs/DATA_CONTRACT.md` | Graph docs and tests. |
| Change generalization diagnostics | `gignn/generalization.py`, `tests/test_generalization_standard.py`, `docs/REPORTS.md` | Generalization docs and tests. |
| Change plotting | `gignn/plotting.py`, `tests/test_plotting.py`, `docs/PLOTTING.md` | Plotting docs and tests. |
| Change checkpoint persistence | `gignn/io/persistence.py`, `gignn/estimators.py`, `tests/test_estimator.py`, `docs/SECURITY.md` | Security docs, tests. |
| Change package metadata or build | `pyproject.toml`, `docs/COMMANDS.md`, `docs/RELEASE.md`, `docs/RUNBOOK.md` | Commands/release/runbook docs. |
| Update docs only | `llms.txt`, `README.md`, related docs | Docs and spec task package. |

## Source Authority

When evidence conflicts, use this order:

1. Tests under `tests/`.
2. Current implementation under `gignn/`.
3. `examples/quickstart.py`.
4. `docs/IMPLEMENTATION_STATUS.md` for current notebook/paper alignment boundaries.
5. `README.md` and `docs/*.md`.
6. Current spec task packages under `.spec/specs/`.

Legacy Trae task packages have been migrated into `.spec/specs/archive/`; original Trae files are preserved under each migrated package's `legacy/` directory. Do not use `.trae` as an active or historical source.

Do not treat archived specs as current implementation status unless the user explicitly asks for history.

## Implementation Boundaries

GIGNN is a library package.

Do not introduce or document these unless implementing them with tests:

- HTTP API endpoints.
- Web server or worker runtime.
- Database persistence.
- Required environment variables.
- Automatic PyPI publishing.
- Raw remote-sensing image acquisition or preprocessing pipeline.
- Service-level auth, rate limiting, CSRF, or deployment controls.

## Public API Boundaries

Stable user entry points are:

- top-level `gignn` exports
- `gignn.geodetector`
- `gignn.graph`

Advanced integration surfaces are documented but less stable before `1.0`:

- `gignn.validation`
- `gignn.training`
- `gignn.models`
- `gignn.io`

## Validation Requirements

Run the most relevant command set after changes.

Docs-only minimum:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import re
root = Path('.')
missing = []
for path in [root/'README.md', root/'llms.txt', root/'llms-full.txt', *sorted((root/'docs').glob('*.md'))]:
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

Standard package validation:

```bash
.venv/bin/python -m pytest
.venv/bin/python examples/quickstart.py
.venv/bin/python -m build
```

Formatting check:

```bash
.venv/bin/python -m black --check gignn tests examples
```

## Documentation Sync Rules

When changing code, update docs by topic:

| Code Change | Documentation Sync |
| --- | --- |
| New top-level export | `docs/API.md`, `docs/FEATURE_OVERVIEW.md`, `llms.txt`, `llms-full.txt`. |
| New estimator behavior | `docs/API.md`, `docs/USER_GUIDE.md`, `docs/DATA_CONTRACT.md`. |
| New report field | `docs/REPORTS.md`, `docs/API.md`, examples if relevant. |
| New graph behavior | `docs/DATA_CONTRACT.md`, `docs/FEATURE_OVERVIEW.md`. |
| New GIGNN core alignment evidence | `docs/IMPLEMENTATION_STATUS.md`, `docs/RAG_INDEX.md`, `llms*.txt`, `Agent.md`. |
| New plotting parameter | `docs/PLOTTING.md`, `docs/API.md`. |
| New command or dependency | `docs/COMMANDS.md`, `docs/CONTRIBUTING.md`, `docs/RELEASE.md`. |
| New security boundary | `docs/SECURITY.md`, `docs/RUNBOOK.md`. |

Update `docs/RAG_INDEX.md` when adding or renaming docs.

## Safety Notes For Agents

- Do not use system Python 3.9 for validation; use `.venv/bin/python` or Python `>=3.10`.
- Do not delete or overwrite user work without checking `git status`.
- Do not commit generated build artifacts unless explicitly requested.
- Treat checkpoint files and local data files as untrusted inputs.
- Do not bypass checkpoint schema/resource validation.
- Do not claim a release is published unless an explicit upload occurred.

## Review Checklist

Before finalizing:

- Public API docs match `gignn/__init__.py` and subpackage `__all__` declarations.
- Data contract matches validation and graph builder tests.
- Report docs match `gignn/reports.py` and `gignn/generalization.py` dataclasses.
- Security docs match checkpoint and path behavior in code.
- README links every user-facing doc.
- `llms.txt` and `llms-full.txt` reflect any changed capability boundaries.
