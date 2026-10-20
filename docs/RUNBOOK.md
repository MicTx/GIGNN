# Runbook

GIGNN is a Python library package rather than a hosted service. This runbook covers local validation, package checks, and common failures.

## Operating Model

| Area | Current State |
| --- | --- |
| Runtime service | None. No HTTP server or worker process is defined. |
| API endpoints | None. There are no route or OpenAPI files. |
| Runtime environment variables | None required by package code. |
| Deployment target | Python wheel and sdist artifacts. |
| Publishing | Manual only. No automatic PyPI upload exists. |

## Standard Validation

Use the project virtual environment or another Python `>=3.10` environment with dependencies installed.

```bash
.venv/bin/python -m pytest
.venv/bin/python examples/quickstart.py
.venv/bin/python -m build
```

For documentation-only validation, run the local Markdown link check from `docs/COMMANDS.md` and compare `llms.txt`, `llms-full.txt`, `docs/AGENTS.md`, and `docs/RAG_INDEX.md` against README navigation and current public API boundaries.

For formatting checks:

```bash
.venv/bin/python -m black --check gignn tests examples
```

## Clean Wheel Smoke

```bash
rm -rf /tmp/gignn-clean-venv
python3.11 -m venv /tmp/gignn-clean-venv
/tmp/gignn-clean-venv/bin/python -m pip install --upgrade pip
/tmp/gignn-clean-venv/bin/python -m pip install dist/gignn-*.whl
cd /tmp
/tmp/gignn-clean-venv/bin/python - <<'PY'
from gignn import GIGNNAutoClassifier, GIGNNClassifier
print(GIGNNClassifier.__name__, GIGNNAutoClassifier.__name__)
PY
```

## Artifact Checks

| Check | Expected Result |
| --- | --- |
| Wheel exists | `dist/gignn-<version>-py3-none-any.whl` |
| sdist exists | `dist/gignn-<version>.tar.gz` |
| License in wheel | `gignn-<version>.dist-info/licenses/LICENSE` present |
| Runtime package import | `from gignn import GIGNNAutoClassifier, GIGNNClassifier` succeeds. |
| Internal task files | `.claude/`, `.spec/`, and `tmp/` are not shipped in wheel. |
| Public metadata | No private local paths, raw internal URLs, or notebook-only paths. |
| LLM documentation links | `llms.txt`, `llms-full.txt`, `docs/AGENTS.md`, and `docs/RAG_INDEX.md` exist and local Markdown links resolve. |

## Common Issues

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `command not found: python` | No `python` shim in shell. | Use `.venv/bin/python` or `python3.11`. |
| Python version is 3.9 | macOS system Python is being used. | Use `.venv/bin/python` or create a Python `>=3.10` venv. |
| `No module named sklearn` | Dependencies are not installed in the active interpreter. | Run `.venv/bin/python -m pip install -e '.[dev]'`. |
| `No module named torch` | PyTorch is missing or installed in a different environment. | Install official PyTorch wheel set, then reinstall GIGNN. |
| `No module named torch_geometric` | PyG is missing. | Install official PyG wheel set for the current Torch/Python platform. |
| Graph construction produced no edges | Geometry strategy or threshold is too sparse. | Increase `k`, increase `threshold`, or provide explicit `edge_index`. |
| Auto workflow reports all graph candidates failed | Every candidate lacks usable train/validation/test partition edges. | Use a denser graph, larger partitions, a wider threshold, or partition-compatible explicit edges. |
| Prediction rejects feature columns | Prediction columns differ from training columns. | Reorder/select columns to match `feature_names_in_`. |
| Checkpoint load rejected | Schema/version, file/member/expanded size, array hash/shape/dtype, tensor count, class count, or feature count violates guards. | Recreate the checkpoint from a trusted compatible model; do not bypass validation. Use `.pt` for `GIGNNClassifier` and `.gignn` for `GIGNNAutoClassifier`. |

## Recovery

This is a library package. Rollback means reverting source changes or reinstalling a previous artifact.

```bash
git revert <commit>
python -m pip install <previous-wheel-path>
```

Do not delete user data or external environments as part of package rollback.

## Escalation

| Situation | Action |
| --- | --- |
| Build artifact contains internal files | Stop release, inspect manifest/source tree, rebuild after excluding leakage. |
| Metadata exposes private path or URL | Stop release, remove metadata leak, rebuild, and re-inspect. |
| Clean install fails due to GIGNN import/API error | Treat as release blocker. |
| Clean install fails only due to PyTorch/PyG wheel resolution | Document platform-specific dependency install path and retry. |
| Checkpoint security guard fails unexpectedly | Inspect checkpoint source and schema before bypassing anything. |
