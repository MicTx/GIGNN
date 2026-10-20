# Release

This document describes the manual release-preparation flow for GIGNN. The repository does not automatically publish to PyPI.

## Release Preconditions

| Check | Requirement |
| --- | --- |
| Working tree | Clean except intended release changes. |
| Version | `pyproject.toml` version matches the intended release. |
| Documentation | README and docs reflect current public API and behavior. |
| Tests | Full test suite passes in a Python `>=3.10` environment. |
| Build | Wheel and sdist build successfully. |
| Artifacts | Wheel and sdist are inspected before upload. |

## Version Source

The package version is defined in `pyproject.toml`:

```toml
[project]
version = "0.1.0"
```

Update documentation examples that mention the exact version when bumping the version.

## Build Artifacts

Build with:

```bash
.venv/bin/python -m build
```

Expected artifact names use this pattern:

```text
dist/gignn-<version>-py3-none-any.whl
dist/gignn-<version>.tar.gz
```

For temporary validation without touching repository `dist/`:

```bash
.venv/bin/python -m build --sdist --wheel --outdir /tmp/gignn-release-build
```

## Release Checklist

1. Confirm version in `pyproject.toml`.
2. Run tests:

```bash
.venv/bin/python -m pytest
```

3. Run documentation link checks from `docs/COMMANDS.md`.
4. Confirm `llms.txt`, `llms-full.txt`, `docs/AGENTS.md`, and `docs/RAG_INDEX.md` still match the README documentation table and current public API boundaries.
5. Run formatting check:

```bash
.venv/bin/python -m black --check gignn tests examples
```

6. Run example smoke:

```bash
.venv/bin/python examples/quickstart.py
```

7. Build artifacts:

```bash
.venv/bin/python -m build
```

8. Validate wheel import from outside the repository:

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

9. Inspect artifacts for accidental internal files or private metadata.
10. Tag and publish only after explicit human approval.

## Artifact Inspection

Recommended checks:

```bash
python -m zipfile -l dist/gignn-<version>-py3-none-any.whl
python -m tarfile -l dist/gignn-<version>.tar.gz
```

Look for:

- License file in wheel metadata.
- `gignn/` package files present.
- `.claude/`, `.spec/`, `.venv/`, `tmp/`, and notebook scratch files absent from wheel.
- No private local paths or internal-only URLs in metadata.

## Repository States

This repository has two states:

- **Private Gitea (source of truth).** The private `origin` remote tracks everything, including `patent/` application materials and `.spec/` internal task records. Do not untrack or gitignore them here.
- **Public GitHub (release state).** The public mirror must never contain `patent/`, `.spec/`, or `tmp/`. Never push this repository's history directly to GitHub — the patent files would remain recoverable from every commit.

Publish to GitHub as a fresh-history export, after the release checklist passes:

```bash
mkdir -p /tmp/gignn-public-src
git -C "$(pwd)" archive --format=tar HEAD \
  | tar -x -C /tmp/gignn-public-src
rm -rf /tmp/gignn-public-src/patent /tmp/gignn-public-src/.spec /tmp/gignn-public-src/tmp
cd /tmp/gignn-public-src
git init && git add -A && git commit -m "GIGNN <version>"
git remote add origin <github-remote>
git push -u origin main
```

`git archive` exports only tracked files, so build outputs, `.venv/`, and `dist/` are excluded automatically; the three `rm -rf` lines cover the internal-only tracked directories.

## Publishing Policy

Publishing is manual. Do not upload to PyPI or TestPyPI unless the release target is explicitly requested and artifact checks pass.

If using Twine, run the upload only after review:

```bash
python -m twine check dist/*
python -m twine upload dist/*
```

`twine` is not part of the current `.[dev]` extra; install it only in the release environment if needed.

## Rollback

If a release commit is wrong, revert the commit:

```bash
git revert <commit>
```

If an installed package is wrong, reinstall a previous reviewed wheel:

```bash
python -m pip install <previous-wheel-path>
```

If a public package was uploaded incorrectly, follow the package index policy for yanking or replacing releases. Do not assume deletion is possible.
