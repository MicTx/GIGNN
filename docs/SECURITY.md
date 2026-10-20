# Security

GIGNN is a local Python library. Its main security boundaries are untrusted tabular data, geometry objects, graph edges, checkpoint files, research-artifact metadata, and output paths supplied by callers.

## Trust Boundaries

| Input | Trust Model | Guard |
| --- | --- | --- |
| Feature table `X` | Untrusted user data | Must be finite numeric pandas `DataFrame`; row/feature limits apply. |
| Labels `y` | Untrusted user data | Length, non-empty, missing values, and ordinal class contract validated. |
| Geometry | Untrusted objects | Only expected coordinate or topology methods are used; automatic graph size is capped. |
| `edge_index` | Untrusted array-like graph input | Shape, dtype, range, non-empty, and edge-count limits enforced. |
| Checkpoint path | Untrusted local path when exposed by an application | Direct and automatic checkpoints reject a symlink that exists at the final path when checked; callers must still confine paths to application-owned roots and prevent concurrent path replacement. |
| Direct checkpoint payload | Untrusted until validated | Save accepts only weights-only-safe containers/scalars/tensors and normalizes NumPy scalars; load uses `weights_only=True` plus exact estimator schema/resource checks. |
| Automatic checkpoint archive | Untrusted until validated | File/member/expanded-size limits, exact member names, duplicate rejection, JSON schema, NPY header/hash/dtype/shape checks, and semantic payload validation. |
| Research artifact output | Local filesystem path | Symlink/non-bundle overwrite guards, bounded tables/metadata, and CSV formula escaping. |
| Plot `output_path` | Local filesystem path | Parent directory is created; applications should constrain paths when exposed to users. |

## Direct Estimator Checkpoints

`GIGNNClassifier.load()` uses:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

Direct checkpoint payloads must contain weights-only-safe values: `None`, strings/bytes, booleans, integers, finite or non-finite Python floats supported by PyTorch serialization, exact base `torch.Tensor` objects, and nested dictionaries/lists/tuples using primitive dictionary keys. NumPy scalar values are converted to their native Python equivalents before writing. Custom objects, tensor subclasses, arbitrary NumPy arrays, cyclic containers, nesting deeper than 100 levels, and payloads above the item limit are rejected before the target is replaced. Estimator checkpoints additionally contain exactly `params`, `state_dict`, `classes`, `n_features_in`, and `feature_names_in`; the loader rejects invalid estimator params, too many classes, invalid feature metadata, non-tensor state values, too many tensors, excessive tensor elements, and files larger than `MAX_CHECKPOINT_BYTES`.

The direct writer fsyncs the temporary file, atomically replaces the target, and fsyncs the parent directory when the platform/filesystem supports directory descriptors. Explicit unsupported-directory errors are treated as a portability limitation; unexpected I/O errors propagate. If parent fsync fails after `os.replace`, the new file may already exist even though `save_checkpoint` raises, meaning content is present but directory-entry durability is not confirmed.

This format remains intended for the lower-level estimator and conventionally uses a `.pt` suffix.

## Automatic Workflow Checkpoints

`GIGNNAutoClassifier.save/load` uses a separate `.gignn` archive. It does not call `torch.load` when reading the archive.

The container is a ZIP file with:

- `metadata.json` as the first and only metadata member;
- `arrays/array-NNNN.npy` numeric array members;
- JSON references binding each array to its kind, path, dtype, shape, element count, and SHA-256.

The loader rejects:

- symlink checkpoint files, missing files, oversized files, and oversized expanded archives;
- duplicate, encrypted, directory, traversal-like, unexpected, or excessively numerous members;
- unsupported compression and metadata that is oversized, malformed, non-finite, duplicated, too deep, or too large;
- object/string NPY arrays, invalid NPY headers, Fortran-order arrays, digest mismatches, unexpected byte lengths, non-finite float arrays, or excessive total elements;
- missing/unreferenced/duplicate array references;
- incompatible archive, payload, or package major/minor versions;
- invalid selector metadata, ordinal classes, target/split indices, reports, training losses, model parameters, or state dict limits.

The writer creates a temporary file in the destination directory, fsyncs it, enforces the final size limit, and atomically replaces the requested ordinary-file path. Applications that expose checkpoint paths must still use a rooted application directory because parent-directory traversal is a caller-level path authorization concern.

## Research Artifact Export

`export_research_artifacts` treats supplied frames, IDs, predictions, probabilities, simulation reports, provenance, and output paths as untrusted inputs. It validates row/cell limits, probability shapes and values, metadata depth/item limits, and existing-output ownership. CSV cells beginning with spreadsheet formula prefixes are escaped. Raw rows are excluded unless `include_raw=True`.

## File Path Guidance

Library functions that write files include:

- `GIGNNClassifier.save(path)`
- `GIGNNAutoClassifier.save(path)`
- `export_research_artifacts(output_dir, ...)`
- plotting functions with `output_path`

If wrapping GIGNN in a web service, CLI, notebook gateway, or batch platform, resolve user identifiers to application-owned paths rather than accepting arbitrary filesystem paths. The direct checkpoint symlink check covers a stable final path at check time; it is not a race-resistant `openat`/`O_NOFOLLOW` path authorization boundary. Applications facing concurrent or adversarial filesystem mutation must enforce that boundary outside GIGNN. Do not disable checkpoint or artifact guards to recover an incompatible file.

## Untrusted Data Guidance

Recommended caller-side checks before invoking GIGNN:

- Limit input file size before loading it into pandas.
- Limit row count and feature count to expected operational bounds.
- Validate feature names and expected schema.
- Validate geometry source and CRS outside GIGNN when spatial semantics matter.
- Prefer explicit `edge_index` for very large datasets instead of automatic geometry graph construction.
- Keep untrusted checkpoint and artifact locations outside source, package, and credential directories.

GIGNN also enforces resource limits in code; see [Data Contract](DATA_CONTRACT.md).

## Dependency Boundary

Core ML execution depends on PyTorch and PyTorch Geometric. Install platform-appropriate wheels from official sources. Avoid mixing unreviewed private package indexes into release environments.

## No Runtime Secrets

The current package code does not require environment variables, API keys, service credentials, databases, or network calls.

## Reporting Security Issues

For private security issues, do not open a public issue with exploit details. Share a minimal reproduction, affected version, expected impact, and relevant artifact hashes with the maintainers through the project's agreed private channel.
