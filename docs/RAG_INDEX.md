# RAG Index

This document helps retrieval systems route questions to the right GIGNN documentation chunks.

## Retrieval Strategy

Use `llms.txt` as the first retrieval hit for broad questions. Use `llms-full.txt` for long-context answers or agent bootstrap. Use the topical docs below for precise answers.

Chunk by heading. Preserve file path, heading, and nearby tables as metadata.

Recommended metadata fields:

| Field | Example |
| --- | --- |
| `project` | `GIGNN` |
| `doc_path` | `docs/API.md` |
| `heading` | `GIGNNAutoClassifier` |
| `audience` | `user`, `developer`, `agent`, `release`, `security` |
| `topic` | `api`, `graph`, `geodetector`, `validation`, `plotting`, `reports` |
| `stability` | `public`, `advanced`, `internal`, `operational` |

## Query Routing Table

| Query Intent | Best Documents | Keywords |
| --- | --- | --- |
| What is GIGNN? | `llms.txt`, `README.md`, `docs/FEATURE_OVERVIEW.md` | overview, purpose, capabilities, GIS, RS, GNN |
| Is GIGNN fully implemented? | `docs/IMPLEMENTATION_STATUS.md`, `docs/API.md`, `tests/test_model_training.py` | implementation status, notebook core, paper reproduction, NaturalBreaks, edge_attr |
| How do I use it end to end? | `docs/USER_GUIDE.md`, `examples/quickstart.py`, `README.md` | quickstart, workflow, fit, predict, simulate |
| What APIs exist? | `docs/API.md`, `llms-full.txt` | API, signature, method, return, exception |
| What does `GIGNNAutoClassifier` do? | `docs/API.md`, `docs/FEATURE_OVERVIEW.md`, `docs/REPORTS.md` | auto, grid, selector, graph_report, model_selection |
| What does `GIGNNClassifier` do? | `docs/API.md`, `docs/FEATURE_OVERVIEW.md`, `docs/SECURITY.md` | classifier, predict_proba, score, save, load |
| What inputs are valid? | `docs/DATA_CONTRACT.md`, `docs/API.md` | X, y, geometry, edge_index, labels, finite, numeric |
| How are graphs built? | `docs/DATA_CONTRACT.md`, `docs/FEATURE_OVERVIEW.md`, `docs/API.md` | knn, distance, contiguity, edge_index, adjacency |
| How does GeoDetector work here? | `docs/FEATURE_OVERVIEW.md`, `docs/API.md` | q statistic, discretization, interaction, selector |
| What reports are produced? | `docs/REPORTS.md`, `docs/API.md` | discovery_report, graph_report, SimulationReport |
| How do I plot results? | `docs/PLOTTING.md`, `docs/API.md` | plot, Figure, Axes, output_path, map, histogram |
| How do I install or test? | `docs/COMMANDS.md`, `docs/CONTRIBUTING.md`, `README.md` | install, pytest, build, black, quickstart |
| How do I release? | `docs/RELEASE.md`, `docs/RUNBOOK.md`, `docs/COMMANDS.md` | release, wheel, sdist, PyPI, artifact |
| What can fail operationally? | `docs/RUNBOOK.md`, `docs/ENV.md`, `docs/SECURITY.md` | troubleshooting, environment, PyTorch, checkpoint |
| Is it secure to load checkpoints? | `docs/SECURITY.md`, `docs/DATA_CONTRACT.md`, `docs/API.md` | checkpoint, weights_only, file path, trusted |
| How should an agent modify this repo? | `docs/AGENTS.md`, `llms.txt`, `.spec/docs/*.md` | agent, task routing, validation, docs sync |
| Where are the reference paper and notebook evidence? | `research/README.md`, `docs/IMPLEMENTATION_STATUS.md` | research evidence, notebook, paper, implementation status |

## Document Cards

### `llms.txt`

Audience: LLMs, coding agents, crawlers.

Use for:

- Short context injection.
- Initial model grounding.
- Quick routing to canonical docs.

Avoid using for:

- Exact report fields.
- Detailed examples.
- Long explanations.

### `llms-full.txt`

Audience: long-context LLMs, agent bootstrap, RAG full-context mode.

Use for:

- Full project summary.
- Capability and API overview.
- Agent task routing.
- Security and verification reminders.

Avoid using as the sole source for exact field-level details when topical docs are available.

### `README.md`

Audience: human users and package evaluators.

Use for:

- Project positioning.
- Install path.
- Quickstart.
- Documentation navigation.

### `docs/IMPLEMENTATION_STATUS.md`

Audience: users, maintainers, agents.

Use for:

- Answering whether the notebook GIGNN core is implemented.
- Distinguishing notebook core implementation from full paper numerical reproduction.
- Finding the audit matrix linking notebook requirements to source and tests.

### `docs/FEATURE_OVERVIEW.md`

Audience: users, maintainers, agents.

Use for:

- Capability hierarchy.
- Understanding how data, GeoDetector, graph, model, reports, plotting, and persistence fit together.

Best chunk headings:

- `Data Preparation`
- `GeoDetector Feature Discovery`
- `Graph Construction`
- `GNN Model and Training`
- `Estimators`
- `Reports and Diagnostics`
- `Plotting`
- `Persistence and Safety Limits`

### `docs/USER_GUIDE.md`

Audience: library users.

Use for:

- End-to-end workflows.
- How to prepare inputs.
- Auto estimator vs direct estimator usage.
- Simulation, diagnostics, plotting, save/load.

### `docs/API.md`

Audience: users, developers, agents.

Use for:

- Public API names.
- Method signatures.
- Return values.
- Learned estimator attributes.
- Exceptions.
- Advanced modules.

Set `stability=public` for top-level, `gignn.geodetector`, and `gignn.graph`. Set `stability=advanced` for `validation`, `training`, `models`, and `io`.

### `docs/DATA_CONTRACT.md`

Audience: users, developers, agents.

Use for:

- Feature table rules.
- Label rules.
- Geometry requirements.
- Edge-index semantics.
- Prediction graph behavior.
- Resource limits.

Prioritize this document for validation-related questions.

### `docs/ARCHITECTURE.md`

Audience: maintainers and coding agents.

Use for:

- Package layout.
- Module boundaries.
- Internal data flow.
- Dependency boundaries.
- Persistence boundary.

### `docs/REPORTS.md`

Audience: users, report consumers, notebook authors.

Use for:

- Report dataclass fields.
- `discovery_report_`, `graph_report_`, `model_selection_report_`, `generalization_report_`.
- `SimulationReport` fields.
- Dataset and diagnostic report fields.

### `docs/PLOTTING.md`

Audience: notebook users and visualization workflows.

Use for:

- Required plotting columns.
- Return objects.
- `output_path` behavior.
- Figure-saving behavior.

### `docs/EXAMPLES.md`

Audience: new users and smoke-test agents.

Use for:

- Bundled quickstart behavior.
- Expected output shape.
- Future example categories.

### `docs/COMMANDS.md`

Audience: developers, release agents, CI authors.

Use for:

- Install commands.
- Test commands.
- Build commands.
- Clean wheel validation.
- Package metadata facts.

### `docs/CONTRIBUTING.md`

Audience: contributors and coding agents.

Use for:

- Development workflow.
- Test map.
- Documentation maintenance matrix.
- PR checklist.

### `docs/RELEASE.md`

Audience: maintainers and release agents.

Use for:

- Manual release checklist.
- Version source.
- Build artifacts.
- Artifact inspection.
- Publishing policy.

### `docs/RUNBOOK.md`

Audience: maintainers and troubleshooting agents.

Use for:

- Operating model.
- Standard validation.
- Clean wheel smoke.
- Common issues.
- Recovery and escalation.

### `docs/SECURITY.md`

Audience: security reviewers, service wrapper authors, agents.

Use for:

- Trust boundaries.
- Checkpoint loading safety.
- File path guidance.
- Untrusted data guidance.
- Runtime secrets.

### `docs/AGENTS.md`

Audience: coding agents.

Use for:

- Task-specific reading order.
- File ownership.
- Documentation sync rules.
- Verification commands.
- Safety notes.

## Chunking Recommendations

- Split Markdown by `##` headings for broad retrieval.
- Split by `###` headings for `docs/API.md`, `docs/FEATURE_OVERVIEW.md`, `docs/REPORTS.md`, and `docs/PLOTTING.md`.
- Keep tables with the preceding heading.
- Keep code blocks with the immediately preceding explanatory paragraph.
- Store `doc_path` and heading path as metadata.
- Prefer smaller chunks for API docs and larger chunks for overview docs.

## Anti-Hallucination Rules

Retrieval answers should not claim:

- GIGNN has HTTP endpoints.
- GIGNN requires environment variables.
- GIGNN automatically publishes to PyPI.
- GIGNN downloads or preprocesses raw remote-sensing imagery.
- GIGNN checkpoint loading is safe for arbitrary untrusted paths in a service context.
- `natural_breaks`/`fisher_jenks` fall back to quantile bins, or `maximum_breaks` is the same as Fisher-Jenks (it is a separate widest-gap method).
- The package fully reproduces every paper experiment, external dataset, baseline, or notebook filesystem artifact.

When uncertain, retrieve source code or tests in addition to docs.
