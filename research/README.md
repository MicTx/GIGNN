# Research Evidence

This directory stores source evidence used to audit GIGNN's implementation against the reference paper and notebook workflow.

Reference paper (published): Ma, X., Zhang, Z., & Song, Y. (2025). Geographically informed graph neural networks. *Spatial Statistics*, 69, 100920. https://doi.org/10.1016/j.spasta.2025.100920

```text
research/
├── papers/
│   └── geographically-informed-graph-neural-networks.pdf
├── notebooks/
│   └── space-prediction-generalization.ipynb
├── experiments/
│   ├── dataset.csv
│   ├── schema.json
│   ├── manifest.json
│   ├── ablation_configs.json
│   └── runs/
└── scripts/
    ├── generate_phase2_dataset.py
    ├── phase2_metrics.py
    └── run_phase2_experiments.py
```

Rules:

- Keep research evidence here instead of `tmp/` or ad hoc root-level folders.
- Use stable ASCII file names for paths referenced by docs, agents, and RAG systems.
- Do not import files from this directory at runtime; package behavior must be proven through `gignn/`, `tests/`, and documented examples.

Phase 2 experiment package:

- `research/experiments/` contains the fixed synthetic lattice, ablation configs, and small run outputs.
- Rebuild data with `research/scripts/generate_phase2_dataset.py`.
- Rerun ablations with `PYTHONPATH=. .venv/bin/python research/scripts/run_phase2_experiments.py`.
