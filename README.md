# GC-II Capability Accounting

Reproducibility package for Generative Calculus II capability attribution experiments.

This repository contains the software, tests, deterministic reproduction workflow, generated result tables, figures, and machine-readable theorem audit used to study capability attribution across resource, information, interface, and rule changes.

## Reproduce

```bash
python -m pip install -r requirements-lock.txt
./reproduce.sh
```

or

```bash
python scripts/reproduce_all.py
pytest -q
```

## Contents

- `src/gcii/`: core implementation
- `scripts/reproduce_all.py`: one-command experiment reproduction
- `tests/`: automated theorem/implementation checks
- `results/`: generated CSV/JSON/audit outputs
- `figures/`: generated publication figures
- `.github/workflows/ci.yml`: CI workflow

## License

Apache License 2.0.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar.
