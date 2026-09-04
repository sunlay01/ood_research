#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m pytest "${repo_root}/tests" -q
python3 -m ood_repr_reg.sweep \
  --config "${repo_root}/configs/regularizer_sweep_smoke.json" \
  --output "${repo_root}/artifacts/EXPL-001-smoke"
python3 -m ood_repr_reg.analyze_sweep \
  --input "${repo_root}/artifacts/EXPL-001-smoke"
