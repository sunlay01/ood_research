#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GATE EVIDENCE_PATH" >&2
  exit 2
fi

gate="$1"
evidence="$2"
case "$gate" in
  DESIGN_GATE|MVP_GATE|CODE_GATE|RESULT_GATE) ;;
  *) echo "unknown gate: $gate" >&2; exit 2 ;;
esac

repo="$(cd "$(dirname "$0")/.." && pwd)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output="$repo/docs/audits/LATENT-001-${gate}-${stamp}.json"
prompt_file="$repo/prompts/supervisor/LATENT-001_supervisor.md"
schema="$repo/schemas/supervisor_verdict.schema.json"

mkdir -p "$repo/docs/audits"
{
  cat "$prompt_file"
  printf '\nAudit gate: %s\nEvidence entry point: %s\n' "$gate" "$evidence"
  printf 'Inspect the frozen contract and the evidence directly. Do not modify anything.\n'
} | codex exec --ephemeral -s read-only -C "$repo" \
    --output-schema "$schema" -o "$output" -

python - "$output" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
verdict = json.loads(path.read_text())["verdict"]
print(f"{path}: {verdict}")
if verdict != "PASS":
    raise SystemExit(1)
PY
