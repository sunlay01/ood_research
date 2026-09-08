#!/usr/bin/env bash
# Shared environment for all Lean projects in this repository.
# Source this file; do not execute it in a child shell.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source $0" >&2
  exit 2
fi

export PATH="${HOME}/.elan/bin:${PATH}"

# mathlib's cache tool uses this directory for official .ltar artifacts.
# Keep this cache shared, while keeping each project's .lake/build private.
export MATHLIB_CACHE_DIR="${MATHLIB_CACHE_DIR:-${HOME}/.cache/mathlib}"

if [[ ! -x "${HOME}/.elan/bin/lake" ]]; then
  echo "Lean/Lake is not installed at ${HOME}/.elan/bin/lake" >&2
  return 1
fi

echo "Lean: $("${HOME}/.elan/bin/lake" --version | head -1)"
echo "MATHLIB_CACHE_DIR=${MATHLIB_CACHE_DIR}"
