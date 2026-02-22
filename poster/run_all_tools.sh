#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTER_DIR="$SCRIPT_DIR"
TOOLS_DIR="$POSTER_DIR/tools"
PYTHON="${PYTHON:-python3}"

shopt -s nullglob
for tool in "$TOOLS_DIR"/*.py; do
  if [[ "$tool" == *"__pycache__"* ]]; then
    continue
  fi
  echo "Running $tool"
  "$PYTHON" "$tool"
done
