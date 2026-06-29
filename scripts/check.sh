#!/bin/bash
# Run all code quality checks (no files are modified).
# Verifies formatting with black, then runs the test suite.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Checking formatting with black..."
uv run black --check --diff backend main.py

echo "==> Running tests..."
uv run pytest

echo "All checks passed."
