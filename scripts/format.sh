#!/bin/bash
# Auto-format all Python code in the repo with black.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Formatting Python code with black..."
uv run black backend main.py
echo "Done."
