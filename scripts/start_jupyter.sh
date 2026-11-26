#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi

if [[ ! -x .venv/bin/jupyter-lab ]]; then
    .venv/bin/python -m pip install -r requirements-dev.txt
fi

exec .venv/bin/jupyter-lab notebooks/Untitled.ipynb "$@"
