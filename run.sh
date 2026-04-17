#!/bin/bash
set -e

ENV_NAME="${1:-pcprocessor}"

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/src"

python main.py