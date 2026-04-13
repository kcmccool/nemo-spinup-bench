#!/bin/bash
# Evaluate the generated restart file using nemo-spinup-evaluation

set -euo pipefail

INDIR="$(dirname "$(realpath "$0")")"
CONFIG="${INDIR}/gen-setup.yaml"
DATA_DIR="${INDIR}/generated"
RESULTS_DIR="${INDIR}/results"

mkdir -p "${RESULTS_DIR}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Evaluation configuration file not found at $CONFIG"
    exit 1
fi

echo "==> Evaluating generated restart file"
nemo-spinup-evaluation \
    --sim-path "${DATA_DIR}" \
    --config "${CONFIG}" \
    --mode restart \
    --results-dir "${RESULTS_DIR}"

echo "Evaluation complete. Results are saved in: ${RESULTS_DIR}"
