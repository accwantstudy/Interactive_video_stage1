#!/usr/bin/env bash
# Source this file before using ARDY: source cluster/activate.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced: source cluster/activate.sh" >&2
    exit 1
fi
ARDY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cmake/3.31.0 eigen/3.4.0
module load slurm/killarney/25.05.9
if [[ ! -f "${ARDY_ROOT}/.venv/bin/activate" ]]; then
    echo "ARDY environment not found. Run: bash cluster/setup.sh" >&2
    return 1
fi
source "${ARDY_ROOT}/.venv/bin/activate"
cd "${ARDY_ROOT}"
