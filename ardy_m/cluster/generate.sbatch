#!/usr/bin/env bash
#SBATCH --job-name=ardy-generate
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/activate.sh
if [[ "$#" -eq 0 ]]; then
    set -- "A person walks in a circle." --duration 5 --output first_motion
fi
srun python scripts/generate.py "$@"
