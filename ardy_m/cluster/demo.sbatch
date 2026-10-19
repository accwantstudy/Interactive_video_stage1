#!/usr/bin/env bash
#SBATCH --job-name=ardy-demo
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/activate.sh
echo "ARDY demo node: $(hostname -f)"
echo "Forward local port 2333 here, then open http://localhost:2333"
srun python scripts/run_demo.py --no-compile
