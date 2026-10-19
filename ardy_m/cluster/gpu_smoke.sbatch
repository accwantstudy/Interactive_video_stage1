#!/usr/bin/env bash
#SBATCH --job-name=ardy-smoke
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/activate.sh
python - <<'PY'
import motion_correction
import torch
assert torch.cuda.is_available(), "Allocated GPU is not visible to PyTorch"
x = torch.arange(8, device="cuda", dtype=torch.float32)
assert x.sum().item() == 28
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("motion_correction:", motion_correction.__file__)
print("ARDY GPU smoke test: PASS")
PY
