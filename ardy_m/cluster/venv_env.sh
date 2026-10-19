# ARDY environment settings shared by both activation entry points.
# This file is sourced automatically by .venv/bin/activate.
if type module >/dev/null 2>&1; then
    # Do not load the Python module here: module load prepends its bin directory
    # and would hide the already-activated virtualenv.
    module load StdEnv/2023 gcc/12.3 cuda/12.6 cmake/3.31.0 eigen/3.4.0
    module load slurm/killarney/25.05.6
fi

# Module loads can prepend system paths after virtualenv activation. Put the
# virtualenv back first so both `python` and console scripts use .venv.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
    hash -r 2>/dev/null || true
fi

# Authentication is private to each user.
export HF_HOME="${HOME}/.cache/huggingface"
export HF_TOKEN_PATH="${HOME}/.config/huggingface/token"

# Downloaded repositories, datasets, and Xet/assets are shared by the project.
export HF_HUB_CACHE="/project/6101778/huggingface/hub"
export HF_DATASETS_CACHE="/project/6101778/huggingface/datasets"
export HF_XET_CACHE="/project/6101778/huggingface/xet"
export HF_ASSETS_CACHE="/project/6101778/huggingface/assets"

_ARDY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TORCH_HOME="${_ARDY_ROOT}/.cache/torch"
export TRITON_CACHE_DIR="${_ARDY_ROOT}/.cache/triton"
export PYTHONNOUSERSITE=1
unset _ARDY_ROOT
umask 0007
