#!/usr/bin/env bash
# Reproduce ARDY with Alliance modules + virtualenv, never Conda.
set -euo pipefail
ARDY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ARDY_ROOT}"
module purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cmake/3.31.0 eigen/3.4.0
module load slurm/killarney/25.05.6
if [[ ! -d .venv ]]; then
    virtualenv --no-download .venv
fi

# Make direct virtualenv activation load ARDY's cache and token locations too.
VENV_HOOK_LINE='source "$(dirname "$VIRTUAL_ENV")/cluster/venv_env.sh"'
if ! grep -Fqx "${VENV_HOOK_LINE}" .venv/bin/activate; then
    printf '\n%s\n' "${VENV_HOOK_LINE}" >> .venv/bin/activate
fi

mkdir -p "${HOME}/.config/huggingface" "${HOME}/.cache/huggingface"
chmod 700 "${HOME}/.config/huggingface" "${HOME}/.cache/huggingface"
mkdir -p /project/6101778/huggingface/{hub,datasets,xet,assets}
chmod 2770 /project/6101778/huggingface /project/6101778/huggingface/{hub,datasets,xet,assets}
source .venv/bin/activate

# Prefer the Alliance wheelhouse.
python -m pip install --no-index --upgrade pip
python -m pip install --no-index "torch==2.9.1"

# These required versions are newer than the current Alliance wheelhouse.
python -m pip install --no-deps "vector-quantize-pytorch==1.25.2"
python -m pip install "gradio==6.26.0"

# Pins prevent resolver backtracking from replacing CUDA-enabled PyTorch.
python -m pip install --no-index \
    "torch==2.9.1" "numpy==1.26.4" "scipy==1.17.0" \
    "hydra-core==1.3.2" "omegaconf==2.3.0" \
    "transformers==5.8.1" "peft==0.19.1" "einops==0.8.2" \
    "safetensors==0.8.0" "rich==14.3.3" "trimesh==4.7.1" \
    "imageio==2.37.3" "msgspec==0.19.0" "nodeenv==1.9.1" \
    "requests==2.34.2" "websockets==15.0.1" "lxml==6.1.1" \
    "einx==0.4.3" "pybind11==2.11.1"

# ARDY needs URDF loading, not yourdfpy's broad optional geometry stack.
python -m pip install --no-index --no-deps "yourdfpy==0.0.56"

# Pinned viewer fork with timeline support.
python -m pip install --no-deps \
    "viser @ git+https://github.com/nv-tlabs/kimodo-viser.git@7c82ad8f8640bad9dff8ded5c5eee908eeb08f11"

# Build the bundled C++17 motion-correction extension.
PYBIND11_CMAKE_DIR="$(python -m pybind11 --cmakedir)"
export CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}:${CMAKE_PREFIX_PATH:-}"
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
mkdir -p .cache/torch .cache/triton cluster/logs outputs
echo "ARDY setup complete. Run: source cluster/activate.sh"
