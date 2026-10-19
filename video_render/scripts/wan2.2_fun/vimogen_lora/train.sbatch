#!/bin/bash
#SBATCH --job-name=vimogen_lora
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --time=2-00:00
#SBATCH --output=/data/pdai/VideoX-Fun_v1/logs/vimogen_lora_%j.log
# train -> validate -> train ... in this single allocation. Re-submitting resumes from the latest checkpoint.
mkdir -p /data/pdai/VideoX-Fun_v1/logs
bash /data/pdai/VideoX-Fun_v1/scripts/wan2.2_fun/vimogen_lora/run_all.sh
