#!/bin/bash
# One srun = one allocation: baseline eval -> [train CHUNK steps -> eval] x N.
# usage: bash scripts/wan2.2_fun/vimogen_lora/launch_depth.sh            (defaults below)
#        NUM_GPU=4 NUM_CPUS=32 TIME=2-00:00 CHUNK=250 TOTAL=6000 bash .../launch_depth.sh
NUM_GPU=${NUM_GPU:-4}
NUM_CPUS=${NUM_CPUS:-32}
TIME=${TIME:-2-00:00}        # day-hour:minute
export NUM_GPU CHUNK TOTAL

source /data/pdai/anaconda3/etc/profile.d/conda.sh
conda activate /data/pdai/anaconda3/envs/VideoX_Fun
cd /data/pdai/VideoX-Fun_v1
mkdir -p logs

srun --gres=gpu:$NUM_GPU -c $NUM_CPUS --time $TIME \
    bash scripts/wan2.2_fun/vimogen_lora/run_all_depth.sh 2>&1 | tee logs/vimogen_lora_depth_$(date +%m%d_%H%M).log
