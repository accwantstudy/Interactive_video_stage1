#!/bin/bash
# One slurm allocation, all GPUs used serially: train CHUNK steps -> validate (30 cases sharded over the GPUs) -> resume training ...
# env: TOTAL (default 3000 steps), CHUNK (default 80 steps, ~1h; tune after seeing the s/step in the log)
cd /data/pdai/VideoX-Fun_v1
PY=/data/pdai/anaconda3/envs/VideoX_Fun/bin/python
D=scripts/wan2.2_fun/vimogen_lora
export OUTPUT_DIR=${OUTPUT_DIR:-models/Wan2.2-Fun-5B-Control-ViMoGen-LoRA-depth}
export META_DIR=${META_DIR:-/data/pdai/datasets/ViMoGen-lora-meta-depth-v2}
VAL_DIR=${VAL_DIR:-samples/vimogen_val_depth}
TOTAL=${TOTAL:-3000}; CHUNK=${CHUNK:-80}
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p $OUTPUT_DIR $VAL_DIR

validate() {  # $1 tag, $2 lora path (empty = baseline)
  local i
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY -u $D/validate_watch.py --tag "$1" ${2:+--ckpt "$2"} \
      --shard $i --num_shards ${#GPUS[@]} --out_dir $VAL_DIR --val_json $META_DIR/val.json &
  done
  wait
}
latest_step() { ls $OUTPUT_DIR | grep -E '^checkpoint-[0-9]+$' | sed 's/checkpoint-//' | sort -n | tail -1; }

[ "$(ls $VAL_DIR/baseline/*.mp4 2>/dev/null | wc -l)" -ge 30 ] || validate baseline ""
step=$(latest_step); step=${step:-0}
while [ $step -lt $TOTAL ]; do
  target=$(( step + CHUNK )); [ $target -gt $TOTAL ] && target=$TOTAL
  MAX_STEPS=$target bash $D/train_depth.sh || { echo "training failed"; exit 1; }
  new=$(latest_step); new=${new:-0}
  [ $new -gt $step ] || { echo "no progress ($step -> $new), abort"; exit 1; }
  step=$new
  cp $OUTPUT_DIR/checkpoint-$step/lora_diffusion_pytorch_model.safetensors $OUTPUT_DIR/lora_step_$step.safetensors
  cp $OUTPUT_DIR/checkpoint-$step/lora_diffusion_pytorch_model_compatible_with_comfyui.safetensors $OUTPUT_DIR/lora_step_${step}_comfyui.safetensors 2>/dev/null
  validate step-$step $OUTPUT_DIR/lora_step_$step.safetensors
done
