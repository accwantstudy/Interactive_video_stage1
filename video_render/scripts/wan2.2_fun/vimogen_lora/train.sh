#!/bin/bash
# Pose-control LoRA finetuning of Wan2.2-Fun-5B-Control on ViMoGen (4x A100 80G).
# usage: MAX_STEPS=250 bash scripts/wan2.2_fun/vimogen_lora/train.sh [extra args]
# Resumes from the latest checkpoint-N/ in $OUTPUT_DIR (LoRA + optimizer + scheduler); saves one at MAX_STEPS.
cd /data/pdai/VideoX-Fun_v1
# CUDA_VISIBLE_DEVICES is set by slurm (see train.sbatch)
export MODEL_NAME="models/Wan2.2-Fun-5B-Control"
export OUTPUT_DIR=${OUTPUT_DIR:-"models/Wan2.2-Fun-5B-Control-ViMoGen-LoRA"}
export TOKENIZERS_PARALLELISM=false
export PATH=/data/pdai/anaconda3/envs/VideoX_Fun/bin:$PATH

# rank 128 / alpha 128 (scale = 1.0 so merge weight 1.0 at inference matches training)
/data/pdai/anaconda3/envs/VideoX_Fun/bin/accelerate launch --mixed_precision="bf16" --multi_gpu --num_processes=${NUM_GPU:-4} --main_process_port=29511 \
  scripts/wan2.2_fun/train_control_lora.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="" \
  --train_data_meta="${META_DIR:-/data/pdai/datasets/ViMoGen-lora-meta-v2}/train.json" \
  --image_sample_size=704 \
  --video_sample_size=704 \
  --token_sample_size=704 \
  --video_sample_stride=1 \
  --video_sample_n_frames=77 \
  --train_batch_size=2 \
  --video_repeat=1 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=8 \
  --max_train_steps=${MAX_STEPS:-3000} \
  --checkpointing_steps=100000 \
  --checkpoints_total_limit=3 \
  --save_state \
  --resume_from_checkpoint=latest \
  --learning_rate=5e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=50 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --ref_aug_prob=0.6 \
  --add_inpaint_info \
  --add_full_ref_image_in_self_attention \
  --boundary_type="full" \
  --rank=128 \
  --network_alpha=128 \
  --target_name="q,k,v,o,ffn.0,ffn.2" \
  --use_peft_lora "$@"
