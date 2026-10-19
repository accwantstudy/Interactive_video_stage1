# ViMoGen pose-control LoRA for Wan2.2-Fun-5B-Control

1. `python build_meta.py` (CPU only) -> /data/pdai/datasets/ViMoGen-lora-meta/{train,val}.json (+ val_refs/). Re-run it once the dwpose folders are complete.
2. `sbatch train.sbatch` — ONE allocation (4 GPUs) that runs `run_all.sh`:
   baseline validation -> [train CHUNK steps -> validate the new LoRA on the 4 GPUs] x N, until TOTAL steps.
   - `CHUNK=250 TOTAL=6000` env vars (tune CHUNK so a chunk is ~1h).
   - LoRA: models/Wan2.2-Fun-5B-Control-ViMoGen-LoRA/lora_step_N.safetensors (+ checkpoint-N/ with optimizer state for resuming)
   - Validation videos [pose | generated | GT]: samples/vimogen_val/{baseline,step-N}/
   - Re-submitting the job resumes from the latest checkpoint-N/.
Inference: `lora_path = .../lora_step_N.safetensors`, `lora_weight = 1.0` (alpha == rank).
