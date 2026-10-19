"""Four-step pose-conditioned streaming with the LongLive non-AR DMD LoRA.

Run from the repository root:
  python examples/wan2.2_fun/predict_v2v_control_ref_5b_streaming_fewstep.py

Reuses the streaming example (16 new frames + 4 history frames). This adapter
was distilled on vanilla Wan2.2-TI2V-5B, NOT Fun-Control. Applying it to the
Fun-Control backbone is an experimental transfer; pose/reference fidelity and
short-window quality must be evaluated. It does not load the AR LongLive model.

Published recipe: rank/alpha 64/64, strength 1, 4 steps, FlowUniPC, shift 5.
CFG=1 is our speed-oriented default, not a universal CFG recommendation from
the adapter author. Use --guidance-scale to evaluate other values (>1 doubles
the transformer batch). TeaCache and CFG skipping are disabled.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

import predict_v2v_control_ref_5b_streaming as streaming


REPO_ID = "Perflow-Shuai/Wan2.2-5B-NonAR-DMD-4Step-LoRA-r64-iter1600"
REVISION = "3e23ff0e949cec32c3dce7213fc55e182c9f142c"
ADAPTER_SHA256 = "da4a75094b4afdf5fbdf47a07530151b9ef47f2fffebc4c8675683ad049c32dc"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LORA_DIR = PROJECT_ROOT / "models/LongLive2-Wan2.2-5B-NonAR-DMD-4Step-LoRA-r64"


def verify_adapter(lora_dir):
    """Verify the pinned generator-only release before loading any large model."""
    lora_dir = Path(lora_dir)
    weight_path = lora_dir / "adapter_model.safetensors"
    config_path = lora_dir / "adapter_config.json"
    if not weight_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Missing adapter_model.safetensors or adapter_config.json in {lora_dir}. "
            f"Download them from https://huggingface.co/{REPO_ID}/tree/{REVISION}"
        )
    digest = hashlib.sha256()
    with weight_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != ADAPTER_SHA256:
        raise ValueError("Adapter SHA-256 mismatch; this entry point expects the pinned non-AR rank-64 release")
    config = json.loads(config_path.read_text())
    expected_targets = {
        f"blocks.{i}.{suffix}" for i in range(30)
        for suffix in (
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2",
        )
    }
    if (config.get("r") != 64 or config.get("lora_alpha") != 64
            or config.get("base_model_name_or_path") != "Wan-AI/Wan2.2-TI2V-5B"
            or set(config.get("target_modules", [])) != expected_targets
            or config.get("use_rslora", False) or config.get("use_dora", False)
            or config.get("fan_in_fan_out", False)):
        raise ValueError("Adapter config does not match the published rank/alpha/target modules")
    return weight_path, config


def validate_lora_mapping(transformer, weight_path, config):
    """Reject missing, extra, partial or shape-incompatible pairs before merging."""
    targets = sorted(config["target_modules"])
    pairs = []
    expected_keys = set()
    rank = config["r"]
    with safe_open(str(weight_path), framework="pt", device="cpu") as weights:
        for target in targets:
            module = transformer.get_submodule(target)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"Expected a Linear module at {target}, got {type(module).__name__}")
            a_key = f"base_model.model.{target}.lora_A.weight"
            b_key = f"base_model.model.{target}.lora_B.weight"
            expected_keys.update((a_key, b_key))
            a_shape = weights.get_slice(a_key).get_shape()
            b_shape = weights.get_slice(b_key).get_shape()
            if a_shape != [rank, module.in_features] or b_shape != [module.out_features, rank]:
                raise ValueError(f"LoRA shape mismatch at {target}: A={a_shape}, B={b_shape}, weight={tuple(module.weight.shape)}")
            pairs.append((target, a_key, b_key))
        if set(weights.keys()) != expected_keys:
            raise ValueError("LoRA file has unexpected or missing tensors; refusing a partial merge")
        for key in sorted(expected_keys):
            if not torch.isfinite(weights.get_tensor(key)).all():
                raise ValueError(f"Non-finite adapter tensor: {key}")
    return pairs


@torch.no_grad()
def merge_dmd_lora(transformer, weight_path, config, strength, merge_device="cpu"):
    """Merge FP32 B@A one layer at a time, before offload/quantization hooks."""
    pairs = validate_lora_mapping(transformer, weight_path, config)
    scale = strength * config["lora_alpha"] / config["r"]
    with safe_open(str(weight_path), framework="pt", device="cpu") as weights:
        for target, a_key, b_key in pairs:
            weight = transformer.get_submodule(target).weight
            if weight.device.type == "meta":
                raise ValueError("LoRA must be merged before installing offload hooks")
            a = weights.get_tensor(a_key).to(device=merge_device, dtype=torch.float32)
            b = weights.get_tensor(b_key).to(device=merge_device, dtype=torch.float32)
            merged = weight.detach().to(device=merge_device, dtype=torch.float32).addmm(b, a, alpha=scale)
            if not torch.isfinite(merged).all():
                raise ValueError(f"Non-finite merged weight: {target}")
            weight.copy_(merged.to(device=weight.device, dtype=weight.dtype))
            del a, b, merged
    print(f"[DMD LoRA] Merged {len(pairs)}/{len(config['target_modules'])} Linear layers; "
          f"rank={config['r']}, alpha={config['lora_alpha']}, strength={strength}", flush=True)
    return len(pairs)


def parse_args(argv=None):
    # Reuse all streaming flags and validation while choosing few-step defaults.
    streaming.num_inference_steps = 4
    streaming.save_path = "samples/wan-videos-fun-control-streaming-fewstep"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=streaming.seed)
    parser.add_argument("--gpu-memory-mode", dest="GPU_memory_mode", default="model_cpu_offload",
                        choices=["model_cpu_offload", "model_full_load", "model_group_offload", "sequential_cpu_offload"])
    parser.add_argument("--merge-device", default="cpu", help="LoRA merge device; cpu or cuda (before model offload)")
    parser.add_argument("--check-only", action="store_true", help="Verify checksum and all target shapes on CPU, then exit")
    args = streaming.parse_args(argv, parser=parser)
    if args.num_inference_steps != 4 and args.lora_strength != 0:
        parser.error("The adapter recipe requires four steps; other step counts are allowed only with --lora-strength 0 for baseline comparisons")
    if not math.isfinite(args.lora_strength) or args.lora_strength < 0:
        parser.error("lora-strength must be finite and nonnegative")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
        parser.error("guidance-scale must be finite and >= 1")
    if not math.isfinite(args.shift) or args.shift <= 0:
        parser.error("shift must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    weight_path, config = verify_adapter(args.lora_dir)
    if args.check_only:
        # Instantiate only architecture metadata; do not allocate 5B parameters.
        model_config = json.loads((PROJECT_ROOT / streaming.model_name / "config.json").read_text())
        with torch.device("meta"):
            transformer = streaming.Wan2_2Transformer3DModel.from_config(model_config)
        pairs = validate_lora_mapping(transformer, weight_path, config)
        print(f"PASS: SHA-256 verified; {len(pairs)} target layers / {2 * len(pairs)} tensors match Fun-Control 5B")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for video inference; use --check-only for CPU compatibility checks")
    streaming.sampler_name = "Flow_Unipc"
    streaming.enable_teacache = False
    streaming.cfg_skip_ratio = 0
    streaming.lora_path = streaming.lora_high_path = None
    args.run_metadata = {
        "adapter_repo": REPO_ID, "adapter_revision": REVISION, "adapter_sha256": ADAPTER_SHA256,
        "adapter_path": str(weight_path), "lora_strength": args.lora_strength,
        "num_inference_steps": args.num_inference_steps, "sampler": "Flow_Unipc", "shift": args.shift,
        "guidance_scale": args.guidance_scale, "seed": args.seed,
        "chunk_frames": args.chunk_frames, "overlap_frames": args.overlap_frames,
        "sample_size": args.sample_size, "fps": args.fps,
        "control_video": args.control_video, "ref_image": args.ref_image,
        "GPU_memory_mode": args.GPU_memory_mode,
        "compatibility": "Experimental transfer from vanilla TI2V-5B to Fun-Control 5B",
    }
    print(f"[Inference] FlowUniPC, steps={args.num_inference_steps}, shift={args.shift}, CFG={args.guidance_scale}; "
          f"{args.chunk_frames} new frames + {args.overlap_frames} history", flush=True)
    print("[Compatibility] LoRA was distilled on vanilla TI2V-5B; Fun-Control transfer quality is not guaranteed.", flush=True)
    streaming.main(args=args, transformer_setup=lambda transformer: merge_dmd_lora(
        transformer, weight_path, config, args.lora_strength, args.merge_device))


if __name__ == "__main__":
    main()
