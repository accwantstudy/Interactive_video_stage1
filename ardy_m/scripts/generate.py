# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cmd line API for generating motions with Ardy.

Generated files are written under outputs/ unless --output contains a path.

Examples:
    python scripts/generate.py "A person walks in a circle." --checkpoints_dir checkpoints
    python scripts/generate.py "A person jumps." --model soma --num_samples 4 --seed 0 --output jump
    python scripts/generate.py "A person walks forward." --model g1 --duration 8.0
    python scripts/generate.py "A person walks." --prompt-switch 3 "A person runs." --duration 6
"""

import argparse
import os

import numpy as np
import torch

from ardy.constraints import load_constraints_lst
from ardy.model import DEFAULT_MODEL, load_model
from ardy.model.loading import get_env_var
from ardy.model.registry import resolve_model_name
from ardy.motion_rep.tools import length_to_mask
from ardy.postprocess import post_process_motion
from ardy.prompt_schedule import PromptSchedule
from ardy.skeleton import SOMASkeleton30
from ardy.tools import seed_everything, to_numpy


def parse_args():
    parser = argparse.ArgumentParser(description="Cmd line API for generating motions with Ardy")
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt describing the motion to generate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model nickname (core/g1/soma, optionally with horizon like core8) or full folder name.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--prompt-switch",
        action="append",
        nargs=2,
        default=[],
        metavar=("SECONDS", "PROMPT"),
        help=(
            "Switch to PROMPT at SECONDS. Repeat for multiple switches; times must align to the "
            "model's motion-token boundary."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate (default: 1)",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=None,
        help="Number of diffusion steps, at most the model's schedule length (num_base_steps, the default).",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        help="Saved constraint list",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output stem name: with one sample writes a single file per format (e.g. test.npz, test.csv); with multiple samples creates a folder and writes test_00.npz, test_01.npz, ... inside it. Used for NPZ and CSV. Bare names are written under outputs/; pass a path (e.g. ./test or results/test) to write elsewhere.",
    )
    parser.add_argument(
        "--history_frames",
        type=int,
        default=None,
        help="History frames visible to each autoregressive step (multiple of the model's token size). Default: one motion token, matching the interactive demo (4 frames for Core).",
    )
    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply motion post-processing to reduce foot skating (default: off, matching the interactive demo; ignored for G1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible results",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        nargs="+",
        default=[2.0, 2.0],
        help="CFG scale(s): one float (text weight only) or two floats [text_weight, constraint_weight] (default: 2.0 2.0).",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="Local dir holding released model folders. Falls back to the CHECKPOINTS_DIR env var; if neither is set the model is downloaded from Hugging Face.",
    )
    return parser.parse_args()


def _resolve_output_base(path: str, default_dir: str = "outputs") -> str:
    """Place bare output names under ``default_dir``; honor explicit (relative or absolute)
    paths."""
    if os.path.dirname(path):
        return path
    return os.path.join(default_dir, path)


def _single_file_path(path: str, ext: str) -> str:
    """Return path for a single output file (no folder).

    Adds ext if missing; creates parent dirs if any.
    """
    if not path.endswith(ext):
        path = path.rstrip(os.sep) + ext
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _output_dir_and_path(path: str, default_base: str, ext: str):
    """Create output folder from path and return (dir_path, path_for_file_with_suffix, base_name).

    If path has an extension, folder name is the path stem; else the path is the folder name.
    base_name is the folder basename for _00, _01, ... when n_samples > 1.
    """
    folder = os.path.splitext(path)[0] if os.path.splitext(path)[1] else path
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(folder.rstrip(os.sep))
    return folder, os.path.join(folder, default_base + ext), base_name


def _select_sample(output: dict, index: int, n_samples: int) -> dict:
    """Extract sample ``index`` from a batched output dict (non-batched entries kept as-is)."""
    return {
        k: (v[index] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
        for k, v in output.items()
    }


def _generate_with_prompt_schedule(
    model,
    schedule: PromptSchedule,
    *,
    num_samples: int,
    diffusion_steps: int,
    cfg_weight,
    history_frames: int,
) -> torch.Tensor:
    """Generate one horizon at a time and change text at schedule boundaries."""
    generated_motion = None
    text_embedding_cache = {}
    generated_frames = 0

    while generated_frames < schedule.num_frames:
        segment_idx = schedule.index_at(generated_frames)
        prompt = schedule.texts[segment_idx]
        segment_end = schedule.end_frames[segment_idx]

        history_length = 0
        history_motion = None
        if generated_motion is not None:
            history_length = min(generated_motion.shape[1], history_frames)
            history_motion = generated_motion[:, -history_length:]

        if prompt not in text_embedding_cache:
            text_embedding_cache[prompt] = model._encode_text([prompt] * num_samples)
        text_feat, text_pad_mask = text_embedding_cache[prompt]

        print(
            f"Generating frames {generated_frames}-{min(segment_end, generated_frames + model.gen_horizon_len) - 1} "
            f"with prompt: {prompt!r}"
        )
        window = model.autoregressive_step(
            num_frames=history_length + model.gen_horizon_len,
            num_denoising_steps=diffusion_steps,
            motion_mask=None,
            observed_motion=None,
            cfg_weight=cfg_weight,
            texts=None,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            init_history_sequence=history_motion,
        )
        new_frames = window[:, history_length:]
        take = min(new_frames.shape[1], segment_end - generated_frames, schedule.num_frames - generated_frames)
        new_frames = new_frames[:, :take]
        generated_motion = (
            new_frames if generated_motion is None else torch.cat((generated_motion, new_frames), dim=1)
        )
        generated_frames += take

    return generated_motion


def save_motion_npz(
    path: str,
    motion_dict: dict,
    fps: float,
    text: str,
    prompt_schedule: PromptSchedule | None = None,
) -> None:
    """Save a motion output dict to ``.npz`` along with fps and the prompt."""
    arrays = {k: np.asarray(v) for k, v in motion_dict.items()}
    arrays["fps"] = np.asarray(fps)
    arrays["text"] = np.asarray(text)
    if prompt_schedule is not None:
        arrays.update(prompt_schedule.to_npz_arrays())
    np.savez(path, **arrays)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    args = parse_args()

    if args.num_samples < 1:
        raise ValueError(f"--num_samples must be >= 1, got {args.num_samples}.")
    if args.duration <= 0:
        raise ValueError(f"--duration must be positive, got {args.duration}.")
    if args.prompt_switch and args.constraints:
        raise ValueError("--prompt-switch cannot currently be combined with --constraints")

    if len(args.cfg_weight) == 1:
        cfg_weight = float(args.cfg_weight[0])
    elif len(args.cfg_weight) == 2:
        cfg_weight = (float(args.cfg_weight[0]), float(args.cfg_weight[1]))
    else:
        raise ValueError("--cfg_weight expects one float (text) or two floats (text, constraint).")

    # Load model
    checkpoints_dir = args.checkpoints_dir or get_env_var("CHECKPOINTS_DIR")
    resolved_model = resolve_model_name(args.model, checkpoints_dir=checkpoints_dir)
    model = load_model(resolved_model, device=device, checkpoints_dir=checkpoints_dir)
    print(f"Loaded model: {resolved_model}")

    fps = model.motion_rep.fps
    num_frames = int(args.duration * fps)
    text = args.prompt.strip()
    try:
        prompt_switches = [(float(seconds), prompt) for seconds, prompt in args.prompt_switch]
    except ValueError as error:
        raise ValueError("Each --prompt-switch time must be a number of seconds") from error
    prompt_schedule = (
        PromptSchedule.from_switches(
            text,
            prompt_switches,
            fps=fps,
            num_frames=num_frames,
            frame_alignment=model.num_frames_per_token,
        )
        if prompt_switches
        else None
    )
    print(f"Will generate '{text}' with {num_frames} frames ({args.duration}s at {fps} fps)")
    if prompt_schedule is not None:
        for prompt_text, start, end in zip(
            prompt_schedule.texts, prompt_schedule.start_frames, prompt_schedule.end_frames
        ):
            print(f"  {start / fps:6.2f}s-{end / fps:6.2f}s: {prompt_text}")

    # The diffusion schedule can only be subsampled: asking for more steps than
    # num_base_steps indexes past the timestep map (CUDA device-side assert).
    num_base_steps = int(model.diffusion.num_base_steps)
    diffusion_steps = args.diffusion_steps if args.diffusion_steps is not None else num_base_steps
    if not 1 <= diffusion_steps <= num_base_steps:
        raise ValueError(
            f"--diffusion_steps must be between 1 and {num_base_steps} "
            f"(this model's num_base_steps); got {diffusion_steps}."
        )
    print(f"Using {diffusion_steps} denoising steps")

    # Match the interactive demo's default: retain one motion token of history.
    patch = model.num_frames_per_token
    history_frames = args.history_frames
    if history_frames is None:
        history_frames = patch
    elif history_frames < patch or history_frames % patch != 0:
        raise ValueError(f"--history_frames must be a positive multiple of {patch} (this model's token size).")
    print(f"Using {history_frames} history frames per autoregressive step")

    # Load constraints
    if args.constraints:
        constraint_lst = load_constraints_lst(args.constraints, model.skeleton)
    else:
        constraint_lst = []

    if constraint_lst:
        print(f"Using {len(constraint_lst)} set of constraints")
        for constraint in constraint_lst:
            print(f"    {type(constraint).__name__} on frames {constraint.frame_indices.tolist()}")
        max_frame_idx = max(int(c.frame_indices.max()) for c in constraint_lst)
        if max_frame_idx >= num_frames:
            raise ValueError(
                f"Constraint frame index {max_frame_idx} exceeds the motion length "
                f"({num_frames} frames = {args.duration}s at {fps} fps); increase --duration."
            )

    if args.seed is not None:
        seed_everything(args.seed)

    # Build the generation inputs (batch of identical prompts/lengths)
    num_samples = args.num_samples
    texts = [text] * num_samples
    lengths = torch.tensor([num_frames] * num_samples, device=device)
    pad_mask = length_to_mask(lengths)
    first_heading_angle = torch.zeros(num_samples, device=device)  # facing +Z

    observed_motion, motion_mask = None, None
    if constraint_lst:
        observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
            constraint_lst,
            lengths,
            to_normalize=True,
            device=device,
        )

    with torch.no_grad():
        if prompt_schedule is None:
            motion = model(
                texts,
                num_frames,
                num_denoising_steps=diffusion_steps,
                pad_mask=pad_mask,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_weight,
                crop_history_length=history_frames,
            )
        else:
            motion = _generate_with_prompt_schedule(
                model,
                prompt_schedule,
                num_samples=num_samples,
                diffusion_steps=diffusion_steps,
                cfg_weight=cfg_weight,
                history_frames=history_frames,
            )
        output = model.motion_rep.inverse(motion, is_normalized=True)

    # G1: postprocessing is disabled (does not work well for this model).
    use_postprocess = "g1" not in resolved_model.lower() and args.postprocess
    if use_postprocess:
        corrected = post_process_motion(
            output["local_rot_mats"],
            output["root_positions"],
            output["foot_contacts"],
            model.skeleton,
            constraint_lst=constraint_lst or None,
        )
        output.update(corrected)

    # Convert SOMA output to somaskel77 for external API
    if isinstance(model.skeleton, SOMASkeleton30):
        output = model.skeleton.output_to_SOMASkeleton77(output)

    output = to_numpy(output)

    n_samples = int(output["posed_joints"].shape[0])
    # Parse the output stem once; all formats (NPZ, CSV) use this base name.
    output_base = _resolve_output_base(args.output)
    output_text = prompt_schedule.summary() if prompt_schedule is not None else text

    # Save the NPZ output
    if n_samples == 1:
        npz_path = _single_file_path(output_base, ".npz")
        print(f"Saving the npz output to {npz_path}")
        save_motion_npz(
            npz_path,
            _select_sample(output, 0, n_samples),
            fps,
            output_text,
            prompt_schedule,
        )
    else:
        out_dir, _, base_name = _output_dir_and_path(output_base, "motion", ".npz")
        print(f"Saving the npz output to {out_dir}/ ({base_name}_00.npz ...)")
        for i in range(n_samples):
            save_motion_npz(
                os.path.join(out_dir, f"{base_name}_{i:02d}.npz"),
                _select_sample(output, i, n_samples),
                fps,
                output_text,
                prompt_schedule,
            )

    # Save the CSV output (MuJoCo qpos) for G1
    if "g1" in resolved_model.lower():
        from ardy.exports.mujoco import MujocoQposConverter

        converter = MujocoQposConverter(model.skeleton)
        qpos = converter.dict_to_qpos(output, device)
        if n_samples == 1:
            csv_path = _single_file_path(output_base, ".csv")
            print(f"Saving the csv output to {csv_path}")
            converter.save_csv(qpos, csv_path)
        else:
            out_dir, _, base_name = _output_dir_and_path(output_base, "qpos", ".csv")
            print(f"Saving the csv output to {out_dir}/ ({base_name}_00.csv ...)")
            converter.save_csv(qpos, os.path.join(out_dir, base_name + ".csv"))


if __name__ == "__main__":
    main()
