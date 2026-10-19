# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constraint-aware inverse kinematics and temporal motion editing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .motion_io import MotionData
from .skeleton import SkeletonBase


SCOPE_ANCESTORS = {
    "Limb": 4,
    "Body": 6,
    "Whole body": None,
}


@dataclass(frozen=True)
class MotionEditResult:
    """Summary of one temporal joint-drag operation."""

    frame_indices: torch.Tensor
    weights: torch.Tensor
    target_error: float
    ancestor_indices: tuple[int, ...]


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def temporal_blend_weights(
    num_frames: int,
    keyframe_idx: int,
    ease_in_frames: int,
    hold_frames: int,
    ease_out_frames: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create an ease-in, hold, ease-out envelope centered on an edit keyframe."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if not 0 <= keyframe_idx < num_frames:
        raise ValueError(f"keyframe_idx {keyframe_idx} is outside a {num_frames}-frame motion")
    if min(ease_in_frames, hold_frames, ease_out_frames) < 0:
        raise ValueError("Temporal durations cannot be negative")

    frame = torch.arange(num_frames, dtype=dtype)
    weights = torch.zeros(num_frames, dtype=dtype)

    if ease_in_frames == 0:
        weights[keyframe_idx] = 1.0
    else:
        start = max(0, keyframe_idx - ease_in_frames)
        ramp_frames = frame[start : keyframe_idx + 1]
        weights[start : keyframe_idx + 1] = smoothstep(
            (ramp_frames - (keyframe_idx - ease_in_frames)) / float(ease_in_frames)
        )

    hold_end = min(num_frames - 1, keyframe_idx + hold_frames)
    weights[keyframe_idx : hold_end + 1] = 1.0

    if ease_out_frames > 0 and hold_end + 1 < num_frames:
        end = min(num_frames - 1, keyframe_idx + hold_frames + ease_out_frames)
        ramp_frames = frame[hold_end + 1 : end + 1]
        weights[hold_end + 1 : end + 1] = smoothstep(
            1.0 - (ramp_frames - (keyframe_idx + hold_frames)) / float(ease_out_frames)
        )
    return weights


def ancestor_chain(
    skeleton: SkeletonBase,
    joint_idx: int,
    scope: str = "Limb",
) -> list[int]:
    """Return IK pivots ordered from the dragged joint's parent toward the root."""
    if not 0 <= joint_idx < skeleton.nbjoints:
        raise ValueError(f"Invalid joint index {joint_idx}")
    if scope not in SCOPE_ANCESTORS:
        raise ValueError(f"Unknown scope {scope!r}; choose one of {list(SCOPE_ANCESTORS)}")

    chain = []
    current = int(skeleton.joint_parents[joint_idx])
    while current >= 0:
        if scope != "Whole body" and current == skeleton.root_idx:
            break
        chain.append(current)
        current = int(skeleton.joint_parents[current])

    limit = SCOPE_ANCESTORS[scope]
    return chain if limit is None else chain[:limit]


def _rotation_between(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    step_size: float,
    max_angle_radians: float,
) -> torch.Tensor:
    """Batched shortest rotations from source vectors to target vectors."""
    eps = 1e-7
    source_norm = torch.linalg.vector_norm(source, dim=-1, keepdim=True)
    target_norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
    valid = (source_norm > eps) & (target_norm > eps)
    a = source / source_norm.clamp_min(eps)
    b = target / target_norm.clamp_min(eps)

    dot = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    cross = torch.cross(a, b, dim=-1)
    cross_norm = torch.linalg.vector_norm(cross, dim=-1, keepdim=True)

    seed_x = torch.zeros_like(a)
    seed_x[..., 0] = 1.0
    seed_y = torch.zeros_like(a)
    seed_y[..., 1] = 1.0
    seed = torch.where((a[..., :1].abs() < 0.9), seed_x, seed_y)
    opposite_axis = torch.cross(a, seed, dim=-1)
    opposite_axis = opposite_axis / torch.linalg.vector_norm(opposite_axis, dim=-1, keepdim=True).clamp_min(eps)
    axis = torch.where(cross_norm > eps, cross / cross_norm.clamp_min(eps), opposite_axis)

    angle = torch.acos(dot) * float(step_size)
    angle = angle.clamp(max=float(max_angle_radians))
    angle = torch.where(valid.squeeze(-1), angle, torch.zeros_like(angle))
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(*axis.shape[:-1], 3, 3)
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(skew)
    sin = torch.sin(angle)[..., None, None]
    cos = torch.cos(angle)[..., None, None]
    return identity + sin * skew + (1.0 - cos) * (skew @ skew)


def solve_position_ik(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    skeleton: SkeletonBase,
    joint_idx: int,
    targets: torch.Tensor,
    *,
    scope: str = "Limb",
    iterations: int = 10,
    tolerance: float = 1e-3,
    step_size: float = 0.75,
    max_step_degrees: float = 25.0,
    whole_body_root_translation: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Solve one positional constraint for a batch of poses with CCD IK.

    Bone lengths remain exact because the solver updates local rotations and always
    recomputes positions with forward kinematics. A whole-body solve may optionally
    absorb a fraction of the final residual as root translation.
    """
    if local_rot_mats.ndim != 4 or root_positions.ndim != 2 or targets.ndim != 2:
        raise ValueError("Expected rotations [B,J,3,3], roots [B,3], and targets [B,3]")
    if not (len(local_rot_mats) == len(root_positions) == len(targets)):
        raise ValueError("Batch dimensions must match")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < step_size <= 1.0:
        raise ValueError("step_size must be in (0, 1]")

    rotations = local_rot_mats.clone()
    roots = root_positions.clone()
    chain = ancestor_chain(skeleton, joint_idx, scope)
    if not chain:
        return rotations, roots, tuple()

    max_step = torch.deg2rad(torch.tensor(max_step_degrees, dtype=rotations.dtype)).item()
    for _ in range(iterations):
        global_rots, positions, _ = skeleton.fk(rotations, roots)
        errors = torch.linalg.vector_norm(targets - positions[:, joint_idx], dim=-1)
        if bool(torch.all(errors <= tolerance)):
            break

        for pivot_idx in chain:
            global_rots, positions, _ = skeleton.fk(rotations, roots)
            pivot = positions[:, pivot_idx]
            delta_world = _rotation_between(
                positions[:, joint_idx] - pivot,
                targets - pivot,
                step_size=step_size,
                max_angle_radians=max_step,
            )
            parent_idx = int(skeleton.joint_parents[pivot_idx])
            if parent_idx < 0:
                delta_local = delta_world
            else:
                parent_global = global_rots[:, parent_idx]
                delta_local = parent_global.transpose(-2, -1) @ delta_world @ parent_global
            candidate = delta_local @ rotations[:, pivot_idx]
            u, _singular_values, vh = torch.linalg.svd(candidate)
            correction = torch.eye(3, dtype=candidate.dtype, device=candidate.device)
            correction = correction.expand(len(candidate), 3, 3).clone()
            correction[:, -1, -1] = torch.sign(torch.linalg.det(u @ vh))
            rotations[:, pivot_idx] = u @ correction @ vh

    if scope == "Whole body" and whole_body_root_translation > 0.0:
        _, positions, _ = skeleton.fk(rotations, roots)
        roots = roots + whole_body_root_translation * (targets - positions[:, joint_idx])

    return rotations, roots, tuple(chain)


def apply_temporal_joint_drag(
    base_motion: MotionData,
    keyframe_idx: int,
    joint_idx: int,
    target_position: torch.Tensor,
    *,
    scope: str = "Limb",
    ease_in_frames: int = 10,
    hold_frames: int = 5,
    ease_out_frames: int = 10,
    iterations: int = 10,
    tolerance: float = 1e-3,
    whole_body_root_translation: float = 0.0,
) -> tuple[MotionData, MotionEditResult]:
    """Apply one world-space joint translation over a smooth temporal envelope."""
    target_position = torch.as_tensor(target_position, dtype=torch.float32).reshape(3)
    weights = temporal_blend_weights(
        base_motion.num_frames,
        keyframe_idx,
        ease_in_frames,
        hold_frames,
        ease_out_frames,
    )
    affected = torch.where(weights > 0.0)[0]
    edited = base_motion.clone()

    displacement = target_position - base_motion.posed_joints[keyframe_idx, joint_idx]
    if joint_idx == base_motion.skeleton.root_idx:
        edited_roots = base_motion.root_positions[affected] + weights[affected, None] * displacement
        edited.replace_pose_frames(affected, base_motion.local_rot_mats[affected], edited_roots)
        chain = tuple()
    else:
        local_subset = base_motion.local_rot_mats[affected]
        root_subset = base_motion.root_positions[affected]
        targets = base_motion.posed_joints[affected, joint_idx] + weights[affected, None] * displacement
        solved_rotations, solved_roots, chain = solve_position_ik(
            local_subset,
            root_subset,
            base_motion.skeleton,
            joint_idx,
            targets,
            scope=scope,
            iterations=iterations,
            tolerance=tolerance,
            whole_body_root_translation=whole_body_root_translation,
        )
        edited.replace_pose_frames(affected, solved_rotations, solved_roots)

    error = torch.linalg.vector_norm(edited.posed_joints[keyframe_idx, joint_idx] - target_position).item()
    return edited, MotionEditResult(
        frame_indices=affected,
        weights=weights,
        target_error=float(error),
        ancestor_indices=chain,
    )
