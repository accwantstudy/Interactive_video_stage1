# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, model-independent helpers for reading and writing ARDY motion files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .motion_rep.tools import compute_heading_angle
from .prompt_schedule import PromptSchedule
from .skeleton import SkeletonBase, build_skeleton


@dataclass
class MotionData:
    """Editable representation of one generated NPZ motion.

    All tensors live on CPU. source_arrays preserves fields that the editor does
    not understand, so saving an edited motion remains forwards compatible with new
    model outputs.
    """

    skeleton: SkeletonBase
    local_rot_mats: torch.Tensor
    global_rot_mats: torch.Tensor
    posed_joints: torch.Tensor
    root_positions: torch.Tensor
    fps: float
    text: str = ""
    foot_contacts: torch.Tensor | None = None
    prompt_schedule: PromptSchedule | None = None
    source_arrays: dict[str, np.ndarray] | None = None

    @property
    def num_frames(self) -> int:
        return int(self.posed_joints.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.posed_joints.shape[1])

    def clone(self) -> "MotionData":
        return MotionData(
            skeleton=self.skeleton,
            local_rot_mats=self.local_rot_mats.clone(),
            global_rot_mats=self.global_rot_mats.clone(),
            posed_joints=self.posed_joints.clone(),
            root_positions=self.root_positions.clone(),
            fps=self.fps,
            text=self.text,
            foot_contacts=None if self.foot_contacts is None else self.foot_contacts.clone(),
            prompt_schedule=self.prompt_schedule,
            source_arrays=None if self.source_arrays is None else dict(self.source_arrays),
        )

    def text_at(self, frame_idx: int) -> str:
        """Return the active prompt at a frame, falling back to legacy scalar text."""
        if self.prompt_schedule is None:
            return self.text
        return self.prompt_schedule.text_at(frame_idx)

    def replace_pose(self, local_rot_mats: torch.Tensor, root_positions: torch.Tensor) -> None:
        """Replace the editable pose and recompute all FK-derived fields."""
        local_rot_mats = local_rot_mats.detach().to(device="cpu", dtype=torch.float32)
        root_positions = root_positions.detach().to(device="cpu", dtype=torch.float32)
        if local_rot_mats.shape != self.local_rot_mats.shape:
            raise ValueError(
                f"Expected local rotations with shape {tuple(self.local_rot_mats.shape)}, "
                f"got {tuple(local_rot_mats.shape)}"
            )
        if root_positions.shape != self.root_positions.shape:
            raise ValueError(
                f"Expected root positions with shape {tuple(self.root_positions.shape)}, "
                f"got {tuple(root_positions.shape)}"
            )
        global_rot_mats, posed_joints, _ = self.skeleton.fk(local_rot_mats, root_positions)
        self.local_rot_mats = local_rot_mats
        self.global_rot_mats = global_rot_mats
        self.posed_joints = posed_joints
        self.root_positions = root_positions

    def replace_pose_frames(
        self,
        frame_indices: torch.Tensor,
        local_rot_mats: torch.Tensor,
        root_positions: torch.Tensor,
    ) -> None:
        """Replace selected frames and run FK only for that temporal window."""
        frame_indices = torch.as_tensor(frame_indices, dtype=torch.long, device="cpu")
        local_rot_mats = local_rot_mats.detach().to(device="cpu", dtype=torch.float32)
        root_positions = root_positions.detach().to(device="cpu", dtype=torch.float32)
        expected_rot_shape = self.local_rot_mats[frame_indices].shape
        expected_root_shape = self.root_positions[frame_indices].shape
        if local_rot_mats.shape != expected_rot_shape:
            raise ValueError(f"Expected selected rotations with shape {expected_rot_shape}, got {local_rot_mats.shape}")
        if root_positions.shape != expected_root_shape:
            raise ValueError(f"Expected selected roots with shape {expected_root_shape}, got {root_positions.shape}")

        global_rot_mats, posed_joints, _ = self.skeleton.fk(local_rot_mats, root_positions)
        self.local_rot_mats[frame_indices] = local_rot_mats
        self.global_rot_mats[frame_indices] = global_rot_mats
        self.posed_joints[frame_indices] = posed_joints
        self.root_positions[frame_indices] = root_positions

    def save(self, path: str | Path) -> Path:
        """Save an edited motion without discarding unknown source fields."""
        output_path = Path(path)
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        arrays = dict(self.source_arrays or {})
        arrays.update(
            local_rot_mats=self.local_rot_mats.numpy(),
            global_rot_mats=self.global_rot_mats.numpy(),
            posed_joints=self.posed_joints.numpy(),
            root_positions=self.root_positions.numpy(),
            fps=np.asarray(self.fps),
            text=np.asarray(self.text),
        )
        if "smooth_root_pos" in arrays:
            arrays["smooth_root_pos"] = self.root_positions.numpy()
        if "global_root_heading" in arrays:
            heading = compute_heading_angle(self.posed_joints, self.skeleton)
            arrays["global_root_heading"] = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1).numpy()
        if self.foot_contacts is not None:
            arrays["foot_contacts"] = self.foot_contacts.numpy()
        if self.prompt_schedule is not None:
            arrays.update(self.prompt_schedule.to_npz_arrays())
        np.savez(output_path, **arrays)
        return output_path


def _tensor(arrays: dict[str, np.ndarray], key: str) -> torch.Tensor | None:
    value = arrays.get(key)
    return None if value is None else torch.from_numpy(np.asarray(value)).to(dtype=torch.float32)


def load_motion_npz(path: str | Path, *, require_rotations: bool = True) -> MotionData:
    """Load a generated motion and reconstruct any omitted FK-derived arrays.

    Args:
        path: ARDY NPZ file.
        require_rotations: If false, a positions-only file is accepted by filling
            identity rotations. This is intended for skeleton-only rendering; such
            a motion should not be edited and saved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    candidates = [arrays.get("posed_joints"), arrays.get("local_rot_mats"), arrays.get("global_rot_mats")]
    shape_source = next((value for value in candidates if value is not None), None)
    if shape_source is None or shape_source.ndim < 3:
        raise ValueError(f"{path} does not contain posed_joints, local_rot_mats, or global_rot_mats")
    num_joints = int(shape_source.shape[1])
    skeleton = build_skeleton(num_joints)

    posed_joints = _tensor(arrays, "posed_joints")
    local_rot_mats = _tensor(arrays, "local_rot_mats")
    global_rot_mats = _tensor(arrays, "global_rot_mats")
    root_positions = _tensor(arrays, "root_positions")

    if root_positions is None and posed_joints is not None:
        root_positions = posed_joints[:, skeleton.root_idx].clone()
    if global_rot_mats is not None and local_rot_mats is None:
        local_rot_mats = skeleton.global_rots_to_local_rots(global_rot_mats)
    if local_rot_mats is not None and root_positions is not None and (
        global_rot_mats is None or posed_joints is None
    ):
        fk_rots, fk_positions, _ = skeleton.fk(local_rot_mats, root_positions)
        global_rot_mats = fk_rots if global_rot_mats is None else global_rot_mats
        posed_joints = fk_positions if posed_joints is None else posed_joints

    if posed_joints is None:
        raise ValueError(f"{path} does not contain enough information to reconstruct joint positions")
    if local_rot_mats is None or global_rot_mats is None:
        if require_rotations:
            raise ValueError(
                f"{path} is positions-only. The motion editor requires local_rot_mats or global_rot_mats."
            )
        identity = torch.eye(3, dtype=torch.float32).expand(posed_joints.shape[0], num_joints, 3, 3).clone()
        local_rot_mats = identity
        global_rot_mats = identity.clone()
    if root_positions is None:
        root_positions = posed_joints[:, skeleton.root_idx].clone()

    foot_array = arrays.get("foot_contacts")
    foot_contacts = None if foot_array is None else torch.from_numpy(np.asarray(foot_array)).clone()
    fps = float(np.asarray(arrays.get("fps", 20.0)).item())
    text = str(np.asarray(arrays.get("text", "")).item())
    prompt_schedule = PromptSchedule.from_npz_arrays(arrays, num_frames=int(posed_joints.shape[0]))
    return MotionData(
        skeleton=skeleton,
        local_rot_mats=local_rot_mats.contiguous(),
        global_rot_mats=global_rot_mats.contiguous(),
        posed_joints=posed_joints.contiguous(),
        root_positions=root_positions.contiguous(),
        fps=fps,
        text=text,
        foot_contacts=None if foot_contacts is None else foot_contacts.contiguous(),
        prompt_schedule=prompt_schedule,
        source_arrays=arrays,
    )
