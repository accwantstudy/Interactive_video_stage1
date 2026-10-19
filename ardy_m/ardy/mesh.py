# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display-independent deformation of the packaged Core, SOMA, and G1 meshes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from .assets import skeleton_asset_path
from .motion_io import MotionData
from .skeleton import CoreSkeleton27, G1Skeleton34, SOMASkeleton30, SOMASkeleton77
from .viz.core_skin import CoreSkin
from .viz.soma_skin import SOMASkin


class MotionMesh:
    """A triangle mesh whose vertices can be evaluated for any motion frame."""

    faces: np.ndarray

    def vertices(self, frame_idx: int) -> np.ndarray:
        raise NotImplementedError


class SkinnedMotionMesh(MotionMesh):
    """Linear-blend-skinned Core or SOMA mesh with a small frame cache."""

    def __init__(self, motion: MotionData, cache_frames: int = 16):
        self.motion = motion
        self.cache_frames = max(1, int(cache_frames))
        if isinstance(motion.skeleton, CoreSkeleton27):
            self.skin = CoreSkin(motion.skeleton)
        elif isinstance(motion.skeleton, (SOMASkeleton30, SOMASkeleton77)):
            self.skin = SOMASkin(motion.skeleton)
        else:
            raise TypeError(f"No deformable skin is available for {type(motion.skeleton).__name__}")
        self.faces = self.skin.faces.detach().cpu().numpy().astype(np.int32, copy=False)
        self._cache_start = 0
        self._cache_end = 0
        self._vertices_cache: np.ndarray | None = None

    def vertices(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if not self._cache_start <= frame_idx < self._cache_end:
            start = (frame_idx // self.cache_frames) * self.cache_frames
            end = min(self.motion.num_frames, start + self.cache_frames)
            with torch.inference_mode():
                vertices = self.skin.skin(
                    self.motion.global_rot_mats[start:end],
                    self.motion.posed_joints[start:end],
                    rot_is_global=True,
                )
            self._vertices_cache = vertices.detach().cpu().numpy()
            self._cache_start = start
            self._cache_end = end
        assert self._vertices_cache is not None
        return self._vertices_cache[frame_idx - self._cache_start]


def _quaternion_matrix(wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(wxyz, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _parse_vector(value: str | None, default: np.ndarray) -> np.ndarray:
    if value is None:
        return default.copy()
    return np.asarray([float(component) for component in value.split()], dtype=np.float64)


def _cluster_once(vertices: np.ndarray, faces: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Simplify a rigid part using deterministic vertex clustering."""
    origin = vertices.min(axis=0)
    cells = np.floor((vertices - origin) / max(cell_size, 1e-12)).astype(np.int64)
    _unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    clustered = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(clustered, inverse, vertices)
    clustered /= counts[:, None]

    remapped = inverse[faces]
    valid = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 2] != remapped[:, 0])
    )
    remapped = remapped[valid]
    if len(remapped):
        canonical = np.sort(remapped, axis=1)
        _unique_faces, first = np.unique(canonical, axis=0, return_index=True)
        remapped = remapped[np.sort(first)]
    return clustered.astype(np.float32), remapped.astype(np.int32)


def _simplify_rigid_part(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= target_faces:
        return vertices.astype(np.float32, copy=False), faces.astype(np.int32, copy=False)
    extent = np.ptp(vertices, axis=0)
    high = max(float(np.linalg.norm(extent)), 1e-6)
    low = high / 100000.0
    best_vertices = vertices
    best_faces = faces
    best_error = abs(len(faces) - target_faces)
    for _ in range(16):
        cell_size = (low + high) * 0.5
        candidate_vertices, candidate_faces = _cluster_once(vertices, faces, cell_size)
        candidate_error = abs(len(candidate_faces) - target_faces)
        if len(candidate_faces) > 0 and candidate_error < best_error:
            best_vertices = candidate_vertices
            best_faces = candidate_faces
            best_error = candidate_error
        if len(candidate_faces) > target_faces:
            low = cell_size
        else:
            high = cell_size
    return (
        np.asarray(best_vertices, dtype=np.float32),
        np.asarray(best_faces, dtype=np.int32),
    )


class G1MotionMesh(MotionMesh):
    """Rigidly articulated G1 STL geometry bound from the packaged MJCF."""

    _mujoco_to_ardy = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    def __init__(self, motion: MotionData, max_faces: int = 30000):
        if not isinstance(motion.skeleton, G1Skeleton34):
            raise TypeError("G1MotionMesh requires a G1Skeleton34 motion")
        try:
            import trimesh
        except ImportError as error:
            raise RuntimeError("G1 mesh rendering requires trimesh; install ardy[video].") from error

        self.motion = motion
        mesh_dir = skeleton_asset_path("g1skel34", "meshes", "g1")
        xml_path = skeleton_asset_path("g1skel34", "xml", "g1.xml")
        root = ET.parse(xml_path).getroot()
        mesh_files = {
            node.get("name"): node.get("file")
            for node in root.findall(".//asset/mesh")
            if node.get("name") and node.get("file")
        }

        records: list[tuple[int, Path, np.ndarray, np.ndarray]] = []
        seen: set[tuple[int, str]] = set()

        def walk_body(body: ET.Element, inherited_joint: int | None) -> None:
            joint_idx = inherited_joint
            joint = body.find("joint")
            if joint is not None and joint.get("name"):
                joint_name = joint.get("name").replace("_joint", "_skel")
                joint_idx = motion.skeleton.bone_index.get(joint_name)
            elif body.find("freejoint") is not None:
                joint_idx = motion.skeleton.root_idx

            if joint_idx is not None:
                for geom in body.findall("geom"):
                    mesh_name = geom.get("mesh")
                    mesh_file = mesh_files.get(mesh_name)
                    if mesh_file is None or (joint_idx, mesh_file) in seen:
                        continue
                    seen.add((joint_idx, mesh_file))
                    geom_pos = _parse_vector(geom.get("pos"), np.zeros(3, dtype=np.float64))
                    geom_rot = _quaternion_matrix(
                        _parse_vector(geom.get("quat"), np.array([1.0, 0.0, 0.0, 0.0]))
                    )
                    geom_pos = self._mujoco_to_ardy @ geom_pos
                    geom_rot = self._mujoco_to_ardy @ geom_rot @ self._mujoco_to_ardy.T
                    records.append((joint_idx, mesh_dir / mesh_file, geom_pos, geom_rot))
            for child in body.findall("body"):
                walk_body(child, joint_idx)

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"No worldbody found in {xml_path}")
        for body in worldbody.findall("body"):
            walk_body(body, None)

        loaded = []
        total_faces = 0
        for joint_idx, mesh_path, geom_pos, geom_rot in records:
            mesh = trimesh.load_mesh(mesh_path, process=True)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.dump()))
            vertices = np.asarray(mesh.vertices, dtype=np.float64) @ self._mujoco_to_ardy.T
            vertices = vertices @ geom_rot.T + geom_pos
            faces = np.asarray(mesh.faces, dtype=np.int32)
            loaded.append((joint_idx, vertices, faces))
            total_faces += len(faces)
        if not loaded:
            raise FileNotFoundError(f"No G1 visual meshes were loaded from {mesh_dir}")

        face_budget = max(1000, int(max_faces))
        vertex_parts = []
        face_parts = []
        joint_parts = []
        vertex_offset = 0
        for joint_idx, vertices, faces in loaded:
            target = max(64, round(face_budget * len(faces) / total_faces))
            vertices, faces = _simplify_rigid_part(vertices, faces, target)
            vertex_parts.append(vertices)
            face_parts.append(faces + vertex_offset)
            joint_parts.append(np.full(len(vertices), joint_idx, dtype=np.int32))
            vertex_offset += len(vertices)
        self.local_vertices = np.concatenate(vertex_parts, axis=0)
        self.faces = np.concatenate(face_parts, axis=0)
        self.vertex_joint_indices = np.concatenate(joint_parts, axis=0)

    def vertices(self, frame_idx: int) -> np.ndarray:
        rotations = self.motion.global_rot_mats[frame_idx].numpy()[self.vertex_joint_indices]
        positions = self.motion.posed_joints[frame_idx].numpy()[self.vertex_joint_indices]
        return np.einsum("vij,vj->vi", rotations, self.local_vertices) + positions


def build_motion_mesh(
    motion: MotionData,
    *,
    cache_frames: int = 16,
    g1_max_faces: int = 30000,
) -> MotionMesh:
    """Build the packaged mesh appropriate for a motion's skeleton."""
    if isinstance(motion.skeleton, (CoreSkeleton27, SOMASkeleton30, SOMASkeleton77)):
        return SkinnedMotionMesh(motion, cache_frames=cache_frames)
    if isinstance(motion.skeleton, G1Skeleton34):
        return G1MotionMesh(motion, max_faces=g1_max_faces)
    raise TypeError(f"No packaged mesh is available for {type(motion.skeleton).__name__}")
