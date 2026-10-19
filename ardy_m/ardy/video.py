# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless skeleton rendering and FFmpeg-backed video writing."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .mesh import build_motion_mesh
from .motion_io import MotionData


@dataclass(frozen=True)
class OrbitCamera:
    """Simple Y-up orbit camera used by the headless renderer."""

    azimuth_degrees: float = 45.0
    elevation_degrees: float = 15.0
    distance: float | None = None
    fov_degrees: float = 45.0
    target: tuple[float, float, float] | None = None
    follow_root: bool = False


def front_facing_camera(
    motion: MotionData, camera: OrbitCamera | None = None, *, frame_idx: int = 0,
) -> OrbitCamera:
    """Resolve azimuth relative to a pose's hip heading, preserving other controls.

    With no camera supplied, use a level front view. With a camera, its azimuth
    is an offset from the pose's heading. Resolve once at export start so later
    turns remain visible, even when following the root.
    """
    if not 0 <= frame_idx < motion.num_frames:
        raise ValueError(f"Facing frame {frame_idx} is outside the motion")
    camera = camera or OrbitCamera(azimuth_degrees=0, elevation_degrees=0)
    positions = motion.posed_joints[frame_idx].numpy()
    right_hip, left_hip = motion.skeleton.hip_joint_idx
    across = positions[right_hip] - positions[left_hip]
    if np.isfinite(across).all() and np.linalg.norm(across[[0, 2]]) < 1e-6:
        bones = motion.skeleton.bone_index
        shoulders = (
            ("right_shoulder_pitch_skel", "left_shoulder_pitch_skel")
            if motion.skeleton.name == "g1skel34" else ("RightArm", "LeftArm")
        )
        if all(name in bones for name in shoulders):
            across = positions[bones[shoulders[0]]] - positions[bones[shoulders[1]]]
    if not np.isfinite(across).all() or np.linalg.norm(across[[0, 2]]) < 1e-6:
        raise ValueError("Cannot infer initial facing direction; use --no-face-front and --azimuth")
    front = np.cross([0.0, 1.0, 0.0], across)
    heading = np.rad2deg(np.arctan2(front[0], front[2]))
    return replace(camera, azimuth_degrees=float(heading + camera.azimuth_degrees))


class FfmpegVideoWriter:
    """Stream RGB frames to FFmpeg without storing them all in memory."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float, *, ffmpeg: str | None = None):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        if self.width < 2 or self.height < 2:
            raise ValueError("Video dimensions must be at least 2x2")
        if self.width % 2 or self.height % 2:
            raise ValueError("Video width and height must be even for yuv420p output")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.path.suffix.lower() not in {".mp4", ".webm"}:
            raise ValueError("Output must end in .mp4 or .webm")

        executable = ffmpeg or shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("FFmpeg was not found on PATH. Install FFmpeg or pass its executable path.")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        codec_args = (
            ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            if self.path.suffix.lower() == ".mp4"
            else ["-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0", "-pix_fmt", "yuv420p"]
        )
        command = [
            executable,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:g}",
            "-i",
            "-",
            "-an",
            *codec_args,
            str(self.path),
        ]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Expected frame shape {(self.height, self.width, 3)}, got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(self._read_error("FFmpeg stopped while writing video")) from error

    def _read_error(self, prefix: str) -> str:
        stderr = b""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read()
            self._process.stderr.close()
        detail = stderr.decode("utf-8", errors="replace").strip()
        return f"{prefix}: {detail}" if detail else prefix

    def close(self) -> Path:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        return_code = self._process.wait()
        stderr = b""
        if self._process.stderr is not None and not self._process.stderr.closed:
            stderr = self._process.stderr.read()
            self._process.stderr.close()
        if return_code:
            detail = stderr.decode("utf-8", errors="replace").strip()
            message = f"FFmpeg exited with status {return_code}"
            raise RuntimeError(f"{message}: {detail}" if detail else message)
        return self.path

    def __enter__(self) -> "FfmpegVideoWriter":
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            if self._process.stdin is not None and not self._process.stdin.closed:
                self._process.stdin.close()
            self._process.terminate()
            self._process.wait()
            if self._process.stderr is not None and not self._process.stderr.closed:
                self._process.stderr.close()


class SkeletonRenderer:
    """Pillow-based 3D skeleton renderer with deterministic camera controls."""

    def __init__(
        self,
        motion: MotionData,
        width: int = 1280,
        height: int = 720,
        camera: OrbitCamera | None = None,
        *,
        supersample: int = 2,
        background: tuple[int, int, int] = (17, 21, 29),
        show_grid: bool = True,
        show_text: bool = False,
        trail_seconds: float = 0.0,
    ):
        if width < 2 or height < 2:
            raise ValueError("width and height must be at least 2")
        if width % 2 or height % 2:
            raise ValueError("width and height must be even")
        if supersample < 1:
            raise ValueError("supersample must be positive")
        self.motion = motion
        self.width = int(width)
        self.height = int(height)
        self.scale = int(supersample)
        self.render_width = self.width * self.scale
        self.render_height = self.height * self.scale
        self.camera = camera or OrbitCamera()
        if not 1.0 <= self.camera.fov_degrees < 179.0:
            raise ValueError("camera field of view must be between 1 and 179 degrees")
        if self.camera.distance is not None and self.camera.distance <= 0:
            raise ValueError("camera distance must be positive")
        self.background = background
        self.show_grid = show_grid
        self.show_text = show_text
        self.trail_frames = max(0, round(float(trail_seconds) * motion.fps))
        try:
            self.font = ImageFont.load_default(size=max(11, 13 * self.scale))
        except TypeError:  # Pillow 9.x did not expose the size argument.
            self.font = ImageFont.load_default()

        positions = motion.posed_joints.numpy()
        roots = positions[:, motion.skeleton.root_idx]
        body_relative = positions - roots[:, None]
        body_min = body_relative.reshape(-1, 3).min(axis=0)
        body_max = body_relative.reshape(-1, 3).max(axis=0)
        self.body_center_offset = (body_min + body_max) * 0.5
        body_radius = np.linalg.norm(body_max - body_min) * 0.5

        if self.camera.target is None and not self.camera.follow_root:
            all_min = positions.reshape(-1, 3).min(axis=0)
            all_max = positions.reshape(-1, 3).max(axis=0)
            self.static_target = (all_min + all_max) * 0.5
            radius = np.linalg.norm(all_max - all_min) * 0.5
        elif self.camera.target is not None:
            self.static_target = np.asarray(self.camera.target, dtype=np.float64)
            radius = body_radius
        else:
            self.static_target = None
            radius = body_radius

        fov = np.deg2rad(self.camera.fov_degrees)
        self.distance = (
            self.camera.distance
            if self.camera.distance is not None
            else max(1.0, 1.25 * radius / np.tan(fov * 0.5))
        )
        self.focal = 0.5 * min(self.render_width, self.render_height) / np.tan(fov * 0.5)

        self.parents = motion.skeleton.joint_parents.numpy()
        self.root_path = roots.copy()
        self.grid_half_extent = max(3.0, min(20.0, self.distance * 0.9))

    def _target_for_frame(self, frame_idx: int) -> np.ndarray:
        if self.camera.follow_root:
            root = self.motion.posed_joints[frame_idx, self.motion.skeleton.root_idx].numpy()
            return root + self.body_center_offset
        assert self.static_target is not None
        return self.static_target

    def _camera_basis(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = self._target_for_frame(frame_idx)
        azimuth = np.deg2rad(self.camera.azimuth_degrees)
        elevation = np.deg2rad(self.camera.elevation_degrees)
        offset = self.distance * np.array(
            [
                np.sin(azimuth) * np.cos(elevation),
                np.sin(elevation),
                np.cos(azimuth) * np.cos(elevation),
            ]
        )
        position = target + offset
        forward = target - position
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)
        right /= right_norm
        up = np.cross(right, forward)
        return position, right, up, forward

    def _project(
        self,
        points: np.ndarray,
        basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        camera_position, right, up, forward = basis
        relative = np.asarray(points) - camera_position
        depth = relative @ forward
        safe_depth = np.maximum(depth, 1e-4)
        x = self.render_width * 0.5 + self.focal * (relative @ right) / safe_depth
        y = self.render_height * 0.5 - self.focal * (relative @ up) / safe_depth
        return np.stack((x, y), axis=-1), depth

    def _draw_grid(
        self,
        draw: ImageDraw.ImageDraw,
        basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        target: np.ndarray,
    ) -> None:
        extent = self.grid_half_extent
        spacing = 1.0
        center_x = np.floor(target[0] / spacing) * spacing
        center_z = np.floor(target[2] / spacing) * spacing
        values = np.arange(-extent, extent + spacing, spacing)
        lines = []
        for value in values:
            lines.append([[center_x + value, 0.0, center_z - extent], [center_x + value, 0.0, center_z + extent]])
            lines.append([[center_x - extent, 0.0, center_z + value], [center_x + extent, 0.0, center_z + value]])
        for line in np.asarray(lines):
            projected, depth = self._project(line, basis)
            if np.all(depth > 0):
                draw.line([tuple(point) for point in projected], fill=(48, 55, 68), width=self.scale)

    def render_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(np.clip(frame_idx, 0, self.motion.num_frames - 1))
        image = Image.new("RGB", (self.render_width, self.render_height), self.background)
        draw = ImageDraw.Draw(image, "RGB")
        basis = self._camera_basis(frame_idx)
        target = self._target_for_frame(frame_idx)

        if self.show_grid:
            self._draw_grid(draw, basis, target)

        if self.trail_frames > 0:
            start = max(0, frame_idx - self.trail_frames)
            trail, trail_depth = self._project(self.root_path[start : frame_idx + 1], basis)
            valid = trail_depth > 0
            if valid.sum() > 1:
                draw.line(
                    [tuple(point) for point in trail[valid]],
                    fill=(72, 145, 230),
                    width=max(1, 2 * self.scale),
                )

        positions = self.motion.posed_joints[frame_idx].numpy()
        projected, depth = self._project(positions, basis)
        depth_order = sorted(range(1, self.motion.num_joints), key=lambda idx: depth[idx], reverse=True)
        for joint_idx in depth_order:
            parent_idx = int(self.parents[joint_idx])
            if parent_idx < 0 or depth[joint_idx] <= 0 or depth[parent_idx] <= 0:
                continue
            shade = int(np.clip(235 - 30 * (depth[joint_idx] / max(self.distance, 1e-4)), 115, 225))
            draw.line(
                [tuple(projected[parent_idx]), tuple(projected[joint_idx])],
                fill=(80, shade, 220),
                width=max(2, 5 * self.scale),
            )

        contact_indices = set()
        if self.motion.foot_contacts is not None and frame_idx < len(self.motion.foot_contacts):
            contacts = self.motion.foot_contacts[frame_idx].bool().numpy()
            foot_indices = np.asarray(self.motion.skeleton.foot_joint_idx, dtype=int)
            count = min(len(contacts), len(foot_indices))
            contact_indices = set(foot_indices[:count][contacts[:count]].tolist())

        radius = 5 * self.scale
        for joint_idx in np.argsort(depth)[::-1]:
            if depth[joint_idx] <= 0:
                continue
            x, y = projected[joint_idx]
            color = (190, 75, 245) if int(joint_idx) in contact_indices else (245, 205, 80)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        if self.show_text:
            elapsed = frame_idx / self.motion.fps
            title = self.motion.text_at(frame_idx).strip() or "ARDY motion"
            draw.rounded_rectangle(
                (
                    18 * self.scale,
                    18 * self.scale,
                    min(self.render_width - 18, (35 + 8 * len(title)) * self.scale),
                    70 * self.scale,
                ),
                radius=8 * self.scale,
                fill=(5, 8, 13),
            )
            draw.text((30 * self.scale, 27 * self.scale), title, fill=(238, 242, 248), font=self.font)
            draw.text(
                (30 * self.scale, 48 * self.scale),
                f"{elapsed:6.2f}s   frame {frame_idx + 1}/{self.motion.num_frames}",
                fill=(145, 160, 181),
                font=self.font,
            )

        if self.scale > 1:
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.asarray(image)


class MeshRenderer(SkeletonRenderer):
    """CPU mesh renderer using the packaged Core, SOMA, or G1 visual geometry."""

    def __init__(
        self,
        motion: MotionData,
        width: int = 1280,
        height: int = 720,
        camera: OrbitCamera | None = None,
        *,
        supersample: int = 2,
        background: tuple[int, int, int] = (17, 21, 29),
        show_grid: bool = True,
        show_text: bool = False,
        trail_seconds: float = 0.0,
        mesh_color: tuple[int, int, int] = (82, 148, 224),
        mesh_cache_frames: int = 16,
        g1_max_faces: int = 30000,
    ):
        super().__init__(
            motion,
            width,
            height,
            camera,
            supersample=supersample,
            background=background,
            show_grid=show_grid,
            show_text=show_text,
            trail_seconds=trail_seconds,
        )
        if len(mesh_color) != 3 or any(not 0 <= channel <= 255 for channel in mesh_color):
            raise ValueError("mesh_color must contain three values between 0 and 255")
        self.mesh_color = np.asarray(mesh_color, dtype=np.float64)
        self.mesh = build_motion_mesh(
            motion,
            cache_frames=mesh_cache_frames,
            g1_max_faces=g1_max_faces,
        )
        self.mesh_faces = np.asarray(self.mesh.faces, dtype=np.int32)

    def render_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(np.clip(frame_idx, 0, self.motion.num_frames - 1))
        image = Image.new("RGB", (self.render_width, self.render_height), self.background)
        draw = ImageDraw.Draw(image, "RGB")
        basis = self._camera_basis(frame_idx)
        target = self._target_for_frame(frame_idx)

        if self.show_grid:
            self._draw_grid(draw, basis, target)
        if self.trail_frames > 0:
            start = max(0, frame_idx - self.trail_frames)
            trail, trail_depth = self._project(self.root_path[start : frame_idx + 1], basis)
            valid_trail = trail_depth > 0
            if valid_trail.sum() > 1:
                draw.line(
                    [tuple(point) for point in trail[valid_trail]],
                    fill=(72, 145, 230),
                    width=max(1, 2 * self.scale),
                )

        vertices = np.asarray(self.mesh.vertices(frame_idx), dtype=np.float64)
        projected, depth = self._project(vertices, basis)
        triangles = vertices[self.mesh_faces]
        projected_triangles = projected[self.mesh_faces]
        triangle_depth = depth[self.mesh_faces]
        centers = triangles.mean(axis=1)
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        normal_lengths = np.linalg.norm(normals, axis=1)
        unit_normals = normals / np.maximum(normal_lengths[:, None], 1e-12)
        camera_position = basis[0]
        facing_camera = np.einsum("ij,ij->i", normals, camera_position - centers) > 0.0

        x = projected_triangles[..., 0]
        y = projected_triangles[..., 1]
        on_screen = (
            (x.max(axis=1) >= 0.0)
            & (x.min(axis=1) < self.render_width)
            & (y.max(axis=1) >= 0.0)
            & (y.min(axis=1) < self.render_height)
        )
        visible = (
            np.all(triangle_depth > 1e-3, axis=1)
            & (normal_lengths > 1e-10)
            & facing_camera
            & on_screen
        )

        light_direction = np.asarray([-0.35, 0.85, 0.40])
        light_direction /= np.linalg.norm(light_direction)
        diffuse = np.clip(unit_normals @ light_direction, 0.0, 1.0)
        view_vectors = camera_position - centers
        view_vectors /= np.maximum(np.linalg.norm(view_vectors, axis=1, keepdims=True), 1e-12)
        rim = 1.0 - np.abs(np.einsum("ij,ij->i", unit_normals, view_vectors))
        brightness = np.clip(0.34 + 0.58 * diffuse + 0.08 * rim, 0.0, 1.0)
        face_colors = np.clip(
            self.mesh_color[None] * brightness[:, None] + 8.0,
            0.0,
            255.0,
        ).astype(np.uint8)

        visible_indices = np.flatnonzero(visible)
        order = visible_indices[np.argsort(triangle_depth[visible_indices].mean(axis=1))[::-1]]
        for face_idx in order:
            draw.polygon(
                [tuple(point) for point in projected_triangles[face_idx]],
                fill=tuple(int(value) for value in face_colors[face_idx]),
            )

        if self.show_text:
            elapsed = frame_idx / self.motion.fps
            title = self.motion.text_at(frame_idx).strip() or "ARDY motion"
            draw.rounded_rectangle(
                (
                    18 * self.scale,
                    18 * self.scale,
                    min(self.render_width - 18, (35 + 8 * len(title)) * self.scale),
                    70 * self.scale,
                ),
                radius=8 * self.scale,
                fill=(5, 8, 13),
            )
            draw.text((30 * self.scale, 27 * self.scale), title, fill=(238, 242, 248), font=self.font)
            draw.text(
                (30 * self.scale, 48 * self.scale),
                f"{elapsed:6.2f}s   frame {frame_idx + 1}/{self.motion.num_frames}",
                fill=(145, 160, 181),
                font=self.font,
            )

        if self.scale > 1:
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.asarray(image)


def export_skeleton_video(
    motion: MotionData,
    output_path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float | None = None,
    camera: OrbitCamera | None = None,
    face_front: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
    supersample: int = 2,
    show_grid: bool = True,
    show_text: bool = False,
    trail_seconds: float = 0.0,
) -> Path:
    """Render a skeleton video; face_front aligns the camera to the first exported pose."""
    start_frame = max(0, int(start_frame))
    end_frame = motion.num_frames if end_frame is None else min(motion.num_frames, int(end_frame))
    if start_frame >= end_frame:
        raise ValueError(f"Empty frame range [{start_frame}, {end_frame})")
    if face_front:
        camera = front_facing_camera(motion, camera, frame_idx=start_frame)
    output_fps = float(motion.fps if fps is None else fps)
    renderer = SkeletonRenderer(
        motion,
        width,
        height,
        camera,
        supersample=supersample,
        show_grid=show_grid,
        show_text=show_text,
        trail_seconds=trail_seconds,
    )
    with FfmpegVideoWriter(output_path, width, height, output_fps) as writer:
        for frame_idx in range(start_frame, end_frame):
            writer.write(renderer.render_frame(frame_idx))
    return Path(output_path)


def export_mesh_video(
    motion: MotionData,
    output_path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float | None = None,
    camera: OrbitCamera | None = None,
    face_front: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
    supersample: int = 2,
    show_grid: bool = True,
    show_text: bool = False,
    trail_seconds: float = 0.0,
    mesh_color: tuple[int, int, int] = (82, 148, 224),
    mesh_cache_frames: int = 16,
    g1_max_faces: int = 30000,
) -> Path:
    """Render a mesh video; face_front aligns the camera to the first exported pose."""
    start_frame = max(0, int(start_frame))
    end_frame = motion.num_frames if end_frame is None else min(motion.num_frames, int(end_frame))
    if start_frame >= end_frame:
        raise ValueError(f"Empty frame range [{start_frame}, {end_frame})")
    if face_front:
        camera = front_facing_camera(motion, camera, frame_idx=start_frame)
    output_fps = float(motion.fps if fps is None else fps)
    renderer = MeshRenderer(
        motion,
        width,
        height,
        camera,
        supersample=supersample,
        show_grid=show_grid,
        show_text=show_text,
        trail_seconds=trail_seconds,
        mesh_color=mesh_color,
        mesh_cache_frames=mesh_cache_frames,
        g1_max_faces=g1_max_faces,
    )
    with FfmpegVideoWriter(output_path, width, height, output_fps) as writer:
        for frame_idx in range(start_frame, end_frame):
            writer.write(renderer.render_frame(frame_idx))
    return Path(output_path)
