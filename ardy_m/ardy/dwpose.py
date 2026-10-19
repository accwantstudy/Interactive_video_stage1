# SPDX-License-Identifier: Apache-2.0
"""Project ARDY motion into a body-only DW-Pose-style control video.

No pose estimator is needed. Body ordering and RGB palette follow DWPose:
https://github.com/IDEA-Research/DWPose/blob/main/ControlNet-v1-1-nightly/annotator/dwpose/util.py
Head is a skeleton-joint proxy for the nose slot; face and hand landmarks are omitted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .motion_io import MotionData
from .video import FfmpegVideoWriter, OrbitCamera, SkeletonRenderer, front_facing_camera


# Body order: head, neck, right arm, left arm, right leg, left leg.
BODY_COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255),
)
BODY_EDGES = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
)


def body_keypoints(motion: MotionData) -> np.ndarray:
    """Return [T, 14, 3] world positions; unavailable head points are NaN.

    Use upper-arm pivots for shoulders (the *Shoulder bones are clavicles),
    and the shoulder midpoint for DW-Pose's neck. No spine, toe, or hand-tip
    joints are rendered. G1 has no head joint and therefore omits the head.
    """
    bones = motion.skeleton.bone_index
    if motion.skeleton.name == "g1skel34":
        names = [None, None]
        for side in ("right", "left"):
            names.extend(f"{side}_{joint}_skel" for joint in ("shoulder_pitch", "elbow", "wrist_yaw"))
        for side in ("right", "left"):
            names.extend(f"{side}_{joint}_skel" for joint in ("hip_pitch", "knee", "ankle_roll"))
    elif motion.skeleton.name in {"cskel27", "somaskel30", "somaskel77"}:
        names = ["Head", None]
        for side in ("Right", "Left"):
            names.extend(f"{side}{joint}" for joint in ("Arm", "ForeArm", "Hand"))
        leg_names = ("UpLeg", "Leg", "Foot") if motion.skeleton.name == "cskel27" else ("Leg", "Shin", "Foot")
        for side in ("Right", "Left"):
            names.extend(f"{side}{joint}" for joint in leg_names)
    else:
        raise ValueError(f"DW-Pose mapping is unavailable for {motion.skeleton.name}")
    positions = motion.posed_joints.numpy()
    points = np.full((motion.num_frames, 14, 3), np.nan, dtype=np.float64)
    for idx, name in enumerate(names):
        if name is not None:
            points[:, idx] = positions[:, bones[name]]
    points[:, 1] = (points[:, 2] + points[:, 5]) * 0.5
    return points


def _frame_range(motion: MotionData, start_frame: int, end_frame: int | None) -> tuple[int, int]:
    start = max(0, int(start_frame))
    end = motion.num_frames if end_frame is None else min(motion.num_frames, int(end_frame))
    if start >= end:
        raise ValueError(f"Empty frame range [{start}, {end})")
    return start, end


class DWPoseRenderer(SkeletonRenderer):
    """Black-background body pose with a fixed initial-facing camera.

    By default azimuth is an offset from the first exported pose's facing
    direction. Set face_front=False for absolute world-space camera angles.
    Auto-fit covers every source joint in the selected clip, including hidden
    head/hand/toe endpoints, with constant distance (no zoom pumping). An explicit
    distance overrides auto-fit. Root following translates but never rotates the
    camera, so turns and rolls remain visible.
    """

    def __init__(
        self,
        motion: MotionData,
        width: int = 1280,
        height: int = 720,
        camera: OrbitCamera | None = None,
        *,
        supersample: int = 2,
        hide_head: bool = False,
        face_front: bool = True,
        start_frame: int = 0,
        end_frame: int | None = None,
        padding: float = 0.08,
    ):
        self.start_frame, self.end_frame = _frame_range(motion, start_frame, end_frame)
        if not np.isfinite(padding) or not 0 <= padding < 0.45:
            raise ValueError("padding must be between 0 and 0.45")
        positions = motion.posed_joints.numpy()
        if not np.isfinite(positions).all():
            raise ValueError("Motion joint positions must be finite")
        camera = camera or OrbitCamera(azimuth_degrees=0, elevation_degrees=0)
        values = [camera.azimuth_degrees, camera.elevation_degrees, camera.fov_degrees]
        values.extend(() if camera.distance is None else (camera.distance,))
        values.extend(() if camera.target is None else camera.target)
        if not np.isfinite(values).all():
            raise ValueError("Camera parameters must be finite")
        if camera.follow_root and camera.target is not None:
            raise ValueError("follow_root and a fixed target are mutually exclusive")
        self.keypoints = body_keypoints(motion)
        self.hide_head = hide_head
        if face_front:
            camera = front_facing_camera(motion, camera, frame_idx=self.start_frame)
        super().__init__(
            motion, width, height, camera, supersample=supersample,
            background=(0, 0, 0), show_grid=False,
        )
        # Four-pixel radius at 512px, scaled with output size and antialiasing.
        self.joint_radius = max(1.0, min(width, height) / 128.0) * self.scale
        self._fit_camera(positions[self.start_frame:self.end_frame], padding)

    def _fit_camera(self, positions: np.ndarray, padding: float) -> None:
        _, right, up, forward = self._camera_basis(self.start_frame)
        axes = np.stack((right, up, forward), axis=1)
        if self.camera.follow_root:
            roots = positions[:, self.motion.skeleton.root_idx]
            points = positions - roots[:, None]
        else:
            points = positions
        coordinates = points @ axes
        if self.camera.target is None:
            center = (coordinates.min(axis=(0, 1)) + coordinates.max(axis=(0, 1))) * 0.5
            if self.camera.follow_root:
                self.body_center_offset = center @ axes.T
            else:
                self.static_target = center @ axes.T
        else:
            center = np.asarray(self.camera.target) @ axes
        relative = coordinates - center
        # True vertical FOV; solve both frustum inequalities for every point,
        # including depth. This also fits portrait output and jumps.
        self.focal = 0.5 * self.render_height / np.tan(np.deg2rad(self.camera.fov_degrees) * 0.5)
        available = np.array([self.render_width, self.render_height]) * (0.5 - padding)
        available -= self.joint_radius + 2 * self.scale
        if np.any(available <= 0):
            raise ValueError("Output dimensions are too small for the pose padding and joint radius")
        if self.camera.distance is None:
            required = np.abs(relative[..., :2]) * self.focal / available - relative[..., 2, None]
            self.distance = float(max(required.max(), 0.01 - relative[..., 2].min(), 0.01))

    def _paint_primitive(
        self,
        canvas: np.ndarray,
        zbuffer: np.ndarray,
        owners: np.ndarray,
        polygon: np.ndarray,
        color: tuple[int, int, int],
        owner: int,
        *,
        endpoints: np.ndarray | None = None,
        depths: np.ndarray,
        joint_edges: tuple[int, ...] = (),
    ) -> None:
        """Rasterize a small bounding box, testing depth at each covered pixel.

        Reciprocal depth is linear along a perspective-projected segment. Using
        average limb depth or skeleton index for draw order can invert crossings.
        Joint discs overlay their own incident limbs, but obey other limbs' depth.
        """
        lower = np.maximum(np.floor(polygon.min(axis=0)), 0).astype(int)
        upper = np.minimum(np.ceil(polygon.max(axis=0)) + 1, [self.render_width, self.render_height]).astype(int)
        x0, y0 = lower
        x1, y1 = upper
        if x0 >= x1 or y0 >= y1:
            return
        mask = Image.new("L", (x1 - x0, y1 - y0))
        draw = ImageDraw.Draw(mask)
        if endpoints is None:
            draw.ellipse(tuple((polygon - lower).ravel()), fill=255)
            depth = depths[0]
        else:
            draw.polygon([tuple(point) for point in polygon - lower], fill=255)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            delta = endpoints[1] - endpoints[0]
            t = np.clip(
                ((xx - endpoints[0, 0]) * delta[0] + (yy - endpoints[0, 1]) * delta[1]) / (delta @ delta),
                0.0, 1.0,
            )
            depth = 1.0 / ((1.0 - t) / depths[0] + t / depths[1])
        old_depth = zbuffer[y0:y1, x0:x1]
        old_owner = owners[y0:y1, x0:x1]
        visible = depth <= old_depth + 1e-8
        if joint_edges:
            visible |= np.isin(old_owner, joint_edges)
        visible &= np.asarray(mask) > 0
        canvas[y0:y1, x0:x1][visible] = color
        # Preserve the nearer incident edge's depth when placing a joint on it,
        # so a subsequent, unrelated joint cannot show through that limb.
        old_depth[visible] = np.minimum(old_depth, depth)[visible]
        old_owner[visible] = owner

    def render_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(np.clip(frame_idx, self.start_frame, self.end_frame - 1))
        canvas = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        zbuffer = np.full((self.render_height, self.render_width), np.inf)
        owners = np.full((self.render_height, self.render_width), -1, dtype=np.int16)
        points, depths = self._project(self.keypoints[frame_idx], self._camera_basis(frame_idx))
        visible = np.isfinite(points).all(axis=-1) & (depths > 0)
        if self.hide_head:
            visible[0] = False
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        for edge_idx, (start, end) in enumerate(BODY_EDGES):
            if not (visible[start] and visible[end]):
                continue
            delta = points[end] - points[start]
            length = np.linalg.norm(delta)
            if length < 1e-6:
                continue
            direction = delta / length
            perpendicular = np.array([-direction[1], direction[0]])
            polygon = (
                (points[start] + points[end]) * 0.5
                + np.cos(angles)[:, None] * delta * 0.5
                + np.sin(angles)[:, None] * perpendicular * self.joint_radius
            )
            color = tuple(int(channel * 0.6) for channel in BODY_COLORS[edge_idx])
            self._paint_primitive(
                canvas, zbuffer, owners, polygon, color, edge_idx,
                endpoints=points[[start, end]], depths=depths[[start, end]],
            )
        radius = self.joint_radius
        for idx in np.flatnonzero(visible):
            bounds = np.array([points[idx] - radius, points[idx] + radius])
            incident = tuple(edge_idx for edge_idx, edge in enumerate(BODY_EDGES) if idx in edge)
            self._paint_primitive(
                canvas, zbuffer, owners, bounds, BODY_COLORS[idx], len(BODY_EDGES) + idx,
                depths=depths[[idx]], joint_edges=incident,
            )
        image = Image.fromarray(canvas)
        if self.scale > 1:
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.asarray(image)


def export_dwpose_video(
    motion: MotionData,
    output_path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float | None = None,
    camera: OrbitCamera | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    supersample: int = 2,
    hide_head: bool = False,
    face_front: bool = True,
    padding: float = 0.08,
) -> Path:
    """Stream a body-only pose MP4/WebM, with no model or GPU dependency."""
    renderer = DWPoseRenderer(
        motion, width, height, camera, supersample=supersample,
        hide_head=hide_head, face_front=face_front, start_frame=start_frame,
        end_frame=end_frame, padding=padding,
    )
    output_fps = motion.fps if fps is None else float(fps)
    if not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("fps must be finite and positive")
    with FfmpegVideoWriter(output_path, width, height, output_fps) as writer:
        for frame_idx in range(renderer.start_frame, renderer.end_frame):
            writer.write(renderer.render_frame(frame_idx))
    return Path(output_path)
