# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export an ARDY motion NPZ to a camera-controlled skeleton, mesh, or DW-Pose video.

Examples:
    python scripts/export_video.py outputs/output.npz
    python scripts/export_video.py outputs/output.npz --render-mode dwpose --hide-head
    python scripts/export_video.py outputs/output.npz --render-mode mesh --face-front --follow-root
    python scripts/export_video.py outputs/output.npz --azimuth 120 --elevation 20 --follow-root
    python scripts/export_video.py outputs/output.npz --output videos/output.mp4 --width 1920 --height 1080
"""

import argparse
import time
from pathlib import Path

from ardy.dwpose import export_dwpose_video
from ardy.motion_io import load_motion_npz
from ardy.video import OrbitCamera, export_mesh_video, export_skeleton_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an ARDY skeleton, mesh, or DW-Pose animation without a browser or GPU."
    )
    parser.add_argument("motion", help="Generated ARDY .npz motion")
    parser.add_argument("-o", "--output", default=None, help="Output .mp4/.webm path (default: beside input)")
    parser.add_argument(
        "--render-mode",
        choices=("skeleton", "mesh", "dwpose"),
        default="skeleton",
        help="Geometry to render (default: skeleton)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Even output width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Even output height (default: 720)")
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS (default: motion FPS)")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to render")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive last frame (default: all)")
    parser.add_argument(
        "--azimuth", type=float, default=None,
        help="Camera azimuth; offset from initial heading with --face-front (default: 0 for front/DW-Pose, otherwise 45)",
    )
    parser.add_argument(
        "--elevation", type=float, default=None, help="Camera elevation (default: 0 for front/DW-Pose views, otherwise 15)",
    )
    parser.add_argument("--hide-head", action="store_true", help="DW-Pose: omit the head point and neck-to-head edge")
    parser.add_argument(
        "--face-front", action=argparse.BooleanOptionalAction, default=None,
        help="Face the first exported pose, then keep the angle fixed (default: enabled for DW-Pose; disabled otherwise)",
    )
    parser.add_argument(
        "--padding", type=float, default=0.08,
        help="DW-Pose: fraction of each image dimension reserved on each side (default: 0.08)",
    )
    parser.add_argument("--distance", type=float, default=None, help="Camera distance (default: fit motion)")
    parser.add_argument("--fov", type=float, default=45.0, help="Vertical field of view in degrees")
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Fixed camera look-at point (default: center the motion)",
    )
    parser.add_argument(
        "--follow-root",
        action="store_true",
        help="Keep the camera centered on the moving character instead of the full path",
    )
    parser.add_argument("--trail-seconds", type=float, default=0.0, help="Root path trail duration (default: disabled)")
    parser.add_argument("--supersample", type=int, default=2, choices=(1, 2, 3), help="Antialiasing factor")
    parser.add_argument("--no-grid", action="store_true", help="Hide the ground grid")
    parser.add_argument("--show-text", action="store_true", help="Show prompt, time, and frame overlay")
    parser.add_argument(
        "--mesh-color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(82, 148, 224),
        help="Mesh color as RGB values from 0 to 255",
    )
    parser.add_argument(
        "--mesh-max-faces",
        type=int,
        default=30000,
        help="Approximate G1 face budget for CPU rendering (default: 30000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_path = Path(args.motion)
    default_output = (
        motion_path.with_name(f"{motion_path.stem}_dwpose.mp4")
        if args.render_mode == "dwpose" else motion_path.with_suffix(".mp4")
    )
    output_path = Path(args.output) if args.output else default_output
    if output_path.resolve() == motion_path.resolve():
        raise ValueError("Video output path cannot overwrite the input motion file")
    if args.follow_root and args.target is not None:
        raise ValueError("--follow-root and --target are mutually exclusive")
    if args.distance is not None and args.distance <= 0:
        raise ValueError("--distance must be positive")
    if not 1.0 <= args.fov < 179.0:
        raise ValueError("--fov must be between 1 and 179 degrees")
    if any(not 0 <= channel <= 255 for channel in args.mesh_color):
        raise ValueError("--mesh-color values must be between 0 and 255")
    if args.mesh_max_faces < 1000:
        raise ValueError("--mesh-max-faces must be at least 1000")

    if args.render_mode == "dwpose" and (args.show_text or args.trail_seconds):
        raise ValueError("DW-Pose output has no overlays; omit --show-text and --trail-seconds")

    motion = load_motion_npz(motion_path, require_rotations=args.render_mode == "mesh")
    face_front = args.face_front if args.face_front is not None else args.render_mode == "dwpose"
    level_view = face_front or args.render_mode == "dwpose"
    camera = OrbitCamera(
        azimuth_degrees=args.azimuth if args.azimuth is not None else (0.0 if level_view else 45.0),
        elevation_degrees=(
            args.elevation if args.elevation is not None else (0.0 if level_view else 15.0)
        ),
        distance=args.distance,
        fov_degrees=args.fov,
        target=None if args.target is None else tuple(args.target),
        follow_root=args.follow_root,
    )
    frame_end = motion.num_frames if args.end_frame is None else min(args.end_frame, motion.num_frames)
    print(
        f"Rendering {frame_end - args.start_frame} {args.render_mode} frames at "
        f"{args.fps or motion.fps:g} fps "
        f"to {output_path}..."
    )
    started = time.perf_counter()
    exporter = {
        "mesh": export_mesh_video, "skeleton": export_skeleton_video, "dwpose": export_dwpose_video,
    }[args.render_mode]
    common_arguments = dict(
        motion=motion,
        output_path=output_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera=camera,
        face_front=face_front,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        supersample=args.supersample,
    )
    if args.render_mode == "dwpose":
        common_arguments.update(hide_head=args.hide_head, padding=args.padding)
    else:
        common_arguments.update(
            show_grid=not args.no_grid, show_text=args.show_text, trail_seconds=args.trail_seconds,
        )
    if args.render_mode == "mesh":
        common_arguments.update(
            mesh_color=tuple(args.mesh_color),
            g1_max_faces=args.mesh_max_faces,
        )
    result = exporter(**common_arguments)
    elapsed = time.perf_counter() - started
    print(f"Saved {result} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
