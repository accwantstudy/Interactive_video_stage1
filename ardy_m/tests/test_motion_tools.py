# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ardy.motion_editing import apply_temporal_joint_drag, temporal_blend_weights
from ardy.motion_io import MotionData, load_motion_npz
from ardy.prompt_schedule import PromptSchedule
from ardy.skeleton import CoreSkeleton27
from ardy.video import (
    MeshRenderer, OrbitCamera, SkeletonRenderer, export_mesh_video, export_skeleton_video, front_facing_camera,
)


def make_motion(num_frames: int = 24) -> MotionData:
    skeleton = CoreSkeleton27()
    local = torch.eye(3).expand(num_frames, skeleton.nbjoints, 3, 3).clone()
    roots = torch.zeros(num_frames, 3)
    roots[:, 2] = torch.linspace(0.0, 0.3, num_frames)
    global_rots, positions, _ = skeleton.fk(local, roots)
    return MotionData(
        skeleton=skeleton,
        local_rot_mats=local,
        global_rot_mats=global_rots,
        posed_joints=positions,
        root_positions=roots,
        fps=20.0,
        text="test motion",
    )


class MotionEditingTests(unittest.TestCase):
    def test_temporal_envelope_is_smooth_and_bounded(self):
        weights = temporal_blend_weights(30, 10, 4, 2, 4)
        self.assertEqual(float(weights[6]), 0.0)
        self.assertEqual(float(weights[10]), 1.0)
        self.assertEqual(float(weights[12]), 1.0)
        self.assertEqual(float(weights[16]), 0.0)
        self.assertTrue(torch.all(weights >= 0.0))
        self.assertTrue(torch.all(weights <= 1.0))
        self.assertTrue(torch.all(weights[6:11][1:] >= weights[6:11][:-1]))
        self.assertTrue(torch.all(weights[12:17][1:] <= weights[12:17][:-1]))

    def test_drag_reaches_target_and_preserves_bone_lengths(self):
        motion = make_motion()
        joint_idx = motion.skeleton.bone_index["LeftHand"]
        frame_idx = 10
        target = motion.posed_joints[frame_idx, joint_idx] + torch.tensor([-0.12, 0.20, 0.06])
        edited, result = apply_temporal_joint_drag(
            motion,
            frame_idx,
            joint_idx,
            target,
            scope="Limb",
            ease_in_frames=4,
            hold_frames=2,
            ease_out_frames=4,
            iterations=20,
        )
        self.assertLess(result.target_error, 0.005)
        self.assertGreater(len(result.ancestor_indices), 1)

        parents = motion.skeleton.joint_parents
        child_indices = torch.where(parents >= 0)[0]
        parent_indices = parents[child_indices]
        original_lengths = torch.linalg.vector_norm(
            motion.posed_joints[:, child_indices] - motion.posed_joints[:, parent_indices], dim=-1
        )
        edited_lengths = torch.linalg.vector_norm(
            edited.posed_joints[:, child_indices] - edited.posed_joints[:, parent_indices], dim=-1
        )
        self.assertLess(float((original_lengths - edited_lengths).abs().max()), 1e-5)
        self.assertTrue(torch.equal(edited.posed_joints[0], motion.posed_joints[0]))
        self.assertTrue(torch.equal(edited.posed_joints[-1], motion.posed_joints[-1]))

    def test_root_drag_moves_the_coherent_body(self):
        motion = make_motion()
        frame_idx = 8
        root_idx = motion.skeleton.root_idx
        displacement = torch.tensor([0.2, 0.1, -0.15])
        target = motion.posed_joints[frame_idx, root_idx] + displacement
        edited, _ = apply_temporal_joint_drag(
            motion,
            frame_idx,
            root_idx,
            target,
            ease_in_frames=0,
            hold_frames=0,
            ease_out_frames=0,
        )
        joint_displacements = edited.posed_joints[frame_idx] - motion.posed_joints[frame_idx]
        expected = displacement.expand_as(joint_displacements)
        self.assertTrue(torch.allclose(joint_displacements, expected, atol=1e-6))


class MotionIOAndVideoTests(unittest.TestCase):
    def test_npz_round_trip_preserves_editable_fields(self):
        motion = make_motion(8)
        motion.foot_contacts = torch.zeros(8, len(motion.skeleton.foot_joint_idx), dtype=torch.bool)
        motion.prompt_schedule = PromptSchedule(("walk", "run"), (0, 4), (4, 8))
        with tempfile.TemporaryDirectory() as directory:
            path = motion.save(Path(directory) / "motion")
            loaded = load_motion_npz(path)
            self.assertEqual(loaded.num_frames, motion.num_frames)
            self.assertEqual(loaded.num_joints, motion.num_joints)
            self.assertEqual(loaded.text, motion.text)
            self.assertTrue(torch.allclose(loaded.local_rot_mats, motion.local_rot_mats))
            self.assertTrue(torch.allclose(loaded.posed_joints, motion.posed_joints))
            self.assertIsNotNone(loaded.foot_contacts)
            self.assertEqual(loaded.foot_contacts.dtype, torch.bool)
            self.assertEqual(loaded.prompt_schedule, motion.prompt_schedule)
            self.assertEqual(loaded.text_at(3), "walk")
            self.assertEqual(loaded.text_at(4), "run")

    def test_headless_renderer_outputs_rgb(self):
        motion = make_motion(4)
        renderer = SkeletonRenderer(
            motion,
            320,
            240,
            OrbitCamera(azimuth_degrees=70.0, elevation_degrees=90.0),
            supersample=1,
        )
        self.assertEqual(renderer.trail_frames, 0)
        frame = renderer.render_frame(0)
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(int(frame.max()), int(frame.min()))
        overlay_frame = SkeletonRenderer(
            motion,
            320,
            240,
            OrbitCamera(azimuth_degrees=70.0, elevation_degrees=90.0),
            supersample=1,
            show_text=True,
        ).render_frame(0)
        self.assertFalse(np.array_equal(frame[:80, :180], overlay_frame[:80, :180]))

    def test_headless_mesh_renderer_outputs_rgb(self):
        motion = make_motion(2)
        renderer = MeshRenderer(
            motion,
            320,
            240,
            OrbitCamera(azimuth_degrees=30.0, elevation_degrees=12.0, follow_root=True),
            supersample=1,
            show_grid=False,
        )
        self.assertEqual(renderer.trail_frames, 0)
        frame = renderer.render_frame(0)
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(int(frame.max()), int(frame.min()))

    def test_skeleton_and_mesh_front_view_uses_first_exported_frame(self):
        motion = make_motion(6)
        local = motion.local_rot_mats.clone()
        local[2:4, motion.skeleton.root_idx] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        local[4:, motion.skeleton.root_idx] = torch.diag(torch.tensor([-1., 1., -1.]))
        motion.replace_pose(local, motion.root_positions)
        original = OrbitCamera(azimuth_degrees=12, elevation_degrees=7, distance=5, follow_root=True)
        for exporter, renderer in ((export_skeleton_video, SkeletonRenderer), (export_mesh_video, MeshRenderer)):
            with self.subTest(mode=renderer.__name__):
                with patch(f"ardy.video.{renderer.__name__}", wraps=renderer) as factory, patch(
                    "ardy.video.FfmpegVideoWriter"
                ) as writer:
                    exporter(motion, "unused.mp4", width=160, height=120, camera=original,
                             face_front=True, start_frame=2, end_frame=5, supersample=1)
                camera = factory.call_args.args[3]
                self.assertAlmostEqual(camera.azimuth_degrees, 102)
                self.assertEqual(camera.elevation_degrees, 7)
                self.assertEqual(camera.distance, 5)
                self.assertTrue(camera.follow_root)
                self.assertEqual(original.azimuth_degrees, 12)
                self.assertEqual(writer.return_value.__enter__.return_value.write.call_count, 3)
                # One fixed camera for the full export; the later 180-degree
                # turn must not reorient it to keep the character facing us.
                self.assertEqual(factory.call_count, 1)

    def test_front_view_without_camera_defaults_to_level_front(self):
        camera = front_facing_camera(make_motion(1))
        self.assertAlmostEqual(camera.azimuth_degrees, 0)
        self.assertEqual(camera.elevation_degrees, 0)

    def test_front_view_falls_back_to_shoulders_and_reports_ambiguous_pose(self):
        motion = make_motion(1)
        right, left = motion.skeleton.hip_joint_idx
        motion.posed_joints[:, right] = motion.posed_joints[:, left]
        self.assertAlmostEqual(front_facing_camera(motion).azimuth_degrees, 0)
        motion.posed_joints[:] = 0
        with self.assertRaisesRegex(ValueError, "Cannot infer initial facing"):
            front_facing_camera(motion)

    def test_cli_front_choice_and_defaults_for_all_modes(self):
        from scripts import export_video

        motion = make_motion(1)
        for mode in ("skeleton", "mesh", "dwpose"):
            for choice in ([], ["--face-front"], ["--no-face-front"]):
                with self.subTest(mode=mode, choice=choice):
                    argv = ["export_video.py", "motion.npz", "--render-mode", mode, *choice]
                    enabled = choice == ["--face-front"] or (not choice and mode == "dwpose")
                    with patch.object(sys, "argv", argv), patch.object(
                        export_video, "load_motion_npz", return_value=motion
                    ), patch.object(export_video, f"export_{mode}_video") as exporter, patch("builtins.print"):
                        export_video.main()
                    arguments = exporter.call_args.kwargs
                    self.assertEqual(arguments["face_front"], enabled)
                    camera = arguments["camera"]
                    self.assertEqual(camera.azimuth_degrees, 0 if enabled or mode == "dwpose" else 45)
                    self.assertEqual(camera.elevation_degrees, 0 if enabled or mode == "dwpose" else 15)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_mp4_export(self):
        motion = make_motion(4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.mp4"
            export_skeleton_video(motion, path, width=320, height=240, supersample=1)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
