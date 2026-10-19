# SPDX-License-Identifier: Apache-2.0
"""Geometry, camera framing, and actual encoding checks for pose export."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ardy.dwpose import BODY_COLORS, DWPoseRenderer, body_keypoints, export_dwpose_video
from ardy.motion_io import MotionData, load_motion_npz
from ardy.skeleton import build_skeleton
from ardy.video import OrbitCamera


def make_motion(num_frames=6, joints=27):
    skeleton = build_skeleton(joints)
    local = torch.eye(3).expand(num_frames, joints, 3, 3).clone()
    roots = torch.zeros(num_frames, 3)
    rotations, positions, _ = skeleton.fk(local, roots)
    return MotionData(skeleton, local, rotations, positions, roots, 20.0)


def yaw(degrees):
    angle = np.deg2rad(degrees)
    return torch.tensor([
        [np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)],
    ], dtype=torch.float32)


class DWPoseTests(unittest.TestCase):
    def test_body_mapping_uses_limb_pivots_and_no_detail_joints(self):
        for count in (27, 30, 34, 77):
            with self.subTest(joints=count):
                motion = make_motion(joints=count)
                points = body_keypoints(motion)
                self.assertEqual(points.shape, (6, 14, 3))
                bones = motion.skeleton.bone_index
                shoulder = "right_shoulder_pitch_skel" if count == 34 else "RightArm"
                wrist = "right_wrist_yaw_skel" if count == 34 else "RightHand"
                np.testing.assert_allclose(points[:, 2], motion.posed_joints[:, bones[shoulder]])
                np.testing.assert_allclose(points[:, 4], motion.posed_joints[:, bones[wrist]])
                np.testing.assert_allclose(points[:, 1], (points[:, 2] + points[:, 5]) / 2)
                self.assertEqual(np.isfinite(points[0]).all(axis=-1).sum(), 13 if count == 34 else 14)
                frame = DWPoseRenderer(motion, 320, 240).render_frame(0)
                self.assertEqual(frame.dtype, np.uint8)
                self.assertGreater(frame.max(), 0)

    def test_initial_view_is_front_facing_for_arbitrary_world_heading(self):
        for angle in (-170, -90, 0, 65, 180):
            with self.subTest(angle=angle):
                motion = make_motion()
                motion.posed_joints = motion.posed_joints @ yaw(angle).T
                renderer = DWPoseRenderer(motion, 512, 512, supersample=1)
                points, depths = renderer._project(renderer.keypoints[0], renderer._camera_basis(0))
                # Anatomical right appears on the left when viewed from the front.
                self.assertLess(points[2, 0], points[5, 0])
                self.assertAlmostEqual(depths[8], depths[11], places=5)
                camera_front = -renderer._camera_basis(0)[3]
                expected_front = (yaw(angle) @ torch.tensor([0.0, 0.0, 1.0])).numpy()
                np.testing.assert_allclose(camera_front, expected_front, atol=1e-6)

    def test_trimmed_clip_faces_its_first_frame_and_preserves_later_turns(self):
        motion = make_motion()
        motion.posed_joints[2:4] = motion.posed_joints[2:4] @ yaw(90).T
        motion.posed_joints[4:] = motion.posed_joints[4:] @ yaw(180).T
        renderer = DWPoseRenderer(motion, start_frame=2)
        np.testing.assert_allclose(-renderer._camera_basis(2)[3], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(renderer._camera_basis(2)[3], renderer._camera_basis(5)[3])
        manual = DWPoseRenderer(motion, camera=OrbitCamera(azimuth_degrees=0, elevation_degrees=0),
                                start_frame=2, face_front=False)
        np.testing.assert_allclose(-manual._camera_basis(2)[3], [0, 0, 1], atol=1e-6)

    def test_all_source_joints_fit_even_during_travel_and_jumps(self):
        motion = make_motion()
        # Combine large depth changes, lateral travel and vertical jumps.
        motion.posed_joints += torch.tensor([
            [0, 0, 0], [5, 0, 3], [-4, 3, -4], [0, 6, 1], [1, -2, 5], [3, 1, -3],
        ])[:, None]
        for width, height in ((720, 1280), (1280, 720), (512, 512)):
            for follow in (False, True):
                with self.subTest(size=(width, height), follow=follow):
                    camera = OrbitCamera(azimuth_degrees=20, elevation_degrees=15, follow_root=follow)
                    renderer = DWPoseRenderer(motion, width, height, camera, supersample=1)
                    for idx in range(motion.num_frames):
                        points, depth = renderer._project(motion.posed_joints[idx].numpy(), renderer._camera_basis(idx))
                        self.assertTrue(np.all(depth > 0))
                        self.assertTrue(np.all(points >= np.array([width, height]) * 0.08))
                        self.assertTrue(np.all(points <= np.array([width, height]) * 0.92))

    def test_fixed_target_auto_fit_and_explicit_distance(self):
        motion = make_motion()
        camera = OrbitCamera(azimuth_degrees=0, elevation_degrees=0, target=(4, 3, 2))
        renderer = DWPoseRenderer(motion, 320, 640, camera, supersample=1)
        points, depth = renderer._project(motion.posed_joints[0].numpy(), renderer._camera_basis(0))
        self.assertTrue(np.all(depth > 0))
        self.assertTrue(np.all(points >= [0, 0]))
        self.assertTrue(np.all(points < [320, 640]))
        renderer = DWPoseRenderer(motion, camera=OrbitCamera(distance=9))
        self.assertEqual(renderer.distance, 9)

    def test_only_exported_frames_affect_framing(self):
        motion = make_motion()
        motion.posed_joints[0] += torch.tensor([100, 0, 0])
        full = DWPoseRenderer(motion)
        trimmed = DWPoseRenderer(motion, start_frame=1)
        self.assertGreater(full.distance, trimmed.distance * 5)

    def test_hide_head_removes_head_and_edge_without_changing_camera(self):
        motion = make_motion()
        full = DWPoseRenderer(motion, 512, 512, supersample=1)
        hidden = DWPoseRenderer(motion, 512, 512, supersample=1, hide_head=True)
        self.assertEqual(full.distance, hidden.distance)
        np.testing.assert_array_equal(full.static_target, hidden.static_target)
        frame = full.render_frame(0)
        hidden_frame = hidden.render_frame(0)
        points, _ = full._project(full.keypoints[0], full._camera_basis(0))
        x, y = np.rint(points[0]).astype(int)
        np.testing.assert_array_equal(frame[y, x], BODY_COLORS[0])
        np.testing.assert_array_equal(hidden_frame[y, x], [0, 0, 0])
        x, y = np.rint((points[0] + points[1]) / 2).astype(int)
        self.assertTrue(frame[y, x].any())
        self.assertFalse(hidden_frame[y, x].any())
        neck_y = int(points[1, 1]) + 6
        np.testing.assert_array_equal(frame[neck_y:], hidden_frame[neck_y:])
        np.testing.assert_array_equal(frame[:20], 0)

    def crossing_renderer(self, keypoints):
        """Place sparse joints at specified (pixel x, pixel y, camera depth)."""
        renderer = DWPoseRenderer(
            make_motion(1), 256, 256,
            OrbitCamera(azimuth_degrees=0, elevation_degrees=0, distance=10, target=(0, 0, 0)),
            supersample=1, face_front=False,
        )
        position, right, up, forward = renderer._camera_basis(0)
        renderer.keypoints[:] = np.nan
        for idx, (x, y, depth) in keypoints.items():
            renderer.keypoints[0, idx] = (
                position + forward * depth
                + right * (x - 128) * depth / renderer.focal
                - up * (y - 128) * depth / renderer.focal
            )
        return renderer

    def test_crossed_legs_draw_the_nearer_leg_in_front(self):
        for right_depth, left_depth, edge_color in ((3, 5, 7), (7, 5, 10), (2, 4, 7)):
            with self.subTest(right_depth=right_depth, left_depth=left_depth):
                renderer = self.crossing_renderer({
                    8: (64, 64, right_depth), 9: (192, 192, right_depth),
                    11: (64, 192, left_depth), 12: (192, 64, left_depth),
                })
                expected = (np.array(BODY_COLORS[edge_color]) * 0.6).astype(np.uint8)
                np.testing.assert_array_equal(renderer.render_frame(0)[128, 128], expected)

    def test_crossing_uses_perspective_depth_instead_of_average_limb_depth(self):
        renderer = self.crossing_renderer({
            8: (64, 64, 2), 9: (192, 192, 10),
            11: (64, 192, 4), 12: (192, 64, 4),
        })
        # Right limb's average depth is 6 (behind left), but its depth at the
        # projected midpoint is 1 / (0.5/2 + 0.5/10) = 3.33 (in front).
        expected = (np.array(BODY_COLORS[7]) * 0.6).astype(np.uint8)
        np.testing.assert_array_equal(renderer.render_frame(0)[128, 128], expected)

    def test_same_limb_can_be_in_front_at_one_crossing_and_behind_at_another(self):
        renderer = self.crossing_renderer({
            8: (32, 128, 2), 9: (224, 128, 14),
            11: (64, 32, 4), 12: (128, 224, 4), 13: (192, 32, 4),
        })
        frame = renderer.render_frame(0)
        np.testing.assert_array_equal(frame[128, 96], (np.array(BODY_COLORS[7]) * 0.6).astype(np.uint8))
        np.testing.assert_array_equal(frame[128, 160], (np.array(BODY_COLORS[11]) * 0.6).astype(np.uint8))

    def test_unrelated_joint_obeys_limb_depth(self):
        for depth in (2, 8):
            with self.subTest(joint_depth=depth):
                renderer = self.crossing_renderer({
                    4: (128, 128, depth), 11: (64, 128, 4), 12: (192, 128, 4),
                })
                expected = BODY_COLORS[4] if depth == 2 else (np.array(BODY_COLORS[10]) * 0.6).astype(np.uint8)
                np.testing.assert_array_equal(renderer.render_frame(0)[128, 128], expected)

    def test_overlapping_joints_use_depth_instead_of_left_right_order(self):
        renderer = self.crossing_renderer({9: (128, 128, 3), 12: (128, 128, 6)})
        np.testing.assert_array_equal(renderer.render_frame(0)[128, 128], BODY_COLORS[9])

    def test_positions_only_file_needs_no_rotations(self):
        motion = make_motion()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "positions.npz"
            np.savez(path, posed_joints=motion.posed_joints.numpy())
            loaded = load_motion_npz(path, require_rotations=False)
            self.assertTrue(np.array_equal(DWPoseRenderer(loaded).render_frame(0),
                                           DWPoseRenderer(motion).render_frame(0)))

    def test_invalid_motion_and_camera_fail_before_encoding(self):
        motion = make_motion()
        for kwargs in ({"start_frame": 6}, {"end_frame": 0}, {"padding": -0.1}, {"padding": 0.5},
                       {"camera": OrbitCamera(azimuth_degrees=float("nan"))},
                       {"camera": OrbitCamera(target=(0, 0, 0), follow_root=True)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DWPoseRenderer(motion, **kwargs)
        motion.posed_joints[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            DWPoseRenderer(motion)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe not installed")
    def test_actual_cli_export_preserves_dimensions_fps_and_frame_range(self):
        motion = make_motion()
        with tempfile.TemporaryDirectory() as directory:
            source = motion.save(Path(directory) / "motion.npz")
            subprocess.run([
                sys.executable, "scripts/export_video.py", str(source), "--render-mode", "dwpose",
                "--width", "240", "--height", "320", "--hide-head", "--start-frame", "1",
                "--end-frame", "4", "--fps", "12", "--supersample", "1",
            ], check=True, capture_output=True)
            mp4 = source.with_name("motion_dwpose.mp4")
            self.assert_video(mp4)
            self.assertTrue(source.exists())


    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe not installed")
    def test_actual_webm_export(self):
        encoders = subprocess.run(["ffmpeg", "-encoders"], check=True, capture_output=True, text=True)
        if "libvpx-vp9" not in encoders.stdout:
            self.skipTest("Installed FFmpeg lacks the libvpx-vp9 encoder")
        with tempfile.TemporaryDirectory() as directory:
            path = export_dwpose_video(make_motion(), Path(directory) / "pose.webm", width=240, height=320,
                                       start_frame=1, end_frame=4, fps=12, supersample=1)
            self.assert_video(path)

    def assert_video(self, path):
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path),
        ], check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        self.assertEqual((stream["width"], stream["height"]), (240, 320))
        self.assertEqual(stream["r_frame_rate"], "12/1")
        self.assertEqual(int(stream["nb_read_frames"]), 3)


if __name__ == "__main__":
    unittest.main()
