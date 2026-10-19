# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from interactive_demo.session_io import SessionIOMixin  # noqa: E402

from ardy.motion_io import load_motion_npz  # noqa: E402
from ardy.skeleton import CoreSkeleton27  # noqa: E402


class _MotionRep:
    def __init__(self, output):
        self.output = output

    def unnormalize(self, motion):
        return motion

    def inverse(self, motion, *, is_normalized):
        self.last_motion = motion
        self.last_is_normalized = is_normalized
        return self.output


class _Demo(SessionIOMixin):
    def __init__(self, session):
        self.client_sessions = {7: session}

    def client_active(self, client_id):
        return client_id in self.client_sessions


class DemoNpzExportTests(unittest.TestCase):
    def test_export_is_standard_npz_with_prompt_schedule(self):
        skeleton = CoreSkeleton27()
        num_frames = 8
        local = torch.eye(3).expand(1, num_frames, skeleton.nbjoints, 3, 3).clone()
        roots = torch.zeros(1, num_frames, 3)
        global_rots, posed_joints, _ = skeleton.fk(local[0], roots[0])
        output = {
            "local_rot_mats": local,
            "global_rot_mats": global_rots.unsqueeze(0),
            "posed_joints": posed_joints.unsqueeze(0),
            "root_positions": roots,
            "foot_contacts": torch.zeros(1, num_frames, len(skeleton.foot_joint_idx)),
        }
        prompts = {
            "first": SimpleNamespace(start_frame=0, text="walk"),
            "second": SimpleNamespace(start_frame=4, text="run"),
        }
        session = SimpleNamespace(
            motion_tensor=torch.zeros(1, num_frames, 1),
            motion_tensor_lock=threading.Lock(),
            motion_rep=_MotionRep(output),
            model_fps=20,
            client=SimpleNamespace(timeline=SimpleNamespace(_prompts=prompts)),
            gui_elements=SimpleNamespace(gui_prompt_text=SimpleNamespace(value="fallback")),
        )

        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "demo_motion"
            saved = _Demo(session).export_motion_npz(7, str(requested))
            self.assertEqual(saved, [str(requested.with_suffix(".npz"))])
            motion = load_motion_npz(saved[0])
            self.assertEqual(motion.num_frames, num_frames)
            self.assertEqual(motion.prompt_schedule.texts, ("walk", "run"))
            self.assertEqual(motion.text_at(3), "walk")
            self.assertEqual(motion.text_at(4), "run")


    def test_dwpose_export_snapshots_multiple_samples_and_cleans_up(self):
        skeleton = CoreSkeleton27()
        local = torch.eye(3).expand(2, skeleton.nbjoints, 3, 3).clone()
        roots = torch.zeros(2, 3)
        rotations, positions, _ = skeleton.fk(local, roots)
        from ardy.motion_io import MotionData

        motion = MotionData(skeleton, local, rotations, positions, roots, 20.0)
        snapshots = []

        def snapshot(_client_id, filepath):
            for idx in range(2):
                path = Path(filepath).with_name(f"motion_{idx:02d}.npz")
                motion.save(path)
                snapshots.append(str(path))
            return snapshots

        demo = _Demo(SimpleNamespace())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pose.mp4"
            with patch.object(demo, "export_motion_npz", side_effect=snapshot), patch(
                "ardy.dwpose.export_dwpose_video"
            ) as render:
                saved = demo.export_motion_dwpose(7, str(output), width=240, height=320,
                                                  hide_head=True, follow_root=True)
                self.assertEqual(saved, [str(output.with_name(f"pose_{idx:02d}.mp4")) for idx in range(2)])
                self.assertEqual(render.call_count, 2)
                for call in render.call_args_list:
                    self.assertEqual(call.args[0].num_frames, 2)
                    self.assertTrue(call.kwargs["hide_head"])
                    self.assertTrue(call.kwargs["camera"].follow_root)
                    self.assertEqual(call.kwargs["width"], 240)
                    self.assertEqual(call.kwargs["height"], 320)
        self.assertTrue(all(not Path(path).exists() for path in snapshots))

    def test_dwpose_export_reports_snapshot_failure(self):
        demo = _Demo(SimpleNamespace())
        with patch.object(demo, "export_motion_npz", return_value=[]):
            with self.assertRaisesRegex(ValueError, "No motion"):
                demo.export_motion_dwpose(7, "pose.mp4")
        with self.assertRaisesRegex(ValueError, "mp4 or .webm"):
            demo.export_motion_dwpose(7, "motion.npz")


if __name__ == "__main__":
    unittest.main()
