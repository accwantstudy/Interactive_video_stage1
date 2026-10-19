# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from ardy.prompt_schedule import PromptSchedule


class PromptScheduleTests(unittest.TestCase):
    def test_builds_sorted_contiguous_schedule(self):
        schedule = PromptSchedule.from_switches(
            "walk",
            [(6.0, "jump"), (3.0, "run")],
            fps=20.0,
            num_frames=180,
            frame_alignment=4,
        )
        self.assertEqual(schedule.texts, ("walk", "run", "jump"))
        self.assertEqual(schedule.start_frames, (0, 60, 120))
        self.assertEqual(schedule.end_frames, (60, 120, 180))
        self.assertEqual(schedule.text_at(59), "walk")
        self.assertEqual(schedule.text_at(60), "run")
        self.assertEqual(schedule.text_at(179), "jump")

    def test_rejects_unaligned_switch(self):
        with self.assertRaisesRegex(ValueError, "not aligned"):
            PromptSchedule.from_switches(
                "walk",
                [(0.25, "run")],
                fps=20.0,
                num_frames=100,
                frame_alignment=4,
            )

    def test_npz_arrays_use_no_python_objects(self):
        schedule = PromptSchedule(("walk", "run"), (0, 20), (20, 40))
        arrays = schedule.to_npz_arrays()
        self.assertNotEqual(arrays["prompt_texts"].dtype, np.dtype(object))
        restored = PromptSchedule.from_npz_arrays(arrays, num_frames=40)
        self.assertEqual(restored, schedule)


if __name__ == "__main__":
    unittest.main()
