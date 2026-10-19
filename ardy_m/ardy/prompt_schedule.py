# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt schedules shared by generation, motion files, and renderers."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class PromptSchedule:
    """Piecewise-constant text prompts over half-open frame ranges."""

    texts: tuple[str, ...]
    start_frames: tuple[int, ...]
    end_frames: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.texts:
            raise ValueError("A prompt schedule must contain at least one prompt")
        if not (len(self.texts) == len(self.start_frames) == len(self.end_frames)):
            raise ValueError("Prompt texts, start frames, and end frames must have equal lengths")
        if any(not text.strip() for text in self.texts):
            raise ValueError("Prompt text cannot be empty")
        if self.start_frames[0] != 0:
            raise ValueError("The first prompt must start at frame 0")
        for index, (start, end) in enumerate(zip(self.start_frames, self.end_frames)):
            if start < 0 or end <= start:
                raise ValueError(f"Invalid prompt frame range [{start}, {end})")
            if index and start != self.end_frames[index - 1]:
                raise ValueError("Prompt frame ranges must be contiguous and non-overlapping")

    @property
    def num_frames(self) -> int:
        return self.end_frames[-1]

    def index_at(self, frame_idx: int) -> int:
        """Return the segment covering ``frame_idx``."""
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < self.num_frames:
            raise IndexError(f"Frame {frame_idx} is outside [0, {self.num_frames})")
        return bisect_right(self.start_frames, frame_idx) - 1

    def text_at(self, frame_idx: int) -> str:
        return self.texts[self.index_at(frame_idx)]

    def summary(self) -> str:
        return " → ".join(self.texts)

    def to_npz_arrays(self) -> dict[str, np.ndarray]:
        """Return non-object arrays safe to load with ``allow_pickle=False``."""
        return {
            "prompt_texts": np.asarray(self.texts, dtype=np.str_),
            "prompt_start_frames": np.asarray(self.start_frames, dtype=np.int64),
            "prompt_end_frames": np.asarray(self.end_frames, dtype=np.int64),
        }

    @classmethod
    def from_npz_arrays(
        cls,
        arrays: Mapping[str, np.ndarray],
        *,
        num_frames: int | None = None,
    ) -> PromptSchedule | None:
        keys = ("prompt_texts", "prompt_start_frames", "prompt_end_frames")
        present = [key in arrays for key in keys]
        if not any(present):
            return None
        if not all(present):
            missing = [key for key, is_present in zip(keys, present) if not is_present]
            raise ValueError(f"Incomplete prompt schedule; missing {', '.join(missing)}")
        schedule = cls(
            texts=tuple(str(text) for text in np.asarray(arrays["prompt_texts"]).reshape(-1)),
            start_frames=tuple(int(frame) for frame in np.asarray(arrays["prompt_start_frames"]).reshape(-1)),
            end_frames=tuple(int(frame) for frame in np.asarray(arrays["prompt_end_frames"]).reshape(-1)),
        )
        if num_frames is not None and schedule.num_frames != int(num_frames):
            raise ValueError(
                f"Prompt schedule ends at frame {schedule.num_frames}, but motion has {num_frames} frames"
            )
        return schedule

    @classmethod
    def from_switches(
        cls,
        initial_prompt: str,
        switches: Iterable[tuple[float, str]],
        *,
        fps: float,
        num_frames: int,
        frame_alignment: int,
    ) -> PromptSchedule:
        """Build a schedule from ``(time_seconds, new_prompt)`` switches."""
        initial_prompt = initial_prompt.strip()
        if not initial_prompt:
            raise ValueError("Initial prompt cannot be empty")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if frame_alignment <= 0:
            raise ValueError("frame_alignment must be positive")

        parsed: list[tuple[int, str, float]] = []
        for seconds, prompt in switches:
            seconds = float(seconds)
            prompt = prompt.strip()
            if not math.isfinite(seconds):
                raise ValueError(f"Prompt switch time must be finite, got {seconds}")
            if not prompt:
                raise ValueError("Prompt switch text cannot be empty")
            frame = round(seconds * fps)
            if frame <= 0 or frame >= num_frames:
                raise ValueError(
                    f"Prompt switch at {seconds:g}s resolves to frame {frame}; it must be inside "
                    f"the motion range (0, {num_frames / fps:g}s)"
                )
            if frame % frame_alignment:
                nearest_frame = round(frame / frame_alignment) * frame_alignment
                max_aligned_frame = ((num_frames - 1) // frame_alignment) * frame_alignment
                nearest_frame = min(max(frame_alignment, nearest_frame), max_aligned_frame)
                raise ValueError(
                    f"Prompt switch at {seconds:g}s resolves to frame {frame}, which is not aligned "
                    f"to the model's {frame_alignment}-frame token size; use approximately "
                    f"{nearest_frame / fps:g}s"
                )
            parsed.append((frame, prompt, seconds))

        parsed.sort(key=lambda item: item[0])
        for previous, current in zip(parsed, parsed[1:]):
            if previous[0] == current[0]:
                raise ValueError(
                    f"Prompt switches at {previous[2]:g}s and {current[2]:g}s resolve to the same frame"
                )

        texts = (initial_prompt, *(prompt for _, prompt, _ in parsed))
        starts = (0, *(frame for frame, _, _ in parsed))
        ends = (*starts[1:], int(num_frames))
        return cls(texts=texts, start_frames=starts, end_frames=ends)
