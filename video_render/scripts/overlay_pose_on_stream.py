#!/usr/bin/env python3
"""Overlay the control (pose) video on a generated stream.mp4 to inspect skeleton alignment (CPU only).

The pose frames are resampled and resized exactly like iter_pose_chunks() in the streaming predict scripts
(timestamp resampling to --fps, cv2.resize to WIDTH x HEIGHT), i.e. what the model actually received as control.
Outputs <out>.mp4 (skeleton alpha-blended on the generated video) and <out>_sheet.png (evenly spaced frames,
each shown as [generated | overlay]).
"""
import argparse

import cv2
import imageio.v2 as imageio
import numpy as np


def pose_frames(path, fps, size_hw):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    ratio = src_fps / fps if src_fps > 0 else 1.0
    idx, frame, t = -1, None, 0
    while True:
        wanted = int(round(t * ratio))
        while idx < wanted:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                return
            idx += 1
        yield cv2.resize(frame, (size_hw[1], size_hw[0]))  # BGR, same stretch as generation
        t += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stream", required=True)
    p.add_argument("--pose", required=True)
    p.add_argument("--out", required=True, help="output prefix without extension")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--gain", type=float, default=1.6, help="brighten the dim skeleton so it is visible on the video")
    p.add_argument("--sheet-frames", type=int, default=6)
    a = p.parse_args()

    cap = cv2.VideoCapture(a.stream)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    poses = pose_frames(a.pose, a.fps, (h, w))
    keep = set(np.linspace(0, n - 1, a.sheet_frames).astype(int).tolist())
    sheet, used = [], 0
    with imageio.get_writer(a.out + ".mp4", fps=a.fps, macro_block_size=16) as writer:
        for i in range(n):
            ok, frame = cap.read()
            pose = next(poses, None)
            if not ok or pose is None:
                print(f"stopped at frame {i}: stream ok={ok}, pose available={pose is not None}")
                break
            mask = pose.max(2) > 25
            boosted = np.clip(pose.astype(np.float32) * a.gain, 0, 255)
            out = frame.copy()
            out[mask] = (a.alpha * boosted[mask] + (1 - a.alpha) * frame[mask]).astype(np.uint8)
            writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            if i in keep:
                sheet.append(np.concatenate([frame, out], axis=1))
            used += 1
    cap.release()
    if sheet:
        cv2.imwrite(a.out + "_sheet.png", cv2.resize(np.concatenate(sheet, axis=0), None, fx=0.5, fy=0.5))
    print(f"wrote {used}/{n} frames -> {a.out}.mp4 and {a.out}_sheet.png (stream {w}x{h}, pose frames available: {used == n})")


if __name__ == "__main__":
    main()
