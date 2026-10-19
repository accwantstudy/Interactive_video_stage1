# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser motion editor with click selection, joint dragging, temporal IK, and video export.

Examples:
    python scripts/edit_motion.py outputs/output.npz
    python scripts/edit_motion.py outputs/output.npz --port 2335 --output outputs/output_edited.npz
"""

# ruff: noqa: E402, I001 -- thread limits must be configured before numerical imports.

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

EDITOR_COMPUTE_THREADS = max(1, int(os.environ.get("ARDY_EDITOR_THREADS", "1")))
for _thread_environment_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_environment_variable] = str(EDITOR_COMPUTE_THREADS)

import numpy as np
import torch
import trimesh
import viser

torch.set_num_threads(EDITOR_COMPUTE_THREADS)
try:
    torch.set_num_interop_threads(EDITOR_COMPUTE_THREADS)
except RuntimeError:
    # This only occurs when imported after another module has already used
    # PyTorch's inter-op pool; the OpenMP/intra-op limit still prevents the crash.
    pass

from ardy.motion_editing import apply_temporal_joint_drag
from ardy.motion_io import MotionData, load_motion_npz
from ardy.video import FfmpegVideoWriter
from ardy.viz.viser_utils import Character


MESH_MODE_BY_JOINT_COUNT = {
    27: "core_skin",
    30: "soma_skin",
    34: "g1_stl",
    77: "soma_skin",
}


class MotionEditor:
    def __init__(self, motion_path: str, output_path: str | None, port: int):
        self.motion_path = Path(motion_path)
        self.motion = load_motion_npz(self.motion_path)
        self.original = self.motion.clone()
        self.drag_base = self.motion.clone()
        self.undo_stack: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.edit_started = False
        self.frame_idx = 0
        self.selected_joint_idx = self._default_joint()
        self.playing = False
        self.exporting = False
        self.suppress_gizmo_callback = False
        self.suppress_frame_callback = False
        self.state_lock = threading.RLock()
        self.output_path = Path(output_path) if output_path else self.motion_path.with_name(
            f"{self.motion_path.stem}_edited.npz"
        )

        self.server = viser.ViserServer(host="0.0.0.0", port=port, label="ARDY Motion Editor")
        self.server.scene.set_up_direction("+y")
        self.server.scene.world_axes.visible = False
        self.server.scene.add_grid(
            "/grid",
            width=20.0,
            height=20.0,
            wxyz=viser.transforms.SO3.from_x_radians(-np.pi / 2.0).wxyz,
            position=(0.0, 0.0001, 0.0),
            fade_distance=20.0,
            infinite_grid=True,
        )
        self.character = self._create_character()
        self.joint_handles = self._create_joint_targets()
        self.gizmo = None
        self._create_gui()
        self._create_gizmo()
        self._show_frame(0)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            positions = self.motion.posed_joints.numpy()
            bounds_min = positions.reshape(-1, 3).min(axis=0)
            bounds_max = positions.reshape(-1, 3).max(axis=0)
            center = (bounds_min + bounds_max) * 0.5
            extent = max(2.5, float(np.linalg.norm(bounds_max - bounds_min)))
            client.camera.look_at = center
            client.camera.position = center + np.array([0.7 * extent, 0.35 * extent, 0.7 * extent])

    def _default_joint(self) -> int:
        names = self.motion.skeleton.bone_order_names
        preferred = (
            self.motion.skeleton.left_hand_joint_names[0]
            if self.motion.skeleton.left_hand_joint_names
            else names[self.motion.skeleton.root_idx]
        )
        return names.index(preferred)

    def _create_character(self) -> Character:
        mesh_mode = MESH_MODE_BY_JOINT_COUNT[self.motion.num_joints]
        try:
            return Character(
                "editable_motion",
                self.server,
                self.motion.skeleton,
                create_skeleton_mesh=True,
                create_skinned_mesh=True,
                visible_skeleton=True,
                visible_skinned_mesh=True,
                mesh_mode=mesh_mode,
            )
        except Exception as error:
            print(f"Mesh visualization unavailable ({error}); using skeleton-only mode.")
            return Character(
                "editable_motion",
                self.server,
                self.motion.skeleton,
                create_skeleton_mesh=True,
                create_skinned_mesh=False,
                visible_skeleton=True,
            )

    def _create_joint_targets(self) -> list:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.035)
        handles = []
        for joint_idx, joint_name in enumerate(self.motion.skeleton.bone_order_names):
            handle = self.server.scene.add_mesh_simple(
                f"/edit_targets/{joint_idx}_{joint_name}",
                vertices=sphere.vertices,
                faces=sphere.faces,
                color=(255, 145, 35) if joint_idx == self.selected_joint_idx else (80, 190, 255),
                opacity=0.55,
                position=self.motion.posed_joints[0, joint_idx].numpy(),
                visible=True,
            )

            def register_click(index: int, target_handle) -> None:
                @target_handle.on_click
                def _(_event) -> None:
                    self.gui_joint.value = self.motion.skeleton.bone_order_names[index]

            register_click(joint_idx, handle)
            handles.append(handle)
        return handles

    def _create_gui(self) -> None:
        info = (
            f"**{self.motion_path.name}**  \n"
            f"{self.motion.num_frames} frames @ {self.motion.fps:g} fps  \n"
            "Pause on a frame, click a blue joint (or choose it below), then drag the orange gizmo."
        )
        self.server.gui.add_markdown(info)

        with self.server.gui.add_folder("Playback", expand_by_default=True):
            self.gui_play = self.server.gui.add_button("Play")
            self.gui_previous = self.server.gui.add_button("Previous frame")
            self.gui_next = self.server.gui.add_button("Next frame")
            self.gui_frame = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=self.motion.num_frames - 1,
                step=1,
                initial_value=0,
            )
            self.gui_show_mesh = self.server.gui.add_checkbox(
                "Show mesh",
                initial_value=self.character.skinned_mesh is not None or self.character.g1_mesh_rig is not None,
            )
            self.gui_show_skeleton = self.server.gui.add_checkbox("Show skeleton", initial_value=True)

        with self.server.gui.add_folder("Drag a joint", expand_by_default=True):
            self.gui_edit_enabled = self.server.gui.add_checkbox("Show joint targets", initial_value=True)
            self.gui_joint = self.server.gui.add_dropdown(
                "Selected joint",
                options=self.motion.skeleton.bone_order_names,
                initial_value=self.motion.skeleton.bone_order_names[self.selected_joint_idx],
            )
            self.gui_scope = self.server.gui.add_dropdown(
                "Body influence",
                options=("Limb", "Body", "Whole body"),
                initial_value="Limb",
                hint="Limb preserves the torso/lower body; broader scopes let the body help reach.",
            )
            self.gui_ease_in = self.server.gui.add_number(
                "Ease in (seconds)", initial_value=0.5, min=0.0, max=5.0, step=0.05
            )
            self.gui_hold = self.server.gui.add_number(
                "Hold (seconds)", initial_value=0.25, min=0.0, max=10.0, step=0.05
            )
            self.gui_ease_out = self.server.gui.add_number(
                "Ease out (seconds)", initial_value=0.5, min=0.0, max=5.0, step=0.05
            )
            self.gui_iterations = self.server.gui.add_slider(
                "IK iterations", min=2, max=30, step=1, initial_value=15
            )
            self.gui_allow_root = self.server.gui.add_slider(
                "Whole-body root reach",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=0.0,
                hint="For Whole body only: translate the body by this fraction of any residual.",
            )
            self.gui_commit = self.server.gui.add_button("Commit edit / start another")
            self.gui_undo = self.server.gui.add_button("Undo")
            self.gui_reset = self.server.gui.add_button("Reset all edits")
            self.gui_status = self.server.gui.add_markdown("Ready.")

        with self.server.gui.add_folder("Save edited motion", expand_by_default=True):
            self.gui_output_path = self.server.gui.add_text("NPZ output", initial_value=str(self.output_path))
            self.gui_save = self.server.gui.add_button("Save motion")

        with self.server.gui.add_folder("Viewport video (mesh or skeleton)", expand_by_default=True):
            default_video = self.output_path.with_suffix(".mp4")
            self.gui_video_path = self.server.gui.add_text("Video output", initial_value=str(default_video))
            self.gui_video_width = self.server.gui.add_number(
                "Width", initial_value=1280, min=320, max=3840, step=2
            )
            self.gui_video_height = self.server.gui.add_number(
                "Height", initial_value=720, min=240, max=2160, step=2
            )
            self.gui_export_video = self.server.gui.add_button(
                "Export current viewport",
                hint="Captures every frame using this browser's current camera and visibility settings.",
            )

        @self.gui_play.on_click
        def _(_event) -> None:
            if self.exporting:
                return
            self.playing = not self.playing
            self.gui_play.label = "Pause" if self.playing else "Play"
            if self.playing:
                self._commit_drag()
                self.gui_edit_enabled.value = False

        @self.gui_previous.on_click
        def _(_event) -> None:
            self.playing = False
            self.gui_play.label = "Play"
            self.gui_frame.value = max(0, self.frame_idx - 1)

        @self.gui_next.on_click
        def _(_event) -> None:
            self.playing = False
            self.gui_play.label = "Play"
            self.gui_frame.value = min(self.motion.num_frames - 1, self.frame_idx + 1)

        @self.gui_frame.on_update
        def _(_event) -> None:
            if self.exporting or self.suppress_frame_callback:
                return
            self.playing = False
            self.gui_play.label = "Play"
            self._commit_drag()
            self._show_frame(int(self.gui_frame.value), update_gizmo=True)

        @self.gui_show_mesh.on_update
        def _(_event) -> None:
            self.character.set_skinned_mesh_visibility(self.gui_show_mesh.value)

        @self.gui_show_skeleton.on_update
        def _(_event) -> None:
            self.character.set_skeleton_visibility(self.gui_show_skeleton.value)

        @self.gui_edit_enabled.on_update
        def _(_event) -> None:
            self._set_edit_visibility(self.gui_edit_enabled.value)

        @self.gui_joint.on_update
        def _(_event) -> None:
            self._select_joint(self.motion.skeleton.bone_order_names.index(self.gui_joint.value))

        for temporal_control in (
            self.gui_scope,
            self.gui_ease_in,
            self.gui_hold,
            self.gui_ease_out,
            self.gui_iterations,
            self.gui_allow_root,
        ):
            temporal_control.on_update(lambda _event: self._reapply_current_drag())

        @self.gui_commit.on_click
        def _(_event) -> None:
            self._commit_drag()
            self.gui_status.content = "Edit committed. Drag again to add another layer."

        @self.gui_undo.on_click
        def _(_event) -> None:
            self._undo()

        @self.gui_reset.on_click
        def _(_event) -> None:
            self._restore_pose(self.original.local_rot_mats, self.original.root_positions)
            self.undo_stack.clear()
            self._commit_drag()
            self.gui_status.content = "Restored the original motion."

        @self.gui_save.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                path = self.motion.save(self.gui_output_path.value)
                self.gui_status.content = f"Saved edited motion to **{path}**."
                if event.client:
                    event.client.add_notification(
                        title="Motion saved", body=str(path), color="green", auto_close_seconds=3.0
                    )
            except Exception as error:
                self.gui_status.content = f"Save failed: {error}"
                if event.client:
                    event.client.add_notification(
                        title="Save failed", body=str(error), color="red", auto_close_seconds=5.0
                    )

        @self.gui_export_video.on_click
        def _(event: viser.GuiEvent) -> None:
            if event.client is None or self.exporting:
                return
            thread = threading.Thread(
                target=self._export_viewport_video,
                args=(
                    event.client,
                    self.gui_video_path.value,
                    int(self.gui_video_width.value),
                    int(self.gui_video_height.value),
                ),
                daemon=True,
            )
            thread.start()

    def _create_gizmo(self) -> None:
        if self.gizmo is not None:
            self.server.scene.remove_by_name(self.gizmo.name)
        self.suppress_gizmo_callback = True
        self.gizmo = self.server.scene.add_transform_controls(
            "/selected_joint_drag",
            scale=0.12,
            line_width=4.0,
            disable_rotations=True,
            disable_axes=False,
            disable_sliders=False,
            depth_test=False,
            position=self.motion.posed_joints[self.frame_idx, self.selected_joint_idx].numpy(),
            visible=self.gui_edit_enabled.value,
        )

        @self.gizmo.on_update
        def _(_event) -> None:
            if self.suppress_gizmo_callback or self.exporting:
                return
            self._apply_gizmo_target()

        self.suppress_gizmo_callback = False

    def _set_edit_visibility(self, visible: bool) -> None:
        for handle in self.joint_handles:
            handle.visible = visible
        if self.gizmo is not None:
            self.gizmo.visible = visible

    def _select_joint(self, joint_idx: int) -> None:
        with self.state_lock:
            self._commit_drag()
            old_idx = self.selected_joint_idx
            self.selected_joint_idx = joint_idx
            self.joint_handles[old_idx].color = (80, 190, 255)
            self.joint_handles[joint_idx].color = (255, 145, 35)
            self._create_gizmo()
            self.gui_status.content = (
                f"Selected **{self.motion.skeleton.bone_order_names[joint_idx]}** at frame {self.frame_idx}."
            )

    def _seconds_to_frames(self, seconds: float) -> int:
        return max(0, round(float(seconds) * self.motion.fps))

    def _push_undo(self) -> None:
        self.undo_stack.append((self.drag_base.local_rot_mats.clone(), self.drag_base.root_positions.clone()))
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def _apply_gizmo_target(self) -> None:
        with self.state_lock:
            if not self.edit_started:
                self._push_undo()
                self.edit_started = True
            target = torch.tensor(self.gizmo.position, dtype=torch.float32)
            edited, result = apply_temporal_joint_drag(
                self.drag_base,
                self.frame_idx,
                self.selected_joint_idx,
                target,
                scope=self.gui_scope.value,
                ease_in_frames=self._seconds_to_frames(self.gui_ease_in.value),
                hold_frames=self._seconds_to_frames(self.gui_hold.value),
                ease_out_frames=self._seconds_to_frames(self.gui_ease_out.value),
                iterations=int(self.gui_iterations.value),
                whole_body_root_translation=float(self.gui_allow_root.value),
            )
            self.motion = edited
            self._show_frame(self.frame_idx, update_gizmo=False)
            chain_names = [self.motion.skeleton.bone_order_names[idx] for idx in result.ancestor_indices]
            self.gui_status.content = (
                f"Edited {len(result.frame_indices)} frames; target error "
                f"**{result.target_error * 1000.0:.1f} mm**.  \n"
                f"IK chain: {', '.join(chain_names) if chain_names else 'root translation'}"
            )

    def _reapply_current_drag(self) -> None:
        if self.edit_started and not self.exporting:
            self._apply_gizmo_target()

    def _commit_drag(self) -> None:
        self.drag_base = self.motion.clone()
        self.edit_started = False

    def _restore_pose(self, rotations: torch.Tensor, roots: torch.Tensor) -> None:
        self.motion.replace_pose(rotations.clone(), roots.clone())
        self._show_frame(self.frame_idx, update_gizmo=True)

    def _undo(self) -> None:
        with self.state_lock:
            if self.edit_started:
                self.edit_started = False
            if not self.undo_stack:
                self.gui_status.content = "Nothing to undo."
                return
            rotations, roots = self.undo_stack.pop()
            self._restore_pose(rotations, roots)
            self._commit_drag()
            self.gui_status.content = "Undid the last drag."

    def _show_frame(self, frame_idx: int, *, update_gizmo: bool = False) -> None:
        frame_idx = int(np.clip(frame_idx, 0, self.motion.num_frames - 1))
        self.frame_idx = frame_idx
        contacts = None
        if self.motion.foot_contacts is not None:
            contacts = self.motion.foot_contacts[frame_idx] > 0.5
        self.character.set_pose(
            self.motion.posed_joints[frame_idx],
            self.motion.global_rot_mats[frame_idx],
            foot_contacts=contacts,
        )
        for idx, handle in enumerate(self.joint_handles):
            handle.position = self.motion.posed_joints[frame_idx, idx].numpy()
        if update_gizmo and self.gizmo is not None:
            self.suppress_gizmo_callback = True
            self.gizmo.position = self.motion.posed_joints[frame_idx, self.selected_joint_idx].numpy()
            self.suppress_gizmo_callback = False
        if not self.exporting and int(self.gui_frame.value) != frame_idx:
            self.suppress_frame_callback = True
            self.gui_frame.value = frame_idx
            self.suppress_frame_callback = False

    def _export_viewport_video(
        self,
        client: viser.ClientHandle,
        path: str,
        width: int,
        height: int,
    ) -> None:
        if width % 2 or height % 2:
            client.add_notification(
                title="Video export failed",
                body="Width and height must be even.",
                color="red",
                auto_close_seconds=5.0,
            )
            return
        self.exporting = True
        self.playing = False
        old_frame = self.frame_idx
        old_target_visibility = self.gui_edit_enabled.value
        self._set_edit_visibility(False)
        notification = client.add_notification(
            title="Exporting video",
            body=f"Rendering {self.motion.num_frames} frames...",
            loading=True,
            with_close_button=False,
        )
        self.gui_status.content = f"Exporting viewport video to **{path}**..."
        try:
            with FfmpegVideoWriter(path, width, height, self.motion.fps) as writer:
                for frame_idx in range(self.motion.num_frames):
                    with self.state_lock:
                        self._show_frame(frame_idx)
                    client.flush()
                    render = client.camera.get_render(height=height, width=width)
                    writer.write(np.asarray(render)[..., :3])
            self.gui_status.content = f"Saved viewport video to **{path}**."
            notification.title = "Video exported"
            notification.body = path
            notification.color = "green"
            notification.auto_close_seconds = 4.0
        except Exception as error:
            self.gui_status.content = f"Video export failed: {error}"
            notification.title = "Video export failed"
            notification.body = str(error)
            notification.color = "red"
            notification.auto_close_seconds = 8.0
        finally:
            notification.loading = False
            notification.with_close_button = True
            self._show_frame(old_frame, update_gizmo=True)
            self._set_edit_visibility(old_target_visibility)
            self.exporting = False

    def run(self) -> None:
        print("ARDY Motion Editor")
        print(f"  Motion: {self.motion_path}")
        print(f"  Compute threads: {EDITOR_COMPUTE_THREADS}")
        print(f"  Browser: {self.server.get_url() if hasattr(self.server, 'get_url') else 'see URL above'}")
        next_tick = time.perf_counter()
        try:
            while True:
                if self.playing and not self.exporting:
                    next_frame = (self.frame_idx + 1) % self.motion.num_frames
                    self._show_frame(next_frame, update_gizmo=True)
                next_tick += 1.0 / self.motion.fps
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
        except KeyboardInterrupt:
            print("Motion editor stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively drag joints in an ARDY motion")
    parser.add_argument("motion", help="Generated ARDY .npz motion")
    parser.add_argument("--output", default=None, help="Default edited .npz output path")
    parser.add_argument("--port", type=int, default=2335, help="Viser server port (default: 2335)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MotionEditor(args.motion, args.output, args.port).run()


if __name__ == "__main__":
    main()
