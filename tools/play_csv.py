#!/usr/bin/env python3
"""Play a robot motion CSV in MuJoCo's native interactive viewer.

CSV layout (one frame per row):

    root_x, root_y, root_z, quat_x, quat_y, quat_z, quat_w, joint_0, ...

The joint columns are applied to the model's scalar joints in XML order.  The
default model is the Tiangong 3 model used by CSVEditor.

Examples:

    python tools/play_csv.py retargeted/motion.csv
    python tools/play_csv.py retargeted/motion.csv --speed 0.25
    python tools/play_csv.py retargeted/motion.csv --no-contact-forces

Viewer controls:

    Mouse left/right/wheel   rotate / pan / zoom
    Space                    pause or resume
    Left / Right             step one frame (and pause)
    PageDown / PageUp        step backward / forward one second
    Up / Down                double / halve playback speed
    R                        restart and pause
    C                        toggle contact points
    F                        toggle contact-force arrows
    T                        toggle pelvis tracking / free camera
    H                        print this help again

The CSV contains positions but no measured forces.  Contact locations come
from collision detection and are useful for inspection.  Displayed force
arrows are MuJoCo's estimate for each frozen pose, not measured dynamic forces.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DEFAULT_MODEL = PROJECT_ROOT / "assets" / "tiangong3" / "urdf" / "tiangong3.xml"
DEFAULT_MODEL = "/home/eai/tushuqi/gitlab/tienKung/CSVEditor/assets/tiangong3/urdf/tiangong3.xml"
DEFAULT_FPS = 120.0


@dataclass(frozen=True)
class ModelLayout:
    root_qpos_adr: int
    root_body_id: int
    joint_qpos_adrs: np.ndarray
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class PendingActions:
    paused: bool
    speed: float
    step_frames: int
    restart: bool
    toggle_contact_points: bool
    toggle_contact_forces: bool
    toggle_tracking: bool
    timing_changed: bool


@dataclass
class PlaybackControls:
    paused: bool
    speed: float
    fps: float
    _step_frames: int = 0
    _restart: bool = False
    _toggle_contact_points: bool = False
    _toggle_contact_forces: bool = False
    _toggle_tracking: bool = False
    _timing_changed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def key_callback(self, key: int) -> None:
        with self._lock:
            if key == glfw.KEY_SPACE:
                self.paused = not self.paused
                self._timing_changed = True
            elif key == glfw.KEY_RIGHT:
                self.paused = True
                self._step_frames += 1
                self._timing_changed = True
            elif key == glfw.KEY_LEFT:
                self.paused = True
                self._step_frames -= 1
                self._timing_changed = True
            elif key == glfw.KEY_PAGE_UP:
                self.paused = True
                self._step_frames += max(1, int(round(self.fps)))
                self._timing_changed = True
            elif key == glfw.KEY_PAGE_DOWN:
                self.paused = True
                self._step_frames -= max(1, int(round(self.fps)))
                self._timing_changed = True
            elif key == glfw.KEY_UP:
                self.speed = min(8.0, self.speed * 2.0)
                self._timing_changed = True
            elif key == glfw.KEY_DOWN:
                self.speed = max(0.03125, self.speed / 2.0)
                self._timing_changed = True
            elif key == glfw.KEY_R:
                self.paused = True
                self._restart = True
                self._timing_changed = True
            elif key == glfw.KEY_C:
                self._toggle_contact_points = True
            elif key == glfw.KEY_F:
                self._toggle_contact_forces = True
            elif key == glfw.KEY_T:
                self._toggle_tracking = True
            elif key == glfw.KEY_H:
                print_controls()

    def pause(self) -> None:
        with self._lock:
            self.paused = True
            self._timing_changed = True

    def consume(self) -> PendingActions:
        with self._lock:
            actions = PendingActions(
                paused=self.paused,
                speed=self.speed,
                step_frames=self._step_frames,
                restart=self._restart,
                toggle_contact_points=self._toggle_contact_points,
                toggle_contact_forces=self._toggle_contact_forces,
                toggle_tracking=self._toggle_tracking,
                timing_changed=self._timing_changed,
            )
            self._step_frames = 0
            self._restart = False
            self._toggle_contact_points = False
            self._toggle_contact_forces = False
            self._toggle_tracking = False
            self._timing_changed = False
            return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="直接用 MuJoCo 原生 Viewer 播放 CSV 动作，并清晰显示接触。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="CSV 动作文件")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo XML/MJCF 模型")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="CSV 帧率")
    parser.add_argument("--speed", type=float, default=1.0, help="播放倍速")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧（含）")
    parser.add_argument("--end-frame", type=int, help="结束帧（含，默认最后一帧）")
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="CSV 第 4 至 7 列的四元数顺序",
    )
    parser.add_argument(
        "--position-unit",
        choices=("auto", "m", "mm"),
        default="auto",
        help="根位置单位；auto 在绝对值超过 50 时按毫米处理",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="播放结束后循环",
    )
    parser.add_argument("--paused", action="store_true", help="打开窗口后保持暂停")
    parser.add_argument(
        "--track-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="相机跟随带 free joint 的根刚体",
    )
    parser.add_argument("--camera-distance", type=float, default=3.0, help="初始相机距离")
    parser.add_argument("--camera-azimuth", type=float, default=90.0, help="初始相机方位角（度）")
    parser.add_argument("--camera-elevation", type=float, default=-15.0, help="初始相机俯仰角（度）")
    parser.add_argument(
        "--contact-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示接触点",
    )
    parser.add_argument(
        "--contact-forces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示接触力箭头",
    )
    parser.add_argument(
        "--contact-marker-scale",
        type=float,
        default=2.0,
        help="接触点标记相对模型默认值的缩放",
    )
    parser.add_argument(
        "--contact-force-scale",
        type=float,
        default=1.0,
        help="接触力箭头长度相对模型默认值的缩放",
    )
    parser.add_argument("--max-contact-lines", type=int, default=8, help="画面中最多列出的接触对数量")
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示 MuJoCo 原生左右控制面板",
    )
    parser.add_argument("--validate-only", action="store_true", help="只检查 CSV/模型和首帧接触，不打开窗口")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps 必须大于 0")
    if args.speed <= 0:
        parser.error("--speed 必须大于 0")
    if args.start_frame < 0:
        parser.error("--start-frame 不能为负数")
    if args.contact_marker_scale <= 0 or args.contact_force_scale <= 0:
        parser.error("接触显示缩放必须大于 0")
    if args.max_contact_lines < 0:
        parser.error("--max-contact-lines 不能为负数")
    return args


def load_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 CSV: {path}")

    frames = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    if frames.size == 0:
        raise ValueError(f"CSV 为空: {path}")
    if frames.ndim == 1:
        frames = frames[np.newaxis, :]

    # CSVEditor normally writes no header, but tolerate one textual header row.
    if len(frames) > 1 and not np.all(np.isfinite(frames[0])):
        if np.all(np.isfinite(frames[1:])):
            frames = frames[1:]

    bad = np.argwhere(~np.isfinite(frames))
    if len(bad):
        row, col = bad[0]
        raise ValueError(f"CSV 第 {row + 1} 行、第 {col + 1} 列不是有效数字")
    return np.ascontiguousarray(frames)


def inspect_model(model: mujoco.MjModel) -> ModelLayout:
    free_joint_ids: list[int] = []
    scalar_joint_ids: list[int] = []
    scalar_joint_names: list[str] = []

    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            free_joint_ids.append(joint_id)
        elif joint_type in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            scalar_joint_ids.append(joint_id)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            scalar_joint_names.append(name or f"joint_{joint_id}")
        else:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise ValueError(f"暂不支持非标量关节: {name or joint_id} (type={joint_type})")

    if len(free_joint_ids) != 1:
        raise ValueError(f"模型必须恰好包含一个 free joint，实际为 {len(free_joint_ids)} 个")

    root_joint_id = free_joint_ids[0]
    return ModelLayout(
        root_qpos_adr=int(model.jnt_qposadr[root_joint_id]),
        root_body_id=int(model.jnt_bodyid[root_joint_id]),
        joint_qpos_adrs=np.asarray(
            [int(model.jnt_qposadr[joint_id]) for joint_id in scalar_joint_ids],
            dtype=np.int32,
        ),
        joint_names=tuple(scalar_joint_names),
    )


def choose_position_scale(frames: np.ndarray, unit: str) -> tuple[float, str]:
    if unit == "m":
        return 1.0, "m"
    if unit == "mm":
        return 0.001, "mm"
    max_abs = float(np.max(np.abs(frames[:, :3])))
    return (0.001, "mm (auto)") if max_abs > 50.0 else (1.0, "m (auto)")


def validate_columns(frames: np.ndarray, layout: ModelLayout) -> None:
    expected = 7 + len(layout.joint_qpos_adrs)
    actual = frames.shape[1]
    if actual != expected:
        raise ValueError(
            f"CSV 有 {actual} 列，但模型需要 {expected} 列 "
            f"(根位姿 7 + 标量关节 {len(layout.joint_qpos_adrs)})"
        )


def apply_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: ModelLayout,
    row: np.ndarray,
    position_scale: float,
    quat_order: str,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    root = layout.root_qpos_adr
    data.qpos[root : root + 3] = row[:3] * position_scale

    quat = row[3:7]
    if quat_order == "xyzw":
        quat = quat[[3, 0, 1, 2]]
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("发现长度接近 0 的根四元数")
    data.qpos[root + 3 : root + 7] = quat / norm
    data.qpos[layout.joint_qpos_adrs] = row[7:]
    mujoco.mj_forward(model, data)


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name:
        return name
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return f"{body_name or f'body#{body_id}'}/geom#{geom_id}"


def contact_lines(model: mujoco.MjModel, data: mujoco.MjData, limit: int) -> list[str]:
    lines: list[str] = []
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(min(int(data.ncon), limit)):
        contact = data.contact[contact_id]
        mujoco.mj_contactForce(model, data, contact_id, force)
        left = geom_name(model, int(contact.geom1))
        right = geom_name(model, int(contact.geom2))
        lines.append(
            f"{left} <-> {right}  gap={float(contact.dist) * 1000:+.2f} mm  "
            f"Fn={float(force[0]):.1f} N"
        )
    remaining = int(data.ncon) - len(lines)
    if remaining > 0:
        lines.append(f"... 另有 {remaining} 个接触")
    return lines


def print_controls() -> None:
    print(
        "\n控制键:\n"
        "  鼠标左键/右键/滚轮  旋转/平移/缩放\n"
        "  Space                 暂停/继续\n"
        "  Left / Right          前后单步一帧\n"
        "  PageDown / PageUp     前后单步一秒\n"
        "  Up / Down             加速/减速\n"
        "  R                     回到起始帧并暂停\n"
        "  C / F                 开关接触点/接触力\n"
        "  T                     切换跟随/自由相机\n"
        "  H                     再次显示帮助\n"
        "  Esc 或关闭窗口         退出\n"
    )


def set_viewer_overlay(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: int,
    start_frame: int,
    end_frame: int,
    fps: float,
    actions: PendingActions,
    contact_points: bool,
    contact_forces: bool,
    tracking: bool,
    max_contact_lines: int,
) -> None:
    if not hasattr(viewer, "set_texts"):
        return
    state = "PAUSED" if actions.paused else "PLAYING"
    status_left = "Frame\nTime\nState\nSpeed\nContacts\nCamera\nContact view"
    status_right = (
        f"{frame}  [{start_frame}, {end_frame}]\n"
        f"{frame / fps:.3f} s\n"
        f"{state}\n"
        f"{actions.speed:g}x\n"
        f"{int(data.ncon)}\n"
        f"{'TRACKING' if tracking else 'FREE'}\n"
        f"points={'on' if contact_points else 'off'}, forces={'on' if contact_forces else 'off'}"
    )
    contacts = contact_lines(model, data, max_contact_lines)
    texts = [
        (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, status_left, status_right),
    ]
    if contacts:
        texts.append(
            (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Active contacts", "\n".join(contacts))
        )
    viewer.set_texts(texts)


def play(args: argparse.Namespace) -> None:
    csv_path = args.csv.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型: {model_path}")

    frames = load_csv(csv_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    layout = inspect_model(model)
    validate_columns(frames, layout)
    position_scale, detected_unit = choose_position_scale(frames, args.position_unit)

    end_frame = len(frames) - 1 if args.end_frame is None else args.end_frame
    if args.start_frame >= len(frames):
        raise ValueError(f"--start-frame 超出范围；CSV 只有 {len(frames)} 帧")
    if end_frame < args.start_frame or end_frame >= len(frames):
        raise ValueError(f"--end-frame 必须在 [{args.start_frame}, {len(frames) - 1}] 内")

    model.vis.scale.contactwidth *= args.contact_marker_scale
    model.vis.scale.contactheight *= args.contact_marker_scale
    model.vis.scale.forcewidth *= args.contact_marker_scale
    model.vis.map.force *= args.contact_force_scale

    displayed_frame = args.start_frame
    apply_frame(
        model,
        data,
        layout,
        frames[displayed_frame],
        position_scale,
        args.quat_order,
    )

    root_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, layout.root_body_id)
    duration = (end_frame - args.start_frame + 1) / args.fps
    print(f"CSV:       {csv_path}")
    print(f"Model:     {model_path}")
    print(f"Frames:    {len(frames)} total, playing {args.start_frame}..{end_frame} ({duration:.2f} s)")
    print(f"Layout:    root={root_body_name or layout.root_body_id}, joints={len(layout.joint_names)}")
    print(f"Position:  {detected_unit}; quaternion={args.quat_order}")
    print(f"Contacts:  frame {displayed_frame} has {int(data.ncon)} contact(s)")
    for line in contact_lines(model, data, args.max_contact_lines):
        print(f"           {line}")

    if args.validate_only:
        print("Validation passed; viewer was not opened (--validate-only).")
        return

    controls = PlaybackControls(paused=args.paused, speed=args.speed, fps=args.fps)
    print_controls()

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=controls.key_callback,
        show_left_ui=args.ui,
        show_right_ui=args.ui,
    ) as viewer:
        contact_points = args.contact_points
        contact_forces = args.contact_forces
        tracking = args.track_camera

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(contact_points)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(contact_forces)
            viewer.cam.distance = args.camera_distance
            viewer.cam.azimuth = args.camera_azimuth
            viewer.cam.elevation = args.camera_elevation
            if tracking:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = layout.root_body_id
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = data.qpos[
                    layout.root_qpos_adr : layout.root_qpos_adr + 3
                ]

        viewer.sync()
        next_frame = displayed_frame + 1
        next_deadline = time.monotonic() + 1.0 / (args.fps * args.speed)
        last_overlay = 0.0
        actions = controls.consume()

        while viewer.is_running():
            now = time.monotonic()
            actions = controls.consume()

            if actions.toggle_contact_points:
                contact_points = not contact_points
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(contact_points)
            if actions.toggle_contact_forces:
                contact_forces = not contact_forces
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(contact_forces)
            if actions.toggle_tracking:
                tracking = not tracking
                with viewer.lock():
                    if tracking:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        viewer.cam.trackbodyid = layout.root_body_id
                    else:
                        viewer.cam.lookat[:] = data.qpos[
                            layout.root_qpos_adr : layout.root_qpos_adr + 3
                        ]
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            target_frame: int | None = None
            if actions.restart:
                target_frame = args.start_frame
            elif actions.step_frames:
                target_frame = int(
                    np.clip(
                        displayed_frame + actions.step_frames,
                        args.start_frame,
                        end_frame,
                    )
                )
            elif not actions.paused and now >= next_deadline:
                if next_frame > end_frame:
                    next_frame = args.start_frame
                target_frame = next_frame

            if target_frame is not None:
                with viewer.lock():
                    apply_frame(
                        model,
                        data,
                        layout,
                        frames[target_frame],
                        position_scale,
                        args.quat_order,
                    )
                displayed_frame = target_frame
                next_frame = displayed_frame + 1

                if displayed_frame == end_frame and not args.loop and not actions.step_frames:
                    controls.pause()
                interval = 1.0 / (args.fps * actions.speed)
                next_deadline += interval
                if next_deadline < now - 4.0 * interval:
                    next_deadline = now + interval

            if actions.timing_changed or actions.paused:
                next_deadline = now + 1.0 / (args.fps * actions.speed)

            viewer.sync()
            if now - last_overlay >= 0.1 or target_frame is not None:
                set_viewer_overlay(
                    viewer,
                    model,
                    data,
                    displayed_frame,
                    args.start_frame,
                    end_frame,
                    args.fps,
                    actions,
                    contact_points,
                    contact_forces,
                    tracking,
                    args.max_contact_lines,
                )
                last_overlay = now

            time.sleep(0.002 if not actions.paused else 0.01)


def main() -> int:
    args = parse_args()
    try:
        play(args)
    except (FileNotFoundError, ValueError, mujoco.FatalError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
