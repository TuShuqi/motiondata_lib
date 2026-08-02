#!/usr/bin/env python3
"""Convert a retargeted motion pickle to the numeric Tiangong CSV format.

Only load pickle files from trusted sources. Python's pickle format can execute
code while it is being deserialized.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motiondata_lib.importers.tiangong3_csv import DEFAULT_FPS  # noqa: E402
from motiondata_lib.model import build_qpos_frames, load_model  # noqa: E402
from motiondata_lib.robot_profiles import (  # noqa: E402
    RobotProfile,
    load_robot_profile,
)
from motiondata_lib.transforms import normalize_quaternions  # noqa: E402
from motiondata_lib.types import MotionClip  # noqa: E402


TARGET_FPS = DEFAULT_FPS
REQUIRED_KEYS = ("fps", "root_pos", "root_rot", "dof_pos")


@dataclass(frozen=True)
class PklMotion:
    fps: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos: np.ndarray
    used_embedded_joint_names: bool

    @property
    def frame_count(self) -> int:
        return int(self.joint_pos.shape[0])


def _as_joint_names(values: Any) -> tuple[str, ...]:
    names = []
    for value in np.asarray(values).reshape(-1).tolist():
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    if len(names) != len(set(names)):
        raise ValueError("PKL joint_names contains duplicate names")
    return tuple(names)


def _validate_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        bad_count = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
        raise ValueError(f"PKL {name} contains {bad_count} NaN or infinite values")


def _make_quaternions_continuous(quat_wxyz: np.ndarray) -> np.ndarray:
    result = normalize_quaternions(quat_wxyz)
    norms = np.linalg.norm(quat_wxyz, axis=1)
    if np.any(norms < 1e-8):
        frames = np.flatnonzero(norms < 1e-8)[:10].tolist()
        raise ValueError(f"PKL root_rot has zero-length quaternions at frames {frames}")
    for frame in range(1, len(result)):
        if np.dot(result[frame - 1], result[frame]) < 0.0:
            result[frame] *= -1.0
    return result


def load_pkl_motion(
    path: Path,
    profile: RobotProfile,
    input_quat_order: str,
) -> PklMotion:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected PKL root object to be a dict, got {type(payload).__name__}")

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"PKL is missing required keys: {', '.join(missing)}")

    fps_values = np.asarray(payload["fps"])
    if fps_values.size != 1:
        raise ValueError(f"PKL fps must be a scalar, got shape {fps_values.shape}")
    fps = float(fps_values.item())
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"PKL fps must be positive and finite, got {fps}")

    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    root_quat = np.asarray(payload["root_rot"], dtype=np.float64)
    joint_pos = np.asarray(payload["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"PKL root_pos must have shape (N, 3), got {root_pos.shape}")
    if root_quat.shape != (len(root_pos), 4):
        raise ValueError(
            f"PKL root_rot must have shape ({len(root_pos)}, 4), got {root_quat.shape}"
        )
    if joint_pos.ndim != 2 or joint_pos.shape[0] != len(root_pos):
        raise ValueError(
            "PKL dof_pos must be 2D and have the same frame count as root_pos; "
            f"got {joint_pos.shape} and {root_pos.shape}"
        )
    if len(root_pos) == 0:
        raise ValueError("PKL does not contain any motion frames")

    for name, values in (
        ("root_pos", root_pos),
        ("root_rot", root_quat),
        ("dof_pos", joint_pos),
    ):
        _validate_finite(name, values)

    used_embedded_joint_names = "joint_names" in payload
    if used_embedded_joint_names:
        source_names = _as_joint_names(payload["joint_names"])
        if len(source_names) != joint_pos.shape[1]:
            raise ValueError(
                f"PKL has {len(source_names)} joint_names but dof_pos has "
                f"{joint_pos.shape[1]} columns"
            )
        source_column = {name: column for column, name in enumerate(source_names)}
        missing_joints = [
            name for name in profile.joint_names if name not in source_column
        ]
        if missing_joints:
            raise ValueError(
                f"PKL joint_names is missing joints required by '{profile.name}': "
                f"{missing_joints}"
            )
        joint_pos = joint_pos[
            :, [source_column[name] for name in profile.joint_names]
        ]
    elif joint_pos.shape[1] != len(profile.joint_names):
        raise ValueError(
            f"PKL dof_pos has {joint_pos.shape[1]} columns, but robot profile "
            f"'{profile.name}' requires {len(profile.joint_names)}. The PKL has no "
            "joint_names array, so a safe reordering is impossible."
        )

    if input_quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    root_quat = _make_quaternions_continuous(root_quat)

    return PklMotion(
        fps=fps,
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        joint_pos=joint_pos,
        used_embedded_joint_names=used_embedded_joint_names,
    )


def _slerp_pairwise(
    quat_wxyz: np.ndarray,
    left_indices: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    q0 = quat_wxyz[left_indices]
    q1 = quat_wxyz[np.minimum(left_indices + 1, len(quat_wxyz) - 1)].copy()
    dots = np.sum(q0 * q1, axis=1)
    negative = dots < 0.0
    q1[negative] *= -1.0
    dots = np.clip(np.abs(dots), 0.0, 1.0)

    result = np.empty_like(q0)
    nearly_equal = dots > 0.9995
    if np.any(nearly_equal):
        alpha = fractions[nearly_equal, None]
        result[nearly_equal] = (
            (1.0 - alpha) * q0[nearly_equal] + alpha * q1[nearly_equal]
        )
    different = ~nearly_equal
    if np.any(different):
        theta = np.arccos(dots[different])
        sin_theta = np.sin(theta)
        alpha = fractions[different]
        weight0 = np.sin((1.0 - alpha) * theta) / sin_theta
        weight1 = np.sin(alpha * theta) / sin_theta
        result[different] = (
            weight0[:, None] * q0[different]
            + weight1[:, None] * q1[different]
        )
    return _make_quaternions_continuous(result)


def resample_motion(motion: PklMotion, target_fps: float = TARGET_FPS) -> PklMotion:
    if np.isclose(motion.fps, target_fps):
        return motion
    if motion.frame_count == 1:
        return PklMotion(
            fps=target_fps,
            root_pos=motion.root_pos.copy(),
            root_quat_wxyz=motion.root_quat_wxyz.copy(),
            joint_pos=motion.joint_pos.copy(),
            used_embedded_joint_names=motion.used_embedded_joint_names,
        )

    source_times = np.arange(motion.frame_count, dtype=np.float64) / motion.fps
    duration = source_times[-1]
    target_count = int(np.floor(duration * target_fps)) + 1
    target_times = np.arange(target_count, dtype=np.float64) / target_fps

    root_pos = np.column_stack(
        [
            np.interp(target_times, source_times, motion.root_pos[:, axis])
            for axis in range(3)
        ]
    )
    joint_pos = np.column_stack(
        [
            np.interp(target_times, source_times, motion.joint_pos[:, column])
            for column in range(motion.joint_pos.shape[1])
        ]
    )
    left = np.searchsorted(source_times, target_times, side="right") - 1
    left = np.clip(left, 0, motion.frame_count - 2)
    interval = source_times[left + 1] - source_times[left]
    fractions = (target_times - source_times[left]) / interval
    root_quat = _slerp_pairwise(motion.root_quat_wxyz, left, fractions)

    return PklMotion(
        fps=target_fps,
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        joint_pos=joint_pos,
        used_embedded_joint_names=motion.used_embedded_joint_names,
    )


def validate_profile_model(profile: RobotProfile, model: mujoco.MjModel) -> None:
    for name in profile.joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise ValueError(
                f"Robot profile joint '{name}' does not exist in {profile.model_path}"
            )
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(f"Robot profile joint '{name}' is not a scalar joint")


def _foot_meshes(model: mujoco.MjModel) -> list[tuple[int, np.ndarray]]:
    foot_meshes = []
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        )
        if not body_name or "ankle_roll" not in body_name.lower():
            continue
        if model.geom_contype[geom_id] == 0:
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[start : start + count], dtype=np.float64
        )
        foot_meshes.append((geom_id, vertices))
    if not foot_meshes:
        raise ValueError(
            "Ground alignment requires collision mesh geoms on ankle_roll bodies, "
            "but none were found in the selected robot model"
        )
    return foot_meshes


def compute_ground_offset(
    motion: PklMotion,
    profile: RobotProfile,
    model: mujoco.MjModel,
    sample_stride: int,
) -> float:
    clip = MotionClip(
        path=profile.model_path,
        display_name="ground-alignment",
        format_name="pkl",
        framerate=motion.fps,
        joint_names=np.asarray(profile.joint_names),
        joint_pos=motion.joint_pos,
        base_pos_w=motion.root_pos,
        base_quat_w=motion.root_quat_wxyz,
    )
    qpos_frames = build_qpos_frames(clip, model)
    data = mujoco.MjData(model)
    foot_meshes = _foot_meshes(model)
    lowest = np.inf
    for qpos in qpos_frames[::sample_stride]:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for geom_id, vertices in foot_meshes:
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            world_z = vertices @ rotation[2] + data.geom_xpos[geom_id, 2]
            lowest = min(lowest, float(world_z.min()))
    return -lowest


def build_csv_array(motion: PklMotion) -> np.ndarray:
    output = np.empty((motion.frame_count, 7 + motion.joint_pos.shape[1]))
    output[:, 0:3] = motion.root_pos
    output[:, 3:7] = motion.root_quat_wxyz[:, [1, 2, 3, 0]]
    output[:, 7:] = motion.joint_pos
    return output


def write_csv_atomic(path: Path, values: np.ndarray, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savetxt(handle, values, delimiter=",", fmt="%.9f")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a trusted retargeted PKL to the 120 Hz numeric CSV format "
            "used by motiondata-lib."
        )
    )
    parser.add_argument("input", type=Path, help="Trusted input PKL file")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output CSV file")
    parser.add_argument(
        "--robot",
        choices=("tiangong3",),
        default="tiangong3",
        help="Robot profile defining model and output joint order (default: tiangong3)",
    )
    parser.add_argument(
        "--input-quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Component order of PKL root_rot (default: xyzw)",
    )
    parser.add_argument(
        "--recenter",
        action="store_true",
        help="Subtract the first root XY position from every frame",
    )
    parser.add_argument(
        "--ground-align",
        action="store_true",
        help="Shift root Z so the lowest sampled foot point is at Z=0",
    )
    parser.add_argument(
        "--ground-sample-stride",
        type=int,
        default=1,
        help="Use every Nth frame for ground alignment (default: 1)",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Additional root Z offset in meters (default: 0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        input_path = args.input.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input PKL does not exist: {input_path}")
        output_path = args.output.resolve()
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. "
                "Pass --overwrite to replace it."
            )
        if args.ground_sample_stride < 1:
            raise ValueError("--ground-sample-stride must be at least 1")
        if not np.isfinite(args.z_offset):
            raise ValueError("--z-offset must be finite")

        profile = load_robot_profile(args.robot)
        model = load_model(profile)
        validate_profile_model(profile, model)
        source = load_pkl_motion(input_path, profile, args.input_quat_order)
        motion = resample_motion(source)

        root_pos = motion.root_pos.copy()
        if args.recenter:
            root_pos[:, 0:2] -= root_pos[0, 0:2]
        motion = PklMotion(
            fps=motion.fps,
            root_pos=root_pos,
            root_quat_wxyz=motion.root_quat_wxyz,
            joint_pos=motion.joint_pos,
            used_embedded_joint_names=motion.used_embedded_joint_names,
        )

        ground_offset = 0.0
        if args.ground_align:
            ground_offset = compute_ground_offset(
                motion,
                profile,
                model,
                args.ground_sample_stride,
            )
        total_z_offset = ground_offset + args.z_offset
        if total_z_offset:
            root_pos = motion.root_pos.copy()
            root_pos[:, 2] += total_z_offset
            motion = PklMotion(
                fps=motion.fps,
                root_pos=root_pos,
                root_quat_wxyz=motion.root_quat_wxyz,
                joint_pos=motion.joint_pos,
                used_embedded_joint_names=motion.used_embedded_joint_names,
            )

        output = build_csv_array(motion)
        write_csv_atomic(output_path, output, args.overwrite)
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    mapping = (
        "reordered from embedded joint_names"
        if source.used_embedded_joint_names
        else f"assumed {profile.name} profile order (PKL has no joint_names)"
    )
    print(f"Input:  {input_path}")
    print(
        f"Source: {source.frame_count} frames @ {source.fps:g} Hz, "
        f"{source.joint_pos.shape[1]} joints"
    )
    if source.fps != motion.fps:
        print(f"Resample: {source.fps:g} -> {motion.fps:g} Hz")
    print(f"Joints: {mapping}")
    if args.recenter:
        print("Root XY: recentered to the first frame")
    if args.ground_align:
        print(f"Ground offset: {ground_offset:+.6f} m")
    if args.z_offset:
        print(f"Manual Z offset: {args.z_offset:+.6f} m")
    print(
        f"Output: {args.output.resolve()} ({motion.frame_count} x {output.shape[1]}, "
        f"{motion.fps:g} Hz, quaternion xyzw)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
