from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from motiondata_lib.importers.common import build_motion_clip
from motiondata_lib.transforms import quat_xyzw_to_wxyz
from motiondata_lib.types import MotionClip, MotionClipRef

if TYPE_CHECKING:
    from motiondata_lib.robot_profiles import RobotProfile


FORMAT_NAME = "tiangong3_pkl"
REQUIRED_KEYS = ("fps", "root_pos", "root_rot", "dof_pos")
# Retargeted TienKung PKLs store root_rot as xyzw.
DEFAULT_QUAT_ORDER = "xyzw"


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


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected PKL root object to be a dict, got {type(payload).__name__}")
    return payload


def can_load(path: Path) -> bool:
    if path.suffix.lower() != ".pkl":
        return False
    try:
        payload = _load_payload(path)
        if not all(key in payload for key in REQUIRED_KEYS):
            return False
        root_pos = np.asarray(payload["root_pos"])
        root_rot = np.asarray(payload["root_rot"])
        dof_pos = np.asarray(payload["dof_pos"])
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            return False
        if root_rot.shape != (root_pos.shape[0], 4):
            return False
        if dof_pos.ndim != 2 or dof_pos.shape[0] != root_pos.shape[0]:
            return False
        return root_pos.shape[0] > 0
    except Exception:
        return False


def load_motion_clip(
    clip_ref: MotionClipRef,
    robot_profile: "RobotProfile",
) -> MotionClip:
    payload = _load_payload(clip_ref.path)
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{clip_ref.path} is missing required keys: {', '.join(missing)}")

    fps_values = np.asarray(payload["fps"])
    if fps_values.size != 1:
        raise ValueError(f"{clip_ref.path} fps must be a scalar, got shape {fps_values.shape}")
    fps = float(fps_values.item())
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{clip_ref.path} fps must be positive and finite, got {fps}")

    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    root_rot = np.asarray(payload["root_rot"], dtype=np.float64)
    joint_pos = np.asarray(payload["dof_pos"], dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{clip_ref.path} root_pos must have shape (N, 3), got {root_pos.shape}")
    if root_rot.shape != (len(root_pos), 4):
        raise ValueError(
            f"{clip_ref.path} root_rot must have shape ({len(root_pos)}, 4), got {root_rot.shape}"
        )
    if joint_pos.ndim != 2 or joint_pos.shape[0] != len(root_pos):
        raise ValueError(
            f"{clip_ref.path} dof_pos must be 2D with the same frame count as root_pos; "
            f"got {joint_pos.shape} and {root_pos.shape}"
        )

    if "joint_names" in payload:
        source_names = _as_joint_names(payload["joint_names"])
        if len(source_names) != joint_pos.shape[1]:
            raise ValueError(
                f"{clip_ref.path} has {len(source_names)} joint_names but dof_pos has "
                f"{joint_pos.shape[1]} columns"
            )
        source_column = {name: column for column, name in enumerate(source_names)}
        missing_joints = [
            name for name in robot_profile.joint_names if name not in source_column
        ]
        if missing_joints:
            raise ValueError(
                f"{clip_ref.path} joint_names is missing joints required by "
                f"'{robot_profile.name}': {missing_joints}"
            )
        joint_pos = joint_pos[:, [source_column[name] for name in robot_profile.joint_names]]
    elif joint_pos.shape[1] != len(robot_profile.joint_names):
        raise ValueError(
            f"{clip_ref.path} dof_pos has {joint_pos.shape[1]} columns, but robot profile "
            f"'{robot_profile.name}' requires {len(robot_profile.joint_names)}. The PKL has no "
            "joint_names array, so a safe reordering is impossible."
        )

    if DEFAULT_QUAT_ORDER == "xyzw":
        base_quat_w = quat_xyzw_to_wxyz(root_rot)
    else:
        base_quat_w = root_rot

    return build_motion_clip(
        clip_ref,
        framerate=fps,
        joint_names=np.asarray(robot_profile.joint_names),
        joint_pos=joint_pos,
        base_pos_w=root_pos,
        base_quat_w=base_quat_w,
    )
