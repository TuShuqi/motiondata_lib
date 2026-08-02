from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from motiondata_lib.importers.common import build_motion_clip
from motiondata_lib.transforms import quat_xyzw_to_wxyz
from motiondata_lib.types import MotionClip, MotionClipRef

if TYPE_CHECKING:
    from motiondata_lib.robot_profiles import RobotProfile


FORMAT_NAME = "tiangong3_csv"
DEFAULT_FPS = 120.0
SOURCE_JOINT_NAMES = (
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
)
EXPECTED_COLUMN_COUNT = 7 + len(SOURCE_JOINT_NAMES)


def can_load(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        with path.open(newline="") as handle:
            values = [value.strip() for value in handle.readline().split(",")]
        if len(values) != EXPECTED_COLUMN_COUNT:
            return False
        [float(value) for value in values]
        return True
    except Exception:
        return False


def load_motion_clip(
    clip_ref: MotionClipRef,
    robot_profile: "RobotProfile",
) -> MotionClip:
    data = np.loadtxt(clip_ref.path, delimiter=",", dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            f"{clip_ref.path} has shape {data.shape}, "
            f"expected (N, {EXPECTED_COLUMN_COUNT})"
        )

    source_column_by_name = {
        name: column for column, name in enumerate(SOURCE_JOINT_NAMES)
    }
    missing_joints = [
        name for name in robot_profile.joint_names if name not in source_column_by_name
    ]
    if missing_joints:
        raise ValueError(
            f"{clip_ref.path} does not contain joints required by robot profile "
            f"'{robot_profile.name}': {missing_joints}"
        )

    joint_columns = [
        7 + source_column_by_name[name] for name in robot_profile.joint_names
    ]
    return build_motion_clip(
        clip_ref,
        framerate=DEFAULT_FPS,
        joint_names=np.asarray(robot_profile.joint_names),
        joint_pos=data[:, joint_columns],
        base_pos_w=data[:, 0:3],
        base_quat_w=quat_xyzw_to_wxyz(data[:, 3:7]),
    )
