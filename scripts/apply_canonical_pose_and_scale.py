# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
from pathlib import Path

import numpy as np

# parse arguments
parser = argparse.ArgumentParser(description="Apply canonical pose and scale.")
parser.add_argument(
    "--canonical_pose_fn",
    type=Path,
    required=True,
    help="Path to canonical pose file.",
)
parser.add_argument(
    "--canonical_scale_fn",
    type=Path,
    required=True,
    help="Path to canonical scale file.",
)
parser.add_argument(
    "--input_bop_gt_fn",
    type=Path,
    required=True,
    help="Path to input scene GT file.",
)
parser.add_argument(
    "--input_bop_camera_fn",
    type=Path,
    required=True,
    help="Path to input scene camera file.",
)
parser.add_argument(
    "--output_bop_gt_fn",
    type=Path,
    required=True,
    help="Path to output scene GT file.",
)
parser.add_argument(
    "--output_bop_camera_fn",
    type=Path,
    required=True,
    help="Path to output scene camera file.",
)
args = parser.parse_args()

# load canonical pose and scale
with open(args.canonical_pose_fn.as_posix(), "r") as f:
    canonical_pose_transform = np.array(json.load(f))
assert canonical_pose_transform.shape == (4, 4)
with open(args.canonical_scale_fn.as_posix(), "r") as f:
    canonical_scale = float(json.load(f)["scale_factor"])

# load scene GT and apply canonical pose to object poses and
# apply canonical scale to object translation
with open(args.input_bop_gt_fn, "r") as f:
    scene_gt = json.load(f)
for view_idx, view in scene_gt.items():
    assert len(view) <= 1, f"Multiple objects in scene GT view {view_idx}. Error."
    for obj in view:
        # construct object pose transform
        obj_transform = np.eye(4)
        obj_transform[:3, :3] = np.array(obj["cam_R_m2c"]).reshape(3, 3)
        obj_transform[:3, 3] = np.array(obj["cam_t_m2c"]).reshape(3)

        # apply canonical pose transform
        obj_transform = obj_transform @ np.linalg.inv(canonical_pose_transform)

        # apply canonical scale to translation
        obj_transform[:3, 3] *= canonical_scale

        # save object pose transform
        obj["cam_R_m2c"] = obj_transform[:3, :3].reshape(-1).tolist()
        obj["cam_t_m2c"] = obj_transform[:3, 3].reshape(-1).tolist()
with open(args.output_bop_gt_fn, "w") as f:
    json.dump(scene_gt, f, indent=2)

# load scene camera and apply canonical scale to
with open(args.input_bop_camera_fn, "r") as f:
    scene_camera = json.load(f)
for camera in scene_camera.values():
    camera["depth_scale"] *= canonical_scale / 1000.0
with open(args.output_bop_camera_fn, "w") as f:
    json.dump(scene_camera, f, indent=2)
