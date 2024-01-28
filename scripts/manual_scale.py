# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import trimesh

# parse arguments
parser = argparse.ArgumentParser(
    description="Get scale factor by comparing mesh bbox vs. known value."
)
parser.add_argument(
    "input_mesh_fn",
    type=Path,
    help="Path to input mesh file.",
)
parser.add_argument(
    "output_scaled_mesh_fn",
    type=Path,
    help="Path to output scaled mesh file.",
)
parser.add_argument(
    "output_fn",
    type=Path,
    help="Path to output file.",
)
parser.add_argument(
    "--scale_factor",
    type=float,
    default=False,
    help="Record and apply a known scale factor.",
)
args = parser.parse_args()

# load mesh and get bounding box dimensions
print(f"Loading mesh from {args.input_mesh_fn}...")
mesh = trimesh.load_mesh(args.input_mesh_fn.as_posix())
bbox = mesh.bounding_box.bounds
bbox_dims = bbox[1] - bbox[0]
depth, width, height = bbox_dims

# check if scale factor is given
if not args.scale_factor:
    # print dimensions
    print(f"Current dimensions of mesh bounding box (in BOP axes):")
    print(f"  height: {height:.1f}")
    print(f"  width: {width:.1f}")
    print(f"  depth: {depth:.1f}")
    print()

    # prompt user for GT dimensions
    print("Please enter the dimensions of the bounding box in mm.")
    print("(To skip a dimension, leave it blank.)")
    blank_as_nan = lambda x: np.nan if x == "" else float(x)
    height_gt = blank_as_nan(input("  height: ").strip())
    width_gt = blank_as_nan(input("  width: ").strip())
    depth_gt = blank_as_nan(input("  depth: ").strip())

    # print given dimensions
    print("Given dimensions:")
    print(f"  height: {height_gt if not np.isnan(height_gt) else ''}")
    print(f"  width: {width_gt if not np.isnan(width_gt) else ''}")
    print(f"  depth: {depth_gt if not np.isnan(depth_gt) else ''}")

    # compute scale factor
    scale_factor_estimates = [
        x
        for x in (
            height_gt / height,
            width_gt / width,
            depth_gt / depth,
        )
        if not np.isnan(x)
    ]
    if len(scale_factor_estimates) == 0:
        raise ValueError("No valid dimensions were given.")
    print(f"Scale factor estimates: {scale_factor_estimates}")

    rel_std = np.std(scale_factor_estimates) / np.mean(scale_factor_estimates)
    if rel_std > 0.1:
        print(
            f"Warning: relative standard deviation of scale factor estimates "
            + "differs by {rel_std*100:.1f}%!"
        )
    scale_factor = np.mean(scale_factor_estimates)
else:
    scale_factor = args.scale_factor

# write scale factor to file
print(f"Scale factor: {scale_factor:.3f}")
with open(args.output_fn.as_posix(), "w") as f:
    json.dump({"scale_factor": scale_factor}, f)
print(f"Scale factor written to {args.output_fn.as_posix()}.")

# apply scale factor to mesh and write to file
mesh.apply_scale(scale_factor)
mesh.export(args.output_scaled_mesh_fn.as_posix())
print(f"Scaled mesh written to {args.output_scaled_mesh_fn.as_posix()}.")
