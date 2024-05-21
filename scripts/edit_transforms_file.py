# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
from pathlib import Path

# parse args
parser = argparse.ArgumentParser(description="Edit transforms file")
parser.add_argument("file", type=Path, help="Input file")
parser.add_argument(
    "--sort_frames",
    action="store_true",
    help="Sort frames by file_path",
)
parser.add_argument(
    "--set_aabb_scale",
    type=int,
    choices=[1, 2, 4, 8, 16, 32, 64, 128],
    default=None,
    help="Set aabb_scale",
)
parser.add_argument(
    "--edit_frame_file_path",
    type=str,
    nargs=2,
    action="append",
    default=[],
    help="Replace ARG_1 by ARG_2 in frame file paths",
)
parser.add_argument(
    "--output_file",
    type=Path,
    default=None,
    help="Output file",
)
parser.add_argument(
    "--remove_omitted_frames_from_dir",
    type=Path,
    default=None,
    help="Remove frames that have no transforms matrix",
)
args = parser.parse_args()

# ensure files/paths exist
assert args.file.exists(), f"Transforms file {args.file} does not exist."
args.output_file = args.output_file or args.file

# read file
print(f"Reading transforms from '{args.file}' ...")
with open(args.file, "r") as f:
    transforms = json.load(f)

# edit transforms file, if requested
if args.sort_frames or args.set_aabb_scale is not None or args.edit_frame_file_path:
    # sorting frames by filename
    if args.sort_frames:
        print(f"    Sorting frames ...")
        transforms["frames"] = sorted(
            transforms["frames"], key=lambda x: x["file_path"]
        )

    # edit aabb_scale
    if args.set_aabb_scale is not None:
        print(f"    Setting aabb_scale to {args.set_aabb_scale} ...")
        transforms["aabb_scale"] = args.set_aabb_scale

    # edit image file path
    for before, after in args.edit_frame_file_path:
        print(f"    Replacing '{before}' with '{after}' in frame file paths ...")
        for frame in transforms["frames"]:
            frame["file_path"] = frame["file_path"].replace(before, after)

    # write file
    with open(args.output_file, "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"Wrote edited transforms file to '{args.output_file}'")

# remove frames with missing transforms
if args.remove_omitted_frames_from_dir is not None:
    image_dir = args.remove_omitted_frames_from_dir
    print(f"Removing frames from '{image_dir}' with missing transforms ...")

    # get list of image files from transforms
    image_files = set([Path(frame["file_path"]).name for frame in transforms["frames"]])
    image_file_exts = set([Path(image_file).suffix for image_file in image_files])

    # ensure all image files exist
    assert all(
        [(image_dir / image_file).exists() for image_file in image_files]
    ), f"Unable to find all image files required by {args.output_file} in {image_dir}. Cannot proceed with removal of omitted frames."

    # iterate over image files in directory and find those that are not in transforms
    image_files_to_remove = []
    for fn in image_dir.iterdir():
        if fn.suffix not in image_file_exts:
            continue
        if fn.name not in image_files:
            image_files_to_remove.append(fn)

    # remove files
    if len(image_files_to_remove) > 0:
        for fn in image_files_to_remove:
            fn.unlink()
        print(f"Removed {len(image_files_to_remove)} file(s).")
    else:
        print("No files to remove.")
