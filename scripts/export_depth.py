# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from bop_toolkit_lib import inout
from tqdm import tqdm


# function to handle writing depth images as NPZ
def write_depth_npz(
    filename: Path, depth: np.ndarray, alpha: np.ndarray, use_fp16: bool = True
) -> None:
    """Saves a depth image to an NPZ file.
    :param filename: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    :param alpha: ndarray with the alpha channel of the depth image to save.
    :param use_fp16: Whether to use FP16 for depth and alpha channels.
    """
    if filename.suffix.lower() != ".npz":
        raise ValueError("Only NPZ format is currently supported.")
    if use_fp16:
        depth = depth.astype(np.float16)
        alpha = alpha.astype(np.float16)
    np.savez_compressed(filename, depth=depth, alpha=alpha)


# parse arguments
parser = argparse.ArgumentParser(description="Render depth images from an NGP scene")
parser.add_argument(
    "snapshot_fn",
    type=Path,
    help="path to a saved scene snapshot",
)
parser.add_argument(
    "output_dir",
    type=Path,
    help="path to a directory to save the depth images",
)
parser.add_argument(
    "--ngp_root",
    type=Path,
    required=True,
    help="path to the NGP root directory",
)
parser.add_argument(
    "--save_as_npz",
    action="store_true",
    help="save depth images as .npz files instead of .png files",
)
parser.add_argument(
    "--colorized_depth_dir",
    type=Path,
    default=None,
    help="path to a directory to save colorized depth images, if desired",
)
parser.add_argument(
    "--depth_scale",
    type=float,
    default=-1,
    help="depth scale (default: -1, i.e. unknown scale)",
)
parser.add_argument(
    "--clip_distance",
    type=float,
    default=-1,
    help="clip distance (default: -1, i.e. no clipping)",
)
parser.add_argument(
    "--median_filter",
    type=int,
    default=5,
    choices=[False, 3, 5],
    help="median filter kernel size when saving as PNG (default: 5)",
)
args = parser.parse_args()

# check args
assert args.snapshot_fn.is_file(), f"Snapshot file not found: {args.snapshot_fn}"
args.output_dir.mkdir(exist_ok=True, parents=True)
assert (
    args.output_dir.is_dir()
), f"Unable to find or create output directory: {args.output_dir}"
assert args.ngp_root.is_dir(), f"NGP root directory not found: {args.ngp_root}"
if args.colorized_depth_dir is not None:
    args.colorized_depth_dir.mkdir(exist_ok=True, parents=True)
    assert (
        args.colorized_depth_dir.is_dir()
    ), f"Unable to find or create colorized depth directory: {args.colorized_depth_dir}"

# import pyngp
sys.path.append(str(args.ngp_root / "build"))
import pyngp as ngp  # noqa

# initialize NGP and load snapshot
testbed = ngp.Testbed()
testbed.load_snapshot(str(args.snapshot_fn))

# set up rendering parameters
testbed.background_color = [0.0, 0.0, 0.0, 0.0]
testbed.render_mode = ngp.RenderMode.Depth
testbed.nerf.render_gbuffer_hard_edges = True

# render a depth image for each input camera pose stored in the snapshot
for idx in tqdm(range(testbed.nerf.training.dataset.n_images)):
    # render the depth image
    testbed.set_camera_to_training_view(idx)
    resolution = testbed.nerf.training.dataset.metadata[idx].resolution
    output = testbed.render(*resolution, 1, True)
    alpha = output[:, :, -1:]
    depth_image = output[:, :, :1]

    # scale and clip depth image
    if args.depth_scale > 0:
        depth_image *= args.depth_scale  # scale to mm, if available
    if args.clip_distance > 0:
        depth_image = np.clip(depth_image, 0.0, args.clip_distance)
        depth_image[depth_image == args.clip_distance] = 0.0
    depth_image = depth_image[:, :, 0]
    alpha = alpha[:, :, 0]

    # apply median filter
    if args.median_filter > 0:
        depth_image = cv2.medianBlur(depth_image, args.median_filter)
        #  note that `cv2.medianBlur` only supports kernel sizes of 3 or 5
        #  for images that are not `uint8`

    # write depth image
    fn_stem = Path(testbed.nerf.training.dataset.paths[idx]).stem
    depth_output_fn = args.output_dir / fn_stem
    if args.save_as_npz:
        depth_output_fn = depth_output_fn.with_suffix(".npz")
        write_depth_npz(
            depth_output_fn,
            depth_image,
            alpha,
        )
    else:
        depth_output_fn = depth_output_fn.with_suffix(".png")
        inout.save_depth(
            str(depth_output_fn),
            depth_image * 1000,  # convert from m to mm
        )

    # write colorized depth image
    if args.colorized_depth_dir:
        colorized_depth_image = np.uint8(depth_image / args.clip_distance * 256)
        colorized_depth_image = cv2.applyColorMap(
            colorized_depth_image, cv2.COLORMAP_TURBO
        )
        colorized_depth_image[depth_image == 0] = 0
        cv2.imwrite(
            str(args.colorized_depth_dir / (fn_stem + ".jpg")),
            colorized_depth_image,
        )
