# This script was adapted from the following script in the BOP toolkit:
#   https://github.com/thodan/bop_toolkit/blob/master/scripts/calc_gt_masks.py
#   MIT License; Copyright (c) 2019 Tomas Hodan (hodantom@cmp.felk.cvut.cz)

import argparse
from pathlib import Path

import numpy as np
from bop_toolkit_lib import inout, misc, renderer, visibility
from tqdm import tqdm

__print = print
from rich import print


# recursively get all instances of a key from a dict
def get_all_from_dict(d: dict, key) -> list:
    def search_dict(d: dict, key) -> list:
        for k, v in d.items():
            if k == key:
                yield v
            elif isinstance(v, dict):
                yield from search_dict(v, key)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        yield from search_dict(x, key)

    return list(search_dict(d, key))


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "bop_scene_dir",
    type=Path,
    help="Path to BOP scene directory",
)
parser.add_argument(
    "--scene_depth_dir",
    type=Path,
    default=None,
    help="Path to BOP scene depth directory (default: 'BOP_SCENE_DIR/depth')",
)
parser.add_argument(
    "--scene_gt_fn",
    type=Path,
    default=None,
    help="Path to BOP scene GT file (default: 'BOP_SCENE_DIR/scene_gt.json')",
)
parser.add_argument(
    "--scene_camera_fn",
    type=Path,
    default=None,
    help="Path to BOP scene camera file (default: 'BOP_SCENE_DIR/scene_camera.json')",
)
parser.add_argument(
    "--scene_mesh_fn_fmt",
    type=str,
    default=None,
    help="Format string for path to BOP scene mesh files (default: 'BOP_SCENE_DIR/obj_{:06d}.ply')",
)
parser.add_argument(
    "--mask_types",
    type=str,
    default=["full"],
    choices=["full", "visible"],
    nargs="+",
    help="Type of mask to generate (default: full)",
)
parser.add_argument(
    "--visible_tolerance",
    type=float,
    default=25,
    help="Visible tolerance in mm (default: 25)",
)
parser.add_argument(
    "--write_object_depth",
    action="store_true",
    help="Write object depth images to BOP scene directory",
)
parser.add_argument(
    "--read_object_depth",
    action="store_true",
    help="Read object depth images from BOP scene directory",
)
parser.add_argument(
    "--full_mask_output_dir",
    type=Path,
    default=None,
    help="Path to full mask output directory (default: 'BOP_SCENE_DIR/mask')",
)
parser.add_argument(
    "--visible_mask_output_dir",
    type=Path,
    default=None,
    help="Path to visible mask output directory (default: 'BOP_SCENE_DIR/mask_visib')",
)
args = parser.parse_args()

# load scene GT and camera
if args.scene_gt_fn is None:
    scene_gt_fn = args.bop_scene_dir / "scene_gt.json"
else:
    scene_gt_fn = args.scene_gt_fn
scene_gt = inout.load_scene_gt(scene_gt_fn)
print(f"Loaded scene annotations from {scene_gt_fn}")

if args.scene_camera_fn is None:
    scene_camera_fn = args.bop_scene_dir / "scene_camera.json"
else:
    scene_camera_fn = args.scene_camera_fn
scene_camera: dict = inout.load_scene_camera(scene_camera_fn)
print(f"Loaded scene camera parameters from {scene_camera_fn}")

# create mask directory
if args.full_mask_output_dir is None:
    args.full_mask_output_dir = args.bop_scene_dir / "mask"
if args.visible_mask_output_dir is None:
    args.visible_mask_output_dir = args.bop_scene_dir / "mask_visib"
mask_output_dirs = [
    args.full_mask_output_dir if mask_type == "full" else args.visible_mask_output_dir
    for mask_type in args.mask_types
]
for mask_output_dir in mask_output_dirs:
    mask_output_dir.mkdir(exist_ok=True)

# get scene/object depth directories
if args.scene_depth_dir is None:
    scene_depth_dir = args.bop_scene_dir / "depth"
else:
    scene_depth_dir = args.scene_depth_dir
object_depth_dir = args.bop_scene_dir / "rendered_depth_object"
if args.write_object_depth:
    object_depth_dir.mkdir(exist_ok=True)

# get image size from scene_camera
image_sizes = get_all_from_dict(scene_camera, "resolution")
image_widths, image_heights = zip(*image_sizes)
assert len(set(image_widths)) == 1 and len(set(image_heights)) == 1, (
    f"This script requires all images to have the same size, "
    f"but the following sizes were found in {scene_camera_fn}: "
    f"{set(image_widths)=}, {set(image_heights)=}"
)
width, height = map(int, [image_widths[0], image_heights[0]])

# initialize renderer
print(f"Initializing renderer")
__print("\033[38;5;240m", end="")  # change color of renderer terminal output
bop_renderer = renderer.create_renderer(
    width, height, renderer_type="vispy", mode="depth"
)
__print("\033[0m", end="")  # reset terminal color

# add object models
object_ids = get_all_from_dict(scene_gt, "obj_id")
for obj_id in list(set(object_ids)):
    if args.scene_mesh_fn_fmt is not None:
        mesh_fn = Path(args.scene_mesh_fn_fmt.format(obj_id))
    else:
        mesh_fn = args.bop_scene_dir / f"obj_{obj_id:06d}.ply"
    print(f"Adding {mesh_fn}")
    bop_renderer.add_object(obj_id, str(mesh_fn))

# iterate over images
print(f"Rendering {' & '.join(args.mask_types)} masks")
for im_id in tqdm(sorted(scene_gt.keys())):
    # get camera intrinsics
    K = scene_camera[im_id]["cam_K"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # load scene depth image, if computing visible mask
    if "visible" in args.mask_types:
        scene_depth_fn = scene_depth_dir / f"{im_id:06d}.png"
        scene_depth_im = inout.load_depth(scene_depth_fn)
        scene_depth_im *= scene_camera[im_id]["depth_scale"]  # to [mm]
        scene_dist_im = misc.depth_im_to_dist_im_fast(scene_depth_im, K)

    # iterate over objects
    for gt_id, gt in enumerate(scene_gt[im_id]):
        # render the object depth image
        object_depth_fn = object_depth_dir / f"{im_id:06d}_{gt_id:06d}.png"
        if args.read_object_depth and object_depth_fn.exists():
            object_depth_im = inout.load_depth(str(object_depth_fn))
        else:
            object_depth_im = bop_renderer.render_object(
                gt["obj_id"], gt["cam_R_m2c"], gt["cam_t_m2c"], fx, fy, cx, cy
            )["depth"]
            if args.write_object_depth:
                inout.save_depth(str(object_depth_fn), object_depth_im)
        object_dist_im = misc.depth_im_to_dist_im_fast(object_depth_im, K)

        # compute masks
        for mask_type, mask_output_dir in zip(args.mask_types, mask_output_dirs):
            if mask_type == "full":
                # mask of the full object silhouette
                mask = object_dist_im > 0
                mask_output_fn = mask_output_dir / f"{im_id:06d}_{gt_id:06d}.png"
                inout.save_im(str(mask_output_fn), 255 * mask.astype(np.uint8))

            elif mask_type == "visible":
                # mask of the visible part of the object silhouette.
                mask = visibility.estimate_visib_mask_gt(
                    scene_dist_im,
                    object_dist_im,
                    args.visible_tolerance,
                    visib_mode="bop19",
                )
                mask_output_fn = mask_output_dir / f"{im_id:06d}_{gt_id:06d}.png"
                inout.save_im(str(mask_output_fn), 255 * mask.astype(np.uint8))

print("Masks written to:")
for p in mask_output_dirs:
    print(f"  {p}")
print(f"Done!")
