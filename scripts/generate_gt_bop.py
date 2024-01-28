# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from rich import print
from tqdm import tqdm

rotation_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
rotate_neg_90_y = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
rotate_90_z = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def mark_path(path) -> str:
    return f"[steel_blue not bold]{path}[/steel_blue not bold]"


def copy_reference_mesh(
    canonical_mesh_path: Path,
    bop_models_dir: Path,
    obj_id: int,
    force_rewrite: bool = False,
) -> Path:
    # check for missing paths
    if not canonical_mesh_path.exists():
        raise FileNotFoundError(
            f"Could not find reference mesh at {canonical_mesh_path}"
        )
    if not bop_models_dir.exists():
        raise FileNotFoundError(
            f"Could not find BOP models directory at {bop_models_dir}"
        )

    # copy mesh
    mesh_export_path = bop_models_dir / f"obj_{obj_id:06d}.ply"
    if canonical_mesh_path.suffix == ".ply" and not force_rewrite:
        shutil.copy2(canonical_mesh_path, mesh_export_path)
    else:
        mesh: trimesh.Trimesh = trimesh.load_mesh(str(canonical_mesh_path))
        mesh.export(str(mesh_export_path))

    return mesh_export_path


# generate scene_gt.json
def get_scene_gt(
    colmap_transforms: dict,
    world_to_model: np.array,
    obj_id: int,
) -> dict:
    # get transform from canonical model to colmap model
    model_to_world = np.linalg.inv(world_to_model)
    canonical_model_to_colmap_model = model_to_world

    # remove scale from model to camera transform
    s = np.linalg.norm(canonical_model_to_colmap_model[:3, 0])
    canonical_model_to_colmap_model_rescaled = canonical_model_to_colmap_model.copy()
    canonical_model_to_colmap_model_rescaled[:3, :3] /= s

    # construct GT pose for each frame
    scene_gt = {}
    for frame in colmap_transforms["frames"]:
        # get frame index
        frame_idx = int(Path(frame["file_path"]).stem)

        # get world to camera transform
        camera_to_world = np.array(frame["transform_matrix"])
        world_to_camera = np.linalg.inv(camera_to_world)

        # get model to camera transform
        model_to_camera = rotation_180 @ world_to_camera @ rotate_neg_90_y @ rotate_90_z
        m2c = model_to_camera @ canonical_model_to_colmap_model_rescaled

        # construct GT pose dict
        gt = {}
        gt["cam_R_m2c"] = m2c[:3, :3].flatten().tolist()
        gt["cam_t_m2c"] = (m2c[:3, -1] / s).tolist()
        gt["obj_id"] = obj_id
        scene_gt[str(int(frame_idx))] = [gt]

    return scene_gt


def get_scene_camera(transforms):
    scene_camera = {}

    cam_K = [transforms["fl_x"], 0.0, transforms["cx"]]
    cam_K += [0.0, transforms["fl_y"], transforms["cy"]]
    cam_K += [0.0, 0.0, 1.0]

    depth_scale = 1.0

    for i, frame in enumerate(transforms["frames"]):
        index = Path(frame["file_path"]).stem

        scene_camera[str(int(index))] = {
            "cam_K": cam_K,
            "depth_scale": depth_scale,
            "resolution": [transforms["w"], transforms["h"]],
            "height": transforms["h"],
            "width": transforms["w"],
        }

    return scene_camera


def remove_bad_frames(scene_gt: dict, scene_base_dir: Path) -> dict:
    removed_frames_fn = scene_base_dir / "removed_frames.json"
    if removed_frames_fn.exists():
        # TODO document this!
        removed_frames = json.load(removed_frames_fn)
        removed_frames = set(removed_frames["removed_frames"])

        for frame in scene_gt:
            if frame in removed_frames:
                scene_gt.pop(frame)

    return scene_gt


def remove_frames_not_in_scene_gt(scene_gt: dict, bop_folder: Path) -> list:
    scene_gt_frames = set([int(frame) for frame in scene_gt.keys()])
    images = sorted((bop_folder / "rgb").iterdir())

    removed_frames = []
    for img in images:
        if not int(img.stem) in scene_gt_frames:
            img.unlink()
            removed_frames.append(img)

    return removed_frames


def copy_images(scene_image_dir: Path, bop_image_dir: Path, scene_gt: dict) -> None:
    # get source and destination directories
    if not scene_image_dir.exists():
        raise FileNotFoundError(
            f"Could not find scene image directory at {scene_image_dir}"
        )
    bop_image_dir.mkdir(exist_ok=True)

    # load removed frames
    removed_frames_fn = bop_scene_dir / "removed_frames.json"
    if removed_frames_fn.exists():
        removed_frames = json.load(removed_frames_fn)
        removed_frames = set(removed_frames["removed_frames"])
    else:
        removed_frames = set()

    # copy images
    print(
        f"Copying images from {mark_path(scene_image_dir)} to {mark_path(bop_image_dir)}"
    )
    for img_fn in tqdm(sorted(scene_image_dir.iterdir())):
        if img_fn.suffix in [".png", ".jpg"] and int(img_fn.stem) not in removed_frames:
            shutil.copy2(img_fn, bop_image_dir)

    if len(removed_frames):
        print(
            f"Did not copy frames listed in {mark_path(removed_frames_fn)}: {removed_frames}"
        )

    print(
        f"Copied {len(list(bop_image_dir.iterdir()))} files from "
        f"{mark_path(scene_image_dir)} to {mark_path(bop_image_dir)}"
    )

    # Remove frames not in scene_gt.json
    removed_frames = remove_frames_not_in_scene_gt(scene_gt, bop_scene_dir)
    print(
        f"Removed {len(removed_frames)} frames from {mark_path(bop_image_dir)} "
        f"that were missing from {mark_path(bop_scene_dir/'scene_gt.json')}"
    )


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bop_base_dir",
        type=Path,
        help="Path to base BOP directory",
    )
    parser.add_argument(
        "--camera_poses_fn",
        type=Path,
        required=True,
        help="Path to camera poses",
    )
    # parser.add_argument(
    #     "--reference_mesh_fn",
    #     type=Path,
    #     # required=True,
    #     default=None,
    #     help="Path to reference mesh",
    # )
    parser.add_argument(
        "--scene_image_dir",
        type=Path,
        # required=True,
        default=None,
        help="Path to scene image directory",
    )
    parser.add_argument(
        "--scene_depth_dir",
        type=Path,
        # required=True,
        default=None,
        help="Path to scene depth image directory",
    )
    # parser.add_argument(
    #     "--alignment_transform_fn",
    #     type=Path,
    #     default=None,
    #     help="Path to alignment transform",
    # )
    parser.add_argument(
        "--bop_object_id",
        type=int,
        default=1,
        help="BOP object id",
    )
    parser.add_argument(
        "--output_fn_tag",
        type=str,
        default="",
        help="Tag to append to output filenames",
    )
    args = parser.parse_args()

    # # copy canonical mesh to BOP models directory
    # # bop_models_dir = args.bop_base_dir / "models"
    # # bop_models_dir.mkdir(exist_ok=True)
    # args.bop_base_dir.mkdir(exist_ok=True)
    # bop_models_dir = args.bop_base_dir
    # # reference_mesh_fn = args.reference_dir / "mesh" / "reference.ply"
    # reference_mesh_fn = args.reference_mesh_fn
    # bop_reference_mesh_fn = copy_reference_mesh(
    #     reference_mesh_fn, bop_models_dir, args.bop_object_id, force_rewrite=True
    # )
    # print(
    #     f"Copied reference mesh {mark_path(reference_mesh_fn)} to "
    #     f"{mark_path(bop_reference_mesh_fn)}"
    # )

    # # load alignment transform
    # if not args.alignment_transform_fn:
    #     alignment_transform_mtx = np.eye(4)
    #     print(f"Using identity matrix for alignment transform.")
    # else:
    #     if not args.alignment_transform_fn.exists():
    #         raise FileNotFoundError(
    #             f"'{args.alignment_transform_fn}' does not exist. Generate this file with 'manual_align.py'."
    #         )
    #     with open(args.alignment_transform_fn, "r") as fp:
    #         alignment_transform = json.load(fp)
    #     alignment_transform_mtx = np.array(alignment_transform["transform"])
    #     print(
    #         f"Loaded alignment transform from {mark_path(args.alignment_transform_fn)}"
    #     )

    # load colmap camera poses
    if not args.camera_poses_fn.exists():
        raise FileNotFoundError(f"'{args.camera_poses_fn}' does not exist.")
    with open(args.camera_poses_fn, "r") as fp:
        colmap_transforms = json.load(fp)
    print(f"Loaded colmap camera poses from {mark_path(args.camera_poses_fn)}")

    # create BOP scene directory
    # bop_scene_dir = args.bop_base_dir / "scenes" / args.bop_id
    # bop_scene_dir.mkdir(exist_ok=True, parents=True)
    bop_scene_dir = args.bop_base_dir
    bop_scene_dir.mkdir(exist_ok=True, parents=True)
    print(f"Writing BOP scene files to {mark_path(bop_scene_dir)}")

    # generate scene_camera.json
    scene_camera = get_scene_camera(colmap_transforms)
    scene_camera_fn = bop_scene_dir / "scene_camera.json"
    scene_camera_fn = scene_camera_fn.with_stem(
        scene_camera_fn.stem + args.output_fn_tag
    )
    with open(scene_camera_fn, "w") as fp:
        json.dump(scene_camera, fp, indent=2)
    print(f"Wrote camera info to {mark_path(scene_camera_fn)}")

    # generate scene_gt.json
    alignment_transform_mtx = np.eye(4)
    scene_gt = get_scene_gt(
        colmap_transforms, alignment_transform_mtx, args.bop_object_id
    )
    scene_gt = remove_bad_frames(scene_gt, bop_scene_dir)
    scene_gt_fn = bop_scene_dir / "scene_gt.json"
    scene_gt_fn = scene_gt_fn.with_stem(scene_gt_fn.stem + args.output_fn_tag)
    with open(scene_gt_fn, "w") as fp:
        json.dump(scene_gt, fp, indent=2)
    print(f"Wrote object poses to {mark_path(scene_gt_fn)}")

    # copy RGB and depth images
    if args.scene_image_dir is not None:
        copy_images(args.scene_image_dir, bop_scene_dir / "rgb", scene_gt)
    if args.scene_depth_dir is not None:
        copy_images(args.scene_depth_dir, bop_scene_dir / "depth_nerf", scene_gt)
