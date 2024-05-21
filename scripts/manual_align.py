# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
from functools import partial
from itertools import cycle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# parse args
parser = argparse.ArgumentParser(description="Fix and simplify a mesh.")
parser.add_argument("mesh_fn", help="path to mesh file", type=Path)
parser.add_argument(
    "--output_mesh_fn",
    type=Path,
    required=True,
    help="path to output file",
)
parser.add_argument(
    "--output_transform_fn",
    type=Path,
    required=True,
    help="path to output file for transformations",
)
parser.add_argument(
    "--output_scale_fn",
    type=Path,
    default=None,
    help="path to output file for scale factor",
)
parser.add_argument(
    "--reference_mesh",
    type=Path,
    default=None,
    help="Path to canonical mesh. If provided, will rescale mesh to match size of canonical mesh",
)
parser.add_argument(
    "--allow_translation",
    action="store_true",
    help="Allow translation of mesh",
)
parser.add_argument(
    "--allow_scaling",
    action="store_true",
    help="Allow scaling of mesh",
)
parser.add_argument(
    "--already_in_canonical_pose",
    action="store_true",
    help="If alignable mesh is already in canonical pose, do not apply initial transform",
)
args = parser.parse_args()

if args.allow_scaling:
    assert (
        args.reference_mesh is not None
    ), "Must provide reference mesh if scaling is allowed"
    assert (
        args.output_scale_fn is not None
    ), "Must provide output scale file if scaling is allowed"

# perform slow imports after parsing args
import open3d as o3d


# class to handle view changes
class ViewHandler:
    __views__ = cycle(
        [
            {  # canonical front view
                "front": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "name": "front",
            },
            {  #  canonical side view
                "front": [0.0, 1.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "name": "side",
            },
            {  #  canonical top view
                "front": [0.0, 0.0, 1.0],
                "up": [-1.0, 0.0, 0.0],
                "name": "top",
            },
        ]
    )

    def next_view(self, vis: o3d.visualization.Visualizer) -> bool:
        # apply next canonical view
        view_status_json = json.loads(vis.get_view_status())
        view = view_status_json["trajectory"][0]

        next_view = next(self.__views__)
        view["front"] = next_view["front"]
        view["up"] = next_view["up"]
        view["lookat"] = [0.0, 0.0, 0.0]
        view["field_of_view"] = 5.0

        vis.set_view_status(json.dumps(view_status_json))
        print(f"[View] Showing canonical {next_view['name']} view.", flush=True)

        return False


# class to handle mesh visibility
class VisibilityHandler:
    def __init__(
        self, mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh
    ):
        self.mesh1 = mesh1
        self.visible1 = True
        self.mesh2 = mesh2
        self.visible2 = True

    def __show_mesh1__(self, vis: o3d.visualization.Visualizer) -> bool:
        if self.visible1:
            return False
        vis.add_geometry(self.mesh1, reset_bounding_box=False)
        self.visible1 = True
        return True

    def __show_mesh2__(self, vis: o3d.visualization.Visualizer) -> bool:
        if self.visible2:
            return False
        vis.add_geometry(self.mesh2, reset_bounding_box=False)
        self.visible2 = True
        return True

    def __hide_mesh1__(self, vis: o3d.visualization.Visualizer) -> bool:
        if not self.visible1:
            return False
        vis.remove_geometry(self.mesh1, reset_bounding_box=False)
        self.visible1 = False
        return True

    def __hide_mesh2__(self, vis: o3d.visualization.Visualizer) -> bool:
        if not self.visible2:
            return False
        vis.remove_geometry(self.mesh2, reset_bounding_box=False)
        self.visible2 = False
        return True

    def show_all(self, vis: o3d.visualization.Visualizer) -> bool:
        return any([self.__show_mesh1__(vis), self.__show_mesh2__(vis)])

    def alternate_visibility(self, vis: o3d.visualization.Visualizer) -> bool:
        if self.visible1 and self.visible2:
            return self.__hide_mesh1__(vis)
        elif self.visible1:
            return any([self.__hide_mesh1__(vis), self.__show_mesh2__(vis)])
        elif self.visible2:
            return any([self.__hide_mesh2__(vis), self.__show_mesh1__(vis)])
        else:
            return self.__show_mesh2__(vis)


# class to handle alignment transform
class TransformHandler:
    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        bbox: o3d.geometry.AxisAlignedBoundingBox,
    ):
        self.transform = np.eye(4)
        self.scale = 1.0
        self.mesh = mesh
        self.bbox = bbox

    def __apply_transform__(
        self,
        vis: o3d.visualization.Visualizer,
        transform: np.ndarray,
    ) -> bool:
        # transform mesh
        self.mesh.transform(transform)
        self.transform = transform @ self.transform

        # recompute bounding box
        bbox = self.mesh.get_axis_aligned_bounding_box()
        bbox.color = self.bbox.color
        vis.remove_geometry(self.bbox, reset_bounding_box=False)
        vis.add_geometry(bbox, reset_bounding_box=False)
        self.bbox = bbox
        return True

    def __rotate_around_vector__(
        self,
        vis: o3d.visualization.Visualizer,
        vector: np.ndarray,
        step_size: float,
    ) -> bool:
        rotation = Rotation.from_rotvec(step_size * vector, degrees=True).as_matrix()
        rotation_4x4 = np.eye(4)
        rotation_4x4[:3, :3] = rotation
        return self.__apply_transform__(vis, rotation_4x4)

    def rotate_in_screen_space(
        self,
        vis: o3d.visualization.Visualizer,
        step_size: float,
    ) -> bool:
        # get front vector
        view_status_json = json.loads(vis.get_view_status())
        view = view_status_json["trajectory"][0]
        front = np.array(view["front"])

        # snap to nearest axis
        front = np.round(front)
        if np.all(front == 0.0):
            return False
        front = front / np.linalg.norm(front)
        if np.sum(np.abs(front)) != 1.0:
            return False

        # apply rotation
        return self.__rotate_around_vector__(vis, front, step_size)

    def handle_rotation_key(
        self,
        vis: o3d.visualization.Visualizer,
        key: int,
        action: int,
        vector: np.ndarray,
    ) -> bool:
        # ensure key is pressed or held
        if key not in [1, 2]:
            return False

        # get modifier keys
        shift_pressed = action in [1, 3, 5]
        ctrl_pressed = action in [2, 3, 6]
        alt_pressed = action in [4, 5, 6]

        # set initial step size
        step_size = -1.0

        # rotate by 90 degs if ctrl is pressed
        if ctrl_pressed:
            step_size = np.sign(step_size) * 90.0

        # reverse direction if shift is pressed
        if shift_pressed:
            step_size *= -1.0

        # scale step size if alt is pressed
        if alt_pressed:
            step_size /= 10.0

        # apply rotation
        return self.__rotate_around_vector__(vis, vector, step_size)

    def handle_translation_key(
        self,
        vis: o3d.visualization.Visualizer,
        key: int,
        action: int,
        vector: np.ndarray,
    ) -> bool:
        # ensure key is pressed or held
        if key not in [1, 2]:
            return False

        # get modifier keys
        shift_pressed = action in [1, 3, 5]
        ctrl_pressed = action in [2, 3, 6]
        alt_pressed = action in [4, 5, 6]

        # set initial step size
        step_size = 1.0

        # translate by 50mm if ctrl is pressed
        if ctrl_pressed:
            step_size = np.sign(step_size) * 50.0

        # reverse direction if shift is pressed
        if shift_pressed:
            step_size *= -1.0

        # scale step size if alt is pressed
        if alt_pressed:
            step_size /= 5.0

        # apply translation
        translation_4x4 = np.eye(4)
        translation_4x4[:3, 3] = step_size * vector
        return self.__apply_transform__(vis, translation_4x4)


# display mesh
def display_and_align(
    alignable_mesh: o3d.geometry.TriangleMesh,
    origin_size: float,
    reference_mesh: o3d.geometry.TriangleMesh = None,
    allow_translation: bool = False,
    allow_scaling: bool = False,
) -> np.ndarray:
    # initialize viewer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # add origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=origin_size, origin=[0, 0, 0]
    )
    vis.add_geometry(origin)

    # add meshes and bounding boxes
    def add_mesh_to_viz(mesh: o3d.geometry.TriangleMesh):
        # compute normals for better shading
        mesh.compute_vertex_normals()

        # add mesh
        vis.add_geometry(mesh)

        # add bounding box
        mesh_bbox = mesh.get_axis_aligned_bounding_box()
        mesh_bbox.color = [0.0, 0.5, 1.0]
        vis.add_geometry(mesh_bbox)

        return mesh_bbox

    alignable_mesh_bbox = add_mesh_to_viz(alignable_mesh)
    if reference_mesh is not None:
        reference_mesh_bbox = add_mesh_to_viz(reference_mesh)
    else:
        reference_mesh_bbox = None

    # color alignable mesh
    alignable_mesh_bbox.color = [1.0, 0.0, 0.0]

    # add key callback for view handler
    view_handler = ViewHandler()
    vis.register_key_callback(ord("V"), view_handler.next_view)
    view_handler.next_view(vis)  # apply canonical front view

    # add key callback for visibility handler
    if reference_mesh is not None:
        visibility_handler = VisibilityHandler(reference_mesh, alignable_mesh)
        vis.register_key_callback(ord("A"), visibility_handler.alternate_visibility)
        vis.register_key_callback(ord("S"), visibility_handler.show_all)

    # add key callbacks for rotation handler
    transform_handler = TransformHandler(alignable_mesh, alignable_mesh_bbox)
    vis.register_key_action_callback(
        ord("R"),
        partial(
            transform_handler.handle_rotation_key,
            vector=np.array([1, 0, 0]),
        ),
    )
    vis.register_key_action_callback(
        ord("G"),
        partial(
            transform_handler.handle_rotation_key,
            vector=np.array([0, 1, 0]),
        ),
    )
    vis.register_key_action_callback(
        ord("B"),
        partial(
            transform_handler.handle_rotation_key,
            vector=np.array([0, 0, 1]),
        ),
    )

    # add key callback for translation handler
    if allow_translation:
        vis.register_key_action_callback(
            ord("X"),
            partial(
                transform_handler.handle_translation_key,
                vector=np.array([1, 0, 0]),
            ),
        )
        vis.register_key_action_callback(
            ord("Y"),
            partial(
                transform_handler.handle_translation_key,
                vector=np.array([0, 1, 0]),
            ),
        )
        vis.register_key_action_callback(
            ord("Z"),
            partial(
                transform_handler.handle_translation_key,
                vector=np.array([0, 0, 1]),
            ),
        )

    # add key callback for scaling handler
    if allow_scaling:
        pass

    # visualize geometries with custom visualizer
    print("Displaying alignment tool.", flush=True)
    print(
        "> Press 'v' to cycle through views, starting with canonical front view",
        flush=True,
    )

    print(
        "> Press 'r', 'g', or 'b' to rotate by 1 degree around the x, y, or z axis",
        flush=True,
    )
    print(
        "    Holding 'shift' will reverse the direction of rotation",
        flush=True,
    )
    print("    Holding 'ctrl' will rotate by 90 degrees", flush=True)
    print("    Holding 'alt' will rotate by 0.1 degrees", flush=True)

    if args.allow_translation:
        print(
            "> Press 'x', 'y', or 'z' to translate by 1mm along the x, y, or z axis",
            flush=True,
        )
        print(
            "    Holding 'shift' will reverse the direction of translation",
            flush=True,
        )
        print("    Holding 'ctrl' will translate by 5cm", flush=True)
        print("    Holding 'alt' will translate by 0.2mm", flush=True)

    if args.reference_mesh is not None:
        print("> Press 'a' to alternate visibility between meshes", flush=True)
        print("> Press 's' to show both meshes", flush=True)

    print("> Press 'q' to quit.", flush=True)
    vis.run()

    # return total rotation
    return transform_handler.transform, transform_handler.scale


# TODO: handle manual scale adjustments when aligning to canonical mesh

# load alignable mesh
print(f"Reading alignable mesh from {args.mesh_fn}", flush=True)
alignable_mesh = o3d.io.read_triangle_mesh(
    args.mesh_fn.as_posix(), enable_post_processing=True
)
alignable_mesh_obb = alignable_mesh.get_oriented_bounding_box()

# load existing alignment transform
if args.output_transform_fn.exists():
    with open(args.output_transform_fn, "r") as f:
        initial_transform = np.array(json.load(f))
# or initialize to identity, if alignable mesh is already in canonical pose
elif args.already_in_canonical_pose:
    initial_transform = np.eye(4)
# or initialize by centering and aligning to oriented bounding box
else:
    # align oriented bounding box to origin
    obb_transform = np.eye(4)
    obb_transform[:3, :3] = alignable_mesh_obb.R
    obb_transform[:3, 3] = alignable_mesh_obb.center
    initial_transform = np.linalg.inv(obb_transform)

# apply initial transform
alignable_mesh.transform(initial_transform)

# load reference mesh, if applicable
if args.reference_mesh:
    print(f"Reading reference mesh from {args.reference_mesh}", flush=True)
    reference_mesh = o3d.io.read_triangle_mesh(
        args.reference_mesh.as_posix(), enable_post_processing=False
    )

    # lighten the reference mesh
    vertex_colors = np.asarray(reference_mesh.vertex_colors)
    vertex_colors = vertex_colors * 0.5 + 0.5
    reference_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

else:
    reference_mesh = None

# display and align
alignment_transform, alignment_scale = display_and_align(
    alignable_mesh,
    origin_size=alignable_mesh_obb.extent[1] * 1.25,
    reference_mesh=reference_mesh,
    allow_translation=args.allow_translation,
    allow_scaling=args.allow_scaling,
)

# save transformation
final_transform = alignment_transform @ initial_transform
with open(args.output_transform_fn, "w") as f:
    json.dump(final_transform.tolist(), f, indent=2)
print(f"Alignment transform saved to {args.output_transform_fn}", flush=True)

# save scale factor
if args.allow_scaling:
    with open(args.output_scale_fn, "w") as f:
        json.dump({"scale_factor": alignment_scale}, f, indent=2)
    print(f"Scale factor saved to {args.output_scale_fn}", flush=True)

# save transformed mesh
alignable_mesh = o3d.io.read_triangle_mesh(
    args.mesh_fn.as_posix(), enable_post_processing=True
)
alignable_mesh.transform(final_transform)
if args.output_mesh_fn is not None:
    o3d.io.write_triangle_mesh(args.output_mesh_fn.as_posix(), alignable_mesh)
    print(f"Aligned mesh written to {args.output_mesh_fn}", flush=True)
print("Done.", flush=True)
