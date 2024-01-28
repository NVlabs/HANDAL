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

import cv2
import numpy as np
from bop_toolkit_lib import inout
from tqdm import tqdm

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "output_mesh_fn",
    type=Path,
    help="output filename for textured mesh",
)
parser.add_argument(
    "--colored_mesh_output_fn",
    type=Path,
    default=None,
    help="output filename for mesh with vertex colors",
)
parser.add_argument(
    "--bop_pose_fn",
    type=Path,
    required=True,
    help="path to BOP pose file",
)
parser.add_argument(
    "--bop_camera_fn",
    type=Path,
    required=True,
    help="path to BOP camera file",
)
parser.add_argument(
    "--rgb_dir",
    type=Path,
    required=True,
    help="path to RGB image directory",
)
parser.add_argument(
    "--depth_dir",
    type=Path,
    required=True,
    help="path to depth directory",
)
parser.add_argument(
    "--mask_dir",
    type=Path,
    required=True,
    help="path to directory with visible instance masks",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="enable debug mode",
)
parser.add_argument(
    "--tsdf_integrate",
    action="store_true",
    help="enable TSDF volume integration",
)
parser.add_argument(
    "--max_views",
    type=int,
    default=None,
    help="maximum number of views to use",
)
parser.add_argument(
    "--export_obj",
    action="store_true",
    help="export OBJ mesh",
)
parser.add_argument(
    "--texture_existing_mesh",
    type=Path,
    default=None,
    help="apply texturing to existing mesh",
)
parser.add_argument(
    "--subdivision_iterations",
    type=int,
    default=1,
    help="number of mesh subdivision iterations during texturing",
)
args = parser.parse_args()

# perform slow imports after parsing arguments
import open3d as o3d

if args.debug:
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
else:
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# load camera trajectory
print(
    f"Loading camera trajectory from \n\t{args.bop_pose_fn} and \n\t{args.bop_camera_fn} ..."
)
with open(args.bop_pose_fn, "r") as f:
    bop_poses = json.load(f)
with open(args.bop_camera_fn, "r") as f:
    bop_cameras = json.load(f)

camera_trajectory_list = []
if args.debug:
    camera_marker_geometries = []
for pose, camera in zip(bop_poses.values(), bop_cameras.values()):
    assert len(pose) == 1, "Expected only one object pose per image"
    pose = pose[0]

    camera_params = o3d.camera.PinholeCameraParameters()
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = np.array(pose["cam_R_m2c"]).reshape(3, 3)
    extrinsics[:3, 3] = np.array(pose["cam_t_m2c"])
    camera_params.extrinsic = extrinsics

    K = np.array(camera["cam_K"]).reshape(3, 3)
    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(camera["width"]),
        height=int(camera["height"]),
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
    )
    camera_trajectory_list.append(camera_params)

    if args.debug:
        camera_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        camera_marker.transform(np.linalg.inv(extrinsics))
        camera_marker_geometries.append(camera_marker)
camera_trajectory = o3d.camera.PinholeCameraTrajectory()
camera_trajectory.parameters = camera_trajectory_list

# load RGB-D images
print(f"Loading RGB-D images from \n\t{args.rgb_dir} and \n\t{args.depth_dir} ...")
rgbd_images = []
rgb_fns = sorted(args.rgb_dir.glob("*.*"))

# subsample views for speed
# TODO replace this with clustering by viewing direction
if args.max_views and len(rgb_fns) > args.max_views:
    debug_views = np.random.permutation(np.arange(len(rgb_fns)))[: args.max_views]
    rgb_fns = [rgb_fns[idx] for idx in debug_views]
    camera_trajectory.parameters = [
        camera_trajectory.parameters[idx] for idx in debug_views
    ]
    if args.debug:
        camera_marker_geometries = [
            camera_marker_geometries[idx] for idx in debug_views
        ]


# function to apply mask to RGB-D images
def apply_mask(
    image: o3d.geometry.Image, mask: o3d.geometry.Image
) -> o3d.geometry.Image:
    image = np.asarray(image)
    mask = np.asarray(mask)
    image[mask == 0] = 0
    image = o3d.geometry.Image(image)
    return image


# load and mask RGB-D images
for rgb_fn in tqdm(rgb_fns):
    # get filenames
    depth_fn = (args.depth_dir / rgb_fn.name).with_suffix(".png")
    mask_fn = args.mask_dir / (rgb_fn.stem + ".png")
    assert depth_fn.is_file(), f"Unable to find depth image: {depth_fn}"
    assert mask_fn.is_file(), f"Unable to find mask image: {mask_fn}"

    # load RGB-D image
    rgb_image = o3d.io.read_image(str(rgb_fn))
    depth_image = o3d.io.read_image(str(depth_fn))
    mask_image = o3d.io.read_image(str(mask_fn))

    # apply mask
    rgb_image = apply_mask(rgb_image, mask_image)
    depth_image = apply_mask(depth_image, mask_image)

    # create RGB-D image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image,
        depth_image,
        depth_scale=1000.0,  # depth is saved in mm
        depth_trunc=1e6,
        convert_rgb_to_intensity=False,
    )
    rgbd_images.append(rgbd_image)

# create mesh from RGB-D images
if not args.texture_existing_mesh:
    # integrate RGB-D images into a single point cloud
    if args.tsdf_integrate:
        # combine point clouds using TSDF volume integration
        print("Combining point clouds using TSDF volume integration ...")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        for rgbd_image, camera_params in tqdm(
            zip(rgbd_images, camera_trajectory.parameters),
            total=len(rgbd_images),
        ):
            volume.integrate(
                rgbd_image,
                camera_params.intrinsic,
                camera_params.extrinsic,
            )
        pcd = volume.extract_point_cloud()
        if args.debug:
            print("Visualizing integrated point cloud [quit viewer to continue] ...")
            o3d.visualization.draw_geometries([pcd])

    else:
        # convert RGB-D images to point clouds and visualize
        print("Converting RGB-D images to point clouds ...")
        pcds = []
        for rgbd_image, camera_params in tqdm(
            zip(rgbd_images, camera_trajectory.parameters)
        ):
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                camera_params.intrinsic,
                camera_params.extrinsic,
            )
            pcd = pcd.voxel_down_sample(voxel_size=0.04)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.05, max_nn=30
                )
            )
            pcd.orient_normals_towards_camera_location(
                np.array(camera_params.extrinsic)[:3, 3]
            )
            pcds.append(pcd)

        # combine point clouds naively
        print("Combining point clouds ...")
        pcd = o3d.geometry.PointCloud()
        for _pcd in tqdm(pcds):
            pcd += _pcd
        del pcds
        if args.debug:
            print("Visualizing combined point cloud [quit viewer to continue] ...")
            o3d.visualization.draw_geometries([pcd])

    # remove point cloud outliers
    pcd, _ = pcd.remove_radius_outlier(nb_points=500, radius=0.05 * 5)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=500, std_ratio=1.0)
    pcd.orient_normals_consistent_tangent_plane(50)
    if args.debug:
        print("Visualizing simplified point cloud [quit viewer to continue] ...")
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # create mesh by Poisson surface reconstruction
    print("Running Poisson surface reconstruction ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=7
    )
    if args.debug:
        print(
            "Visualizing Poisson surface reconstruction [quit viewer to continue] ..."
        )
        o3d.visualization.draw_geometries([mesh])

    # remove low density vertices from mesh
    print("Remove low density vertices ...")
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    if args.debug:
        print(
            "Visualizing mesh after removing low density vertices [quit viewer to continue] ..."
        )
        o3d.visualization.draw_geometries([mesh])

    # lightly smooth mesh
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=10, lambda_filter=0.2, mu=-0.21
    )

# otherwise, load existing mesh
else:
    print(f"Loading existing mesh from {args.texture_existing_mesh} ...")
    mesh = o3d.io.read_triangle_mesh(str(args.texture_existing_mesh))

# remove duplicated triangles and vertices
print("Removing duplicated triangles and vertices ...")
mesh = mesh.remove_duplicated_triangles()
mesh = mesh.remove_duplicated_vertices()

# remove non-manifold edges
print("Removing non-manifold edges ...")
max_repair_attempts = 3
for _ in range(max_repair_attempts):
    # check if mesh is manifold
    if mesh.is_edge_manifold() and mesh.is_vertex_manifold():
        break
    else:
        print("\tMesh is not manifold, attempting to repair ...")

    # remove clusters disconnected from the primary mesh
    cluster_idx_per_triangle, num_triangles_per_cluster, _ = map(
        np.asarray, mesh.cluster_connected_triangles()
    )
    triangles_to_remove = (
        num_triangles_per_cluster[cluster_idx_per_triangle]
        < num_triangles_per_cluster.max()
    )
    mesh.remove_triangles_by_mask(triangles_to_remove)

    # apply various mesh repair operations
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()

    # remove non-manifold edges
    mesh = mesh.remove_non_manifold_edges()

else:
    if not (mesh.is_edge_manifold() and mesh.is_vertex_manifold()):
        print("Unable to remove all non-manifold edges ...")

# subdivide mesh to increase resolution before color mapping
print("Subdividing mesh ...")
mesh = mesh.subdivide_loop(number_of_iterations=args.subdivision_iterations)
print(mesh)

# run color map pipeline
print("Running color map pipeline ...")
mesh, camera_trajectory = o3d.pipelines.color_map.run_non_rigid_optimizer(
    mesh,
    rgbd_images,
    camera_trajectory,
    o3d.pipelines.color_map.NonRigidOptimizerOption(
        maximum_iteration=50,
        maximum_allowable_depth=10.0,
    ),
)
if args.debug:
    print("Visualizing camera trajectory [quit viewer to continue] ...")
    o3d.visualization.draw_geometries([mesh])

# output mesh with vertex colors, if requested
if args.colored_mesh_output_fn:
    print(f"Saving mesh with vertex colors to {args.colored_mesh_output_fn} ...")
    o3d.io.write_triangle_mesh(
        str(args.colored_mesh_output_fn),
        mesh,
    )

# create texture map
print("Creating texture map ...")
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
mesh.compute_vertex_normals()
mesh.compute_uvatlas()
mesh.vertex["diffuse"] = mesh.vertex.colors
mesh.material.set_default_properties()
texture_tensors = mesh.bake_vertex_attr_textures(1024, {"diffuse"})
if args.debug:
    print("Visualizing texture map [quit viewer to continue] ...")
    o3d.visualization.draw([mesh])

# export texture map
print("Exporting texture map ...")
output_fn_texture = args.output_mesh_fn.with_suffix(".png")
texture_diffuse = texture_tensors["diffuse"].numpy()
texture_diffuse = (texture_diffuse * 255).astype(np.uint8)
texture_diffuse = cv2.cvtColor(texture_diffuse, cv2.COLOR_RGB2BGR)
cv2.imwrite(
    str(output_fn_texture),
    texture_diffuse,
)

# export OBJ mesh
if args.export_obj:
    print("Exporting OBJ mesh ...")
    del mesh.triangle.normals  # OBJ doesn't support triangle normals
    del mesh.vertex.colors  # only use texture map for color
    output_fn_obj = args.output_mesh_fn.with_suffix(".obj")
    o3d.t.io.write_triangle_mesh(
        str(output_fn_obj),
        mesh,
        write_triangle_uvs=True,
    )
    print(f"Exported mesh to {output_fn_obj}")

    # add reference to texture image within .mtl file
    output_fn_mtl = output_fn_obj.with_suffix(".mtl")
    with open(output_fn_mtl, "a") as f:
        f.write(f"map_Kd {output_fn_texture.name}\n")
    print(f"Appended texture map to {output_fn_mtl}")

# export mesh in PLY format
print("Exporting mesh in PLY format ...")
output_fn_ply = args.output_mesh_fn.with_suffix(".ply")
inout.save_ply2(
    output_fn_ply,
    mesh.vertex.positions.numpy(),
    pts_colors=None,
    pts_normals=mesh.vertex.normals.numpy(),
    faces=mesh.triangle.indices.numpy(),
    texture_uv=None,
    texture_uv_face=mesh.triangle.texture_uvs.numpy().reshape(-1, 6),
    texture_file=output_fn_texture.name,
    extra_header_comments=None,
)
print(f"Exported mesh to {output_fn_ply}")

# done
print("Done.")
