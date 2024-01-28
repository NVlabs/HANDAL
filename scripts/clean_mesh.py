# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def open3d_filter_triangle_clusters(
    mesh: o3d.geometry.TriangleMesh,
    small_component_threshold: float = 0.20,
):
    n_triangles, _ = np.asarray(mesh.triangles).shape
    (
        triangle_clusters,
        cluster_n_triangles,
        cluster_area,
    ) = map(np.asarray, mesh.cluster_connected_triangles())
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < (
        small_component_threshold * n_triangles
    )
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()


def open3d_cleanup(
    mesh_fn: Path,
    output_fn: Path,
    small_component_threshold: float = 0.20,
) -> None:
    # read mesh
    mesh = o3d.io.read_triangle_mesh(mesh_fn.as_posix())

    # remove small connected components
    open3d_filter_triangle_clusters(
        mesh, small_component_threshold=small_component_threshold
    )

    # convert to point cloud
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.5)
    pcd.estimate_normals()

    # re-mesh
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
    open3d_filter_triangle_clusters(
        mesh, small_component_threshold=small_component_threshold
    )

    # save mesh
    o3d.io.write_triangle_mesh(output_fn.as_posix(), mesh)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_fn",
        type=Path,
        help="mesh file to clean",
    )
    parser.add_argument(
        "output_fn",
        type=Path,
        help="Output mesh file (default='cleaned.ply')",
        default=None,
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=2,
        help="Number of cleaning iterations",
    )
    args = parser.parse_args()

    # set default output filename
    if args.output_fn is None:
        args.output_fn = args.input_fn.parent / "cleaned.ply"

    # apply Open3D mesh cleanup
    print("Applying Open3D mesh cleanup...")
    open3d_cleanup(mesh_fn=args.input_fn, output_fn=args.output_fn)
    print(f"Saved cleaned mesh to '{args.output_fn}'")
