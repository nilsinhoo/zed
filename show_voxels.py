import open3d as o3d
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="pointcloud.ply")

args = parser.parse_args()

INPUT_DIR = "ply"

# 1) Laden
pcd = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, args.filename))
print(pcd)

# 2) VoxelGrid erstellen
voxel_size = 0.05
vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

num_voxels = len(vg.get_voxels())
print(f"VoxelGrid with {num_voxels} voxels.")

o3d.visualization.draw_geometries(
    [vg],
    window_name="Punktwolke"
)