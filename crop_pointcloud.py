import open3d as o3d
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--source", type=str, default="pointcloud.ply")
parser.add_argument("--target", type=str, default="cropped.ply")

args = parser.parse_args()

INPUT_DIR = "ply"
OUTPUT_DIR = "ply"

# 1) Laden
pcd = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, args.source))
print(pcd)

# 2) Croppen
half = 1.5
bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=(-half, 0, -half),
    max_bound=(half, 2*half, half)
)

pcd_cropped = pcd.crop(bbox)
print(pcd_cropped)


path = os.path.join(OUTPUT_DIR, args.target)
print(f"saving to {path}")
o3d.io.write_point_cloud(path, pcd_cropped)