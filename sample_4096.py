import open3d as o3d
import numpy as np
import os

INPUT_DIR = "ply"
OUTPUT_DIR = "ply"

filename_input = "mit_nils.ply"

pcd = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, filename_input))

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Zentrieren
centroid = points.mean(axis=0)
points_centered = points - centroid

# Skalieren
scale = np.max(np.linalg.norm(points_centered, axis=1))
points_norm = points_centered / scale

