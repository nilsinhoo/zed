import open3d as o3d
import numpy as np
import os

INPUT_DIR = "ply"
OUTPUT_DIR = "ply"

filename_input = "cropped_mit_nils.ply"
filename_output = "cropped_mit_nils_4096.ply"

pcd = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, filename_input))

points = np.asarray(pcd.points)
# colors = np.asarray(pcd.colors)

# Zentrieren
centroid = points.mean(axis=0)
points_centered = points - centroid

# Skalieren
scale = np.max(np.linalg.norm(points_centered, axis=1))
points_norm = points_centered / scale

# Sampling
num_points = 4096
N = points_norm.shape[0]

if N >= num_points:
    # ohne Zurücklegen
    idx = np.random.choice(N, num_points, replace=True)
else:
    # mit Zurücklegen
    idx = np.random.choice(N, num_points, replace=True)

pcd_sampled = pcd.select_by_index(idx)

o3d.visualization.draw_geometries([pcd_sampled])

o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, filename_output))
print("Gespeichert unter:", filename_output)