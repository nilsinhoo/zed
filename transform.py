import open3d as o3d
import numpy as np
import os

INPUT_DIR = "ply"

filename_source = "cropped_mit_nils.ply"
filename_target = "cropped_ohne_nils.ply"

# 1) Laden
pcd_source = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, filename_source))
pcd_target = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, filename_target))

pcd_source.translate((0.3, 0.0, 0.0))

pcd_source.paint_uniform_color([1, 0.7, 0])
pcd_target.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries([pcd_source, pcd_target])
