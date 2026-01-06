import open3d as o3d
import numpy as np
import argparse
import os
import copy

# parser = argparse.ArgumentParser()
# parser.add_argument("--filename", type=str, default="pointcloud.ply")

# args = parser.parse_args()
# filename = args.filename

# filename_source = "cropped_mit_nils.ply"
filename_source = "cropped_ohne_nils_4.ply"
filename_target = "merged_2.ply"
filename_output = "merged_3.ply"

INPUT_DIR = "ply"
OUTPUT_DIR = "ply"

def preprocess(pcd, voxel_size):
    # Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Normalen schätzen
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    
    # FPFH Features berechnen
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def draw_registration_result(source, target, transformation):
    pcd_source_temp = copy.deepcopy(source)
    pcd_target_temp = copy.deepcopy(target)

    pcd_source_temp.paint_uniform_color([1, 0.7, 0])
    pcd_target_temp.paint_uniform_color([0, 0.65, 0.9])

    pcd_source_temp.transform(transformation)

    o3d.visualization.draw_geometries([pcd_source_temp, pcd_target_temp])

def draw_geometries(target, source, window_name="Visualisierung"):
    target.paint_uniform_color([1, 0.7, 0])
    source.paint_uniform_color([0, 0.65, 0.9])

    o3d.visualization.draw_geometries([target, source], window_name=window_name)

# Laden
pcd_source = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, filename_source))
pcd_target = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, filename_target))

print("Source Pointcloud:", pcd_source)
print("Target Pointcloud:", pcd_target)

# Vorverarbeitung
voxel_size = 0.05
pcd_source_down, pcd_source_fpfh = preprocess(pcd_source, voxel_size)
pcd_target_down, pcd_target_fpfh = preprocess(pcd_target, voxel_size)

print("Source Pointcloud Downsampled:", pcd_source_down)
print("Target Pointcloud Downsampled:", pcd_target_down)

# Globale Registrierung
distance_treshold = voxel_size * 1.5
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    pcd_source_down, pcd_target_down, pcd_source_fpfh, pcd_target_fpfh, True, distance_treshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    3, [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_treshold),
    ],
    o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
)

print(result_ransac)

# Transformation anwenden
pcd_source_down_transformed_global = copy.deepcopy(pcd_source_down)
pcd_source_down_transformed_global.transform(result_ransac.transformation)

# Visualisieren
draw_geometries(pcd_source_down_transformed_global, pcd_target_down, window_name="Globale Registrierung")

# Lokale Registrierung
print("Lokale Registrierung")

pcd_source_down_transformed_global.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel_size, max_nn=30)
)

icp_threshold = 0.5 * voxel_size
result_icp = o3d.pipelines.registration.registration_icp(
    pcd_source_down_transformed_global, pcd_target_down,
    max_correspondence_distance=icp_threshold,
    init=result_ransac.transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
)

print(result_icp)

# Transformation anwenden
pcd_source_down_transformed_local = copy.deepcopy(pcd_source_down)
pcd_source_down_transformed_local.transform(result_icp.transformation)

# Visualisieren
draw_geometries(pcd_source_down_transformed_local, pcd_target_down, window_name="Lokale Registrierung")

# Mergen
pcd_merged = pcd_source_down_transformed_local + pcd_target_down

# Downsamplen
pcd_merged_down = pcd_merged.voxel_down_sample(voxel_size)
pcd_merged_down.paint_uniform_color([0, 0.65, 0.9])

# Speichern
output_path = os.path.join(INPUT_DIR, filename_output)
o3d.io.write_point_cloud(output_path, pcd_merged_down)
print("Gespeichert unter:", output_path)

# Visualisieren
o3d.visualization.draw_geometries([pcd_merged_down], window_name="Lokale Registrierung")
