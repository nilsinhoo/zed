import open3d as o3d
import numpy as np
import argparse
import os
import copy

# parser = argparse.ArgumentParser()
# parser.add_argument("--filename", type=str, default="pointcloud.ply")

# args = parser.parse_args()
# filename = args.filename

# filename_source = "cropped_nils_stehend.ply"
filename_source = "cropped_mit_nils.ply"
filename_target = "merged_3.ply"

INPUT_DIR = "ply"

voxel_size = 0.05

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

def draw_clusters(target, clusters):
    target.paint_uniform_color([0.7, 0.7, 0.7])
    pcds = [clusters[0]]
    pcds.append(target)
    print(type(pcds))

    o3d.visualization.draw_geometries(pcds, window_name="Cluster")

def save_pointcloud(pcd, filename, voxel_size=voxel_size):
    pcd_out = copy.deepcopy(pcd)
    pcd_out.voxel_down_sample(voxel_size)

    o3d.io.write_point_cloud(os.path.join(INPUT_DIR, filename), pcd_out)

# 1) Laden
pcd_source = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, filename_source))
pcd_target = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, filename_target))

print("Source Pointcloud:", pcd_source)
print("Target Pointcloud:", pcd_target)

# 2) Vorverarbeitung

pcd_source_down, pcd_source_fpfh = preprocess(pcd_source, voxel_size)
pcd_target_down, pcd_target_fpfh = preprocess(pcd_target, voxel_size)

print("Source Pointcloud Downsampled:", pcd_source_down)
print("Target Pointcloud Downsampled:", pcd_target_down)

save_pointcloud(pcd_target_down, "pcd_target.ply")

# 3) Globale Registrierung
print("Globale Registrierung")

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

# 6) Lokale Registrierung
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
save_pointcloud(pcd_source_down_transformed_local, "pcd_registrated.ply")

# 6) Differenz-Punktwolke bilden

# Distanzen berechnen
dists = np.asarray(pcd_source_down_transformed_local.compute_point_cloud_distance(pcd_target_down))

# Schwellwert zur Bestimmung neuer Punkte
diff_threshold = 3.0 * voxel_size
mask_new = dists > diff_threshold
idx_new = np.where(mask_new)[0]

# Wolke erzeugen
pcd_source_diff = pcd_source_down_transformed_local.select_by_index(idx_new)

# 7) Differenz visualisieren
draw_geometries(pcd_target_down, pcd_source_diff, window_name="Differenz")
save_pointcloud(pcd_source_diff, "pcd_difference.ply")

# 8) Clustern
labels = np.array(
    pcd_source_diff.cluster_dbscan(eps=0.2, min_points=10, print_progress=True)
)

print(f"Anzahl gefundener Cluster: {labels.max() + 1}")
clusters = []

colors = [
    (0.4, 0.6, 0.8),
    (0.6, 0.7, 0.6),
    (0.8, 0.6, 0.7),
]

for i in range(labels.max() + 1):
    color_index = i % len(colors)
    indices = np.where(labels == i)[0]
    cluster_pcd = pcd_source_diff.select_by_index(indices)
    cluster_pcd.paint_uniform_color(colors[color_index])
    print(type(cluster_pcd))
    clusters.append(cluster_pcd)

print("Vorhandene Cluster:", len(clusters))

if len(clusters) > 0:
    draw_clusters(pcd_target_down, clusters)
