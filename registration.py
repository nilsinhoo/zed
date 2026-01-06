import open3d as o3d
import numpy as np
import argparse
import os

INPUT_DIR = "ply"

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="current.ply")
parser.add_argument("--target", type=str, default="reference.ply")
args = parser.parse_args()

print("Open3D:", o3d.__version__)

def preprocess(pcd, voxel):
    # 1. Downsampling
    pcd_down = pcd.voxel_down_sample(voxel)

    # 2. Normalen auf der *downsampled* Cloud berechnen
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=2.0 * voxel,
            max_nn=30
        )
    )

    # Optional, aber oft sinnvoll:
    pcd_down.orient_normals_consistent_tangent_plane(100)

    # 3. FPFH Features berechnen
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=5.0 * voxel,
            max_nn=100
        )
    )

    return pcd_down, fpfh


print("Punktwolken laden...")

# --- Load (klassische CPU-API) ---
source = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, args.source))
target = o3d.io.read_point_cloud(os.path.join(INPUT_DIR, args.target))

print(source)
print(target)

# Optional: grobe Ausreißer entfernen (oft hilfreich)
# source, _ = source.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
# target, _ = target.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

print("Punktwolken vorverarbeiten...")

voxel = 0.05  # <-- anpassen! (Einheiten deiner Daten: z.B. 1cm = 0.01 bei Metern)
src_down, src_fpfh = preprocess(source, voxel)
tgt_down, tgt_fpfh = preprocess(target, voxel)

print("Starte globale Registrierung...")

# --- Global registration (RANSAC) ---
dist_thresh = 1.5 * voxel
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    src_down, tgt_down, src_fpfh, tgt_fpfh,
    mutual_filter=True,
    max_correspondence_distance=dist_thresh,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
)

print("Verfeinere Registrierung mit ICP...")

# --- Refine with ICP (point-to-plane) ---
source.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30)
)
target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30)
)

icp_thresh = 0.5 * voxel
result_icp = o3d.pipelines.registration.registration_icp(
    source, target,
    max_correspondence_distance=icp_thresh,
    init=result_ransac.transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

T = result_icp.transformation.copy()

print("Finale Transformation:\n", T)
print("Fitness:", result_icp.fitness, "RMSE:", result_icp.inlier_rmse)

# Quelle in Ziel-KS transformieren
source_aligned = source.transform(T)

# Punkte vergleichen
diff_thresh = 2.0 * voxel  # anpassen (je nach Messrauschen)

dists = np.asarray(source_aligned.compute_point_cloud_distance(target))
mask_diff = dists > diff_thresh

source_diff = source_aligned.select_by_index(np.where(mask_diff)[0])
source_same = source_aligned.select_by_index(np.where(~mask_diff)[0])

dists_t = np.asarray(target.compute_point_cloud_distance(source_aligned))
mask_diff_t = dists_t > diff_thresh
target_diff = target.select_by_index(np.where(mask_diff_t)[0])

# Einfärben
source_same.paint_uniform_color([0.6, 0.6, 0.6])
source_diff.paint_uniform_color([1.0, 0.0, 0.0])

# Achtung: paint_uniform_color arbeitet in-place und gibt None zurück,
# also vorher eine Kopie machen, wenn du target noch brauchst.
target_vis = target.clone() if hasattr(target, "clone") else target  # für ältere Open3D-Versionen ggf. copy
target_vis.paint_uniform_color([0.3, 0.8, 0.3])

# Nur die Unterschiede anzeigen:
# o3d.visualization.draw_geometries([target_vis, source_same, source_diff])
o3d.visualization.draw_geometries([source_diff])
# oder symmetrisch:
# o3d.visualization.draw_geometries([source_diff, target_diff])
