import open3d as o3d
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["all", "diff", "ref", "cur"])
args = parser.parse_args()

voxel_size = 0.03  # 3 cm (typisch 0.02–0.05)

# 1) Laden
pcd_ref = o3d.io.read_point_cloud("reference.ply")
pcd_cur = o3d.io.read_point_cloud("current.ply")

# 2) Optional, aber empfehlenswert: gleiche Downsample-Strategie
pcd_ref = pcd_ref.voxel_down_sample(voxel_size)
pcd_cur = pcd_cur.voxel_down_sample(voxel_size)

# 3) Referenz in belegte Voxel umwandeln (Set von Voxel-Indizes)
vg_ref = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_ref, voxel_size)
ref_voxels = {tuple(v.grid_index) for v in vg_ref.get_voxels()}

# 4) Aktuelle Punkte: nur behalten, deren Voxel NICHT in der Referenz belegt ist
cur_pts = np.asarray(pcd_cur.points)
cur_idx = np.floor(cur_pts / voxel_size).astype(np.int32)
mask_dynamic = np.array([tuple(i) not in ref_voxels for i in cur_idx], dtype=bool)

diff_pts = cur_pts[mask_dynamic]

pcd_diff = o3d.geometry.PointCloud()
pcd_diff.points = o3d.utility.Vector3dVector(diff_pts)

# pcd_diff = pcd_diff.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)[0]


# 5) Visualisierung
pcd_ref.paint_uniform_color([0.7, 0.7, 0.7])   # grau
pcd_cur.paint_uniform_color([0.0, 0.5, 1.0])   # blau
pcd_diff.paint_uniform_color([1.0, 0.0, 0.0])  # rot

if args.mode == "all":
    o3d.visualization.draw_geometries(
        [pcd_ref, pcd_cur, pcd_diff],
        window_name="Reference (gray) | Current (blue) | Difference (red)"
    )
elif args.mode == "diff":
    o3d.visualization.draw_geometries(
        [pcd_diff],
        window_name="Difference (red)"
    )
elif args.mode == "cur":
    o3d.visualization.draw_geometries(
        [pcd_cur],
        window_name="Current (blue)"
    )
elif args.mode == "ref":
    o3d.visualization.draw_geometries(
        [pcd_ref],
        window_name="Reference (gray)"
    )