import open3d as o3d
import numpy as np
import pyzed.sl as sl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--reference", action="store_true")
args = parser.parse_args()


zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
init_params.coordinate_units = sl.UNIT.METER
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print(f"ZED open failed: {status}")
    exit(1)


def grab_pointcloud(zed):
    runtime_params = sl.RuntimeParameters()

    point_cloud = sl.Mat()

    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        return None

    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

    pc = point_cloud.get_data()

    points = pc[:, :, :3].reshape(-1, 3)

    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    return points


points = grab_pointcloud(zed)

pcd = o3d.geometry.PointCloud() # PointCloud Objekt erzeugen
pcd.points = o3d.utility.Vector3dVector(points)

pcd = pcd.voxel_down_sample(voxel_size=0.03)
# pcd.paint_uniform_color([0.7,0.7,0.7])

if args.reference:
    print("saving to reference")
    o3d.io.write_point_cloud("reference.ply", pcd)
else:
    print("saving to current")
    o3d.io.write_point_cloud("current.ply", pcd)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Punktwolke"
)