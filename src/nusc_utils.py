import os
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points

CLASS_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
}

def project_radar(nusc, root, sample):
    """All RADAR_FRONT points (unfiltered) -> CAM_FRONT. Returns (points[18,N] in camera frame, uv[3,N])."""
    RadarPointCloud.disable_filters()
    rsd = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
    csd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    pc = RadarPointCloud.from_file(os.path.join(root, rsd["filename"]))
    cs = nusc.get("calibrated_sensor", rsd["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix); pc.translate(np.array(cs["translation"]))
    pose = nusc.get("ego_pose", rsd["ego_pose_token"])
    pc.rotate(Quaternion(pose["rotation"]).rotation_matrix); pc.translate(np.array(pose["translation"]))
    pose = nusc.get("ego_pose", csd["ego_pose_token"])
    pc.translate(-np.array(pose["translation"])); pc.rotate(Quaternion(pose["rotation"]).rotation_matrix.T)
    cs = nusc.get("calibrated_sensor", csd["calibrated_sensor_token"])
    pc.translate(-np.array(cs["translation"])); pc.rotate(Quaternion(cs["rotation"]).rotation_matrix.T)
    uv = view_points(pc.points[:3], np.array(cs["camera_intrinsic"]), normalize=True)
    return pc.points, uv

def boxes_2d(nusc, sample, W=1600, H=900, min_vis=2, min_w=12, min_h=25):
    """GT 3D boxes -> 2D boxes (cls, x1, y1, x2, y2) in CAM_FRONT pixels, 3 classes."""
    _, boxes, K = nusc.get_sample_data(sample["data"]["CAM_FRONT"])
    out = []
    for b in boxes:
        cls = CLASS_MAP.get(b.name)
        if cls is None:
            continue
        if int(nusc.get("sample_annotation", b.token)["visibility_token"]) < min_vis:
            continue
        corners = b.corners()
        if corners[2].min() <= 0.1:          # any corner behind camera: skip for now
            continue
        uv = view_points(corners, np.array(K), normalize=True)
        x1, y1 = np.clip(uv[0].min(), 0, W - 1), np.clip(uv[1].min(), 0, H - 1)
        x2, y2 = np.clip(uv[0].max(), 0, W - 1), np.clip(uv[1].max(), 0, H - 1)
        if x2 - x1 < min_w or y2 - y1 < min_h:
            continue
        out.append((cls, x1, y1, x2, y2, float(b.center[2])))
    return out