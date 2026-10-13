import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points

ROOT = r"D:\camera-radar-fusion\data\raw\v1.0-mini"
OUT = r"D:\camera-radar-fusion\outputs\figures"
nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)
RadarPointCloud.disable_filters()

def project_radar(sample):
    rsd = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
    csd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    pc = RadarPointCloud.from_file(os.path.join(ROOT, rsd["filename"]))
    cs = nusc.get("calibrated_sensor", rsd["calibrated_sensor_token"])      # radar -> ego
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix); pc.translate(np.array(cs["translation"]))
    pose = nusc.get("ego_pose", rsd["ego_pose_token"])                       # ego -> global
    pc.rotate(Quaternion(pose["rotation"]).rotation_matrix); pc.translate(np.array(pose["translation"]))
    pose = nusc.get("ego_pose", csd["ego_pose_token"])                       # global -> ego (camera time)
    pc.translate(-np.array(pose["translation"])); pc.rotate(Quaternion(pose["rotation"]).rotation_matrix.T)
    cs = nusc.get("calibrated_sensor", csd["calibrated_sensor_token"])      # ego -> camera
    pc.translate(-np.array(cs["translation"])); pc.rotate(Quaternion(cs["rotation"]).rotation_matrix.T)
    uv = view_points(pc.points[:3], np.array(cs["camera_intrinsic"]), normalize=True)
    return pc.points, uv, csd["filename"]

def nth_sample(scene_name, n):
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    tok = scene["first_sample_token"]
    for _ in range(n):
        tok = nusc.get("sample", tok)["next"]
    return nusc.get("sample", tok)

os.makedirs(OUT, exist_ok=True)
for name in ["scene-0061", "scene-1077"]:
    sample = nth_sample(name, 10)
    pts, uv, cam_file = project_radar(sample)
    img = Image.open(os.path.join(ROOT, cam_file)); W, H = img.size
    inside = (pts[2] >= 0) & (uv[2] > 0.1) & (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H)
    pts = np.asarray(pts)
    valid = (pts[14] == 0) & (pts[11] == 3)   # invalid_state==0 and ambig_state==3
    print(name, ": %d points total, %d in image, %d in image and valid" % (pts.shape[1], inside.sum(), (inside & valid).sum()))
    plt.figure(figsize=(16, 9)); plt.imshow(img)
    plt.scatter(uv[0][inside & valid], uv[1][inside & valid], c="lime", s=45, label="valid (M=1)")
    plt.scatter(uv[0][inside & ~valid], uv[1][inside & ~valid], c="red", s=45, label="flagged (M=0)")
    plt.legend(); plt.axis("off"); plt.title(name)
    plt.savefig(os.path.join(OUT, "overlay_%s.png" % name), bbox_inches="tight", dpi=80); plt.close()