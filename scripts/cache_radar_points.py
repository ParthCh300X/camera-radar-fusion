import os, sys
import numpy as np
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from src.nusc_utils import project_radar

ROOT = os.path.join(PROJ, "data", "raw", "v1.0-mini")
OUT = os.path.join(PROJ, "data", "processed", "mini")
os.makedirs(OUT, exist_ok=True)
nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)
UP_M, DOWN_M = 2.0, 0.5     # vertical extent assumption (metres above / below the radar point)

cache, kept = {}, []
for s in nusc.sample:
    pts, uv = project_radar(nusc, ROOT, s)
    rsd = nusc.get("sample_data", s["data"]["RADAR_FRONT"])
    raw = RadarPointCloud.from_file(os.path.join(ROOT, rsd["filename"])).points   # same column order
    csd = nusc.get("sample_data", s["data"]["CAM_FRONT"])
    f = np.array(nusc.get("calibrated_sensor", csd["calibrated_sensor_token"])["camera_intrinsic"])[1, 1]
    z = pts[2]
    ok = (z > 0.1) & (uv[0] >= 0) & (uv[0] < 1600)
    r = np.maximum(np.hypot(raw[0], raw[1]), 1e-3)
    vr = (raw[0] * raw[8] + raw[1] * raw[9]) / r            # radial speed from compensated velocity
    m = ((raw[14] == 0) & (raw[11] == 3)).astype(np.float32)
    P = np.stack([uv[0], uv[1] - f * UP_M / z, uv[1] + f * DOWN_M / z, z, vr, m, raw[15], raw[5]], axis=1)[ok]
    cache[s["token"]] = P.astype(np.float32)
    kept.append(len(P))

np.savez_compressed(os.path.join(OUT, "radar_points.npz"), **cache)
print("cached %d samples; points/sample mean %.1f, min %d, max %d" % (len(cache), np.mean(kept), min(kept), max(kept)))