import os, sys, json
import numpy as np
from PIL import Image
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from src.nusc_utils import project_radar, boxes_2d

ROOT = os.path.join(PROJ, "data", "raw", "nuscenes_trainval")
OUT = os.path.join(PROJ, "data", "processed", "subset")
CLS = {"car": 0, "pedestrian": 1, "truck": 2}
UP_M, DOWN_M = 2.0, 0.5                      # vertical extent assumption (m above / below radar point)
sub = json.load(open(os.path.join(PROJ, "docs", "subset_scenes.json")))
nusc = NuScenes(version="v1.0-trainval", dataroot=ROOT, verbose=False)
by = {s["name"]: s for s in nusc.scene}
for sp in sub:
    os.makedirs(os.path.join(OUT, "images", sp), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", sp), exist_ok=True)

index, cache, n = [], {}, 0
box_counts = {sp: {c: 0 for c in CLS} for sp in sub}
cond_counts = {}
for sp, items in sub.items():
    for it in items:
        tok = by[it["scene"]]["first_sample_token"]
        while tok:
            s = nusc.get("sample", tok)
            # image (resized to 640x360)
            src = os.path.join(ROOT, nusc.get("sample_data", s["data"]["CAM_FRONT"])["filename"])
            Image.open(src).convert("RGB").resize((640, 360), Image.BILINEAR).save(
                os.path.join(OUT, "images", sp, tok + ".jpg"), quality=95)
            # labels (YOLO format, 3 classes)
            lines = []
            for cls, x1, y1, x2, y2, _ in boxes_2d(nusc, s):
                box_counts[sp][cls] += 1
                lines.append("%d %.6f %.6f %.6f %.6f" % (CLS[cls], (x1 + x2) / 2 / 1600, (y1 + y2) / 2 / 900,
                                                         (x2 - x1) / 1600, (y2 - y1) / 900))
            with open(os.path.join(OUT, "labels", sp, tok + ".txt"), "w") as f:
                f.write("\n".join(lines))
            # radar points: [u, v_top, v_bot, depth, v_radial, M, pdh0, rcs]
            pts, uv = project_radar(nusc, ROOT, s)
            rsd = nusc.get("sample_data", s["data"]["RADAR_FRONT"])
            raw = RadarPointCloud.from_file(os.path.join(ROOT, rsd["filename"])).points
            csd = nusc.get("sample_data", s["data"]["CAM_FRONT"])
            fy = np.array(nusc.get("calibrated_sensor", csd["calibrated_sensor_token"])["camera_intrinsic"])[1, 1]
            z = pts[2]
            ok = (z > 0.1) & (uv[0] >= 0) & (uv[0] < 1600)
            r = np.maximum(np.hypot(raw[0], raw[1]), 1e-3)
            vr = (raw[0] * raw[8] + raw[1] * raw[9]) / r
            m = ((raw[14] == 0) & (raw[11] == 3)).astype(np.float32)
            P = np.stack([uv[0], uv[1] - fy * UP_M / z, uv[1] + fy * DOWN_M / z, z, vr, m, raw[15], raw[5]], axis=1)[ok]
            cache[tok] = P.astype(np.float32)
            index.append({"token": tok, "split": sp, "scene": it["scene"], "condition": it["condition"]})
            cond_counts[(sp, it["condition"])] = cond_counts.get((sp, it["condition"]), 0) + 1
            n += 1
            if n % 200 == 0:
                print("processed", n, "samples")
            tok = s["next"]

json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
np.savez_compressed(os.path.join(OUT, "radar_points.npz"), **cache)
print("DONE: %d samples, mean radar points/sample %.1f" % (n, np.mean([len(v) for v in cache.values()])))
for sp in sub:
    print(sp, "images:", sum(i["split"] == sp for i in index), "boxes:", box_counts[sp],
          "| images by condition:", {c: cond_counts.get((sp, c), 0) for c in ("clear", "rain", "night")})