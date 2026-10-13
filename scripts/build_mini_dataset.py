import os, sys, json
import numpy as np
from PIL import Image
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from nuscenes.nuscenes import NuScenes
from src.nusc_utils import boxes_2d

ROOT = os.path.join(PROJ, "data", "raw", "v1.0-mini")
OUT = os.path.join(PROJ, "data", "processed", "mini")
CLS = {"car": 0, "pedestrian": 1, "truck": 2}
VAL_SCENES = {"scene-0553", "scene-1094"}      # one day, one night; frozen
nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)
for sp in ("train", "val"):
    os.makedirs(os.path.join(OUT, "images", sp), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", sp), exist_ok=True)

index, counts = [], {sp: {c: 0 for c in CLS} for sp in ("train", "val")}
for s in nusc.sample:
    scene = nusc.get("scene", s["scene_token"])["name"]
    sp = "val" if scene in VAL_SCENES else "train"
    tok = s["token"]
    src = os.path.join(ROOT, nusc.get("sample_data", s["data"]["CAM_FRONT"])["filename"])
    Image.open(src).convert("RGB").resize((640, 360), Image.BILINEAR).save(
        os.path.join(OUT, "images", sp, tok + ".jpg"), quality=95)
    lines = []
    for cls, x1, y1, x2, y2, _ in boxes_2d(nusc, s):
        counts[sp][cls] += 1
        lines.append("%d %.6f %.6f %.6f %.6f" % (CLS[cls], (x1 + x2) / 2 / 1600, (y1 + y2) / 2 / 900,
                                                 (x2 - x1) / 1600, (y2 - y1) / 900))
    with open(os.path.join(OUT, "labels", sp, tok + ".txt"), "w") as f:
        f.write("\n".join(lines))
    index.append({"token": tok, "split": sp, "scene": scene})

json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
for sp in counts:
    print(sp, "images:", sum(i["split"] == sp for i in index), "boxes:", counts[sp])