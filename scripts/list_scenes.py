import os, sys, csv
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version="v1.0-trainval", dataroot=os.path.join(PROJ, "data", "raw", "nuscenes_trainval"), verbose=False)
rows = []
for i, s in enumerate(nusc.scene):
    d = s["description"].lower()
    rows.append([s["name"], i // 85 + 1, int("night" in d), int("rain" in d), s["nbr_samples"], s["description"]])
os.makedirs(os.path.join(PROJ, "docs"), exist_ok=True)
with open(os.path.join(PROJ, "docs", "trainval_scenes.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["scene", "chunk", "night", "rain", "samples", "description"]); w.writerows(rows)
print("total scenes:", len(rows))
print("chunk | scenes | night | rain")
for c in range(1, 11):
    r = [x for x in rows if x[1] == c]
    print("%5d | %6d | %5d | %4d" % (c, len(r), sum(x[2] for x in r), sum(x[3] for x in r)))