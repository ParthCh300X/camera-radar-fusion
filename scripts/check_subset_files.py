import os, json
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from nuscenes.nuscenes import NuScenes

ROOT = os.path.join(PROJ, "data", "raw", "nuscenes_trainval")
nusc = NuScenes(version="v1.0-trainval", dataroot=ROOT, verbose=False)
sub = json.load(open(os.path.join(PROJ, "docs", "subset_scenes.json")))
by = {s["name"]: s for s in nusc.scene}
stats = {}
for items in sub.values():
    for it in items:
        row = stats.setdefault(it["chunk"], [0, 0, 0, 0]); row[0] += 1
        tok = by[it["scene"]]["first_sample_token"]
        while tok:
            smp = nusc.get("sample", tok); row[1] += 1
            for key, j in (("CAM_FRONT", 2), ("RADAR_FRONT", 3)):
                if not os.path.exists(os.path.join(ROOT, nusc.get("sample_data", smp["data"][key])["filename"])):
                    row[j] += 1
            tok = smp["next"]
print("chunk | scenes | samples | missing CAM_FRONT | missing RADAR_FRONT")
for c in sorted(stats):
    print("%5d | %6d | %7d | %17d | %19d" % (c, *stats[c]))