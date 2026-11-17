import os, json
from collections import Counter
PROJ = r"D:\camera-radar-fusion"
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version="v1.0-trainval", dataroot=os.path.join(PROJ, "data", "raw", "nuscenes_trainval"), verbose=False)
sub = json.load(open(os.path.join(PROJ, "docs", "subset_scenes.json")))
by = {s["name"]: s for s in nusc.scene}
logs = {}
for sp, items in sub.items():
    logs[sp] = {}
    for it in items:
        lg = nusc.get("log", by[it["scene"]]["log_token"])
        logs[sp][it["scene"]] = (lg["location"], lg["logfile"])
for sp in logs:
    print("%-5s scenes %2d | distinct logs %2d | scenes per location: %s" % (
        sp, len(logs[sp]), len({v[1] for v in logs[sp].values()}), dict(Counter(v[0] for v in logs[sp].values()))))
tr, va, te = ({v[1] for v in logs[s].values()} for s in ("train", "val", "test"))
print("val logs also in train:  %d of %d" % (len(va & tr), len(va)))
print("test logs also in train: %d of %d" % (len(te & tr), len(te)))
print("test logs also in val:   %d of %d" % (len(te & va), len(te)))