import os
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

nusc = NuScenes(version="v1.0-trainval", dataroot=os.path.join(PROJ, "data", "raw", "nuscenes_trainval"), verbose=False)
sp = create_splits_scenes()
print("chunk | split | clear-day | rain-day | night")
for c in (9, 10):
    sc = nusc.scene[(c - 1) * 85: c * 85]
    for name in ("train", "val"):
        g = [s for s in sc if s["name"] in set(sp[name])]
        night = sum("night" in s["description"].lower() for s in g)
        rain = sum(("rain" in s["description"].lower()) and ("night" not in s["description"].lower()) for s in g)
        print("%5d | %5s | %9d | %8d | %5d" % (c, name, len(g) - night - rain, rain, night))