import os, json, random
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

nusc = NuScenes(version="v1.0-trainval", dataroot=os.path.join(PROJ, "data", "raw", "nuscenes_trainval"), verbose=False)
sp = create_splits_scenes()
official_train = set(sp["train"])
random.seed(0)
idx = {s["name"]: i for i, s in enumerate(nusc.scene)}
by = {s["name"]: s for s in nusc.scene}

def cond(s):
    d = s["description"].lower()
    return "night" if "night" in d else ("rain" if "rain" in d else "clear")

pool = {}
for c in (9, 10):
    for s in nusc.scene[(c - 1) * 85: c * 85]:
        pool.setdefault(("train" if s["name"] in official_train else "val", cond(s)), []).append(s["name"])

def take(key, n):
    names = sorted(pool[key]); random.shuffle(names); return names[:n]

subset = {"train": [], "val": [], "test": []}
for c, n_tr, n_va in (("clear", 16, 4), ("rain", 8, 2), ("night", 16, 4)):
    names = take(("train", c), n_tr + n_va)
    subset["train"] += names[:n_tr]; subset["val"] += names[n_tr:]
for c, n in (("clear", 8), ("rain", 6), ("night", 8)):
    subset["test"] += take(("val", c), n)

out = {}
for name, scenes in subset.items():
    out[name] = [{"scene": n, "condition": cond(by[n]), "chunk": idx[n] // 85 + 1, "samples": by[n]["nbr_samples"]} for n in sorted(scenes)]
    conds = {c: sum(x["condition"] == c for x in out[name]) for c in ("clear", "rain", "night")}
    print("%-5s scenes %2d | samples %4d | %s" % (name, len(out[name]), sum(x["samples"] for x in out[name]), conds))
json.dump(out, open(os.path.join(PROJ, "docs", "subset_scenes.json"), "w"), indent=1)