import os, sys
import numpy as np
import torch
import torch.nn.functional as F
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, "scripts"))
import train as T
from torch.utils.data import DataLoader
from src.dataset import FusionDataset, collate
from src.corruptions import make_corruption

ROOT = os.path.join(PROJ, "data", "processed", "subset")
dev = torch.device("cuda")
ck = torch.load(os.path.join(PROJ, "outputs", "checkpoints", "b3_best.pt"), map_location="cpu", weights_only=False)
net = T.Net(ck["mode"], ck["channels"]); net.load_state_dict(ck["model"]); net = net.to(dev).eval()
print("b3 checkpoint from epoch", ck["epoch"])
STR = [8, 16, 32]
print("condition          | stride | gate: empty cells | radar valid (M=1) | radar flagged (M=0)")
for c in ["clean", "noise:high", "misalign:large", "sparsity:10", "dropout:severe"]:
    kind, level = (c.split(":") + [None])[:2]
    ds = FusionDataset(ROOT, "val", corrupt=make_corruption(kind, level))
    acc = {(j, k): [0.0, 0] for j in range(3) for k in ("empty", "valid", "flag")}
    for imgs, radars, labs, toks in DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate, num_workers=0):
        imgs, radars = T.pad_batch(imgs, radars)
        r = radars.to(dev)
        with torch.no_grad():
            net(imgs.to(dev), r)
        for j, s in enumerate(STR):
            g = net.gates[j][:, 0]
            occ = F.avg_pool2d((r[:, 0:1] > 0).float(), s, s)[:, 0]
            val = F.avg_pool2d(((r[:, 0:1] > 0) & (r[:, 2:3] > 0.5)).float(), s, s)[:, 0]
            for k, m in (("empty", occ == 0), ("valid", (occ > 0) & (val / occ.clamp(min=1e-6) > 0.5)),
                         ("flag", (occ > 0) & (val / occ.clamp(min=1e-6) <= 0.5))):
                acc[(j, k)][0] += float(g[m].sum()); acc[(j, k)][1] += int(m.sum())
    for j, s in enumerate(STR):
        v = [acc[(j, k)][0] / acc[(j, k)][1] if acc[(j, k)][1] else float("nan") for k in ("empty", "valid", "flag")]
        print("%-18s | %6d | %17.3f | %17.3f | %19.3f" % (c, s, *v))