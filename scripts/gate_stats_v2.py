import os, sys
import numpy as np
import torch
import torch.nn.functional as F
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, "scripts"))
import train as T
from torch.utils.data import DataLoader
from src.dataset import FusionDataset, collate
from src.corruptions import clutter, misalign

ROOT = os.path.join(PROJ, "data", "processed", "subset")
dev = torch.device("cuda")
CONDS = {"clean": lambda P: P,
         "clutter 30 ghosts": lambda P: clutter(P, 30),
         "clutter 60 ghosts": lambda P: clutter(P, 60),
         "misalign 60 px": lambda P: misalign(P, 60, mark=True)}
names = sys.argv[1:] or ["b3_v2"]
for name in names:
    ck = torch.load(os.path.join(PROJ, "outputs", "checkpoints", name + "_best.pt"), map_location="cpu", weights_only=False)
    net = T.Net(ck["mode"], ck["channels"]); net.load_state_dict(ck["model"]); net = net.to(dev).eval()
    print("\n== %s (checkpoint from epoch %d) ==" % (name, ck["epoch"]))
    print("%-18s | stride | gate on GOOD radar | gate on BAD radar | separation | cells good / bad" % "condition")
    for c, fn in CONDS.items():
        ds = FusionDataset(ROOT, "val", corrupt=fn)
        acc = {(j, k): [0.0, 0] for j in range(3) for k in ("good", "bad")}
        for imgs, radars, labs, toks in DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate, num_workers=0):
            imgs, radars = T.pad_batch(imgs, radars)
            r = radars.to(dev)
            with torch.no_grad():
                net(imgs.to(dev), r)
            for j, s in enumerate((8, 16, 32)):
                g = net.gates[j][:, 0].float()
                occ = F.avg_pool2d((r[:, 0:1] > 0).float(), s, s)[:, 0]
                frac = F.avg_pool2d(r[:, 3:4], s, s)[:, 0] / occ.clamp(min=1e-6)
                for k, m in (("good", (occ > 0) & (frac <= 0.5)), ("bad", (occ > 0) & (frac > 0.5))):
                    acc[(j, k)][0] += float(g[m].sum()); acc[(j, k)][1] += int(m.sum())
        for j, s in enumerate((8, 16, 32)):
            gm = acc[(j, "good")][0] / max(acc[(j, "good")][1], 1)
            bm = acc[(j, "bad")][0] / acc[(j, "bad")][1] if acc[(j, "bad")][1] else float("nan")
            print("%-18s | %6d | %18.3f | %17.3f | %10.3f | %d / %d" % (c, s, gm, bm, gm - bm, acc[(j, "good")][1], acc[(j, "bad")][1]))