import os, sys
import numpy as np
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from nuscenes.nuscenes import NuScenes
from src.nusc_utils import project_radar, boxes_2d

ROOT = os.path.join(PROJ, "data", "raw", "v1.0-mini")
nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)
TOL, SHIFT_PX, SHIFT_M = 3.0, 150, 8.0

def hit(boxes, u, v, z):
    h = np.zeros(len(u), bool)
    for _, x1, y1, x2, y2, zc in boxes:
        h |= (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2) & (np.abs(z - zc) <= TOL)
    return h

NULLS = {"real": (0, 0.0), "shift_px": (SHIFT_PX, 0.0), "shift_depth": (0, SHIFT_M)}
acc = {(g, t): [0, 0] for g in ["valid", "flagged"] for t in NULLS}
for s in nusc.sample:
    pts, uv = project_radar(nusc, ROOT, s)
    boxes = boxes_2d(nusc, s)
    inside = (pts[2] > 0.1) & (uv[0] >= 0) & (uv[0] < 1600) & (uv[1] >= 0) & (uv[1] < 900)
    valid = (pts[14] == 0) & (pts[11] == 3)
    for g, m in [("valid", inside & valid), ("flagged", inside & ~valid)]:
        for t, (du, dz) in NULLS.items():
            h = hit(boxes, uv[0] + du, uv[1], pts[2] + dz)
            acc[(g, t)][0] += h[m].sum()
            acc[(g, t)][1] += m.sum()

print("depth-consistent hit rate (in 2D box AND |depth - box depth| <= %.1f m)" % TOL)
for (g, t), (a, n) in acc.items():
    print("%-8s %-12s %5.1f%%  (%d / %d)" % (g, t, 100 * a / n, a, n))