import os, sys
import numpy as np
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from src.corruptions import ALL_CONDITIONS, make_corruption

z = np.load(os.path.join(PROJ, "data", "processed", "subset", "radar_points.npz"))
pts = [z[t] for t in list(z.files)[:300]]
print("condition          | points/sample | valid fraction | deterministic")
for kind, level in ALL_CONDITIONS:
    f = make_corruption(kind, level)
    n, vf, det = [], [], True
    for P in pts:
        Q, R = f(P), f(P)
        det &= Q.shape == R.shape and np.array_equal(Q, R)
        n.append(len(Q)); vf.append(Q[:, 5].mean() if len(Q) else np.nan)
    print("%-18s | %13.1f | %14.3f | %s" % (kind + ("/" + level if level else ""), np.mean(n), np.nanmean(vf), det))