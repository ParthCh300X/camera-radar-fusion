import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
PROJ = r"D:\camera-radar-fusion"
sys.path.insert(0, PROJ)
from src.dataset import FusionDataset

ROOT = os.path.join(PROJ, "data", "processed", "subset")
OUT = os.path.join(PROJ, "results_pack", "Normal")
os.makedirs(OUT, exist_ok=True)
COL = ["yellow", "cyan", "orange"]
ds = FusionDataset(ROOT, "val")
for k, c in enumerate(("clear", "rain", "night")):
    i = next((j for j, it in enumerate(ds.items) if it["condition"] == c and len(ds[j][2]) >= 6), None)
    if i is None:
        continue
    img, radar, lab, tok = ds[i]
    im = img.permute(1, 2, 0).numpy()
    D, V, M = radar[0].numpy(), radar[1].numpy(), radar[2].numpy()
    has = D > 0
    ov = np.zeros((360, 640, 4)); ov[has & (M == 1)] = [0, 1, 0, .55]; ov[has & (M == 0)] = [1, 0, 0, .55]
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
    ax[0].imshow(im); ax[0].set_title("Camera image (%s scene)" % c)
    ax[1].imshow(im); ax[1].imshow(ov)
    for cl, cx, cy, w_, h_ in lab.numpy():
        ax[1].add_patch(patches.Rectangle(((cx - w_ / 2) * 640, (cy - h_ / 2) * 360), w_ * 640, h_ * 360, fill=False, ec=COL[int(cl)], lw=1.8))
    ax[1].set_title("True boxes + radar (green = valid, red = flagged)")
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "%d_camera_radar_truth_%s.png" % (k * 2 + 1, c)), dpi=90); plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(20, 4.5))
    ax[0].imshow(im); ax[0].imshow(np.ma.masked_where(~has, D), cmap="viridis", alpha=.8); ax[0].set_title("D: distance (dark = near)")
    ax[1].imshow(im); ax[1].imshow(np.ma.masked_where(~has, V), cmap="coolwarm", vmin=-1, vmax=1, alpha=.8); ax[1].set_title("V: speed toward/away (white = static)")
    ax[2].imshow(im); ax[2].imshow(np.ma.masked_where(~has, M), cmap="RdYlGn", vmin=0, vmax=1, alpha=.8); ax[2].set_title("M: green = trustworthy, red = flagged")
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "%d_radar_maps_%s.png" % (k * 2 + 2, c)), dpi=90); plt.close()
    print("saved", c, "val image index", i)