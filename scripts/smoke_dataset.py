import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from src.dataset import FusionDataset, collate

ROOT = os.path.join(PROJ, "data", "processed", "mini")
ds = FusionDataset(ROOT, "train")
imgs, radars, labs, toks = next(iter(DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate, num_workers=0)))
print("train samples:", len(ds), "| batch images:", tuple(imgs.shape), "radar:", tuple(radars.shape))
print("image range: %.2f-%.2f" % (imgs.min(), imgs.max()))
for c, n in enumerate(["D", "V", "M"]):
    print("radar %s range: %.2f to %.2f" % (n, radars[:, c].min(), radars[:, c].max()))
print("labels per image in batch:", [len(l) for l in labs])

dsv = FusionDataset(ROOT, "val")
COL = ["yellow", "cyan", "orange"]
fig, axs = plt.subplots(1, 2, figsize=(18, 5.5))
for ax, scene in zip(axs, ["scene-0553", "scene-1094"]):
    k = [j for j, it in enumerate(dsv.items) if it["scene"] == scene][10]
    im, rd, lb, _ = dsv[k]
    ax.imshow(im.permute(1, 2, 0).numpy())
    D, M = rd[0].numpy(), rd[2].numpy()
    ov = np.zeros((360, 640, 4)); ov[(D > 0) & (M == 1)] = [0, 1, 0, .45]; ov[(D > 0) & (M == 0)] = [1, 0, 0, .45]
    ax.imshow(ov)
    for c, cx, cy, w, h in lb.numpy():
        ax.add_patch(patches.Rectangle(((cx - w / 2) * 640, (cy - h / 2) * 360), w * 640, h * 360,
                                       fill=False, ec=COL[int(c)], lw=1.5))
    ax.set_title("%s: GT boxes + radar (green valid, red flagged)" % scene); ax.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(PROJ, "outputs", "figures", "boxes_radar.png"), dpi=80)
print("saved outputs\\figures\\boxes_radar.png")