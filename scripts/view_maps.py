import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from nuscenes.nuscenes import NuScenes
from src.radar_maps import rasterize

ROOT = os.path.join(PROJ, "data", "raw", "v1.0-mini")
nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)
cache = np.load(os.path.join(PROJ, "data", "processed", "mini", "radar_points.npz"))

for name in ["scene-0061", "scene-1077"]:
    tok = next(s for s in nusc.scene if s["name"] == name)["first_sample_token"]
    for _ in range(10):
        tok = nusc.get("sample", tok)["next"]
    s = nusc.get("sample", tok)
    img = np.asarray(Image.open(os.path.join(ROOT, nusc.get("sample_data", s["data"]["CAM_FRONT"])["filename"])).resize((640, 360)))
    D, V, M = rasterize(cache[tok])
    has = D > 0
    mask_ov = np.zeros((360, 640, 4)); mask_ov[has & (M == 1)] = [0, 1, 0, 0.7]; mask_ov[has & (M == 0)] = [1, 0, 0, 0.7]
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    ax[0, 0].imshow(img); ax[0, 0].set_title("camera")
    ax[0, 1].imshow(img); ax[0, 1].imshow(np.ma.masked_where(~has, D), cmap="viridis", alpha=0.75); ax[0, 1].set_title("D (depth/100)")
    ax[1, 0].imshow(img); ax[1, 0].imshow(np.ma.masked_where(~has, V), cmap="coolwarm", vmin=-1, vmax=1, alpha=0.75); ax[1, 0].set_title("V (radial velocity)")
    ax[1, 1].imshow(img); ax[1, 1].imshow(mask_ov); ax[1, 1].set_title("M (green = valid, red = flagged)")
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(PROJ, "outputs", "figures", "maps_%s.png" % name), dpi=70); plt.close()
    print(name, "points:", len(cache[tok]), "D>0 pixels:", int(has.sum()))