import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PROJ = r"D:\camera-radar-fusion"
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, "scripts"))
import train as T
from src.dataset import FusionDataset
from src.corruptions import misalign

ROOT = os.path.join(PROJ, "data", "processed", "subset")
OUT = os.path.join(PROJ, "results_pack", "B3"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
name = sys.argv[1] if len(sys.argv) > 1 else "b3_v2"
ck = torch.load(os.path.join(PROJ, "outputs", "checkpoints", name + "_best.pt"), map_location="cpu", weights_only=False)
net = T.Net(ck["mode"], ck["channels"]); net.load_state_dict(ck["model"]); net = net.to(dev).eval()
clean = FusionDataset(ROOT, "val")
shifted = FusionDataset(ROOT, "val", corrupt=lambda P: misalign(P, 60, mark=True))

def up(t):
    return F.interpolate(t[None, None], size=(T.HP, T.W), mode="nearest")[0, 0][T.PAD:T.HP - T.PAD].numpy()

@torch.no_grad()
def gate_maps(ds, i):
    img, radar, lab, tok = ds[i]
    im, rd = T.pad_batch(img[None], radar[None])
    net(im.to(dev), rd.to(dev))
    out = []
    for j, s in enumerate((8, 16, 32)):
        g = net.gates[j][0, 0].float().cpu()
        occ = F.avg_pool2d((rd[:, 0:1] > 0).float(), s, s)[0, 0]
        out.append(np.ma.masked_where(up(occ) == 0, up(g)))
    return img.permute(1, 2, 0).numpy(), radar, out

for c in ("clear", "rain", "night"):
    i = next((j for j, it in enumerate(clean.items) if it["condition"] == c and len(clean[j][2]) >= 6), None)
    if i is None:
        continue
    fig, axs = plt.subplots(2, 4, figsize=(24, 8.6))
    for r, (ds, label) in enumerate(((clean, "clean radar"), (shifted, "radar shifted 60 px"))):
        im, radar, gm = gate_maps(ds, i)
        D, M = radar[0].numpy(), radar[2].numpy()
        ov = np.zeros((360, 640, 4)); ov[(D > 0) & (M == 1)] = [0, 1, 0, .6]; ov[(D > 0) & (M == 0)] = [1, 0, 0, .6]
        axs[r, 0].imshow(im); axs[r, 0].imshow(ov); axs[r, 0].set_title("%s: radar (green valid, red flagged)" % label)
        for k, s in enumerate((8, 16, 32)):
            axs[r, k + 1].imshow(im); axs[r, k + 1].imshow(gm[k], cmap="viridis", vmin=0, vmax=1, alpha=.7)
            axs[r, k + 1].set_title("%s: gate at stride %d (bright = radar let through)" % (label, s))
    for a in axs.ravel():
        a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "gate_maps_%s.png" % c), dpi=70); plt.close()
    print("saved", c)