import os, sys, csv, argparse
import numpy as np
import torch
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, "scripts"))
import train as T
from torch.utils.data import DataLoader
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.metrics import ap_per_class
from src.dataset import FusionDataset, collate
from src.corruptions import make_corruption

ROOT = os.path.join(PROJ, "data", "processed", "subset")

def load(name, dev):
    ck = torch.load(os.path.join(PROJ, "outputs", "checkpoints", name + "_best.pt"), map_location="cpu", weights_only=False)
    if "mode" in ck:
        net = T.Net(ck["mode"], ck["channels"]); net.load_state_dict(ck["model"]); filt = ck["filtered"]
    else:                                   # early B0 checkpoint: plain detector weights
        net = T.Net("b0", "dvm"); net.det.load_state_dict(ck["model"]); filt = False
    return net.to(dev).eval(), filt

def score(rows):
    if not rows or sum(len(r[3]) for r in rows) == 0 or sum(len(r[1]) for r in rows) == 0:
        return float("nan"), float("nan"), {}
    res = ap_per_class(np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows]),
                       np.concatenate([r[2] for r in rows]), np.concatenate([r[3] for r in rows]))
    ap, cl = res[5], res[6]
    return float(ap[:, 0].mean()), float(ap.mean()), {T.NAMES[int(c)]: float(ap[i, 0]) for i, c in enumerate(cl)}

@torch.no_grad()
def run(net, filt, split, kind, level, dev, batch):
    corr = make_corruption(kind, level)
    f = (lambda P: T.keep_valid(corr(P))) if filt else corr
    ds = FusionDataset(ROOT, split, corrupt=f)
    cond = {it["token"]: it["condition"] for it in ds.items}
    rows = []
    for imgs, radars, labs, toks in DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=0):
        imgs, radars = T.pad_batch(imgs, radars)
        dets = non_max_suppression(net(imgs.to(dev), radars.to(dev)), conf_thres=0.001, iou_thres=0.7, max_det=300)
        for d, l, tk in zip(dets, labs, toks):
            l = T.norm_labels(l).to(dev); gcls = l[:, 0]
            gb = torch.stack([(l[:, 1] - l[:, 3] / 2) * T.W, (l[:, 2] - l[:, 4] / 2) * T.HP,
                              (l[:, 1] + l[:, 3] / 2) * T.W, (l[:, 2] + l[:, 4] / 2) * T.HP], 1) if len(l) else torch.zeros(0, 4, device=dev)
            rows.append((T.correct_matrix(d, gb, gcls), d[:, 4].cpu().numpy(), d[:, 5].cpu().numpy(), gcls.cpu().numpy(), cond[tk]))
    m50, m5095, per = score(rows)
    byc = {c: score([r for r in rows if r[4] == c])[0] for c in ("clear", "rain", "night")}
    return m50, m5095, per, byc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conds", nargs="+", default=["clean"])      # clean or kind:level, e.g. dropout:complete
    ap.add_argument("--split", default="val")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        sys.exit("The test split is for the final evaluation only: add --final if this is it.")
    dev = torch.device("cuda")
    out = os.path.join(PROJ, "outputs", "logs", "eval_%s.csv" % a.split)
    new = not os.path.exists(out)
    f = open(out, "a", newline=""); w = csv.writer(f)
    if new:
        w.writerow(["split", "model", "kind", "level", "map50", "map5095", "car", "pedestrian", "truck", "clear", "rain", "night"])
    print("%-12s %-18s | mAP50 mAP50-95 | car   ped   truck | clear rain  night" % ("model", "condition"))
    for name in a.models:
        net, filt = load(name, dev)
        for c in a.conds:
            kind, level = (c.split(":") + [None])[:2]
            m50, m5095, per, byc = run(net, filt, a.split, kind, level, dev, a.batch)
            print("%-12s %-18s | %.3f  %.3f    | %.3f %.3f %.3f | %.3f %.3f %.3f" % (
                name, c, m50, m5095, per.get("car", np.nan), per.get("pedestrian", np.nan), per.get("truck", np.nan),
                byc["clear"], byc["rain"], byc["night"]))
            w.writerow([a.split, name, kind, level, m50, m5095, per.get("car"), per.get("pedestrian"), per.get("truck"), byc["clear"], byc["rain"], byc["night"]]); f.flush()

if __name__ == "__main__":
    main()