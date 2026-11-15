import os, sys, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJ = r"D:\camera-radar-fusion"
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, "scripts"))
import train as T
from torch.utils.data import DataLoader
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, ConfusionMatrix
from src.dataset import FusionDataset, collate
from src.corruptions import make_corruption

ROOT = os.path.join(PROJ, "data", "processed", "subset")
PACK = os.path.join(PROJ, "results_pack")
FOLDERS = {"B0": "b0", "B1": "b1", "B2": "b2"}
TITLES = {"B0": "B0: camera-only YOLO", "B1": "B1: camera + radar (early fusion)", "B2": "B2: camera + radar (mid fusion)"}
COL = ["yellow", "cyan", "orange"]
dev = torch.device("cuda")

def load(name):
    ck = torch.load(os.path.join(PROJ, "outputs", "checkpoints", name + "_best.pt"), map_location="cpu", weights_only=False)
    if "mode" in ck:
        net = T.Net(ck["mode"], ck["channels"]); net.load_state_dict(ck["model"]); filt = ck["filtered"]
    else:
        net = T.Net("b0", "dvm"); net.det.load_state_dict(ck["model"]); filt = False
    return net.to(dev).eval(), filt

def gt_boxes(l):
    l = T.norm_labels(l).to(dev)
    if not len(l):
        return torch.zeros(0, 4, device=dev), l[:, 0]
    gb = torch.stack([(l[:, 1] - l[:, 3] / 2) * T.W, (l[:, 2] - l[:, 4] / 2) * T.HP,
                      (l[:, 1] + l[:, 3] / 2) * T.W, (l[:, 2] + l[:, 4] / 2) * T.HP], 1)
    return gb, l[:, 0]

@torch.no_grad()
def collect(net, filt, kind, level, keep=False):
    corr = make_corruption(kind, level)
    f = (lambda P: T.keep_valid(corr(P))) if filt else corr
    ds = FusionDataset(ROOT, "val", corrupt=f)
    cond = {it["token"]: it["condition"] for it in ds.items}
    rows, store = [], []
    for imgs, radars, labs, toks in DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate, num_workers=0):
        imgs, radars = T.pad_batch(imgs, radars)
        dets = non_max_suppression(net(imgs.to(dev), radars.to(dev)), conf_thres=0.001, iou_thres=0.7, max_det=300)
        for d, l, tk in zip(dets, labs, toks):
            gb, gc = gt_boxes(l)
            rows.append((T.correct_matrix(d, gb, gc), d[:, 4].cpu().numpy(), d[:, 5].cpu().numpy(), gc.cpu().numpy(), cond[tk]))
            if keep:
                store.append((d, gb, gc))
    return rows, store

def score(rows, plot_dir=None):
    kw = {}
    if plot_dir is not None:
        kw = dict(plot=True, save_dir=Path(plot_dir), names={i: n for i, n in enumerate(T.NAMES)})
    res = ap_per_class(np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows]),
                       np.concatenate([r[2] for r in rows]), np.concatenate([r[3] for r in rows]), **kw)
    ap, cl = res[5], res[6]
    return float(ap[:, 0].mean()), float(ap.mean()), {T.NAMES[int(c)]: float(ap[i, 0]) for i, c in enumerate(cl)}

@torch.no_grad()
def predict(net, img, radar, conf=0.35):
    im, rd = T.pad_batch(img[None], radar[None])
    return non_max_suppression(net(im.to(dev), rd.to(dev)), conf_thres=conf, iou_thres=0.7, max_det=100)[0].cpu().numpy()

base = FusionDataset(ROOT, "val")
PICK = {}
for c in ("clear", "rain", "night"):
    PICK[c] = next((i for i, it in enumerate(base.items) if it["condition"] == c and len(base[i][2]) >= 6), None)
print("picked val images:", PICK)

for folder, name in FOLDERS.items():
    out = os.path.join(PACK, folder); os.makedirs(out, exist_ok=True)
    print("---", folder, name)
    net, filt = load(name)
    fusion = name != "b0"

    # 1-3: detection pictures
    dsm = FusionDataset(ROOT, "val", corrupt=T.keep_valid if filt else None)
    for k, c in enumerate(("clear", "rain", "night")):
        if PICK[c] is None:
            continue
        img, radar, lab, tok = dsm[PICK[c]]
        det = predict(net, img, radar)
        fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
        for a in ax:
            a.imshow(img.permute(1, 2, 0).numpy()); a.axis("off")
        for cl, cx, cy, w_, h_ in lab.numpy():
            ax[0].add_patch(patches.Rectangle(((cx - w_ / 2) * 640, (cy - h_ / 2) * 360), w_ * 640, h_ * 360, fill=False, ec=COL[int(cl)], lw=1.8))
        ax[0].set_title("Ground truth (%d objects)" % len(lab))
        if fusion:
            D, M = radar[0].numpy(), radar[2].numpy()
            ov = np.zeros((360, 640, 4)); ov[(D > 0) & (M == 1)] = [0, 1, 0, .30]; ov[(D > 0) & (M == 0)] = [1, 0, 0, .30]
            ax[1].imshow(ov)
        for x1, y1, x2, y2, cf, kk in det:
            ax[1].add_patch(patches.Rectangle((x1, y1 - T.PAD), x2 - x1, y2 - y1, fill=False, ec=COL[int(kk)], lw=1.8))
        ax[1].set_title("%s: %d detections (confidence >= 0.35)" % (TITLES[folder], len(det)))
        fig.suptitle("%s scene | yellow = car, cyan = pedestrian, orange = truck%s" % (c, " | radar: green valid, red flagged" if fusion else ""))
        plt.tight_layout(); plt.savefig(os.path.join(out, "%d_detections_%s.png" % (k + 1, c)), dpi=90); plt.close()

    # 4: training curve
    rows_log = list(csv.DictReader(open(os.path.join(PROJ, "outputs", "logs", name + "_log.csv"))))
    ep = [int(r["epoch"]) for r in rows_log]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(ep, [float(r["train_box"]) for r in rows_log], label="box loss"); ax[0].plot(ep, [float(r["train_cls"]) for r in rows_log], label="class loss")
    ax[0].set_title("Training loss (lower is better)"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(ep, [float(r["val_map50"]) for r in rows_log], label="mAP50"); ax[1].plot(ep, [float(r["val_map5095"]) for r in rows_log], label="mAP50-95")
    ax[1].set_title("Validation score (higher is better)"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=.3); ax[1].legend()
    fig.suptitle(TITLES[folder]); plt.tight_layout(); plt.savefig(os.path.join(out, "4_training_curve.png"), dpi=110); plt.close()

    # numbers + 5: PR curve
    rows, store = collect(net, filt, "clean", None, keep=True)
    m50, m5095, per = score(rows, plot_dir=out)
    byc = {c: score([r for r in rows if r[4] == c])[0] for c in ("clear", "rain", "night")}
    for f_, n_ in (("PR_curve.png", "5_PR_curve.png"),):
        if os.path.exists(os.path.join(out, f_)):
            os.replace(os.path.join(out, f_), os.path.join(out, n_))
    for f_ in ("P_curve.png", "R_curve.png", "F1_curve.png"):
        if os.path.exists(os.path.join(out, f_)):
            os.remove(os.path.join(out, f_))

    # 6: confusion matrix
    try:
        cm = ConfusionMatrix(nc=3)
        for d, gb, gc in store:
            cm.process_batch(d, gb, gc)
        cm.plot(normalize=True, save_dir=Path(out), names=list(T.NAMES))
        s_ = os.path.join(out, "confusion_matrix_normalized.png")
        if os.path.exists(s_):
            os.replace(s_, os.path.join(out, "6_confusion_matrix.png"))
    except Exception as e:
        print("confusion matrix skipped:", e)

    # 7: results card
    nr = "n/a (no radar input)"
    if fusion:
        rows_nr, _ = collect(net, filt, "dropout", "complete")
        nr = "%.3f" % score(rows_nr)[0]
    cells = [["mAP50 (validation)", "%.3f" % m50], ["mAP50-95 (validation)", "%.3f" % m5095],
             ["car / pedestrian / truck AP50", "%.3f / %.3f / %.3f" % (per.get("car", 0), per.get("pedestrian", 0), per.get("truck", 0))],
             ["mAP50 clear / rain / night", "%.3f / %.3f / %.3f" % (byc["clear"], byc["rain"], byc["night"])],
             ["mAP50 with all radar removed", nr],
             ["parameters", "%.2f M" % (sum(p.numel() for p in net.parameters()) / 1e6)],
            ["peak GPU memory (training)", ("%.2f GB" % max(float(r["peak_gb"]) for r in rows_log)) if "peak_gb" in rows_log[0] else "not logged (early run)"],
             ["average epoch time", "%.0f s" % np.mean([float(r["sec"]) for r in rows_log])]]
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=["metric", TITLES[folder]], loc="center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 1.6)
    plt.tight_layout(); plt.savefig(os.path.join(out, "7_results_card.png"), dpi=130); plt.close()
    print(folder, "done | mAP50 %.3f | mAP50-95 %.3f" % (m50, m5095))

for f in sorted(os.listdir(PACK)):
    print(f, ":", sorted(os.listdir(os.path.join(PACK, f))))