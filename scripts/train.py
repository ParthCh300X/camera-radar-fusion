import os, sys, time, argparse, csv
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
os.environ["PIN_MEMORY"] = "False"
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, box_iou
from src.dataset import FusionDataset, collate

H0, W, HP, PAD = 360, 640, 384, 12
IOUV = torch.linspace(0.5, 0.95, 10)
NAMES = ["car", "pedestrian", "truck"]

def keep_valid(P):                         # B2-filtered control: drop M=0 radar points
    return P[P[:, 5] > 0.5]

def pad_batch(imgs, radars):
    return (F.pad(imgs, (0, 0, PAD, PAD), value=114 / 255), F.pad(radars, (0, 0, PAD, PAD), value=0.0))

def norm_labels(l):
    l = l.clone()
    if len(l):
        l[:, 2] = (l[:, 2] * H0 + PAD) / HP
        l[:, 4] = l[:, 4] * H0 / HP
    return l

def loss_batch(labs, dev):
    idx, cls, box = [], [], []
    for i, l in enumerate(labs):
        if len(l):
            idx.append(torch.full((len(l),), float(i))); cls.append(l[:, 0:1]); box.append(l[:, 1:5])
    if idx:
        return {"batch_idx": torch.cat(idx).to(dev), "cls": torch.cat(cls).to(dev), "bboxes": torch.cat(box).to(dev)}
    return {"batch_idx": torch.zeros(0, device=dev), "cls": torch.zeros(0, 1, device=dev), "bboxes": torch.zeros(0, 4, device=dev)}

def correct_matrix(det, gt_boxes, gt_cls):
    correct = np.zeros((det.shape[0], 10), dtype=bool)
    if det.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return correct
    iou = box_iou(gt_boxes, det[:, :4])
    same = gt_cls[:, None] == det[:, 5]
    for i, thr in enumerate(IOUV):
        x = torch.where(iou.ge(thr) & same)
        if x[0].shape[0]:
            m = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                m = m[m[:, 2].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct

@torch.no_grad()
def evaluate(net, loader, dev):
    net.eval()
    tp, conf, pcls, tcls = [], [], [], []
    for imgs, radars, labs, _ in loader:
        imgs, radars = pad_batch(imgs, radars)
        preds = net(imgs.to(dev), radars.to(dev))
        dets = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.7, max_det=300)
        for d, l in zip(dets, labs):
            l = norm_labels(l).to(dev)
            gcls = l[:, 0]
            gb = torch.stack([(l[:, 1] - l[:, 3] / 2) * W, (l[:, 2] - l[:, 4] / 2) * HP,
                              (l[:, 1] + l[:, 3] / 2) * W, (l[:, 2] + l[:, 4] / 2) * HP], 1) if len(l) else torch.zeros(0, 4, device=dev)
            tp.append(correct_matrix(d, gb, gcls)); conf.append(d[:, 4].cpu().numpy()); pcls.append(d[:, 5].cpu().numpy()); tcls.append(gcls.cpu().numpy())
    res = ap_per_class(np.concatenate(tp), np.concatenate(conf), np.concatenate(pcls), np.concatenate(tcls))
    ap, classes = res[5], res[6]
    per = {NAMES[int(c)]: float(ap[i, 0]) for i, c in enumerate(classes)}
    return float(ap[:, 0].mean()), float(ap.mean()), per

def build_detector():
    det = DetectionModel("yolov8n.yaml", nc=3, verbose=False)
    sd = torch.load(os.path.join(PROJ, "yolov8n.pt"), map_location="cpu", weights_only=False)["model"].float().state_dict()
    own = det.state_dict()
    det.load_state_dict({k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}, strict=False)
    det.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    return det

def zconv(c):
    cv = nn.Conv2d(c, c, 1); nn.init.zeros_(cv.weight); nn.init.zeros_(cv.bias); return cv

class RadarEncoder(nn.Module):
    """Radar maps (3,384,640) -> features at strides 8/16/32, fed in through zero-initialised 1x1 convs."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(Conv(3, 16, 3, 2), Conv(16, 32, 3, 2), Conv(32, 64, 3, 2))
        self.s16, self.s32 = Conv(64, 128, 3, 2), Conv(128, 256, 3, 2)
        self.adapt = nn.ModuleDict({"4": zconv(64), "6": zconv(128), "9": zconv(256)})
    def forward(self, r):
        p3 = self.stem(r); p4 = self.s16(p3); p5 = self.s32(p4)
        return {"4": self.adapt["4"](p3), "6": self.adapt["6"](p4), "9": self.adapt["9"](p5)}

class Net(nn.Module):
    def __init__(self, mode, channels):
        super().__init__()
        self.mode = mode
        self.det = build_detector()
        self.register_buffer("cmask", torch.tensor([float(c in channels) for c in "dvm"]))
        if mode == "b1":
            c0 = self.det.model[0].conv
            new = nn.Conv2d(6, c0.out_channels, c0.kernel_size, c0.stride, c0.padding, bias=False)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :3] = c0.weight
            self.det.model[0].conv = new
        if mode == "b2":
            self.enc = RadarEncoder()

    def forward(self, imgs, radars):
        radars = radars * self.cmask.view(1, 3, 1, 1)
        if self.mode == "b0":
            return self.det(imgs)
        if self.mode == "b1":
            return self.det(torch.cat([imgs, radars], 1))
        feats = self.enc(radars)
        y, x = [], imgs
        for m in self.det.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if str(m.i) in feats:
                x = x + feats[str(m.i)]
            y.append(x if m.i in self.det.save else None)
        return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="b0", choices=["b0", "b1", "b2"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--channels", default="dvm")
    ap.add_argument("--filtered", action="store_true")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    name = a.name or a.model
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("cuda")
    root = os.path.join(PROJ, "data", "processed", "subset")
    cor = keep_valid if a.filtered else None
    tr = FusionDataset(root, "train", corrupt=cor); va = FusionDataset(root, "val", corrupt=cor)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, collate_fn=collate, num_workers=4, drop_last=True)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, collate_fn=collate, num_workers=2)
    net = Net(a.model, a.channels).to(dev); crit = v8DetectionLoss(net.det)
    print("model %s | channels %s | filtered %s | parameters %.2f M" % (a.model, a.channels, a.filtered, sum(p.numel() for p in net.parameters()) / 1e6))
    opt = torch.optim.AdamW(net.parameters(), lr=1.4e-3, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda")
    total = a.epochs * len(tl); step = 0; best = -1.0
    ck = os.path.join(PROJ, "outputs", "checkpoints"); lg = os.path.join(PROJ, "outputs", "logs"); os.makedirs(ck, exist_ok=True)

    if a.overfit:
        imgs, radars, labs, _ = next(iter(tl)); imgs, radars = pad_batch(imgs[:8], radars[:8]); labs = [norm_labels(l) for l in labs[:8]]
        net.train()
        for it in range(150):
            with torch.autocast("cuda"):
                loss, items = crit(net(imgs.to(dev), radars.to(dev)), loss_batch(labs, dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if it % 15 == 0 or it == 149:
                print("iter %3d | box %.3f cls %.3f dfl %.3f" % (it, *items.tolist()))
        return

    torch.cuda.reset_peak_memory_stats()
    last = os.path.join(ck, name + "_last.pt"); start_ep = 0
    resuming = a.resume and os.path.exists(last)
    if resuming:
        s = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        start_ep, best = s["epoch"], s["best"]; step = start_ep * len(tl)
        print("resuming from epoch", start_ep)
    log = open(os.path.join(lg, name + "_log.csv"), "a" if resuming else "w", newline=""); w = csv.writer(log)
    if not resuming:
        w.writerow(["epoch", "train_box", "train_cls", "train_dfl", "val_map50", "val_map5095", "sec", "peak_gb"])
    for ep in range(start_ep, a.epochs):
        net.train(); t0 = time.time(); acc = torch.zeros(3)
        for imgs, radars, labs, _ in tl:
            lr_scale = min(1.0, (step + 1) / (3 * len(tl))) * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * step / total)))
            for g in opt.param_groups: g["lr"] = 1.4e-3 * lr_scale
            flip = torch.rand(len(imgs)) < 0.5
            imgs[flip] = imgs[flip].flip(-1); radars[flip] = radars[flip].flip(-1)
            labs = [l.clone() for l in labs]
            for i in torch.nonzero(flip).flatten().tolist():
                if len(labs[i]): labs[i][:, 1] = 1 - labs[i][:, 1]
            imgs, radars = pad_batch(imgs, radars); labs = [norm_labels(l) for l in labs]
            with torch.autocast("cuda"):
                loss, items = crit(net(imgs.to(dev), radars.to(dev)), loss_batch(labs, dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            acc += items.cpu(); step += 1
        m50, m5095, per = evaluate(net, vl, dev)
        acc /= len(tl); sec = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 1e9
        w.writerow([ep + 1, *acc.tolist(), m50, m5095, sec, peak]); log.flush()
        print("epoch %2d/%d | box %.3f cls %.3f dfl %.3f | val mAP50 %.3f mAP50-95 %.3f | %.0fs | %.2fG" % (ep + 1, a.epochs, *acc.tolist(), m50, m5095, sec, peak))
        if m5095 > best:
            best = m5095
            torch.save({"model": net.state_dict(), "mode": a.model, "channels": a.channels, "filtered": a.filtered,
                        "epoch": ep + 1, "map50": m50, "map5095": m5095}, os.path.join(ck, name + "_best.pt"))
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": ep + 1, "best": best}, last)
    print("best val mAP50-95: %.3f | per-class AP50 (last epoch): %s" % (best, {k: round(v, 3) for k, v in per.items()}))

if __name__ == "__main__":
    main()