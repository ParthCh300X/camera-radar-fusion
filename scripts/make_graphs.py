import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = r"D:\camera-radar-fusion"
LOGS = os.path.join(PROJ, "outputs", "logs")
OUT = os.path.join(PROJ, "docs", "ppt_assets")
os.makedirs(OUT, exist_ok=True)

def read_log(name):
    p = os.path.join(LOGS, name + "_log.csv")
    if not os.path.exists(p):
        return None
    return [float(r["val_map50"]) for r in csv.DictReader(open(p))]

def load_eval(fname):
    d = {}
    p = os.path.join(LOGS, fname)
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            d[(r["model"], r["kind"], r["level"])] = r
    return d

def val(d, m, k, l, col):
    r = d.get((m, k, l))
    try:
        return float(r[col])
    except Exception:
        return np.nan

E = load_eval("eval_val.csv")
MODELS = ["b0", "b1", "b2", "b2_filtered", "b0_aug", "b2_aug", "b2_aug2", "b3", "b3_v2"]

# fig1: training curves
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for a, group, title in ((ax[0], ["b0", "b1", "b2", "b2_filtered"], "Plain training"),
                        (ax[1], ["b0_aug", "b2_aug", "b2_aug2", "b3", "b3_v2"], "With augmentation")):
    for m in group:
        v = read_log(m)
        if v:
            a.plot(range(1, len(v) + 1), v, label=m, lw=1.8)
    a.set_title(title); a.set_xlabel("epoch"); a.set_ylabel("validation mAP50"); a.set_ylim(0.2, 0.65); a.grid(alpha=0.3); a.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_training_curves.png"), dpi=130); plt.close()

# fig2: by condition (clean radar)
fig, ax = plt.subplots(figsize=(14, 5)); w = 0.26
for j, col in enumerate(("clear", "rain", "night")):
    ax.bar(np.arange(len(MODELS)) + (j - 1) * w, [val(E, m, "clean", "", col) for m in MODELS], w, label=col)
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=15); ax.set_ylabel("validation mAP50")
ax.set_title("Detection score by condition (clean radar)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_by_condition.png"), dpi=130); plt.close()

# fig3: robustness lines (only models that were evaluated on that corruption)
GROUPS = {"Sparsity (share of points kept, %)": ("sparsity", ["75", "50", "25", "10"]),
          "Noise on depth and speed": ("noise", ["low", "mid", "high"]),
          "Misalignment": ("misalign", ["small", "medium", "large"]),
          "Dropout (radar blanked)": ("dropout", ["partial", "severe", "complete"]),
          "Ghost points (false returns)": ("clutter", ["low", "mid", "high"])}
fig, axs = plt.subplots(1, 5, figsize=(24, 4.8))
for a, (title, (kind, levels)) in zip(axs, GROUPS.items()):
    for m in MODELS:
        ys = [val(E, m, "clean", "", "map50")] + [val(E, m, kind, l, "map50") for l in levels]
        if np.isnan(ys[1:]).all() or m == "b0" or m == "b0_aug":
            continue
        a.plot(range(len(ys)), ys, marker="o", label=m)
    a.set_xticks(range(len(levels) + 1)); a.set_xticklabels(["clean"] + levels)
    a.set_title(title); a.set_ylim(0.54, 0.62); a.grid(alpha=0.3)
axs[0].set_ylabel("validation mAP50"); axs[0].legend(fontsize=8)
plt.suptitle("Robustness to bad radar (camera-only models are flat by construction and omitted)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_robustness.png"), dpi=130); plt.close()

# fig4: do the models use the radar? radar present vs all radar removed
fm = ["b1", "b2", "b2_filtered", "b2_aug", "b2_aug2", "b3", "b3_v2"]
fig, ax = plt.subplots(figsize=(11, 5)); w = 0.35
ax.bar(np.arange(len(fm)) - w / 2, [val(E, m, "clean", "", "map50") for m in fm], w, label="radar present")
ax.bar(np.arange(len(fm)) + w / 2, [val(E, m, "dropout", "complete", "map50") for m in fm], w, label="all radar removed")
ax.set_xticks(range(len(fm))); ax.set_xticklabels(fm, rotation=15); ax.set_ylim(0.5, 0.62); ax.set_ylabel("validation mAP50")
ax.set_title("Do the models use the radar?"); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_radar_removed.png"), dpi=130); plt.close()

# fig5: gate reaction to 60 px radar misalignment (numbers from gate_stats_v2 output)
labels = ["stride 8", "stride 16", "stride 32"]
v2_clean, v2_shift = [0.890, 0.915, 0.938], [0.885, 0.697, 0.170]
v1_clean, v1_shift = [0.271, 0.250, 0.097], [0.275, 0.255, 0.102]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, (c, s, t) in zip(ax, ((v1_clean, v1_shift, "B3 v1 (gate not supervised)"), (v2_clean, v2_shift, "B3-v2 (gate supervised)"))):
    a.bar(np.arange(3) - 0.18, c, 0.36, label="clean radar"); a.bar(np.arange(3) + 0.18, s, 0.36, label="radar shifted 60 px")
    a.set_xticks(range(3)); a.set_xticklabels(labels); a.set_ylim(0, 1); a.set_title(t); a.grid(axis="y", alpha=0.3)
ax[0].set_ylabel("mean gate value (0 = closed, 1 = open)"); ax[0].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_gate_misalignment.png"), dpi=130); plt.close()

# fig6: darkened camera (needs the dark evaluations)
D1, D2 = load_eval("eval_val_dark.csv"), load_eval("eval_val_darker.csv")
if D1 and D2:
    ms = ["b0", "b1", "b2", "b0_aug", "b2_aug", "b2_aug2", "b3", "b3_v2"]
    fig, ax = plt.subplots(figsize=(13, 5))
    for j, (lab, dd) in enumerate((("normal camera", E), ("darkened (x0.3)", D1), ("very dark (x0.15)", D2))):
        ax.bar(np.arange(len(ms)) + (j - 1) * 0.27, [val(dd, m, "clean", "", "map50") for m in ms], 0.27, label=lab)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=15); ax.set_ylabel("validation mAP50")
    ax.set_title("Camera made darker at test time"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig6_dark.png"), dpi=130); plt.close()
print("saved:", sorted(f for f in os.listdir(OUT) if f.startswith("fig")))