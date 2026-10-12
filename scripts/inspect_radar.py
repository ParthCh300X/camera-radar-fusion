import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

ROOT = r"D:\camera-radar-fusion\data\raw\v1.0-mini"
nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)

FIELDS = ["x", "y", "z", "dyn_prop", "id", "rcs", "vx", "vy", "vx_comp", "vy_comp",
          "is_quality_valid", "ambig_state", "x_rms", "y_rms", "invalid_state",
          "pdh0", "vx_rms", "vy_rms"]

print("SCENES")
for s in nusc.scene:
    print(" ", s["name"], "|", s["description"], "|", s["nbr_samples"], "samples")

def radar_path(sample):
    sd = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
    return os.path.join(ROOT, sd["filename"])

# Point counts per sample: default filters vs no filters
filt, raw = [], []
for s in nusc.sample:
    p = radar_path(s)
    RadarPointCloud.default_filters()
    filt.append(RadarPointCloud.from_file(p).points.shape[1])
    RadarPointCloud.disable_filters()
    raw.append(RadarPointCloud.from_file(p).points.shape[1])
print("\nPOINTS PER SAMPLE (RADAR_FRONT, all 404 samples)")
print("  default filters: mean %.1f, min %d, max %d" % (np.mean(filt), min(filt), max(filt)))
print("  no filters:      mean %.1f, min %d, max %d" % (np.mean(raw), min(raw), max(raw)))

# Field statistics on the first sample, unfiltered
RadarPointCloud.disable_filters()
pts = RadarPointCloud.from_file(radar_path(nusc.sample[0])).points
print("\nFIELD STATS (first sample, unfiltered, %d points)" % pts.shape[1])
for i, name in enumerate(FIELDS):
    print("  %-17s min %9.2f  max %9.2f  mean %9.2f" % (name, pts[i].min(), pts[i].max(), pts[i].mean()))

# Flag distributions across the whole mini set, unfiltered
allp = np.concatenate([RadarPointCloud.from_file(radar_path(s)).points for s in nusc.sample], axis=1)
print("\nFLAG COUNTS (all samples, unfiltered, %d points)" % allp.shape[1])
for name in ["invalid_state", "ambig_state", "is_quality_valid", "dyn_prop"]:
    vals, cnts = np.unique(allp[FIELDS.index(name)], return_counts=True)
    print(" ", name, dict(zip(vals.astype(int).tolist(), cnts.tolist())))
vals, cnts = np.unique(allp[FIELDS.index("pdh0")], return_counts=True)
print("  pdh0", dict(zip(vals.astype(int).tolist(), cnts.tolist())))