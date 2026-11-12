# Results log

Single source for numbers and figures used in the report and slides. Only measured values go here; anything not yet measured is marked TBD.

Last updated: 13 Nov 2026 (Phase 4 started)

---

## 1. Design decisions and deviations (for Methods and viva)

| Topic | Decision | Reason |
|---|---|---|
| Task | 2D object detection (image plane) | 4-6 GB GPU, solo project; CRF-Net itself is 2D (to be verified in the paper). The deck says 3D: **supervisor sign-off needed**. 3D is a stretch goal only |
| Classes | car, pedestrian, truck (ids 0, 1, 2) | 23 raw nuScenes categories are too rare to learn at this scale; benchmark uses 10 |
| Baselines | B0 camera-only, B1 early fusion, B2 mid fusion, B3 ours (reliability gate) | B1/B2 are our reimplementations of the CRF-Net-style and CenterFusion-style paradigms in one shared framework, not the published networks |
| Extra control | B2-filtered: ungated mid fusion with devkit-filtered radar points | Shows whether the learned gate beats a simple hard-filter heuristic |
| Radar frames | Keyframes only, single sweep | Keyframe-only download is about 10 GB instead of about 70 GB; matches the single-sweep design |
| Reliability supervision | No ground-truth reliability exists: detection loss plus corruption-aware augmentation, optional auxiliary loss on injected-bad points (proxy target, labelled as such) | Avoids inventing labels |

---

## 2. Radar statistics (nuScenes v1.0-mini, RADAR_FRONT, 404 samples)

- 18 fields per point; **z is always 0** (no elevation information).
- Points per sample: about 120 unfiltered (mean 120.4, min 52, max 125); about 60 with the devkit default filter (mean 60.2).
- Only **51.6%** of returns have `invalid_state == 0` (25,108 of 48,659); 90% have `ambig_state == 3`.
- `pdh0` (false-alarm probability level): values 1, 3, 6 observed (29,163 / 16,196 / 3,300). Level meanings to be verified against the nuScenes radar documentation.
- `is_quality_valid` is 1 for every point: unused.
- Raw `vx`, `vy` include ego motion: use `vx_comp`, `vy_comp`.
- After projection into the camera (in front of camera, inside image width): mean 99.4 points per sample on mini (min 32, max 120); 97.9 on the trainval subset.
- Mini has 7 day and 3 night scenes (scene-1077, 1094 after rain, 1100).

### Radar representation (rasterised at 640x360, scale 0.4 from 1600x900)
- **D** = depth / 100 m (clipped to 1)
- **V** = radial velocity / 15 m/s (clipped to [-1, 1]); radial velocity = projection of (`vx_comp`, `vy_comp`) on the line of sight
- **M** = validity mask: 1 if `invalid_state == 0` and `ambig_state == 3`, else 0
- Each point becomes a vertical strip from 0.5 m below to 2.0 m above the radar point (assumption, since radar has no height), 5 px wide at 640 scale; nearest point wins on overlap
- All points are kept (no filtering); M marks reliability. Corruptions act on points before rasterising.

---

## 3. Projection and alignment check (mini)

First attempt, in-box hit rate without depth (inconclusive, scenes are crowded):

| points | real | sideways shift 150 px |
|---|---|---|
| valid (M=1) | 37.7% (7598 / 20162) | 36.4% |
| flagged (M=0) | 36.3% (7254 / 19984) | 36.6% |

Depth-consistent hit rate (inside 2D box and |depth - box depth| <= 3 m):

| points | real | sideways shift 150 px | depth shift 8 m |
|---|---|---|---|
| valid (M=1) | **11.1%** (2242 / 20162) | 6.0% | 1.4% |
| flagged (M=0) | **6.0%** (1195 / 19984) | 3.3% | 0.8% |

Takeaways:
- Alignment is sound: real is about 1.8x the sideways null and about 8x the depth null.
- M carries signal: valid points hit about 1.8x more often than flagged points.
- Flagged points still beat their own null (6.0% vs 3.3%), so they are not pure noise. This supports a soft gate over a hard filter.
- The rates are conservative: radar reflects off the near surface, but depth is compared with the box centre.

---

## 4. 2D label rules

- Source: 3D annotations projected into CAM_FRONT.
- Class mapping: `vehicle.car` to car; `vehicle.truck` to truck; `human.pedestrian.adult`, `.child`, `.construction_worker`, `.police_officer` to pedestrian.
- Kept if visibility bin >= 2 (at least 40% of the object visible), box at least 12x25 px at 1600x900 (about 5x10 px at 640x360), and no corner behind the camera (limitation: objects partly behind the camera are dropped).
- Boxes are clipped to the image and stored in YOLO format (normalised cx, cy, w, h).
- Mini debug split: 8 train scenes, 2 val scenes (scene-0553 day, scene-1094 night after rain). First build used a smaller filter (8x12 px): 323 train / 81 val images. Counts after tightening the filter: TBD (rebuild output not yet recorded).

---

## 5. Trainval subset (final dataset)

### Scene selection
- Source: nuScenes v1.0-trainval, **chunks 9 and 10** (85 scenes each, in `scene.json` order). Condition from the scene description: night, rain (without night), otherwise clear.
- Chunk-to-blob-part mapping verified: all selected scenes have their CAM_FRONT and RADAR_FRONT files, 0 missing.
- Downloaded keyframe-only blobs: part 9 = 4.48 GB (full 31.21 GB), part 10 = 5.78 GB (full 38.87 GB).
- Train and val scenes come from the official nuScenes **train** split (val held out); test scenes come from the official nuScenes **val** split. Random seed 0. Exact scene lists: `docs/subset_scenes.json`.

Condition availability per 85-scene chunk (from scene descriptions):

| chunk | night | rain (any) |
|---|---|---|
| 1-4, 7 | 0 | 0 |
| 5 | 0 | 37 |
| 6 | 0 | 56 |
| 8 | 0 | 34 |
| 9 | 14 | 22 |
| 10 | 85 | 16 |

| chunk | split | clear-day | rain-day | night |
|---|---|---|---|---|
| 9 | train | 26 | 10 | 14 |
| 9 | val | 23 | 12 | 0 |
| 10 | train | 0 | 0 | 70 |
| 10 | val | 0 | 0 | 15 |

Selected scenes by chunk: chunk 9 = 47 scenes / 1900 samples; chunk 10 = 25 scenes / 1004 samples.

### Final dataset (cached at `data/processed/subset`, images resized to 640x360)

| split | scenes (clear / rain / night) | images | images (clear / rain / night) | car | pedestrian | truck |
|---|---|---|---|---|---|---|
| train | 40 (16 / 8 / 16) | 1611 | 646 / 321 / 644 | 4671 | 2143 | 333 |
| val | 10 (4 / 2 / 4) | 406 | 164 / 81 / 161 | 1207 | 560 | 78 |
| test | 22 (8 / 6 / 8) | 887 | 325 / 242 / 320 | 2284 | 491 | 440 |
| **total** | **72** | **2904** | 1135 / 644 / 1125 | 8162 | 3194 | 851 |

Notes for analysis:
- Night is about 39% of images (deliberately over-represented; nuScenes overall is far lower). Report overall and per-condition metrics.
- Test has proportionally more trucks and fewer pedestrians than train (truck: 440 test vs 333 train; pedestrian: 491 test vs 2143 train). Report per-class AP, and treat truck AP with caution (only 333 training boxes).
- Training dataset check: 1611 / 406 / 887 images load correctly through `FusionDataset`.

---

## 6. Compute (RTX 2050, 4 GB VRAM, Windows)

Timing run, YOLOv8n (2.69 M parameters, 6.9 GFLOPs), batch 16, 640 input, AMP on, 4 workers:

- Training: **43 s per epoch** (101 batches, 2.31 it/s); validation about 6 s.
- Peak GPU memory: **2.32 GB**.
- Inference: 3.1 ms per image (plus 2.8 ms preprocess, 3.2 ms postprocess).
- One epoch of pretrained weights gave mAP50 0.248 (recall 0.038). **Not a result**, timing only.
- Windows workaround: `CUDA error: resource already mapped` in the pin-memory thread; fixed with `$env:PIN_MEMORY = "False"`.

---

## 7. Results (fill from actual runs only)

### Clean-data detection (test split, mAP@0.5 unless stated)

| model | mAP@0.5 | mAP@0.5:0.95 | car | pedestrian | truck | latency (ms) | peak VRAM (GB) |
|---|---|---|---|---|---|---|---|
| B0 camera-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B1 early fusion | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B2 mid fusion | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B2-filtered | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B3 reliability gate (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Robustness grid (model x corruption x level), per-condition results, ablations
TBD (Phases 8 and 9).

---

## 8. Figures (`docs/ppt_assets`)

- `overlay_scene-0061.png`, `overlay_scene-1077.png`: radar points on camera (valid green, flagged red)
- `maps_scene-0061.png`, `maps_scene-1077.png`: camera, D, V and M maps
- `boxes_radar.png`: ground-truth 2D boxes with radar strips, day and night (mini val)
- `b0_stock_curves.png`: B0 training curves (to add after the run)
