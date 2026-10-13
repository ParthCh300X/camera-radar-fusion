# Results log (copy numbers into slides from here)

## Radar statistics (nuScenes mini, RADAR_FRONT, 404 samples)
- Points per sample: ~120 unfiltered, ~60 with devkit default filter.
- Only 51.6% of returns have invalid_state == 0 (25,108 of 48,659); 90% have ambig_state == 3.
- z is always 0 (no elevation): vertical extent assumed (+2.0 m above, 0.5 m below).

## Projection check (depth-consistent in-box hit rate)
| points | real | sideways shift 150 px | depth shift 8 m |
|---|---|---|---|
| valid (M=1) | 11.1% | 6.0% | 1.4% |
| flagged (M=0) | 6.0% | 3.3% | 0.8% |

Takeaway: alignment is sound, and M separates better from worse radar (about 1.8x).

## Figures (docs/ppt_assets)
- overlay_scene-0061.png, overlay_scene-1077.png: radar points on camera (valid green / flagged red)
- maps_scene-0061.png, maps_scene-1077.png: camera, D, V, M maps