# nuScenes radar notes (RADAR_FRONT, mini)

- 18 fields per point; z is always 0 (no elevation info).
- ~120 points/sample unfiltered, ~60 with devkit default filter (invalid_state==0, ambig_state==3 etc.).
- Velocity: use vx_comp, vy_comp (ego-motion compensated), not vx, vy.
- is_quality_valid is always 1: unused.
- Validity mask M = (invalid_state == 0) AND (ambig_state == 3).
- pdh0 = false-alarm probability level (higher = worse); mapping to be verified in docs.
- Pipeline keeps ALL points; M marks reliability. Control B2-filtered uses devkit-filtered points only.
- Height: radar has none; vertical extension range to be fixed in Phase 2 (documented assumption).
- Sweeps: start with 1 sweep; test 3-5 later.
- Mini scenes: 7 day, 3 night (1077, 1094 after rain, 1100).
