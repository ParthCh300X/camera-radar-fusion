import numpy as np

def rasterize(P, w=640, h=360, scale=0.4, half_w=2, d_max=100.0, v_max=15.0):
    """P: (N,8 or 9) [u, v_top, v_bot, depth, v_radial, M, pdh0, rcs, (bad)], pixels at 1600x900.
    Returns (4,h,w): D, V, M and BAD (1 where a point was corrupted on purpose; used only as a training target)."""
    out = np.zeros((4, h, w), np.float32)
    if len(P) == 0:
        return out
    for row in P[np.argsort(-P[:, 3])]:
        u, vt, vb, z, vr, m = row[:6]
        bad = row[8] if len(row) > 8 else 0.0
        x = int(round(u * scale))
        y1, y2 = int(max(0, round(vt * scale))), int(min(h - 1, round(vb * scale)))
        x1, x2 = max(0, x - half_w), min(w, x + half_w + 1)
        if y2 < y1 or x1 >= x2:
            continue
        out[0, y1:y2 + 1, x1:x2] = min(z / d_max, 1.0)
        out[1, y1:y2 + 1, x1:x2] = np.clip(vr / v_max, -1.0, 1.0)
        out[2, y1:y2 + 1, x1:x2] = m
        out[3, y1:y2 + 1, x1:x2] = bad
    return out