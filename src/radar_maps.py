import numpy as np

def rasterize(P, w=640, h=360, scale=0.4, half_w=2, d_max=100.0, v_max=15.0):
    """P: (N,8) [u, v_top, v_bot, depth, v_radial, M, pdh0, rcs], full-res (1600x900) pixels.
    Returns (3,h,w): D = depth/d_max, V = radial velocity/v_max in [-1,1], M = validity (1/0).
    Nearest point wins where strips overlap. Empty pixels are 0 in all channels."""
    out = np.zeros((3, h, w), np.float32)
    if len(P) == 0:
        return out
    for u, vt, vb, z, vr, m, _, _ in P[np.argsort(-P[:, 3])]:   # far first, near drawn last
        x = int(round(u * scale))
        y1, y2 = int(max(0, round(vt * scale))), int(min(h - 1, round(vb * scale)))
        x1, x2 = max(0, x - half_w), min(w, x + half_w + 1)
        if y2 < y1 or x1 >= x2:
            continue
        out[0, y1:y2 + 1, x1:x2] = min(z / d_max, 1.0)
        out[1, y1:y2 + 1, x1:x2] = np.clip(vr / v_max, -1.0, 1.0)
        out[2, y1:y2 + 1, x1:x2] = m
    return out