import numpy as np

SEED = 1234
# P columns: 0 u, 1 v_top, 2 v_bot, 3 depth, 4 radial velocity, 5 M, 6 pdh0, 7 rcs (pixels at 1600x900)

def _rng(P, salt):
    return np.random.default_rng((int(abs(float(P.sum())) * 1e3) + salt + SEED) % (2 ** 32))

def sparsity(P, keep, salt=1):
    if len(P) == 0:
        return P
    return P[_rng(P, salt).random(len(P)) < keep]

def noise(P, sigma_d, sigma_v, salt=2):
    if len(P) == 0:
        return P
    rng = _rng(P, salt); P = P.copy()
    P[:, 3] = np.maximum(P[:, 3] + rng.normal(0, sigma_d, len(P)), 0.5)
    P[:, 4] = P[:, 4] + rng.normal(0, sigma_v, len(P))
    return P

def misalign(P, shift_px, salt=3):
    if len(P) == 0:
        return P
    rng = _rng(P, salt); P = P.copy()
    dx = shift_px * rng.choice([-1.0, 1.0]); dy = 0.5 * shift_px * rng.choice([-1.0, 1.0])
    P[:, 0] += dx; P[:, 1] += dy; P[:, 2] += dy
    return P

def dropout(P, frac, salt=4):
    if len(P) == 0 or frac <= 0:
        return P
    if frac >= 1:
        return P[:0]
    a = _rng(P, salt).uniform(0, 1600 * (1 - frac))
    return P[(P[:, 0] < a) | (P[:, 0] > a + 1600 * frac)]

LEVELS = {
    "sparsity": {"75": 0.75, "50": 0.50, "25": 0.25, "10": 0.10},          # fraction of points kept
    "noise": {"low": (1.0, 0.5), "mid": (2.0, 1.0), "high": (4.0, 2.0)},   # sigma depth (m), sigma velocity (m/s)
    "misalign": {"small": 10, "medium": 30, "large": 60},                  # pixels at 1600x900
    "dropout": {"partial": 1 / 3, "severe": 2 / 3, "complete": 1.0},       # fraction of image width blanked
}
ALL_CONDITIONS = [("clean", None)] + [(k, l) for k in LEVELS for l in LEVELS[k]]

def make_corruption(kind, level):
    if kind == "clean":
        return lambda P: P
    v = LEVELS[kind][level]
    if kind == "sparsity":
        return lambda P: sparsity(P, v)
    if kind == "noise":
        return lambda P: noise(P, *v)
    if kind == "misalign":
        return lambda P: misalign(P, v)
    return lambda P: dropout(P, v)

def random_corruption(P):
    """Training-time augmentation: different every call. Half of the samples stay clean."""
    if np.random.rand() < 0.5:
        return P
    kind = list(LEVELS)[np.random.randint(len(LEVELS))]
    level = list(LEVELS[kind])[np.random.randint(len(LEVELS[kind]))]
    salt = int(np.random.randint(1, 10 ** 9))
    v = LEVELS[kind][level]
    if kind == "sparsity":
        return sparsity(P, v, salt)
    if kind == "noise":
        return noise(P, *v, salt=salt)
    if kind == "misalign":
        return misalign(P, v, salt)
    return dropout(P, v, salt)