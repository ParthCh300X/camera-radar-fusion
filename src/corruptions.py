import numpy as np

SEED = 1234
# P columns: 0 u, 1 v_top, 2 v_bot, 3 depth, 4 radial velocity, 5 M, 6 pdh0, 7 rcs, 8 bad (added by corruptions that mark points)

def _rng(P, salt):
    s = P.sum() if P.shape[1] == 8 else P[:, :8].sum()
    return np.random.default_rng((int(abs(float(s)) * 1e3) + salt + SEED) % (2 ** 32))

def _ensure(P):
    return P if P.shape[1] > 8 else np.concatenate([P, np.zeros((len(P), 1), P.dtype)], 1)

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

def misalign(P, shift_px, salt=3, mark=False):
    if len(P) == 0:
        return P
    rng = _rng(P, salt); P = P.copy()
    dx = shift_px * rng.choice([-1.0, 1.0]); dy = 0.5 * shift_px * rng.choice([-1.0, 1.0])
    P[:, 0] += dx; P[:, 1] += dy; P[:, 2] += dy
    if mark:
        P = _ensure(P); P[:, 8] = 1.0
    return P

def dropout(P, frac, salt=4):
    if len(P) == 0 or frac <= 0:
        return P
    if frac >= 1:
        return P[:0]
    a = _rng(P, salt).uniform(0, 1600 * (1 - frac))
    return P[(P[:, 0] < a) | (P[:, 0] > a + 1600 * frac)]

def clutter(P, n, salt=5):
    """Ghost returns: n false points. Each copies a real point's depth, velocity and height but sits at a random horizontal position."""
    P = _ensure(P)
    if n <= 0 or len(P) == 0:
        return P
    rng = _rng(P, salt)
    ghost = P[rng.integers(0, len(P), n)].copy()
    ghost[:, 0] = rng.uniform(0, 1600, n)
    ghost[:, 8] = 1.0
    return np.concatenate([P, ghost], 0)

LEVELS = {
    "sparsity": {"75": 0.75, "50": 0.50, "25": 0.25, "10": 0.10},
    "noise": {"low": (1.0, 0.5), "mid": (2.0, 1.0), "high": (4.0, 2.0)},
    "misalign": {"small": 10, "medium": 30, "large": 60},
    "dropout": {"partial": 1 / 3, "severe": 2 / 3, "complete": 1.0},
    "clutter": {"low": 10, "mid": 30, "high": 60},
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
    if kind == "clutter":
        return lambda P: clutter(P, v)
    return lambda P: dropout(P, v)

def random_corruption(P):
    """Training-time augmentation, different every call; half of the samples stay clean.
    Ghost points and shifts of 30 px or more are marked 'bad' (column 8) so the gate can be supervised on them."""
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
        return misalign(P, v, salt, mark=v >= 30)
    if kind == "clutter":
        return clutter(P, v, salt)
    return dropout(P, v, salt)