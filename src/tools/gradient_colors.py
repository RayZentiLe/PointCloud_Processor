"""
Vectorised height-to-colour mapping.

Red (high) ─► Purple / deep blue (low) via HSV,
matching the reference Open3D script's colour scheme.
"""

import numpy as np


def compute_gradient_colors(values: np.ndarray,
                            min_val: float,
                            max_val: float) -> np.ndarray:
    """
    Map an array of scalar values to an (N, 3) float32 RGB array.

    * ``t = 1`` → red   (hue 0°)
    * ``t = 0`` → purple (hue ≈ 300°)
    * Values outside [min_val, max_val] are clamped to the
      nearest endpoint colour (manual-mode behaviour).

    Parameters
    ----------
    values : ndarray, shape (N,)
        Raw coordinate values (e.g. Z of every point).
    min_val, max_val : float
        The range that maps onto the full colour ramp.

    Returns
    -------
    colors : ndarray, shape (N, 3), dtype float32, in [0, 1]
    """
    n = len(values)
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)

    if max_val == min_val:
        return np.full((n, 3), 0.5, dtype=np.float32)

    t = (values - min_val) / (max_val - min_val)
    t = np.clip(t, 0.0, 1.0)           # clamp beyond range

    # hue: 0 (red) at t=1  →  0.83 (purple) at t=0
    h = 0.83 * (1.0 - t)
    s = np.ones(n, dtype=np.float64)
    v = np.ones(n, dtype=np.float64)

    r, g, b = _hsv_to_rgb_vec(h, s, v)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ── internal ─────────────────────────────────────────────────────

def _hsv_to_rgb_vec(h, s, v):
    """Vectorised HSV → RGB (all inputs same-shape arrays)."""
    i = (h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    # branchless select
    conds = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(conds, [v, q, p, p, t, v])
    g = np.select(conds, [t, v, v, q, p, p])
    b = np.select(conds, [p, p, t, v, v, q])
    return r, g, b