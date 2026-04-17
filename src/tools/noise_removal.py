import numpy as np
from scipy.spatial import cKDTree
from core.layer import MaskGroup


def run_noise_removal(points, indices, total_count,
                      mesh_vertices, threshold, progress_cb):
    """Point-to-mesh-vertex distance filter.

    Returns a MaskGroup (positive = clean, negative = noise).
    """
    if indices is None:
        indices = np.arange(total_count, dtype=np.int64)

    progress_cb(10)
    tree = cKDTree(mesh_vertices)
    progress_cb(30)

    distances, _ = tree.query(points)
    progress_cb(80)

    clean_local = distances < threshold

    full_mask = np.zeros(total_count, dtype=bool)
    full_mask[indices] = clean_local
    progress_cb(100)

    return MaskGroup(
        filter_name="noise_removal",
        mask=full_mask,
        positive_name="clean",
        negative_name="noise",
        positive_visible=True,
        negative_visible=True,
        positive_color=None,
        negative_color=(1.0, 0.6, 0.0),
    )