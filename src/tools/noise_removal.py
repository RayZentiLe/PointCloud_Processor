import sys
import numpy as np
from scipy.spatial import cKDTree
from core.layer import MaskGroup


def run_noise_removal(points, indices, total_count,
                      mesh_vertices, threshold, progress_cb):
    print(f"[NoiseRemoval] Starting: {len(points)} points, "
          f"{len(mesh_vertices)} mesh verts, threshold={threshold}",
          file=sys.stderr)

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

    clean_n = int(np.sum(full_mask))
    noise_n = int(total_count - clean_n)
    print(f"[NoiseRemoval] Done: clean={clean_n}, noise={noise_n}",
          file=sys.stderr)
    progress_cb(100)

    mg = MaskGroup(
        filter_name="noise_removal",
        mask=full_mask,
        positive_name="clean",
        negative_name="noise",
        positive_visible=True,
        negative_visible=True,
        positive_color=None,
        negative_color=None,
    )
    # Set default colors for display (can be changed by user)
    mg.positive_color_mode = "solid"     # Green for clean by default
    mg.negative_color_mode = "solid"     # Orange for noise by default
    mg.positive_solid_color = (0.2, 0.8, 0.2)  # Green
    mg.negative_solid_color = (1.0, 0.6, 0.0)  # Orange
    
    return mg