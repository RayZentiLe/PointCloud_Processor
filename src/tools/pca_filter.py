import sys
import numpy as np
from scipy.spatial import cKDTree
from core.layer import MaskGroup


def run_pca_filter(points, indices, total_count,
                   radius, threshold, k_neighbors,
                   chunk_size, progress_cb):
    print(f"[PCA] Starting: {len(points)} points, radius={radius}, "
          f"threshold={threshold}, k={k_neighbors}", file=sys.stderr)

    if indices is None:
        indices = np.arange(total_count, dtype=np.int64)

    if len(points) == 0:
        raise ValueError("No points to process")

    progress_cb(0)
    print("[PCA] Building KDTree...", file=sys.stderr)
    tree = cKDTree(points)
    kept_local = np.zeros(len(points), dtype=bool)

    n_chunks = max(1, len(points) // chunk_size)
    chunks = np.array_split(np.arange(len(points)), n_chunks)
    print(f"[PCA] Processing {n_chunks} chunks...", file=sys.stderr)

    for ci, chunk_idx in enumerate(chunks):
        chunk_pts = points[chunk_idx]
        neighborhoods = tree.query_ball_point(chunk_pts, radius)

        for j, neighbors in enumerate(neighborhoods):
            if len(neighbors) < max(3, k_neighbors):
                continue
            neighbor_pts = points[neighbors]
            cov = np.cov(neighbor_pts, rowvar=False)
            try:
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]
                if eigenvalues[0] > 1e-10:
                    planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
                    if planarity > threshold:
                        kept_local[chunk_idx[j]] = True
            except np.linalg.LinAlgError:
                continue

        pct = int((ci + 1) / len(chunks) * 100)
        progress_cb(pct)
        if ci % max(1, len(chunks) // 10) == 0:
            print(f"[PCA] {pct}%", file=sys.stderr)

    full_mask = np.zeros(total_count, dtype=bool)
    full_mask[indices] = kept_local

    kept_n = int(np.sum(full_mask))
    reject_n = int(total_count - kept_n)
    print(f"[PCA] Done: kept={kept_n}, rejected={reject_n}", file=sys.stderr)

    return MaskGroup(
        filter_name="pca_filter",
        mask=full_mask,
        positive_name="pca_kept",
        negative_name="pca_rejected",
        positive_visible=True,
        negative_visible=True,
        positive_color=None,
        negative_color=(1.0, 0.3, 0.3),
    )