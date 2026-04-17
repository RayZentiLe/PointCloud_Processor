import numpy as np
from scipy.spatial import cKDTree
from core.layer import MaskGroup


def run_pca_filter(points, indices, total_count,
                   radius, threshold, k_neighbors,
                   chunk_size, progress_cb):
    """PCA planarity filter for point clouds.

    Parameters
    ----------
    points : (M, 3) subset to evaluate
    indices : int array mapping subset back to parent, or None for all
    total_count : length of parent point array
    radius : neighbourhood search radius
    threshold : planarity threshold (0-1)
    k_neighbors : minimum neighbours required
    chunk_size : points per batch
    progress_cb : callable(int 0-100)

    Returns
    -------
    MaskGroup
    """
    if indices is None:
        indices = np.arange(total_count, dtype=np.int64)

    progress_cb(0)
    tree = cKDTree(points)
    kept_local = np.zeros(len(points), dtype=bool)

    n_chunks = max(1, len(points) // chunk_size)
    chunks = np.array_split(np.arange(len(points)), n_chunks)

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

        progress_cb(int((ci + 1) / len(chunks) * 100))

    full_mask = np.zeros(total_count, dtype=bool)
    full_mask[indices] = kept_local

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