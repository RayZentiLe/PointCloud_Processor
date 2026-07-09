import sys
import gc
import numpy as np
from scipy.spatial import cKDTree
from core.layer import MaskGroup


def _pca_debug(message):
    print(f"[PCA][DEBUG] {message}", file=sys.stderr, flush=True)


def run_pca_filter(points, indices, total_count,
                   radius, threshold, k_neighbors,
                   chunk_size, progress_cb, cancel_cb):
    _pca_debug(
        f"Starting: points={len(points)}, total_count={total_count}, "
        f"radius={radius}, threshold={threshold}, k={k_neighbors}, "
        f"chunk_size={chunk_size}, has_indices={indices is not None}"
    )

    if indices is None:
        indices = np.arange(total_count, dtype=np.int64)
        _pca_debug("No sublayer indices provided; using full layer indices")
    else:
        _pca_debug(f"Received sublayer indices: count={len(indices)}")

    if len(points) == 0:
        raise ValueError("No points to process")

    progress_cb(0)
    _pca_debug("Building KDTree")
    tree = cKDTree(points)
    _pca_debug("KDTree built successfully")
    kept_local = np.zeros(len(points), dtype=bool)

    n_chunks = max(1, len(points) // chunk_size)
    chunks = np.array_split(np.arange(len(points)), n_chunks)
    _pca_debug(f"Prepared {n_chunks} chunks for processing")

    for ci, chunk_idx in enumerate(chunks):
        # Check for cancellation
        if cancel_cb():
            _pca_debug(f"Cancellation requested before chunk {ci + 1}/{len(chunks)}; cleaning up")
            # Clean up memory
            del tree
            del kept_local
            del chunks
            del chunk_idx
            gc.collect()
            return None

        chunk_number = ci + 1
        chunk_start = int(chunk_idx[0]) if len(chunk_idx) else -1
        chunk_end = int(chunk_idx[-1]) if len(chunk_idx) else -1
        _pca_debug(
            f"Chunk {chunk_number}/{len(chunks)} start: size={len(chunk_idx)}, "
            f"index_range={chunk_start}-{chunk_end}"
        )

        chunk_pts = points[chunk_idx]
        _pca_debug(f"Chunk {chunk_number}: querying neighborhoods")
        neighborhoods = tree.query_ball_point(chunk_pts, radius)
        _pca_debug(f"Chunk {chunk_number}: neighborhoods ready")

        processed_neighbors = 0
        skipped_small = 0
        kept_in_chunk = 0

        for j, neighbors in enumerate(neighborhoods):
            if len(neighbors) < max(3, k_neighbors):
                skipped_small += 1
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
                        kept_in_chunk += 1
                processed_neighbors += 1
            except np.linalg.LinAlgError:
                _pca_debug(
                    f"Chunk {chunk_number}: LinAlgError at local point {j} "
                    f"with {len(neighbors)} neighbors"
                )
                continue

        pct = int((ci + 1) / len(chunks) * 100)
        progress_cb(pct)
        _pca_debug(
            f"Chunk {chunk_number}/{len(chunks)} done: processed={processed_neighbors}, "
            f"skipped_small={skipped_small}, kept={kept_in_chunk}, progress={pct}%"
        )

    full_mask = np.zeros(total_count, dtype=bool)
    _pca_debug("Combining local mask into full-size mask")
    full_mask[indices] = kept_local

    kept_n = int(np.sum(full_mask))
    reject_n = int(total_count - kept_n)
    _pca_debug(f"Done: kept={kept_n}, rejected={reject_n}")

    # Clean up intermediate data
    del tree
    del kept_local
    del chunks
    gc.collect()
    _pca_debug("Intermediate PCA data cleaned up")

    mg = MaskGroup(
        filter_name="pca_filter",
        mask=full_mask,
        positive_name="pca_kept",
        negative_name="pca_rejected",
        positive_visible=True,
        negative_visible=True,
        positive_color=None,
        negative_color=None,
    )
    # Set default colors for display (can be changed by user)
    mg.positive_color_mode = "solid"     # Green for kept by default
    mg.negative_color_mode = "solid"     # Red for rejected by default
    mg.positive_solid_color = (0.2, 0.8, 0.2)  # Green
    mg.negative_solid_color = (1.0, 0.3, 0.3)  # Red
    _pca_debug("MaskGroup created successfully")
    
    return mg