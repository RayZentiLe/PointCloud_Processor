import numpy as np
import open3d as o3d
from core.layer import MaskGroup


def run_mesh_filter(vertices, faces, face_indices, total_face_count,
                    progress_cb):
    """Keep-largest-component filter for meshes.

    Parameters
    ----------
    vertices : (V, 3) full vertex array
    faces : (F, 3) full face array
    face_indices : which faces to evaluate (None = all)
    total_face_count : len(full faces)

    Returns
    -------
    (MaskGroup, n_components, largest_count, small_count)
    """
    if face_indices is None:
        face_indices = np.arange(total_face_count, dtype=np.int64)

    subset_faces = faces[face_indices]

    # remap vertices so open3d gets a compact mesh
    used_vert_ids = np.unique(subset_faces.ravel())
    vert_map = np.full(len(vertices), -1, dtype=np.int64)
    vert_map[used_vert_ids] = np.arange(len(used_vert_ids))
    remapped = vert_map[subset_faces]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices[used_vert_ids])
    mesh.triangles = o3d.utility.Vector3iVector(remapped.astype(np.int32))
    progress_cb(20)

    cluster_ids, counts, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    counts = np.asarray(counts)
    progress_cb(60)

    if len(counts) == 0:
        raise ValueError("No connected components found")

    largest_id = int(counts.argmax())
    local_mask = cluster_ids == largest_id

    full_mask = np.zeros(total_face_count, dtype=bool)
    full_mask[face_indices] = local_mask
    progress_cb(90)

    n_components = len(counts)
    largest_count = int(counts[largest_id])
    small_count = int(np.sum(~local_mask))
    progress_cb(100)

    mg = MaskGroup(
        filter_name="mesh_filter",
        mask=full_mask,
        positive_name="largest",
        negative_name="small_components",
        positive_visible=True,
        negative_visible=True,
        positive_color=None,
        negative_color=(1.0, 0.5, 0.0),
    )
    return mg, n_components, largest_count, small_count