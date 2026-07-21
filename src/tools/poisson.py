import sys
import numpy as np
import open3d as o3d
from core.layer import MeshLayer, MaskGroup


def run_poisson(points, colors, depth, scale,
                density_quantile, linear_fit, progress_cb, cancel_cb):
    print(f"[Poisson] Starting: {len(points)} points, depth={depth}, "
          f"scale={scale}, dq={density_quantile}", file=sys.stderr)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    progress_cb(5)

    # Outlier removal
    if len(pcd.points) > 20:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=4.0)
    print(f"[Poisson] After outlier removal: {len(pcd.points)} points",
          file=sys.stderr)
    progress_cb(15)

    if len(pcd.points) < 10:
        raise ValueError(f"Too few points after outlier removal ({len(pcd.points)})")

    # Compute adaptive radius from bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    mean_extent = np.mean(extent)
    if mean_extent < 1e-10:
        raise ValueError("Point cloud has zero extent (all points identical?)")

    nn_radius = mean_extent * 0.02
    nn_max = 30
    print(f"[Poisson] Extent={extent}, normal radius={nn_radius:.6f}",
          file=sys.stderr)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=nn_radius, max_nn=nn_max))

    # Try consistent orientation
    try:
        pcd.orient_normals_consistent_tangent_plane(k=15)
    except Exception as e:
        print(f"[Poisson] orient_normals fallback: {e}", file=sys.stderr)
        normals = np.asarray(pcd.normals)
        if np.mean(normals[:, 2]) < 0:
            normals *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)

    progress_cb(35)
    print("[Poisson] Running Poisson reconstruction...", file=sys.stderr)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit)
    progress_cb(70)

    print(f"[Poisson] Raw mesh: {len(mesh.triangles)} triangles",
          file=sys.stderr)

    if not mesh.has_triangles() or len(mesh.triangles) == 0:
        raise ValueError("Poisson produced no triangles")

    # Density-based trimming
    densities = np.asarray(densities)
    if len(densities) > 0 and density_quantile > 0:
        thresh = np.quantile(densities, density_quantile)
        mask = densities < thresh
        mesh.remove_vertices_by_mask(mask)
        print(f"[Poisson] After density trim: {len(mesh.triangles)} triangles",
              file=sys.stderr)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    progress_cb(82)

    cluster_ids, counts, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    counts = np.asarray(counts)

    if len(counts) == 0:
        raise ValueError("No connected components found after Poisson reconstruction")

    largest_id = int(counts.argmax())
    largest_mask = cluster_ids == largest_id

    mg = MaskGroup(
        filter_name="mesh_filter",
        mask=largest_mask,
        positive_name="largest",
        negative_name="small_components",
        positive_visible=True,
        negative_visible=True,
        positive_color=None,
        negative_color=None,
    )
    mg.positive_color_mode = "solid"
    mg.negative_color_mode = "solid"
    mg.positive_solid_color = (0.2, 0.8, 0.2)
    mg.negative_solid_color = (1.0, 0.5, 0.0)

    n_components = len(counts)
    largest_count = int(counts[largest_id])
    small_count = int(np.sum(~largest_mask))
    print(f"[Poisson] Connected components: {n_components}, "
          f"largest={largest_count}, small={small_count}", file=sys.stderr)
    progress_cb(90)

    mesh.compute_vertex_normals()

    layer = MeshLayer(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.triangles, dtype=np.int64),
        vertex_normals=(np.asarray(mesh.vertex_normals, dtype=np.float64)
                        if mesh.has_vertex_normals() else None),
        vertex_colors=(np.asarray(mesh.vertex_colors, dtype=np.float64)
                       if mesh.has_vertex_colors() else None),
        modified=True,
    )
    print(f"[Poisson] Done: {layer.face_count} faces, "
          f"{layer.vertex_count} vertices", file=sys.stderr)
    progress_cb(100)
    return layer, mg, n_components, largest_count, small_count