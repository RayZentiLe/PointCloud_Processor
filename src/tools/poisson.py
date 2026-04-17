import numpy as np
import open3d as o3d
from core.layer import MeshLayer


def run_poisson(points, colors, depth, scale,
                density_quantile, linear_fit, progress_cb):
    """Poisson surface reconstruction.  Returns an unnamed MeshLayer."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    progress_cb(5)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=4.0)
    progress_cb(15)

    if len(pcd.points) < 10:
        raise ValueError("Too few points after outlier removal")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05, max_nn=30))

    normals = np.asarray(pcd.normals)
    if np.mean(normals[:, 2]) < 0:
        normals *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    progress_cb(35)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit)
    progress_cb(70)

    if not mesh.has_triangles() or len(mesh.triangles) == 0:
        raise ValueError("Poisson produced no triangles")

    densities = np.asarray(densities)
    thresh = np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(densities < thresh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
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
    progress_cb(100)
    return layer