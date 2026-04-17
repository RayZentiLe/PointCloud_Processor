import os
import numpy as np
import open3d as o3d
from core.layer import PointCloudLayer, MeshLayer


def load_file(filepath: str):
    """Load a point-cloud or mesh file.  Returns PointCloudLayer or MeshLayer."""
    ext = filepath.lower().rsplit(".", 1)[-1]

    # Try as mesh first for formats that can hold triangles
    if ext in ("ply", "obj", "stl", "off", "gltf", "glb"):
        try:
            mesh = o3d.io.read_triangle_mesh(filepath)
            if mesh.has_triangles() and len(mesh.triangles) > 0:
                return _mesh_to_layer(mesh, filepath)
        except Exception:
            pass

    # Fall back to point cloud
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd.has_points() and len(pcd.points) > 0:
        return _pcd_to_layer(pcd, filepath)

    raise ValueError(f"Could not read any geometry from {filepath}")


def _pcd_to_layer(pcd, filepath) -> PointCloudLayer:
    name = os.path.splitext(os.path.basename(filepath))[0]
    layer = PointCloudLayer(
        name=name,
        points=np.asarray(pcd.points, dtype=np.float64),
        source_path=filepath,
    )
    if pcd.has_colors():
        layer.colors = np.asarray(pcd.colors, dtype=np.float64)
    if pcd.has_normals():
        layer.normals = np.asarray(pcd.normals, dtype=np.float64)
    return layer


def _mesh_to_layer(mesh, filepath) -> MeshLayer:
    name = os.path.splitext(os.path.basename(filepath))[0]
    layer = MeshLayer(
        name=name,
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.triangles, dtype=np.int64),
        source_path=filepath,
    )
    if mesh.has_vertex_colors():
        layer.vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if mesh.has_vertex_normals():
        layer.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    return layer