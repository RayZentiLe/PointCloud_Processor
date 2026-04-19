import numpy as np
import open3d as o3d
from core.layer import PointCloudLayer, MeshLayer
from core.layer_manager import LayerManager


def export_point_cloud(layer: PointCloudLayer, path: str,
                       sublayer_name: str = None, binary: bool = True):
    if sublayer_name is not None:
        mask = LayerManager.get_sublayer_mask(layer, sublayer_name)
        points = layer.points[mask]
        colors = layer.colors[mask] if layer.colors is not None else None
        normals = layer.normals[mask] if layer.normals is not None else None
    else:
        points = layer.points
        colors = layer.colors
        normals = layer.normals

    if path.lower().endswith('.txt'):
        _export_txt(points, colors, normals, path)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        o3d.io.write_point_cloud(path, pcd,
                                 write_ascii=not binary,
                                 print_progress=False)


def export_mesh(layer: MeshLayer, path: str,
                sublayer_name: str = None, binary: bool = True):
    if sublayer_name is not None:
        mask = LayerManager.get_sublayer_mask(layer, sublayer_name)
        face_indices = np.where(mask)[0]
        faces = layer.faces[face_indices]

        used_verts = np.unique(faces.ravel())
        vert_map = np.full(layer.vertex_count, -1, dtype=np.int64)
        vert_map[used_verts] = np.arange(len(used_verts))

        vertices = layer.vertices[used_verts]
        faces = vert_map[faces]
        vertex_colors = (layer.vertex_colors[used_verts]
                         if layer.vertex_colors is not None else None)
        vertex_normals = (layer.vertex_normals[used_verts]
                          if layer.vertex_normals is not None else None)
    else:
        vertices = layer.vertices
        faces = layer.faces
        vertex_colors = layer.vertex_colors
        vertex_normals = layer.vertex_normals

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    if vertex_colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    if vertex_normals is not None:
        mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)

    o3d.io.write_triangle_mesh(path, mesh,
                               write_ascii=not binary,
                               print_progress=False)


def _export_txt(points, colors, normals, path):
    """Export point cloud to TXT format (space separated x y z [r g b])."""
    data = points
    if colors is not None:
        data = np.hstack([points, colors])
    np.savetxt(path, data, fmt='%.6f')