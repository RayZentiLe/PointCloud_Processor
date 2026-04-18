import uuid
import numpy as np
from enum import Enum, auto


class LayerType(Enum):
    POINT_CLOUD = auto()
    MESH = auto()


class MaskGroup:
    """Binary mask + sublayer names for a point cloud or mesh."""
    def __init__(self, filter_name=None, name=None, mask=None,
                 positive_name="kept", negative_name="rejected",
                 positive_visible=True, negative_visible=True,
                 positive_color=None, negative_color=None):
        self.id = str(uuid.uuid4())
        self.name = name or filter_name or "mask"
        self.filter_name = filter_name or name or "mask"
        self.mask = mask
        self.positive_name = positive_name
        self.negative_name = negative_name
        self.positive_visible = positive_visible
        self.negative_visible = negative_visible
        self.positive_color = positive_color
        self.negative_color = negative_color

    @property
    def positive_count(self):
        return int(self.mask.sum()) if self.mask is not None else 0

    @property
    def negative_count(self):
        return int((~self.mask).sum()) if self.mask is not None else 0


class PointCloudLayer:
    def __init__(self, name="Point Cloud", points=None, colors=None, normals=None,
                 source_path=None, modified=False):
        self.id = str(uuid.uuid4())
        self.name = name
        self.points = (np.asarray(points, dtype=np.float32)
                       if points is not None else np.empty((0, 3), dtype=np.float32))
        self.colors = (np.asarray(colors, dtype=np.float32)
                       if colors is not None else None)
        self.normals = (np.asarray(normals, dtype=np.float32)
                        if normals is not None else None)
        self.source_path = source_path
        self.modified = modified
        self.visible = True
        self.display_color = None
        self.mask_groups = []
        self.render_props = {
            "color_mode": "original",        # "original" | "solid" | "height_gradient"
            "solid_color": [0.7, 0.7, 0.7],  # RGB 0-1, used when color_mode == "solid"
            "gradient_axis": "Z",             # "X"|"Y"|"Z"|"-X"|"-Y"|"-Z"
            "render_mode": "points",          # "points" | "decoration"
            "point_size": 2,                  # 1-20
        }

    @property
    def point_count(self):
        return len(self.points)


class MeshLayer:
    def __init__(self, name="Mesh", vertices=None, faces=None, vertex_colors=None,
                 face_normals=None, vertex_normals=None,
                 source_path=None, modified=False):
        self.id = str(uuid.uuid4())
        self.name = name
        self.vertices = (np.asarray(vertices, dtype=np.float32)
                         if vertices is not None else np.empty((0, 3), dtype=np.float32))
        self.faces = (np.asarray(faces, dtype=np.int32)
                      if faces is not None else np.empty((0, 3), dtype=np.int32))
        self.vertex_colors = (np.asarray(vertex_colors, dtype=np.float32)
                              if vertex_colors is not None else None)
        self.face_normals = (np.asarray(face_normals, dtype=np.float32)
                             if face_normals is not None else None)
        self.vertex_normals = (np.asarray(vertex_normals, dtype=np.float32)
                               if vertex_normals is not None else None)
        self.source_path = source_path
        self.modified = modified
        self.visible = True
        self.display_color = None
        self.mask_groups = []
        self.render_props = {
            "color_mode": "original",        # "original" | "solid" | "height_gradient"
            "solid_color": [0.7, 0.7, 0.7],  # RGB 0-1, used when color_mode == "solid"
            "gradient_axis": "Z",             # "X"|"Y"|"Z"|"-X"|"-Y"|"-Z"
            "render_mode": "normal",          # "normal" | "fancy"
        }

    @property
    def vertex_count(self):
        return len(self.vertices)

    @property
    def face_count(self):
        return len(self.faces)