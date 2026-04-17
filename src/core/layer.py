import numpy as np
import uuid
from dataclasses import dataclass, field
from enum import Enum


class LayerType(Enum):
    POINT_CLOUD = "point_cloud"
    MESH = "mesh"


@dataclass
class MaskGroup:
    """One filter operation produces one MaskGroup with two complementary sides.

    For PointCloudLayer the mask is bool[N] over points.
    For MeshLayer the mask is bool[F] over faces.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filter_name: str = ""
    mask: np.ndarray = None
    positive_name: str = ""
    negative_name: str = ""
    positive_visible: bool = True
    negative_visible: bool = True
    positive_color: tuple = None  # (r,g,b) 0-1 or None
    negative_color: tuple = None  # (r,g,b) 0-1 or None

    @property
    def positive_count(self) -> int:
        if self.mask is not None:
            return int(np.sum(self.mask))
        return 0

    @property
    def negative_count(self) -> int:
        if self.mask is not None:
            return int(np.sum(~self.mask))
        return 0


@dataclass
class PointCloudLayer:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    points: np.ndarray = None       # (N, 3) float64
    colors: np.ndarray = None       # (N, 3) float64  0-1
    normals: np.ndarray = None      # (N, 3) float64
    visible: bool = True
    display_color: tuple = None     # (r,g,b) 0-1 or None
    mask_groups: list = field(default_factory=list)
    source_path: str = None
    modified: bool = False

    @property
    def point_count(self) -> int:
        if self.points is not None:
            return len(self.points)
        return 0

    @property
    def layer_type(self) -> LayerType:
        return LayerType.POINT_CLOUD


@dataclass
class MeshLayer:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    vertices: np.ndarray = None      # (V, 3) float64
    faces: np.ndarray = None         # (F, 3) int64
    vertex_colors: np.ndarray = None # (V, 3) float64  0-1
    vertex_normals: np.ndarray = None
    visible: bool = True
    display_color: tuple = None
    mask_groups: list = field(default_factory=list)
    source_path: str = None
    modified: bool = False

    @property
    def face_count(self) -> int:
        if self.faces is not None:
            return len(self.faces)
        return 0

    @property
    def vertex_count(self) -> int:
        if self.vertices is not None:
            return len(self.vertices)
        return 0

    @property
    def layer_type(self) -> LayerType:
        return LayerType.MESH