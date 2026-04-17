import numpy as np
from PySide6.QtCore import QObject, Signal
from core.layer import PointCloudLayer, MeshLayer, MaskGroup, LayerType


class LayerManager(QObject):
    layer_added = Signal(str)
    layer_removed = Signal(str)
    layer_modified = Signal(str)
    layer_renamed = Signal(str)
    visibility_changed = Signal(str)
    mask_added = Signal(str, str)       # layer_id, mask_group_id
    mask_removed = Signal(str, str)     # layer_id, mask_group_id
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._point_clouds: dict[str, PointCloudLayer] = {}
        self._meshes: dict[str, MeshLayer] = {}
        self._selected_layer_id: str | None = None
        self._selected_sublayer_name: str | None = None

    # ── properties ───────────────────────────────────────────────

    @property
    def point_clouds(self) -> dict[str, PointCloudLayer]:
        return dict(self._point_clouds)

    @property
    def meshes(self) -> dict[str, MeshLayer]:
        return dict(self._meshes)

    @property
    def selected_layer_id(self):
        return self._selected_layer_id

    @property
    def selected_sublayer_name(self):
        return self._selected_sublayer_name

    # ── queries ──────────────────────────────────────────────────

    def get_layer(self, layer_id):
        if layer_id in self._point_clouds:
            return self._point_clouds[layer_id]
        if layer_id in self._meshes:
            return self._meshes[layer_id]
        return None

    def get_all_layers(self):
        layers = []
        layers.extend(self._point_clouds.values())
        layers.extend(self._meshes.values())
        return layers

    def get_selected_layer(self):
        if self._selected_layer_id:
            return self.get_layer(self._selected_layer_id)
        return None

    def first_point_cloud(self):
        """Return the first point cloud or None."""
        for pc in self._point_clouds.values():
            return pc
        return None

    def first_mesh(self):
        """Return the first mesh or None."""
        for m in self._meshes.values():
            return m
        return None

    # ── add / remove ─────────────────────────────────────────────

    def add_point_cloud(self, layer: PointCloudLayer):
        layer.name = self._ensure_unique_layer_name(layer.name)
        self._point_clouds[layer.id] = layer
        self.layer_added.emit(layer.id)

    def add_mesh(self, layer: MeshLayer):
        layer.name = self._ensure_unique_layer_name(layer.name)
        self._meshes[layer.id] = layer
        self.layer_added.emit(layer.id)

    def remove_layer(self, layer_id):
        removed = False
        if layer_id in self._point_clouds:
            del self._point_clouds[layer_id]
            removed = True
        elif layer_id in self._meshes:
            del self._meshes[layer_id]
            removed = True
        if removed:
            if self._selected_layer_id == layer_id:
                self._selected_layer_id = None
                self._selected_sublayer_name = None
            self.layer_removed.emit(layer_id)
        return removed

    # ── rename ───────────────────────────────────────────────────

    def rename_layer(self, layer_id, new_name):
        layer = self.get_layer(layer_id)
        if layer is None:
            return False, "Layer not found"
        for other in self.get_all_layers():
            if other.id != layer_id and other.name == new_name:
                return False, f"Name '{new_name}' is already in use by another layer"
        layer.name = new_name
        self.layer_renamed.emit(layer_id)
        return True, ""

    def rename_sublayer(self, layer_id, mask_group_id, is_positive, new_name):
        layer = self.get_layer(layer_id)
        if layer is None:
            return False, "Layer not found"
        mg = self._find_mask_group(layer, mask_group_id)
        if mg is None:
            return False, "Mask group not found"
        # uniqueness inside this layer
        for m in layer.mask_groups:
            if m.id == mask_group_id:
                other_side = m.negative_name if is_positive else m.positive_name
                if other_side == new_name:
                    return False, f"Name '{new_name}' already in use"
                continue
            if m.positive_name == new_name or m.negative_name == new_name:
                return False, f"Name '{new_name}' already in use"
        if is_positive:
            mg.positive_name = new_name
        else:
            mg.negative_name = new_name
        self.layer_modified.emit(layer_id)
        return True, ""

    # ── visibility ───────────────────────────────────────────────

    def set_layer_visibility(self, layer_id, visible):
        layer = self.get_layer(layer_id)
        if layer is not None:
            layer.visible = visible
            self.visibility_changed.emit(layer_id)

    def set_sublayer_visibility(self, layer_id, mask_group_id, is_positive, visible):
        layer = self.get_layer(layer_id)
        if layer is None:
            return
        mg = self._find_mask_group(layer, mask_group_id)
        if mg is None:
            return
        if is_positive:
            mg.positive_visible = visible
        else:
            mg.negative_visible = visible
        self.visibility_changed.emit(layer_id)

    # ── color ────────────────────────────────────────────────────

    def set_layer_color(self, layer_id, color):
        layer = self.get_layer(layer_id)
        if layer is not None:
            layer.display_color = color
            self.layer_modified.emit(layer_id)

    def set_sublayer_color(self, layer_id, mask_group_id, is_positive, color):
        layer = self.get_layer(layer_id)
        if layer is None:
            return
        mg = self._find_mask_group(layer, mask_group_id)
        if mg is None:
            return
        if is_positive:
            mg.positive_color = color
        else:
            mg.negative_color = color
        self.layer_modified.emit(layer_id)

    # ── masks ────────────────────────────────────────────────────

    def add_mask_group(self, layer_id, mask_group: MaskGroup):
        layer = self.get_layer(layer_id)
        if layer is None:
            return
        mask_group.positive_name = self._ensure_unique_sublayer_name(
            layer, mask_group.positive_name)
        mask_group.negative_name = self._ensure_unique_sublayer_name(
            layer, mask_group.negative_name)
        layer.mask_groups.append(mask_group)
        layer.modified = True
        self.mask_added.emit(layer_id, mask_group.id)

    def remove_mask_group(self, layer_id, mask_group_id):
        layer = self.get_layer(layer_id)
        if layer is None:
            return False
        for i, mg in enumerate(layer.mask_groups):
            if mg.id == mask_group_id:
                layer.mask_groups.pop(i)
                self.mask_removed.emit(layer_id, mask_group_id)
                return True
        return False

    # ── selection ────────────────────────────────────────────────

    def set_selection(self, layer_id, sublayer_name=None):
        changed = (self._selected_layer_id != layer_id or
                   self._selected_sublayer_name != sublayer_name)
        self._selected_layer_id = layer_id
        self._selected_sublayer_name = sublayer_name
        if changed:
            self.selection_changed.emit()

    # ── static helpers ───────────────────────────────────────────

    @staticmethod
    def get_sublayer_mask(layer, sublayer_name) -> np.ndarray:
        for mg in layer.mask_groups:
            if mg.positive_name == sublayer_name:
                return mg.mask.copy()
            if mg.negative_name == sublayer_name:
                return ~mg.mask
        if isinstance(layer, PointCloudLayer):
            n = layer.point_count
        elif isinstance(layer, MeshLayer):
            n = layer.face_count
        else:
            n = 0
        return np.ones(n, dtype=bool)

    @staticmethod
    def get_sublayer_mask_group_info(layer, sublayer_name):
        for mg in layer.mask_groups:
            if mg.positive_name == sublayer_name:
                return mg, True
            if mg.negative_name == sublayer_name:
                return mg, False
        return None, None

    # ── private ──────────────────────────────────────────────────

    def _find_mask_group(self, layer, mask_group_id):
        for mg in layer.mask_groups:
            if mg.id == mask_group_id:
                return mg
        return None

    def _ensure_unique_layer_name(self, name):
        existing = {l.name for l in self.get_all_layers()}
        if name not in existing:
            return name
        i = 2
        while f"{name}_{i}" in existing:
            i += 1
        return f"{name}_{i}"

    def _ensure_unique_sublayer_name(self, layer, name):
        existing = set()
        for mg in layer.mask_groups:
            existing.add(mg.positive_name)
            existing.add(mg.negative_name)
        if name not in existing:
            return name
        base = name
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"