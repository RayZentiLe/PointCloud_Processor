import numpy as np
from PySide6.QtCore import QObject, Signal
from core.layer import PointCloudLayer, MeshLayer, MaskGroup, LayerType, clone_layer


class LayerManager(QObject):
    layer_added = Signal(str)
    layer_removed = Signal(str)
    layer_modified = Signal(str)
    layer_renamed = Signal(str)
    visibility_changed = Signal(str)
    mask_added = Signal(str, str)       # layer_id, mask_group_id
    mask_removed = Signal(str, str)     # layer_id, mask_group_id

    selection_changed = Signal(object)
    layer_order_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._point_clouds: dict[str, PointCloudLayer] = {}
        self._meshes: dict[str, MeshLayer] = {}
        self._selected_layer_id: str | None = None
        self._selected_sublayer_name: str | None = None
        self._selected_layer_ids: list[str] = []

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

    @property
    def selected_layer_ids(self):
        return list(self._selected_layer_ids)

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

    def reorder_layer(self, layer_id, target_layer_id, place_after=False):
        source = self.get_layer(layer_id)
        target = self.get_layer(target_layer_id)
        if source is None or target is None:
            return False
        if type(source) is not type(target):
            return False

        container = self._point_clouds if isinstance(source, PointCloudLayer) else self._meshes
        keys = list(container.keys())
        if layer_id not in keys or target_layer_id not in keys or layer_id == target_layer_id:
            return False

        keys.remove(layer_id)
        target_index = keys.index(target_layer_id)
        insert_index = target_index + (1 if place_after else 0)
        keys.insert(insert_index, layer_id)
        reordered = {key: container[key] for key in keys}
        container.clear()
        container.update(reordered)
        self.layer_order_changed.emit()
        return True

    def reorder_layers(self, layer_ids, target_layer_id, place_after=False):
        if not layer_ids:
            return False
        target = self.get_layer(target_layer_id)
        if target is None:
            return False

        unique_ids = []
        seen = set()
        for layer_id in layer_ids:
            if layer_id in seen:
                continue
            layer = self.get_layer(layer_id)
            if layer is None or type(layer) is not type(target):
                return False
            unique_ids.append(layer_id)
            seen.add(layer_id)

        if target_layer_id in seen:
            return False

        container = self._point_clouds if isinstance(target, PointCloudLayer) else self._meshes
        keys = list(container.keys())
        if any(layer_id not in keys for layer_id in unique_ids):
            return False

        remaining = [key for key in keys if key not in seen]
        target_index = remaining.index(target_layer_id)
        insert_index = target_index + (1 if place_after else 0)
        new_keys = remaining[:insert_index] + unique_ids + remaining[insert_index:]
        reordered = {key: container[key] for key in new_keys}
        container.clear()
        container.update(reordered)
        self.layer_order_changed.emit()
        return True

    def combine_layers(self, layer_ids, name=None):
        unique_ids = []
        seen = set()
        for layer_id in layer_ids:
            if layer_id in seen:
                continue
            layer = self.get_layer(layer_id)
            if layer is None:
                return None, "Layer not found"
            unique_ids.append(layer_id)
            seen.add(layer_id)

        if len(unique_ids) < 2:
            return None, "Select at least two layers."

        layers = [self.get_layer(layer_id) for layer_id in unique_ids]
        first = layers[0]
        if any(type(layer) is not type(first) for layer in layers[1:]):
            return None, "Can only combine same-type layers."

        requested_name = (name or "").strip()
        if requested_name:
            result_name = requested_name
        else:
            result_name = self._ensure_unique_layer_name("combined")

        if isinstance(first, PointCloudLayer):
            points = np.vstack([layer.points for layer in layers])
            combined = PointCloudLayer(name=result_name, points=points, modified=True)
            if any(layer.colors is not None for layer in layers):
                color_chunks = []
                for layer in layers:
                    if layer.colors is not None:
                        color_chunks.append(layer.colors)
                    else:
                        color_chunks.append(np.full((len(layer.points), 3), 0.5, dtype=np.float32))
                combined.colors = np.vstack(color_chunks)
            if all(layer.normals is not None for layer in layers):
                combined.normals = np.vstack([layer.normals for layer in layers])
            self.add_point_cloud(combined)
            for layer_id in unique_ids:
                self.remove_layer(layer_id)
            return combined, ""

        if isinstance(first, MeshLayer):
            vertex_chunks = []
            face_chunks = []
            vertex_offset = 0
            for layer in layers:
                vertex_chunks.append(layer.vertices)
                face_chunks.append(layer.faces + vertex_offset)
                vertex_offset += len(layer.vertices)
            combined = MeshLayer(
                name=result_name,
                vertices=np.vstack(vertex_chunks),
                faces=np.vstack(face_chunks),
                modified=True,
            )
            if all(layer.vertex_colors is not None for layer in layers):
                combined.vertex_colors = np.vstack([layer.vertex_colors for layer in layers])
            if all(layer.face_normals is not None for layer in layers):
                combined.face_normals = np.vstack([layer.face_normals for layer in layers])
            if all(layer.vertex_normals is not None for layer in layers):
                combined.vertex_normals = np.vstack([layer.vertex_normals for layer in layers])
            self.add_mesh(combined)
            for layer_id in unique_ids:
                self.remove_layer(layer_id)
            return combined, ""

        return None, "Can only combine same-type layers."

    def copy_layers(self, layer_ids):
        copied_ids = []
        seen = set()
        for layer_id in layer_ids:
            if layer_id in seen:
                continue
            layer = self.get_layer(layer_id)
            if layer is None:
                continue
            seen.add(layer_id)
            copied_name = self._ensure_unique_layer_name(f"{layer.name} copy")
            copied_layer = clone_layer(layer, name=copied_name)
            if isinstance(copied_layer, PointCloudLayer):
                self.add_point_cloud(copied_layer)
            elif isinstance(copied_layer, MeshLayer):
                self.add_mesh(copied_layer)
            copied_ids.append(copied_layer.id)
        return copied_ids

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

    # ── colour ───────────────────────────────────────────────────

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

    # ── render properties ────────────────────────────────────────

    def set_render_prop(self, layer_id, key, value):
        """Update one key in layer.render_props and notify."""
        layer = self.get_layer(layer_id)
        if layer is not None:
            layer.render_props[key] = value
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

    def transfer_points_between_sublayers(self, layer_id, from_sublayer_name, to_sublayer_name, indices):
        layer = self.get_layer(layer_id)
        if not isinstance(layer, PointCloudLayer):
            return False, "Point cloud layer not found"
        if indices is None:
            return False, "No indices provided"

        point_indices = np.asarray(indices, dtype=np.int64).ravel()
        if point_indices.size == 0:
            return True, ""

        valid_mask = (point_indices >= 0) & (point_indices < layer.point_count)
        point_indices = np.unique(point_indices[valid_mask])
        if point_indices.size == 0:
            return True, ""

        from_info = self.get_sublayer_mask_group_info(layer, from_sublayer_name)
        to_info = self.get_sublayer_mask_group_info(layer, to_sublayer_name)
        from_mg, from_is_positive = from_info
        to_mg, to_is_positive = to_info

        if from_mg is None:
            return False, f"Source sublayer '{from_sublayer_name}' not found"
        if to_mg is None:
            return False, f"Target sublayer '{to_sublayer_name}' not found"
        if from_mg is to_mg and from_is_positive == to_is_positive:
            return False, "Source and target sublayers must be different"

        from_membership = from_mg.mask if from_is_positive else ~from_mg.mask
        movable_indices = point_indices[from_membership[point_indices]]
        if movable_indices.size == 0:
            return True, ""

        if from_is_positive:
            from_mg.mask[movable_indices] = False
        else:
            from_mg.mask[movable_indices] = True

        if to_is_positive:
            to_mg.mask[movable_indices] = True
        else:
            to_mg.mask[movable_indices] = False

        layer.modified = True
        self.layer_modified.emit(layer_id)
        self.visibility_changed.emit(layer_id)
        return True, ""

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
        changed = (
            self._selected_layer_id != layer_id or
            self._selected_sublayer_name != sublayer_name
        )

        self._selected_layer_id = layer_id
        self._selected_sublayer_name = sublayer_name

        if changed:
            layer = self.get_layer(layer_id)
            self.selection_changed.emit(layer)

    def set_selected_layers(self, layer_ids, primary_layer_id=None, sublayer_name=None):
        unique_ids = []
        seen = set()
        for layer_id in layer_ids:
            if layer_id in seen or self.get_layer(layer_id) is None:
                continue
            unique_ids.append(layer_id)
            seen.add(layer_id)

        if primary_layer_id not in seen:
            primary_layer_id = unique_ids[0] if unique_ids else None

        changed = (
            self._selected_layer_ids != unique_ids or
            self._selected_layer_id != primary_layer_id or
            self._selected_sublayer_name != sublayer_name
        )

        self._selected_layer_ids = unique_ids
        self._selected_layer_id = primary_layer_id
        self._selected_sublayer_name = sublayer_name

        if changed:
            layer = self.get_layer(primary_layer_id)
            self.selection_changed.emit(layer)



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
        if name == "combined":
            i = 1
            while f"combined ({i})" in existing:
                i += 1
            return f"combined ({i})"
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