from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QComboBox, QDialogButtonBox, QGroupBox, QLabel,
)


class NoiseDialog(QDialog):
    def __init__(self, meshes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Noise Removal Settings")
        self.setMinimumWidth(380)
        self._meshes = meshes
        self._mesh_by_id = {m.id: m for m in meshes}

        layout = QVBoxLayout(self)

        info = QLabel(
            "Remove points far from a reference mesh surface.\n"
            "Points with distance > threshold are marked as noise.")
        info.setWordWrap(True)
        layout.addWidget(info)

        grp = QGroupBox("Parameters")
        form = QFormLayout(grp)

        self._mesh_combo = QComboBox()
        for m in meshes:
            self._mesh_combo.addItem(m.name, m.id)
        self._mesh_combo.currentIndexChanged.connect(self._on_mesh_selected)
        form.addRow("Reference Mesh:", self._mesh_combo)

        self._sublayer_combo = QComboBox()
        form.addRow("Mesh Sublayer:", self._sublayer_combo)

        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.0001, 1000.0)
        self._threshold.setDecimals(4)
        self._threshold.setValue(1.0)
        self._threshold.setSingleStep(0.01)
        form.addRow("Distance Threshold:", self._threshold)

        layout.addWidget(grp)

        bbox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

        # Initialize sublayers for the first mesh
        self._on_mesh_selected()

    def _on_mesh_selected(self):
        """Update sublayer combo when mesh selection changes."""
        self._sublayer_combo.blockSignals(True)
        self._sublayer_combo.clear()
        
        idx = self._mesh_combo.currentIndex()
        if idx < 0:
            self._sublayer_combo.blockSignals(False)
            return
        
        mesh_id = self._mesh_combo.itemData(idx)
        mesh = self._mesh_by_id.get(mesh_id)
        
        if mesh is None:
            self._sublayer_combo.blockSignals(False)
            return
        
        # Add "Whole Mesh" option
        self._sublayer_combo.addItem("Whole Mesh", None)
        
        # Add all mask sublayers
        for mg in mesh.mask_groups:
            self._sublayer_combo.addItem(f"{mg.positive_name} ({mg.positive_count:,})", mg.positive_name)
            self._sublayer_combo.addItem(f"{mg.negative_name} ({mg.negative_count:,})", mg.negative_name)
        
        self._sublayer_combo.blockSignals(False)

    def get_params(self):
        idx = self._mesh_combo.currentIndex()
        sublayer_idx = self._sublayer_combo.currentIndex()
        return {
            "mesh_id": self._mesh_combo.itemData(idx),
            "threshold": self._threshold.value(),
            "mesh_sublayer": self._sublayer_combo.itemData(sublayer_idx),
        }