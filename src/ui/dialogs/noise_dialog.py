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
        form.addRow("Reference Mesh:", self._mesh_combo)

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

    def get_params(self):
        idx = self._mesh_combo.currentIndex()
        return {
            "mesh_id": self._mesh_combo.itemData(idx),
            "threshold": self._threshold.value(),
            "mesh_sublayer": None,
        }