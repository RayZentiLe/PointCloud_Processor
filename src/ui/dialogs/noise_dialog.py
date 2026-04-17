from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout,
    QDoubleSpinBox, QComboBox,
    QDialogButtonBox, QLabel,
)


class NoiseDialog(QDialog):
    def __init__(self, meshes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Noise Removal")
        self.setMinimumWidth(380)
        self._meshes = meshes

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Remove points far from a reference mesh surface"))

        form = QFormLayout()

        self.mesh_combo = QComboBox()
        for m in meshes:
            self.mesh_combo.addItem(m.name, m.id)
        self.mesh_combo.currentIndexChanged.connect(self._update_sublayers)
        form.addRow("Reference Mesh:", self.mesh_combo)

        self.sublayer_combo = QComboBox()
        form.addRow("Mesh Sublayer:", self.sublayer_combo)
        self._update_sublayers()

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.001, 100.0)
        self.threshold.setValue(0.5)
        self.threshold.setDecimals(4)
        self.threshold.setSingleStep(0.1)
        form.addRow("Distance Threshold:", self.threshold)

        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _update_sublayers(self):
        self.sublayer_combo.clear()
        self.sublayer_combo.addItem("(entire mesh)", None)
        idx = self.mesh_combo.currentIndex()
        if 0 <= idx < len(self._meshes):
            mesh = self._meshes[idx]
            for mg in mesh.mask_groups:
                self.sublayer_combo.addItem(mg.positive_name, mg.positive_name)
                self.sublayer_combo.addItem(mg.negative_name, mg.negative_name)

    def get_params(self):
        return dict(
            mesh_id=self.mesh_combo.currentData(),
            mesh_sublayer=self.sublayer_combo.currentData(),
            threshold=self.threshold.value(),
        )