from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QSpinBox, QCheckBox, QDialogButtonBox, QGroupBox, QLabel,
)


class PoissonDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Poisson Reconstruction Settings")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Screened Poisson surface reconstruction.\n"
            "Higher depth = more detail but slower.\n"
            "Density quantile trims low-confidence regions.")
        info.setWordWrap(True)
        layout.addWidget(info)

        grp = QGroupBox("Parameters")
        form = QFormLayout(grp)

        self._depth = QSpinBox()
        self._depth.setRange(1, 14)
        self._depth.setValue(9)
        form.addRow("Octree Depth:", self._depth)

        self._scale = QDoubleSpinBox()
        self._scale.setRange(1.0, 5.0)
        self._scale.setDecimals(2)
        self._scale.setValue(1.1)
        self._scale.setSingleStep(0.1)
        form.addRow("Scale:", self._scale)

        self._dq = QDoubleSpinBox()
        self._dq.setRange(0.0, 0.99)
        self._dq.setDecimals(3)
        self._dq.setValue(0.05)
        self._dq.setSingleStep(0.01)
        form.addRow("Density Quantile Trim:", self._dq)

        self._linear = QCheckBox("Enabled")
        self._linear.setChecked(False)
        form.addRow("Linear Fit:", self._linear)

        layout.addWidget(grp)

        bbox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    def get_params(self):
        return {
            "depth": self._depth.value(),
            "scale": self._scale.value(),
            "density_quantile": self._dq.value(),
            "linear_fit": self._linear.isChecked(),
        }