from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QCheckBox,
    QDialogButtonBox, QLabel,
)


class PoissonDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Poisson Reconstruction")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Poisson surface reconstruction parameters"))

        form = QFormLayout()

        self.depth = QSpinBox()
        self.depth.setRange(4, 14)
        self.depth.setValue(8)
        form.addRow("Octree Depth:", self.depth)

        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.5, 5.0)
        self.scale.setValue(1.1)
        self.scale.setDecimals(2)
        form.addRow("Scale:", self.scale)

        self.density_quantile = QDoubleSpinBox()
        self.density_quantile.setRange(0.0, 0.5)
        self.density_quantile.setValue(0.05)
        self.density_quantile.setDecimals(3)
        self.density_quantile.setSingleStep(0.01)
        form.addRow("Density Quantile:", self.density_quantile)

        self.linear_fit = QCheckBox()
        self.linear_fit.setChecked(False)
        form.addRow("Linear Fit:", self.linear_fit)

        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_params(self):
        return dict(
            depth=self.depth.value(),
            scale=self.scale.value(),
            density_quantile=self.density_quantile.value(),
            linear_fit=self.linear_fit.isChecked(),
        )