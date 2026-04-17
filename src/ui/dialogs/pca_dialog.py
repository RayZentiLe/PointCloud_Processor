from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QDialogButtonBox, QLabel,
)


class PCADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PCA Filter Parameters")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("PCA-based planarity filter"))

        form = QFormLayout()

        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.01, 100.0)
        self.radius.setValue(0.5)
        self.radius.setDecimals(3)
        self.radius.setSingleStep(0.1)
        form.addRow("Search Radius:", self.radius)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setValue(0.3)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.05)
        form.addRow("Planarity Threshold:", self.threshold)

        self.k_neighbors = QSpinBox()
        self.k_neighbors.setRange(3, 200)
        self.k_neighbors.setValue(10)
        form.addRow("Min Neighbors:", self.k_neighbors)

        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(1000, 1_000_000)
        self.chunk_size.setValue(50_000)
        self.chunk_size.setSingleStep(10_000)
        form.addRow("Chunk Size:", self.chunk_size)

        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_params(self):
        return dict(
            radius=self.radius.value(),
            threshold=self.threshold.value(),
            k_neighbors=self.k_neighbors.value(),
            chunk_size=self.chunk_size.value(),
        )