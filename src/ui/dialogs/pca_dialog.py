from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QSpinBox, QDialogButtonBox, QGroupBox, QLabel,
)


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class _NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class PCADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PCA Filter Settings")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)

        info = QLabel(
            "PCA-based planarity filter.\n"
            "Points in planar neighborhoods (high planarity) are kept.\n"
            "Adjust radius and threshold for your data scale.")
        info.setWordWrap(True)
        layout.addWidget(info)

        grp = QGroupBox("Parameters")
        form = QFormLayout(grp)

        self._radius = _NoWheelDoubleSpinBox()
        self._radius.setRange(0.001, 1000.0)
        self._radius.setDecimals(4)
        self._radius.setValue(0.50)
        self._radius.setSingleStep(0.01)
        form.addRow("Search Radius:", self._radius)

        self._threshold = _NoWheelDoubleSpinBox()
        self._threshold.setRange(0.0, 1.0)
        self._threshold.setDecimals(3)
        self._threshold.setValue(0.3)
        self._threshold.setSingleStep(0.05)
        form.addRow("Planarity Threshold:", self._threshold)

        self._k = _NoWheelSpinBox()
        self._k.setRange(3, 200)
        self._k.setValue(10)
        form.addRow("Min Neighbors (k):", self._k)

        self._chunk = _NoWheelSpinBox()
        self._chunk.setRange(100, 1000000)
        self._chunk.setValue(50000)
        self._chunk.setSingleStep(10000)
        form.addRow("Chunk Size:", self._chunk)

        layout.addWidget(grp)

        bbox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    def get_params(self):
        return {
            "radius": self._radius.value(),
            "threshold": self._threshold.value(),
            "k_neighbors": self._k.value(),
            "chunk_size": self._chunk.value(),
        }