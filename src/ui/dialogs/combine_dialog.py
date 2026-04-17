from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox,
    QLineEdit, QDialogButtonBox, QGroupBox, QLabel, QMessageBox,
)
from core.layer import PointCloudLayer, MeshLayer


class CombineDialog(QDialog):
    def __init__(self, layer_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine Layers")
        self.setMinimumWidth(380)
        self.lm = layer_manager

        layout = QVBoxLayout(self)

        info = QLabel("Merge two same-type layers into one.")
        info.setWordWrap(True)
        layout.addWidget(info)

        grp = QGroupBox("Select Layers")
        form = QFormLayout(grp)

        self._combo_a = QComboBox()
        self._combo_b = QComboBox()
        all_layers = self.lm.get_all_layers()
        for l in all_layers:
            tag = "PC" if isinstance(l, PointCloudLayer) else "Mesh"
            label = f"{l.name} [{tag}]"
            self._combo_a.addItem(label, l.id)
            self._combo_b.addItem(label, l.id)
        if self._combo_b.count() > 1:
            self._combo_b.setCurrentIndex(1)

        form.addRow("Layer A:", self._combo_a)
        form.addRow("Layer B:", self._combo_b)

        self._name_edit = QLineEdit("combined")
        form.addRow("Result Name:", self._name_edit)

        layout.addWidget(grp)

        bbox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self._validate)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    def _validate(self):
        id_a = self._combo_a.currentData()
        id_b = self._combo_b.currentData()
        if id_a == id_b:
            QMessageBox.warning(self, "Combine", "Select two different layers.")
            return
        a = self.lm.get_layer(id_a)
        b = self.lm.get_layer(id_b)
        if type(a) != type(b):
            QMessageBox.warning(
                self, "Combine",
                "Both layers must be the same type\n"
                "(both point clouds or both meshes).")
            return
        self.accept()

    def get_params(self):
        return {
            "layer_a_id": self._combo_a.currentData(),
            "layer_b_id": self._combo_b.currentData(),
            "name": self._name_edit.text().strip() or "combined",
        }