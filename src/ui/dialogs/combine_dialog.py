from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout,
    QComboBox, QLineEdit,
    QDialogButtonBox, QLabel, QMessageBox,
)
from core.layer import PointCloudLayer, MeshLayer


class CombineDialog(QDialog):
    def __init__(self, layer_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine Layers")
        self.setMinimumWidth(400)
        self.lm = layer_manager

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select two layers of the same type"))

        form = QFormLayout()

        self.type_combo = QComboBox()
        self.type_combo.addItems(["Point Cloud", "Mesh"])
        self.type_combo.currentIndexChanged.connect(self._refresh)
        form.addRow("Type:", self.type_combo)

        self.combo_a = QComboBox()
        self.combo_b = QComboBox()
        form.addRow("Layer A:", self.combo_a)
        form.addRow("Layer B:", self.combo_b)

        self.name_edit = QLineEdit()
        form.addRow("New Name:", self.name_edit)

        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._validate)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._refresh()

    def _refresh(self):
        self.combo_a.clear()
        self.combo_b.clear()
        layers = (list(self.lm.point_clouds.values())
                  if self.type_combo.currentIndex() == 0
                  else list(self.lm.meshes.values()))
        for layer in layers:
            self.combo_a.addItem(layer.name, layer.id)
            self.combo_b.addItem(layer.name, layer.id)
        if self.combo_a.count() >= 2:
            self.combo_b.setCurrentIndex(1)
        a = self.combo_a.currentText()
        b = self.combo_b.currentText()
        if a and b:
            self.name_edit.setText(f"{a}+{b}")

    def _validate(self):
        if self.combo_a.count() < 2:
            QMessageBox.warning(self, "Combine",
                                "Need at least 2 layers of the same type.")
            return
        if self.combo_a.currentData() == self.combo_b.currentData():
            QMessageBox.warning(self, "Combine",
                                "Select two different layers.")
            return
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Combine", "Enter a name.")
            return
        self.accept()

    def get_params(self):
        return dict(
            layer_a_id=self.combo_a.currentData(),
            layer_b_id=self.combo_b.currentData(),
            name=self.name_edit.text().strip(),
        )