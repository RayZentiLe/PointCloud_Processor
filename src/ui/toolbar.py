from PySide6.QtWidgets import QToolBar
from PySide6.QtCore import Signal


class Toolbar(QToolBar):
    open_requested = Signal()
    pca_requested = Signal()
    poisson_requested = Signal()
    mesh_filter_requested = Signal()
    noise_removal_requested = Signal()
    export_requested = Signal()
    combine_requested = Signal()

    def __init__(self, layer_manager, parent=None):
        super().__init__("Main Toolbar", parent)
        self.layer_manager = layer_manager

        self.addAction("📂 Open", self.open_requested.emit)
        self.addSeparator()
        self.addAction("PCA Filter", self.pca_requested.emit)
        self.addAction("Poisson", self.poisson_requested.emit)
        self.addAction("Mesh Filter", self.mesh_filter_requested.emit)
        self.addAction("Noise Removal", self.noise_removal_requested.emit)
        self.addSeparator()
        self.addAction("Combine", self.combine_requested.emit)
        self.addAction("💾 Export", self.export_requested.emit)