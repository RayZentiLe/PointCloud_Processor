from PySide6.QtWidgets import QToolBar, QMenu, QToolButton
from PySide6.QtCore import Signal, Qt


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
        self.dock_widgets = {}  # Store references to dock widgets

        self.addAction("📂 Open", self.open_requested.emit)
        self.addSeparator()
        self.addAction("PCA Filter", self.pca_requested.emit)
        self.addAction("Poisson", self.poisson_requested.emit)
        self.addAction("Mesh Filter", self.mesh_filter_requested.emit)
        self.addAction("Noise Removal", self.noise_removal_requested.emit)
        self.addSeparator()
        self.addAction("Combine", self.combine_requested.emit)
        self.addAction("💾 Export", self.export_requested.emit)
        self.addSeparator()
        
        # Add Windows dropdown menu
        self._create_windows_menu()

    def _create_windows_menu(self):
        """Create the Windows dropdown menu for panel visibility."""
        menu = QMenu("Windows", self)
        
        # Add actions for each panel
        self.layers_action = menu.addAction("Layers")
        self.layers_action.setCheckable(True)
        self.layers_action.setChecked(True)
        self.layers_action.triggered.connect(self._toggle_layers)
        
        self.properties_action = menu.addAction("Properties")
        self.properties_action.setCheckable(True)
        self.properties_action.setChecked(True)
        self.properties_action.triggered.connect(self._toggle_properties)
        
        self.log_action = menu.addAction("Log")
        self.log_action.setCheckable(True)
        self.log_action.setChecked(True)
        self.log_action.triggered.connect(self._toggle_log)
        
        # Add menu button
        menu_button = QToolButton(self)
        menu_button.setText("Windows")
        menu_button.setMenu(menu)
        menu_button.setPopupMode(QToolButton.InstantPopup)
        self.addWidget(menu_button)

    def set_dock_widgets(self, layers_dock, properties_dock, log_dock):
        """Set the dock widget references for panel toggling."""
        self.dock_widgets['layers'] = layers_dock
        self.dock_widgets['properties'] = properties_dock
        self.dock_widgets['log'] = log_dock
        
        # Update checkbox states based on current visibility
        if 'layers' in self.dock_widgets:
            self.layers_action.setChecked(self.dock_widgets['layers'].isVisible())
        if 'properties' in self.dock_widgets:
            self.properties_action.setChecked(self.dock_widgets['properties'].isVisible())
        if 'log' in self.dock_widgets:
            self.log_action.setChecked(self.dock_widgets['log'].isVisible())

    def _toggle_layers(self):
        """Toggle Layers panel visibility."""
        if 'layers' in self.dock_widgets:
            self.dock_widgets['layers'].setVisible(self.layers_action.isChecked())

    def _toggle_properties(self):
        """Toggle Properties panel visibility."""
        if 'properties' in self.dock_widgets:
            self.dock_widgets['properties'].setVisible(self.properties_action.isChecked())

    def _toggle_log(self):
        """Toggle Log panel visibility."""
        if 'log' in self.dock_widgets:
            self.dock_widgets['log'].setVisible(self.log_action.isChecked())