from PySide6.QtWidgets import QWidgetAction, QToolBar, QMenu, QToolButton, QLabel
from PySide6.QtCore import Signal, Qt


HELP_SECTIONS = [
    {
        "title": "General",
        "items": [
            ("Open files", "Use the toolbar Open button or press Ctrl+O."),
            ("Export selected layer", "Use the Export button or press Ctrl+E after selecting a layer."),
            ("Drag and drop", "Drop supported files directly into the application window to load them."),
            ("Undo / Redo", "Use the toolbar buttons to undo or redo supported actions.\nSupported: Combine layer(s), Delete layer(s),\nCross Section point transfer."),
        ],
    },
    {
        "title": "Layers Panel",
        "items": [
            ("Right click layer", "Open actions such as Rename, Assign Colour,\nCombine, Export, Delete, and Set Camera to Layer."),
            ("Multi-select layers", "Use Shift/Ctrl+Left Click in the Layers panel\nto combine or delete several layers."),
            ("Reorder layers", "Drag selected layers in the Layers panel to change their order."),
            ("Toggle visibility", "Use the checkbox beside each layer or sublayer to show or hide it."),
        ],
    },
    {
        "title": "Viewport",
        "items": [
            ("Navigation", "Use mouse drag and wheel to pan and zoom the scene\naccording to the current view mode."),
            ("View mode toggle", "Use the button above the viewport to switch\nbetween Fixed Z Plane View and orbit view."),
            ("Background colour", "Right click the viewport background to change the background colour."),
            ("Focus camera", "Right click a layer in the Layers panel and choose Set Camera to Layer."),
        ],
    },
    {
        "title": "Cross Section",
        "items": [
            ("Open panel", "Use the Cross Section toolbar button or enable\nthe panel from Windows."),
            ("Define normal", "Click Define Normal, then click two points in\nthe viewport to define the cross-section direction."),
            ("Move section plane", "Press W to move forward and S to move backward through the cross section."),
            ("Preview visible clouds", "The preview uses all visible point clouds\nonce a valid cross-section direction is defined."),
            ("Transfer points between layers", "Procedure:\n1. Open Cross Section.\n2. Define the normal.\n3. Adjust the plane with W/S if needed.\n4. Left drag to select points in the preview.\n5. Choose From and To layer entries.\n6. Then the selected points are\n   transferred automatically."),
        ],
    },
    {
        "title": "Windows / Panels",
        "items": [
            ("Show or hide panels", "Use the Windows menu in the toolbar to toggle\nLayers, Properties, Cross Section, and Log panels."),
            ("Font size", "Use Windows > Font Size to switch between\nSmall, Medium, Large, and Extra Large text."),
            ("Cross Section panel", "The Cross Section panel is hidden by default and appears when enabled."),
        ],
    },
    {
        "title": "Tips",
        "items": [
            ("Look for right click menus", "Several advanced actions are available only\nfrom context menus."),
            ("Selection matters", "Many tools act on the currently selected layer,\nso select the target layer first."),
            ("Visible layers affect preview", "Cross-section preview and some operations\nuse only layers that are currently visible."),
            ("Orbit view note", "Point picking for cross section works in fixed Z plane view, not orbit view."),
        ],
    },
]


class Toolbar(QToolBar):
    open_requested = Signal()
    undo_requested = Signal()
    redo_requested = Signal()
    auto_denoise_requested = Signal()
    pca_requested = Signal()
    poisson_requested = Signal()
    mesh_filter_requested = Signal()
    noise_removal_requested = Signal()
    cross_section_requested = Signal()
    export_requested = Signal()
    font_size_changed = Signal(int)  # New signal for font size changes

    def __init__(self, layer_manager, parent=None):
        super().__init__("Main Toolbar", parent)
        self.layer_manager = layer_manager
        self.dock_widgets = {}  # Store references to dock widgets
        self.setMovable(False)
        self.setFloatable(False)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)

        self.addAction("📂 Open", self.open_requested.emit)
        self.addAction("↶ Undo", self.undo_requested.emit)
        self.addAction("↷ Redo", self.redo_requested.emit)
        self.addSeparator()
        self.addAction("Auto Denoise", self.auto_denoise_requested.emit)
        self.addAction("PCA Filter", self.pca_requested.emit)
        self.addAction("Poisson", self.poisson_requested.emit)
        self.addAction("Noise Removal", self.noise_removal_requested.emit)
        self.addAction("Cross Section", self.cross_section_requested.emit)
        self.addSeparator()
        self.addAction("💾 Export", self.export_requested.emit)
        self.addSeparator()
        
        # Add Windows dropdown menu
        self._create_windows_menu()
        self._create_help_menu()

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
        
        self.cross_section_action = menu.addAction("Cross Section")
        self.cross_section_action.setCheckable(True)
        self.cross_section_action.setChecked(False)
        self.cross_section_action.triggered.connect(self._toggle_cross_section)
        
        self.log_action = menu.addAction("Log")
        self.log_action.setCheckable(True)
        self.log_action.setChecked(True)
        self.log_action.triggered.connect(self._toggle_log)
        
        # Add font size submenu
        menu.addSeparator()
        font_menu = menu.addMenu("Font Size")
        self._create_font_size_menu(font_menu)
        
        # Add menu button
        menu_button = QToolButton(self)
        menu_button.setText("Windows")
        menu_button.setMenu(menu)
        menu_button.setPopupMode(QToolButton.InstantPopup)
        menu_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addWidget(menu_button)

    def _create_help_menu(self):
        menu = QMenu("Help", self)
        for section in HELP_SECTIONS:
            section_menu = menu.addMenu(section["title"])
            for label_text, description in section["items"]:
                detail_menu = section_menu.addMenu(label_text)
                detail_label = QLabel(description, detail_menu)
                detail_label.setWordWrap(True)
                detail_label.setMinimumWidth(280)
                detail_label.setMaximumWidth(320)
                detail_label.setMargin(8)
                detail_label.setStyleSheet("color: palette(text);")

                detail_action = QWidgetAction(detail_menu)
                detail_action.setDefaultWidget(detail_label)
                detail_menu.addAction(detail_action)

        menu_button = QToolButton(self)
        menu_button.setText("Help")
        menu_button.setMenu(menu)
        menu_button.setPopupMode(QToolButton.InstantPopup)
        menu_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addWidget(menu_button)

    def set_dock_widgets(self, layers_dock, properties_dock, log_dock, cross_section_dock=None):
        """Set the dock widget references for panel toggling."""
        self.dock_widgets['layers'] = layers_dock
        self.dock_widgets['properties'] = properties_dock
        self.dock_widgets['log'] = log_dock
        if cross_section_dock is not None:
            self.dock_widgets['cross_section'] = cross_section_dock
        
        # Update checkbox states based on current visibility
        if 'layers' in self.dock_widgets:
            self.layers_action.setChecked(self.dock_widgets['layers'].isVisible())
        if 'properties' in self.dock_widgets:
            self.properties_action.setChecked(self.dock_widgets['properties'].isVisible())
        if 'cross_section' in self.dock_widgets:
            self.cross_section_action.setChecked(self.dock_widgets['cross_section'].isVisible())
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

    def _toggle_cross_section(self):
        """Toggle Cross Section panel visibility."""
        if 'cross_section' in self.dock_widgets:
            self.dock_widgets['cross_section'].setVisible(self.cross_section_action.isChecked())

    def _toggle_log(self):
        """Toggle Log panel visibility."""
        if 'log' in self.dock_widgets:
            self.dock_widgets['log'].setVisible(self.log_action.isChecked())

    def _create_font_size_menu(self, font_menu):
        """Create the font size submenu with size options."""
        self.font_size_actions = {}
        
        # Font size options
        sizes = [
            ("Small", 10),
            ("Medium", 12),
            ("Large", 14),
            ("Extra Large", 16)
        ]
        
        for name, size in sizes:
            action = font_menu.addAction(name)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, s=size: self._set_font_size(s))
            self.font_size_actions[size] = action
        
        # Set default (Medium) as checked
        self.font_size_actions[12].setChecked(True)

    def _set_font_size(self, size):
        """Set the font size and update action states."""
        # Uncheck all actions
        for action in self.font_size_actions.values():
            action.setChecked(False)
        
        # Check the selected action
        if size in self.font_size_actions:
            self.font_size_actions[size].setChecked(True)
        
        # Emit the signal
        self.font_size_changed.emit(size)