"""
Properties panel – displays and edits visual / colour properties of the
currently-selected layer or sublayer.

Designed as a standalone widget so that layer_panel.py stays focused on
the tree structure and context-menu actions.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QPushButton,
    QLabel, QSlider, QGroupBox, QHBoxLayout, QColorDialog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from core.layer_manager import LayerManager
from core.layer import PointCloudLayer, MeshLayer

# ordered list used by the combo box and for mapping
_AXIS_ITEMS = ["Z (up)", "-Z (down)", "X", "-X", "Y", "-Y"]
_AXIS_KEYS  = ["Z",      "-Z",        "X", "-X", "Y", "-Y"]


class PropertiesPanel(QWidget):
    """Right-side dock widget for layer display properties."""

    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(parent)
        self.lm = layer_manager
        self._updating = False          # prevents signal feedback loops

        self._build_ui()
        self._connect()
        self._refresh()

    # ── build ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(10)

        # placeholder when nothing selected
        self._lbl_empty = QLabel("No layer selected")
        self._lbl_empty.setAlignment(Qt.AlignCenter)
        self._lbl_empty.setStyleSheet("color: #888;")
        root.addWidget(self._lbl_empty)

        # ── layer info ──
        self._grp_info = QGroupBox("Layer")
        fl = QFormLayout()
        fl.setContentsMargins(6, 6, 6, 6)
        self._lbl_name  = QLabel("—")
        self._lbl_name.setWordWrap(True)
        self._lbl_type  = QLabel("—")
        self._lbl_count = QLabel("—")
        fl.addRow("Name:", self._lbl_name)
        fl.addRow("Type:", self._lbl_type)
        fl.addRow("Size:", self._lbl_count)
        self._grp_info.setLayout(fl)
        root.addWidget(self._grp_info)

        # ── display ──
        self._grp_disp = QGroupBox("Display")
        dl = QFormLayout()
        dl.setContentsMargins(6, 6, 6, 6)

        self._cmb_mode = QComboBox()
        self._cmb_mode.addItems(["Original", "Solid Colour", "Height Gradient"])
        dl.addRow("Colour mode:", self._cmb_mode)

        self._btn_solid = self._make_color_btn()
        dl.addRow("Solid colour:", self._btn_solid)

        self._cmb_axis = QComboBox()
        self._cmb_axis.addItems(_AXIS_ITEMS)
        dl.addRow("Gradient axis:", self._cmb_axis)

        # point-size row (point-clouds only)
        ps_row = QHBoxLayout()
        ps_row.setContentsMargins(0, 0, 0, 0)
        self._sl_ps = QSlider(Qt.Horizontal)
        self._sl_ps.setRange(1, 20)
        self._sl_ps.setValue(2)
        self._lbl_ps = QLabel("2")
        self._lbl_ps.setMinimumWidth(18)
        ps_row.addWidget(self._sl_ps, 1)
        ps_row.addWidget(self._lbl_ps)
        self._w_ps = QWidget()
        self._w_ps.setLayout(ps_row)
        dl.addRow("Point size:", self._w_ps)

        self._grp_disp.setLayout(dl)
        root.addWidget(self._grp_disp)

        # ── sublayer colour ──
        self._grp_sub = QGroupBox("Sublayer")
        sl_lay = QFormLayout()
        sl_lay.setContentsMargins(6, 6, 6, 6)
        self._lbl_sub = QLabel("—")
        self._lbl_sub.setWordWrap(True)

        sc_row = QHBoxLayout()
        sc_row.setContentsMargins(0, 0, 0, 0)
        self._btn_sub = self._make_color_btn()
        self._btn_sub_reset = QPushButton("Reset")
        self._btn_sub_reset.setFixedWidth(48)
        self._btn_sub_reset.setToolTip(
            "Clear explicit colour (inherit from layer)")
        sc_row.addWidget(self._btn_sub, 1)
        sc_row.addWidget(self._btn_sub_reset)
        self._w_sub_clr = QWidget()
        self._w_sub_clr.setLayout(sc_row)

        self._lbl_sub_hint = QLabel("")
        self._lbl_sub_hint.setStyleSheet("color: #888; font-size: 11px;")

        sl_lay.addRow("Name:", self._lbl_sub)
        sl_lay.addRow("Colour:", self._w_sub_clr)
        sl_lay.addRow("", self._lbl_sub_hint)
        self._grp_sub.setLayout(sl_lay)
        root.addWidget(self._grp_sub)

        root.addStretch(1)

    @staticmethod
    def _make_color_btn():
        b = QPushButton()
        b.setFixedHeight(24)
        b.setMinimumWidth(48)
        b.setCursor(Qt.PointingHandCursor)
        return b

    # ── wiring ───────────────────────────────────────────────────

    def _connect(self):
        lm = self.lm
        lm.selection_changed.connect(self._refresh)
        lm.layer_modified.connect(lambda _: self._refresh())
        lm.layer_renamed.connect(lambda _: self._refresh())
        lm.mask_added.connect(lambda a, b: self._refresh())
        lm.mask_removed.connect(lambda a, b: self._refresh())
        lm.layer_removed.connect(lambda _: self._refresh())

        self._cmb_mode.currentIndexChanged.connect(self._on_mode)
        self._btn_solid.clicked.connect(self._on_pick_solid)
        self._cmb_axis.currentIndexChanged.connect(self._on_axis)
        self._sl_ps.valueChanged.connect(self._on_ps)
        self._btn_sub.clicked.connect(self._on_pick_sub)
        self._btn_sub_reset.clicked.connect(self._on_sub_reset)

    # ── refresh ──────────────────────────────────────────────────

    def _refresh(self):
        self._updating = True
        try:
            self._do_refresh()
        finally:
            self._updating = False

    def _do_refresh(self):
        layer = self.lm.get_selected_layer()
        sname = self.lm.selected_sublayer_name
        has = layer is not None

        self._lbl_empty.setVisible(not has)
        self._grp_info.setVisible(has)
        self._grp_disp.setVisible(has)
        self._grp_sub.setVisible(False)

        if not has:
            return

        # ── info ──
        self._lbl_name.setText(layer.name)
        is_pc = isinstance(layer, PointCloudLayer)
        if is_pc:
            self._lbl_type.setText("Point Cloud")
            self._lbl_count.setText(f"{layer.point_count:,} points")
        else:
            self._lbl_type.setText("Mesh")
            self._lbl_count.setText(
                f"{layer.face_count:,} faces · "
                f"{layer.vertex_count:,} verts")

        # ── display ──
        props = layer.render_props
        cm = props.get("color_mode", "original")
        self._cmb_mode.setCurrentIndex(
            {"original": 0, "solid": 1, "height_gradient": 2}.get(cm, 0))

        sc = props.get("solid_color", [0.7, 0.7, 0.7])
        self._paint_btn(self._btn_solid, sc)
        self._btn_solid.setEnabled(cm == "solid")

        ax = props.get("gradient_axis", "Z")
        idx = _AXIS_KEYS.index(ax) if ax in _AXIS_KEYS else 0
        self._cmb_axis.setCurrentIndex(idx)
        self._cmb_axis.setEnabled(cm == "height_gradient")

        self._w_ps.setVisible(is_pc)
        if is_pc:
            ps = props.get("point_size", 2)
            self._sl_ps.setValue(ps)
            self._lbl_ps.setText(str(ps))

        # ── sublayer ──
        if sname and layer.mask_groups:
            mg, is_pos = self.lm.get_sublayer_mask_group_info(
                layer, sname)
            if mg is not None:
                self._grp_sub.setVisible(True)
                self._lbl_sub.setText(sname)
                explicit = (mg.positive_color if is_pos
                            else mg.negative_color)
                if explicit is not None:
                    self._paint_btn(self._btn_sub, explicit)
                    self._lbl_sub_hint.setText("Explicit colour set")
                    self._btn_sub_reset.setEnabled(True)
                else:
                    if cm == "solid":
                        self._paint_btn(self._btn_sub, sc)
                        self._lbl_sub_hint.setText(
                            "Inherited (solid colour)")
                    elif cm == "height_gradient":
                        self._paint_btn(self._btn_sub, [0.5, 0.5, 0.5])
                        self._lbl_sub_hint.setText(
                            "Inherited (height gradient)")
                    else:
                        self._paint_btn(self._btn_sub, [0.5, 0.5, 0.5])
                        self._lbl_sub_hint.setText(
                            "Inherited (original colours)")
                    self._btn_sub_reset.setEnabled(False)

    # ── colour button helper ─────────────────────────────────────

    @staticmethod
    def _paint_btn(btn, rgb):
        r, g, b = (int(c * 255) for c in rgb)
        btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); "
            f"border: 1px solid #666; border-radius: 2px;")

    # ── callbacks ────────────────────────────────────────────────

    def _on_mode(self, idx):
        if self._updating:
            return
        layer = self.lm.get_selected_layer()
        if not layer:
            return
        modes = ["original", "solid", "height_gradient"]
        self.lm.set_render_prop(layer.id, "color_mode", modes[idx])

    def _on_pick_solid(self):
        layer = self.lm.get_selected_layer()
        if not layer:
            return
        cur = layer.render_props.get("solid_color", [0.7, 0.7, 0.7])
        c = QColorDialog.getColor(
            QColor.fromRgbF(*cur), self, "Solid Colour")
        if c.isValid():
            self.lm.set_render_prop(
                layer.id, "solid_color",
                [c.redF(), c.greenF(), c.blueF()])

    def _on_axis(self, idx):
        if self._updating:
            return
        layer = self.lm.get_selected_layer()
        if not layer:
            return
        self.lm.set_render_prop(
            layer.id, "gradient_axis", _AXIS_KEYS[idx])

    def _on_ps(self, val):
        if self._updating:
            return
        self._lbl_ps.setText(str(val))
        layer = self.lm.get_selected_layer()
        if not layer:
            return
        self.lm.set_render_prop(layer.id, "point_size", val)

    def _on_pick_sub(self):
        layer = self.lm.get_selected_layer()
        sname = self.lm.selected_sublayer_name
        if not layer or not sname:
            return
        mg, is_pos = self.lm.get_sublayer_mask_group_info(layer, sname)
        if mg is None:
            return
        cur = (mg.positive_color if is_pos else mg.negative_color)
        if cur is None:
            cur = layer.render_props.get("solid_color", [0.7, 0.7, 0.7])
        c = QColorDialog.getColor(
            QColor.fromRgbF(*cur), self, "Sublayer Colour")
        if c.isValid():
            self.lm.set_sublayer_color(
                layer.id, mg.id, is_pos,
                (c.redF(), c.greenF(), c.blueF()))

    def _on_sub_reset(self):
        layer = self.lm.get_selected_layer()
        sname = self.lm.selected_sublayer_name
        if not layer or not sname:
            return
        mg, is_pos = self.lm.get_sublayer_mask_group_info(layer, sname)
        if mg is None:
            return
        self.lm.set_sublayer_color(layer.id, mg.id, is_pos, None)