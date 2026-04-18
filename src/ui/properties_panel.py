"""
Properties panel – layer info + visual appearance controls.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QPushButton, QColorDialog, QGroupBox, QFormLayout,
    QRadioButton, QButtonGroup, QDoubleSpinBox, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from core.layer import PointCloudLayer, MeshLayer

# ── direction look-up tables ────────────────────────────────────
#               0     1     2     3     4     5
_DIR_ITEMS = ["X", "-X", "Y", "-Y", "Z", "-Z"]
_DIR_AXIS  = [ 0,    0,   1,    1,   2,    2 ]   # coord column
_DIR_FLIP  = [False, True, False, True, False, True]
_DIR_DEFAULT_COMBO = 4    # "Z"


def _combo_index(axis: int, flip: bool) -> int:
    """Map (axis, flip) back to the combo-box row."""
    for i, (a, f) in enumerate(zip(_DIR_AXIS, _DIR_FLIP)):
        if a == axis and f == flip:
            return i
    return _DIR_DEFAULT_COMBO


class PropertiesPanel(QWidget):
    """Right-dock panel showing layer info and visual property controls."""

    visual_changed = Signal()           # viewport listens to this

    def __init__(self, layer_manager, parent=None):
        super().__init__(parent)
        self.lm = layer_manager
        self._building = False          # guard against echo loops
        self._auto_mins = None          # ndarray(3,) or None
        self._auto_maxs = None

        self._build_ui()
        self._connect_signals()
        self._refresh()

    # ================================================================
    #  BUILD
    # ================================================================

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Layer info ───────────────────────────────────────────
        info_grp = QGroupBox("Layer Info")
        ifl = QFormLayout(info_grp)
        ifl.setContentsMargins(8, 14, 8, 8)
        ifl.setSpacing(4)

        self.lbl_name  = QLabel("—")
        self.lbl_name.setWordWrap(True)
        self.lbl_type  = QLabel("—")
        self.lbl_count = QLabel("—")
        ifl.addRow("Name:", self.lbl_name)
        ifl.addRow("Type:", self.lbl_type)
        ifl.addRow("Count:", self.lbl_count)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        ifl.addRow(sep)

        self.lbl_x = QLabel("—")
        self.lbl_y = QLabel("—")
        self.lbl_z = QLabel("—")
        ifl.addRow("X range:", self.lbl_x)
        ifl.addRow("Y range:", self.lbl_y)
        ifl.addRow("Z range:", self.lbl_z)

        root.addWidget(info_grp)

        # ── Appearance ───────────────────────────────────────────
        vis_grp = QGroupBox("Appearance")
        vl = QVBoxLayout(vis_grp)
        vl.setContentsMargins(8, 14, 8, 8)
        vl.setSpacing(6)

        # --- point size (top) --------------------------------
        ps_row = QHBoxLayout()
        ps_row.addWidget(QLabel("Point size:"))
        self.sl_ps = QSlider(Qt.Horizontal)
        self.sl_ps.setRange(1, 20)
        self.sl_ps.setValue(2)
        ps_row.addWidget(self.sl_ps, 1)
        self.lbl_ps = QLabel("2")
        self.lbl_ps.setMinimumWidth(18)
        ps_row.addWidget(self.lbl_ps)
        vl.addLayout(ps_row)

        # --- colour scheme combo ------------------------------
        cs_row = QHBoxLayout()
        cs_row.addWidget(QLabel("Color:"))
        self.cmb_scheme = QComboBox()
        self.cmb_scheme.addItems(["Original", "Solid", "Gradient"])
        cs_row.addWidget(self.cmb_scheme, 1)
        vl.addLayout(cs_row)

        # --- solid-only controls ------------------------------
        self.w_solid = QWidget()
        sl = QHBoxLayout(self.w_solid)
        sl.setContentsMargins(0, 4, 0, 0)
        sl.addWidget(QLabel("Pick color:"))
        self.btn_color = QPushButton("")
        self.btn_color.setMinimumHeight(26)
        self._solid_qc = QColor(51, 153, 255)
        self._apply_swatch()
        sl.addWidget(self.btn_color, 1)
        vl.addWidget(self.w_solid)

        # --- gradient-only controls ---------------------------
        self.w_grad = QWidget()
        gl = QVBoxLayout(self.w_grad)
        gl.setContentsMargins(0, 4, 0, 0)
        gl.setSpacing(6)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Axis:"))
        self.cmb_dir = QComboBox()
        self.cmb_dir.addItems(_DIR_ITEMS)
        self.cmb_dir.setCurrentIndex(_DIR_DEFAULT_COMBO)
        dir_row.addWidget(self.cmb_dir, 1)
        gl.addLayout(dir_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Range:"))
        self.rb_auto   = QRadioButton("Auto")
        self.rb_manual = QRadioButton("Manual")
        self.rb_auto.setChecked(True)
        self.bg_mode = QButtonGroup(self)
        self.bg_mode.addButton(self.rb_auto, 0)
        self.bg_mode.addButton(self.rb_manual, 1)
        mode_row.addWidget(self.rb_auto)
        mode_row.addWidget(self.rb_manual)
        mode_row.addStretch()
        gl.addLayout(mode_row)

        mm = QFormLayout()
        mm.setContentsMargins(0, 0, 0, 0)
        self.sp_min = QDoubleSpinBox()
        self.sp_min.setRange(-1e7, 1e7)
        self.sp_min.setDecimals(4)
        self.sp_min.setSingleStep(0.1)
        self.sp_min.setEnabled(False)
        mm.addRow("Min:", self.sp_min)

        self.sp_max = QDoubleSpinBox()
        self.sp_max.setRange(-1e7, 1e7)
        self.sp_max.setDecimals(4)
        self.sp_max.setSingleStep(0.1)
        self.sp_max.setEnabled(False)
        mm.addRow("Max:", self.sp_max)

        gl.addLayout(mm)
        vl.addWidget(self.w_grad)

        root.addWidget(vis_grp)
        root.addStretch(1)

    # ================================================================
    #  SIGNALS
    # ================================================================

    def _connect_signals(self):
        self.lm.selection_changed.connect(self._refresh)
        if hasattr(self.lm, "layers_changed"):
            self.lm.layers_changed.connect(self._refresh)

        self.sl_ps.valueChanged.connect(self._on_ps)
        self.cmb_scheme.currentIndexChanged.connect(self._on_scheme)
        self.btn_color.clicked.connect(self._on_pick_color)
        self.cmb_dir.currentIndexChanged.connect(self._on_grad_param)
        self.bg_mode.buttonClicked.connect(self._on_mode)
        self.sp_min.valueChanged.connect(self._on_grad_param)
        self.sp_max.valueChanged.connect(self._on_grad_param)

    # ================================================================
    #  REFRESH (model → UI)
    # ================================================================

    def _refresh(self):
        self._building = True
        layer = self.lm.get_selected_layer()

        if layer is None:
            for lbl in (self.lbl_name, self.lbl_type, self.lbl_count,
                        self.lbl_x, self.lbl_y, self.lbl_z):
                lbl.setText("—")
            self.w_solid.setVisible(False)
            self.w_grad.setVisible(False)
            self._building = False
            return

        # ── info labels ──────────────────────────────
        self.lbl_name.setText(layer.name)

        if isinstance(layer, PointCloudLayer):
            self.lbl_type.setText("Point Cloud")
            self.lbl_count.setText(f"{layer.point_count:,} points")
            pts = layer.points
        elif isinstance(layer, MeshLayer):
            self.lbl_type.setText("Mesh")
            self.lbl_count.setText(
                f"{layer.face_count:,} faces · {layer.vertex_count:,} verts")
            pts = layer.vertices
        else:
            pts = np.empty((0, 3))

        if pts is not None and len(pts) > 0:
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
            self.lbl_x.setText(f"{lo[0]:.4f}  →  {hi[0]:.4f}")
            self.lbl_y.setText(f"{lo[1]:.4f}  →  {hi[1]:.4f}")
            self.lbl_z.setText(f"{lo[2]:.4f}  →  {hi[2]:.4f}")
            self._auto_mins = lo
            self._auto_maxs = hi
        else:
            self.lbl_x.setText("—")
            self.lbl_y.setText("—")
            self.lbl_z.setText("—")
            self._auto_mins = self._auto_maxs = None

        # ── restore per-layer visual state ───────────
        self.sl_ps.setValue(getattr(layer, "vis_point_size", 2))
        self.lbl_ps.setText(str(self.sl_ps.value()))

        scheme = getattr(layer, "vis_color_scheme", "Original")
        idx = self.cmb_scheme.findText(scheme)
        if idx >= 0:
            self.cmb_scheme.setCurrentIndex(idx)

        sc = getattr(layer, "vis_solid_color", None)
        if sc:
            self._solid_qc = QColor(
                int(sc[0] * 255), int(sc[1] * 255), int(sc[2] * 255))
            self._apply_swatch()

        # vis_gradient_dir is always 0-2, vis_gradient_flip is bool
        axis = getattr(layer, "vis_gradient_dir", 2)
        flip = getattr(layer, "vis_gradient_flip", False)
        self.cmb_dir.setCurrentIndex(_combo_index(axis, flip))

        gmode = getattr(layer, "vis_gradient_mode", "auto")
        self.rb_auto.setChecked(gmode == "auto")
        self.rb_manual.setChecked(gmode == "manual")
        self.sp_min.setEnabled(gmode == "manual")
        self.sp_max.setEnabled(gmode == "manual")

        gmin = getattr(layer, "vis_gradient_min", None)
        gmax = getattr(layer, "vis_gradient_max", None)
        if gmode == "manual" and gmin is not None and gmax is not None:
            self.sp_min.setValue(gmin)
            self.sp_max.setValue(gmax)
        elif self._auto_mins is not None:
            self.sp_min.setValue(self._auto_mins[axis])
            self.sp_max.setValue(self._auto_maxs[axis])

        self._update_visibility()
        self._building = False

    # ================================================================
    #  UI HELPERS
    # ================================================================

    def _update_visibility(self):
        scheme = self.cmb_scheme.currentText()
        self.w_solid.setVisible(scheme == "Solid")
        self.w_grad.setVisible(scheme == "Gradient")

    def _apply_swatch(self):
        self.btn_color.setStyleSheet(
            f"background-color: {self._solid_qc.name()}; "
            f"border: 1px solid #888; min-height: 26px;")

    def _cur_axis_flip(self):
        """Return (axis_column, flip) from the current combo selection."""
        ci = self.cmb_dir.currentIndex()
        return _DIR_AXIS[ci], _DIR_FLIP[ci]

    # ================================================================
    #  SLOTS  (UI → model)
    # ================================================================

    def _on_ps(self, val):
        self.lbl_ps.setText(str(val))
        if self._building:
            return
        layer = self.lm.get_selected_layer()
        if layer:
            layer.vis_point_size = val
            self._notify()

    def _on_scheme(self, _idx):
        self._update_visibility()
        if self._building:
            return
        layer = self.lm.get_selected_layer()
        if layer:
            layer.vis_color_scheme = self.cmb_scheme.currentText()
            self._notify()

    def _on_pick_color(self):
        c = QColorDialog.getColor(self._solid_qc, self, "Solid Color")
        if not c.isValid():
            return
        self._solid_qc = c
        self._apply_swatch()
        layer = self.lm.get_selected_layer()
        if layer:
            layer.vis_solid_color = (c.redF(), c.greenF(), c.blueF())
            self._notify()

    def _on_mode(self):
        manual = self.rb_manual.isChecked()
        self.sp_min.setEnabled(manual)
        self.sp_max.setEnabled(manual)

        axis, _flip = self._cur_axis_flip()

        if not manual and self._auto_mins is not None:
            self._building = True
            self.sp_min.setValue(self._auto_mins[axis])
            self.sp_max.setValue(self._auto_maxs[axis])
            self._building = False

        if self._building:
            return
        layer = self.lm.get_selected_layer()
        if layer:
            layer.vis_gradient_mode = "manual" if manual else "auto"
            if manual:
                layer.vis_gradient_min = self.sp_min.value()
                layer.vis_gradient_max = self.sp_max.value()
            else:
                layer.vis_gradient_min = None
                layer.vis_gradient_max = None
            self._notify()

    def _on_grad_param(self):
        if self._building:
            return
        layer = self.lm.get_selected_layer()
        if layer is None:
            return

        axis, flip = self._cur_axis_flip()
        layer.vis_gradient_dir  = axis      # always 0, 1, or 2
        layer.vis_gradient_flip = flip      # True for negative dirs

        if self.rb_auto.isChecked() and self._auto_mins is not None:
            self._building = True
            self.sp_min.setValue(self._auto_mins[axis])
            self.sp_max.setValue(self._auto_maxs[axis])
            self._building = False
            layer.vis_gradient_min = None
            layer.vis_gradient_max = None
        else:
            layer.vis_gradient_min = self.sp_min.value()
            layer.vis_gradient_max = self.sp_max.value()

        self._notify()

    def _notify(self):
        self.visual_changed.emit()
        if hasattr(self.lm, "visual_changed"):
            self.lm.visual_changed.emit()

    # ================================================================
    #  PUBLIC API  (viewport reads these)
    # ================================================================

    def get_gradient_range(self, layer):
        """
        Return ``(axis, min_val, max_val, is_manual)`` for the given
        layer's current gradient settings.

        *axis* is always 0, 1, or 2 (safe for ``points[:, axis]``).
        When a negative direction is active, min/max are swapped so the
        colour ramp reverses automatically.
        """
        axis = getattr(layer, "vis_gradient_dir", 2)
        flip = getattr(layer, "vis_gradient_flip", False)
        mode = getattr(layer, "vis_gradient_mode", "auto")

        if isinstance(layer, PointCloudLayer):
            coords = layer.points[:, axis]
        elif isinstance(layer, MeshLayer):
            coords = layer.vertices[:, axis]
        else:
            return axis, 0.0, 1.0, False

        if mode == "manual":
            mn = getattr(layer, "vis_gradient_min", None)
            mx = getattr(layer, "vis_gradient_max", None)
            if mn is not None and mx is not None:
                mn, mx = float(mn), float(mx)
                if flip:
                    mn, mx = mx, mn
                return axis, mn, mx, True

        mn = float(coords.min())
        mx = float(coords.max())
        if flip:
            mn, mx = mx, mn
        return axis, mn, mx, False