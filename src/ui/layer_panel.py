import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMenu, QInputDialog, QMessageBox, QGroupBox, QLabel,
    QSizePolicy, QSplitter,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QColor
from core.layer_manager import LayerManager
from core.layer import PointCloudLayer, MeshLayer


_R_LID = Qt.UserRole          # layer id
_R_MGID = Qt.UserRole + 1     # mask-group id
_R_POS = Qt.UserRole + 2      # is-positive bool
_R_TYPE = Qt.UserRole + 3     # "pc_grp" | "mesh_grp" | "layer" | "sub"


class LayerPanel(QWidget):
    export_requested = Signal(str, object)       # layer_id, sublayer_name|None
    delete_requested = Signal(str)               # layer_id
    delete_mask_requested = Signal(str, str)      # layer_id, mg_id

    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(parent)
        self.lm = layer_manager

        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(0)

        # ── splitter: tree on top, properties on bottom ──────────
        self._splitter = QSplitter(Qt.Vertical)
        lo.addWidget(self._splitter)

        # --- tree ---
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Info"])
        self.tree.setColumnWidth(0, 200)
        self.tree.header().setStretchLastSection(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._ctx_menu)
        self.tree.itemChanged.connect(self._on_changed)
        self.tree.currentItemChanged.connect(self._on_sel)
        self._splitter.addWidget(self.tree)

        # --- properties panel (shell) ---
        self._props_box = QGroupBox("Properties")
        self._props_box.setVisible(False)
        self._props_layout = QVBoxLayout(self._props_box)
        self._props_layout.setContentsMargins(6, 6, 6, 6)
        self._props_label = QLabel("No properties to display.")
        self._props_label.setAlignment(Qt.AlignCenter)
        self._props_label.setStyleSheet("color: #888;")
        self._props_layout.addWidget(self._props_label)
        self._splitter.addWidget(self._props_box)

        # tree gets most space, props starts small
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, False)

        # remember user-chosen sizes so we can restore after hide/show
        self._last_sizes = None
        self._splitter.splitterMoved.connect(self._on_splitter_moved)

        # top-level groups
        self._pc_grp = self._make_group("📁 Point Clouds", "pc_grp")
        self._mesh_grp = self._make_group("📁 Meshes", "mesh_grp")

        # connect manager
        self.lm.layer_added.connect(lambda _: self._rebuild())
        self.lm.layer_removed.connect(lambda _: self._rebuild())
        self.lm.layer_modified.connect(lambda _: self._rebuild())
        self.lm.layer_renamed.connect(lambda _: self._rebuild())
        self.lm.mask_added.connect(lambda a, b: self._rebuild())
        self.lm.mask_removed.connect(lambda a, b: self._rebuild())

    # ── helpers ──────────────────────────────────────────────────

    def _make_group(self, label, tag):
        item = QTreeWidgetItem(self.tree, [label, ""])
        item.setData(0, _R_TYPE, tag)
        item.setExpanded(True)
        item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
        return item

    @staticmethod
    def _color_icon(rgb):
        r, g, b = (int(c * 255) for c in rgb)
        px = QPixmap(16, 16)
        px.fill(QColor(r, g, b))
        return QIcon(px)

    # ── splitter memory ──────────────────────────────────────────

    def _on_splitter_moved(self, pos, index):
        if self._props_box.isVisible():
            self._last_sizes = self._splitter.sizes()

    # ── rebuild ──────────────────────────────────────────────────

    def _rebuild(self):
        self.tree.blockSignals(True)

        saved_lid = self.lm.selected_layer_id
        saved_sub = self.lm.selected_sublayer_name

        for grp in (self._pc_grp, self._mesh_grp):
            while grp.childCount():
                grp.removeChild(grp.child(0))

        item_to_select = None

        for layer in self.lm.point_clouds.values():
            item = self._add_layer_item(self._pc_grp, layer, is_pc=True)
            if layer.id == saved_lid:
                if saved_sub is None:
                    item_to_select = item
                else:
                    for ci in range(item.childCount()):
                        child = item.child(ci)
                        if child.text(0) == saved_sub:
                            item_to_select = child
                            break
                    if item_to_select is None:
                        item_to_select = item

        for layer in self.lm.meshes.values():
            item = self._add_layer_item(self._mesh_grp, layer, is_pc=False)
            if layer.id == saved_lid:
                if saved_sub is None:
                    item_to_select = item
                else:
                    for ci in range(item.childCount()):
                        child = item.child(ci)
                        if child.text(0) == saved_sub:
                            item_to_select = child
                            break
                    if item_to_select is None:
                        item_to_select = item

        self._pc_grp.setText(1, f"{len(self.lm.point_clouds)} layers")
        self._mesh_grp.setText(1, f"{len(self.lm.meshes)} layers")

        self.tree.blockSignals(False)

        if item_to_select is not None:
            self.tree.setCurrentItem(item_to_select)

        self._update_props()

    def _add_layer_item(self, parent, layer, is_pc):
        item = QTreeWidgetItem(parent)
        item.setText(0, layer.name)
        if is_pc:
            item.setText(1, f"{layer.point_count:,} pts")
        else:
            item.setText(1, f"{layer.face_count:,} faces")
        item.setData(0, _R_LID, layer.id)
        item.setData(0, _R_TYPE, "layer")
        item.setCheckState(0, Qt.Checked if layer.visible else Qt.Unchecked)
        base_clr = layer.display_color or (0.6, 0.6, 0.6)
        item.setIcon(0, self._color_icon(base_clr))
        item.setExpanded(True)

        for mg in layer.mask_groups:
            p = QTreeWidgetItem(item)
            p.setText(0, mg.positive_name)
            cnt_key = "pts" if is_pc else "faces"
            p.setText(1, f"{mg.positive_count:,} {cnt_key}")
            p.setData(0, _R_LID, layer.id)
            p.setData(0, _R_MGID, mg.id)
            p.setData(0, _R_POS, True)
            p.setData(0, _R_TYPE, "sub")
            p.setCheckState(0, Qt.Checked if mg.positive_visible else Qt.Unchecked)
            clr = mg.positive_color or layer.display_color or (0.6, 0.6, 0.6)
            p.setIcon(0, self._color_icon(clr))

            n = QTreeWidgetItem(item)
            n.setText(0, mg.negative_name)
            n.setText(1, f"{mg.negative_count:,} {cnt_key}")
            n.setData(0, _R_LID, layer.id)
            n.setData(0, _R_MGID, mg.id)
            n.setData(0, _R_POS, False)
            n.setData(0, _R_TYPE, "sub")
            n.setCheckState(0, Qt.Checked if mg.negative_visible else Qt.Unchecked)
            clr = mg.negative_color or layer.display_color or (0.6, 0.6, 0.6)
            n.setIcon(0, self._color_icon(clr))

        return item

    # ── checkbox / selection ─────────────────────────────────────

    def _on_changed(self, item, col):
        if col != 0:
            return
        tp = item.data(0, _R_TYPE)
        lid = item.data(0, _R_LID)
        vis = item.checkState(0) == Qt.Checked
        if tp == "layer":
            self.lm.set_layer_visibility(lid, vis)
        elif tp == "sub":
            self.lm.set_sublayer_visibility(
                lid, item.data(0, _R_MGID), item.data(0, _R_POS), vis)

    def _on_sel(self, cur, _prev):
        if cur is None:
            self._update_props()
            return
        tp = cur.data(0, _R_TYPE)
        if tp == "layer":
            self.lm.set_selection(cur.data(0, _R_LID), None)
        elif tp == "sub":
            self.lm.set_selection(cur.data(0, _R_LID), cur.text(0))
        self._update_props()

    # ── properties panel ─────────────────────────────────────────

    def _update_props(self):
        """Show properties box when a layer or sublayer is selected, hide otherwise."""
        cur = self.tree.currentItem()
        if cur is None:
            self._show_props(False)
            return

        tp = cur.data(0, _R_TYPE)
        if tp not in ("layer", "sub"):
            self._show_props(False)
            return

        lid = cur.data(0, _R_LID)
        layer = self.lm.get_layer(lid)
        if layer is None:
            self._show_props(False)
            return

        if tp == "layer":
            self._props_box.setTitle(f"Properties — {layer.name}")
        elif tp == "sub":
            self._props_box.setTitle(f"Properties — {cur.text(0)}")

        self._show_props(True)

    def _show_props(self, visible):
        was = self._props_box.isVisible()
        if visible == was:
            return

        if visible:
            self._props_box.setVisible(True)
            # restore last user sizes, or use a sensible default
            if self._last_sizes:
                self._splitter.setSizes(self._last_sizes)
            else:
                total = self._splitter.height()
                props_h = max(120, total // 4)
                self._splitter.setSizes([total - props_h, props_h])
        else:
            # save current sizes before hiding
            self._last_sizes = self._splitter.sizes()
            self._props_box.setVisible(False)

    # ── context menu ─────────────────────────────────────────────

    def _ctx_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        tp = item.data(0, _R_TYPE)
        lid = item.data(0, _R_LID)
        menu = QMenu(self)

        if tp == "layer":
            layer = self.lm.get_layer(lid)
            if layer is None:
                return

            menu.addAction("Select",
                           lambda: self._select_item(item))
            menu.addAction("Rename",
                           lambda: self._rename_layer(lid))
            menu.addSeparator()
            menu.addAction("Export",
                           lambda: self.export_requested.emit(lid, None))
            menu.addAction("Delete",
                           lambda: self.delete_requested.emit(lid))

        elif tp == "sub":
            mgid = item.data(0, _R_MGID)
            sname = item.text(0)
            layer = self.lm.get_layer(lid)
            if layer is None:
                return

            menu.addAction("Select",
                           lambda: self._select_item(item))
            menu.addAction("Rename",
                           lambda: self._rename_sub(lid, mgid,
                                                    item.data(0, _R_POS)))
            menu.addSeparator()
            menu.addAction("Export Sublayer",
                           lambda: self.export_requested.emit(lid, sname))
            menu.addAction("Delete Mask Group",
                           lambda: self.delete_mask_requested.emit(lid, mgid))

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    # ── actions ──────────────────────────────────────────────────

    def _select_item(self, item):
        self.tree.setCurrentItem(item)

    def _rename_layer(self, lid):
        layer = self.lm.get_layer(lid)
        if not layer:
            return
        txt, ok = QInputDialog.getText(
            self, "Rename Layer", "New name:", text=layer.name)
        if ok and txt.strip():
            ok2, msg = self.lm.rename_layer(lid, txt.strip())
            if not ok2:
                QMessageBox.warning(self, "Rename", msg)

    def _rename_sub(self, lid, mgid, is_pos):
        layer = self.lm.get_layer(lid)
        if not layer:
            return
        mg = None
        for m in layer.mask_groups:
            if m.id == mgid:
                mg = m
                break
        if mg is None:
            return
        cur = mg.positive_name if is_pos else mg.negative_name
        txt, ok = QInputDialog.getText(
            self, "Rename Sublayer", "New name:", text=cur)
        if ok and txt.strip():
            ok2, msg = self.lm.rename_sublayer(lid, mgid, is_pos, txt.strip())
            if not ok2:
                QMessageBox.warning(self, "Rename", msg)