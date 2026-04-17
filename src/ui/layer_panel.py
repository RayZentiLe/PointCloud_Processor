from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMenu, QInputDialog, QColorDialog, QMessageBox,
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
    combine_requested = Signal(str, str)          # layer_id_a, layer_id_b
    run_pca_requested = Signal()
    run_poisson_requested = Signal()
    run_mesh_filter_requested = Signal()
    run_noise_removal_requested = Signal()

    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(parent)
        self.lm = layer_manager

        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Info"])
        self.tree.setColumnWidth(0, 200)
        self.tree.header().setStretchLastSection(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._ctx_menu)
        self.tree.itemChanged.connect(self._on_changed)
        self.tree.currentItemChanged.connect(self._on_sel)
        lo.addWidget(self.tree)

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

    # ── rebuild ──────────────────────────────────────────────────

    def _rebuild(self):
        self.tree.blockSignals(True)

        # clear children
        for grp in (self._pc_grp, self._mesh_grp):
            while grp.childCount():
                grp.removeChild(grp.child(0))

        for layer in self.lm.point_clouds.values():
            self._add_layer_item(self._pc_grp, layer, is_pc=True)
        for layer in self.lm.meshes.values():
            self._add_layer_item(self._mesh_grp, layer, is_pc=False)

        self._pc_grp.setText(1, f"{len(self.lm.point_clouds)} layers")
        self._mesh_grp.setText(1, f"{len(self.lm.meshes)} layers")

        self.tree.blockSignals(False)

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
            # positive
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

            # negative
            n = QTreeWidgetItem(item)
            n.setText(0, mg.negative_name)
            n.setText(1, f"{mg.negative_count:,} {cnt_key}")
            n.setData(0, _R_LID, layer.id)
            n.setData(0, _R_MGID, mg.id)
            n.setData(0, _R_POS, False)
            n.setData(0, _R_TYPE, "sub")
            n.setCheckState(0, Qt.Checked if mg.negative_visible else Qt.Unchecked)
            clr = mg.negative_color or (1.0, 0.3, 0.3)
            n.setIcon(0, self._color_icon(clr))

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
            self.lm.set_selection(None, None)
            return
        tp = cur.data(0, _R_TYPE)
        if tp == "layer":
            self.lm.set_selection(cur.data(0, _R_LID), None)
        elif tp == "sub":
            self.lm.set_selection(cur.data(0, _R_LID), cur.text(0))
        else:
            self.lm.set_selection(None, None)

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
            menu.addAction("Rename",
                           lambda: self._rename_layer(lid))
            menu.addAction("Set Color",
                           lambda: self._set_layer_color(lid))
            menu.addSeparator()

            tools = menu.addMenu("Run Tool")
            if isinstance(layer, PointCloudLayer):
                tools.addAction("PCA Filter",
                                lambda: self._sel_run(lid, None, "pca"))
                tools.addAction("Poisson",
                                lambda: self._sel_run(lid, None, "poisson"))
                tools.addAction("Noise Removal",
                                lambda: self._sel_run(lid, None, "noise"))
            else:
                tools.addAction("Keep Largest Component",
                                lambda: self._sel_run(lid, None, "mf"))

            combine = menu.addMenu("Combine With")
            others = (self.lm.point_clouds if isinstance(layer, PointCloudLayer)
                      else self.lm.meshes)
            for o in others.values():
                if o.id != lid:
                    combine.addAction(
                        o.name,
                        lambda oid=o.id: self.combine_requested.emit(lid, oid))
            if combine.isEmpty():
                combine.setEnabled(False)

            menu.addSeparator()
            menu.addAction("Export",
                           lambda: self.export_requested.emit(lid, None))
            menu.addAction("Delete",
                           lambda: self.delete_requested.emit(lid))

        elif tp == "sub":
            mgid = item.data(0, _R_MGID)
            is_pos = item.data(0, _R_POS)
            sname = item.text(0)
            layer = self.lm.get_layer(lid)
            if layer is None:
                return

            menu.addAction("Rename",
                           lambda: self._rename_sub(lid, mgid, is_pos))
            menu.addAction("Set Color",
                           lambda: self._set_sub_color(lid, mgid, is_pos))
            menu.addSeparator()

            tools = menu.addMenu("Run Tool")
            if isinstance(layer, PointCloudLayer):
                tools.addAction("PCA Filter",
                                lambda: self._sel_run(lid, sname, "pca"))
                tools.addAction("Poisson",
                                lambda: self._sel_run(lid, sname, "poisson"))
                tools.addAction("Noise Removal",
                                lambda: self._sel_run(lid, sname, "noise"))
            else:
                tools.addAction("Keep Largest Component",
                                lambda: self._sel_run(lid, sname, "mf"))

            menu.addSeparator()
            menu.addAction("Export Sublayer",
                           lambda: self.export_requested.emit(lid, sname))
            menu.addAction("Delete Mask Group",
                           lambda: self.delete_mask_requested.emit(lid, mgid))

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    # ── actions ──────────────────────────────────────────────────

    def _sel_run(self, lid, sname, tool):
        self.lm.set_selection(lid, sname)
        sig = {"pca": self.run_pca_requested,
               "poisson": self.run_poisson_requested,
               "noise": self.run_noise_removal_requested,
               "mf": self.run_mesh_filter_requested}[tool]
        sig.emit()

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

    def _set_layer_color(self, lid):
        layer = self.lm.get_layer(lid)
        if not layer:
            return
        cur = layer.display_color or (0.6, 0.6, 0.6)
        c = QColorDialog.getColor(
            QColor(*(int(x * 255) for x in cur)), self, "Layer Color")
        if c.isValid():
            self.lm.set_layer_color(lid, (c.redF(), c.greenF(), c.blueF()))

    def _set_sub_color(self, lid, mgid, is_pos):
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
        cur = (mg.positive_color if is_pos else mg.negative_color) or (0.6, 0.6, 0.6)
        c = QColorDialog.getColor(
            QColor(*(int(x * 255) for x in cur)), self, "Sublayer Color")
        if c.isValid():
            self.lm.set_sublayer_color(
                lid, mgid, is_pos,
                (c.redF(), c.greenF(), c.blueF()))