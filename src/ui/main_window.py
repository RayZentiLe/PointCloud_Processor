import os
import sys
import gc
import colorsys
import traceback
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QFileDialog,
    QMessageBox, QProgressBar, QInputDialog,
    QDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer

from core.layer_manager import LayerManager
from core.layer import PointCloudLayer, MeshLayer
# Import Viewport lazily in _build_ui to handle environments where VTK
# cannot be imported (e.g. blocked by policy). A lightweight fallback
# viewport will be used in that case to keep the UI functional.
from ui.layer_panel import LayerPanel
from ui.properties_panel import PropertiesPanel
from ui.toolbar import Toolbar
from ui.log_panel import LogPanel
from ui.dialogs.loading_dialog import LoadingDialog
# Defer heavy I/O imports (open3d, exporters) until runtime to avoid
# import-time failures in environments without those packages.
from workers.task_runner import TaskRunner


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Processor")
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        self.lm = LayerManager(self)
        self._worker: TaskRunner | None = None
        self._loading_dialog: LoadingDialog | None = None
        self._undo_stack = []
        self._redo_stack = []

        self._build_ui()
        self._build_menus()
        self._connect()
        QTimer.singleShot(0, self._load_startup_test_files)

    # ── UI setup ─────────────────────────────────────────────────

    def _build_ui(self):
        self.setDockNestingEnabled(False)
        self.setDockOptions(QMainWindow.AnimatedDocks)

        # Try to import the rich VTK-based viewport. If VTK is unavailable
        # (import errors), fall back to a lightweight placeholder so the
        # rest of the UI (docks, panels) can still be used.
        try:
            from ui.viewport import Viewport
            self.viewport = Viewport(self.lm, self)
        except Exception as e:
            from PySide6.QtWidgets import QLabel
            print(f"[MainWindow] Viewport import failed: {e}", file=sys.stderr)
            class ViewportFallback(QLabel):
                def __init__(self, *a, **k):
                    super().__init__("Viewport unavailable: VTK import failed.\nCheck application logs.")
                def fit_all(self):
                    return
                def focus_camera_on_layer(self, layer_id):
                    return
                def enable_cross_section_mode(self, layer):
                    return
                def disable_cross_section_mode(self):
                    return
                def set_cross_section_pick_mode(self, mode):
                    return
                def set_cross_section_direction_confirmed(self, confirmed):
                    return
                def clear_cross_section_direction_points(self):
                    return
                def clear_cross_section_reference_point(self):
                    return
                def get_cross_section_state(self):
                    return {"active": False, "layer_id": None, "direction_points": [], "reference_point": None, "direction_xy": None, "thickness": 1.0, "mode": "none"}

            self.viewport = ViewportFallback()

        self.setCentralWidget(self.viewport)
        self.set_app_font_size(12)  # Default to Medium size

        # left dock – layer tree
        self.layer_panel = LayerPanel(self.lm, self)
        self.layers_dock = QDockWidget("Layers", self)
        self.layers_dock.setObjectName("LayersDock")
        self.layers_dock.setWidget(self.layer_panel)
        self.layers_dock.setMinimumWidth(290)
        self.layers_dock.setFeatures(QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.layers_dock)

        # right dock – properties
        self.props_panel = PropertiesPanel(self.lm, self)
        self.properties_dock = QDockWidget("Properties", self)
        self.properties_dock.setObjectName("PropertiesDock")
        self.properties_dock.setWidget(self.props_panel)
        self.properties_dock.setMinimumWidth(350)
        self.properties_dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.properties_dock.setFeatures(
            QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.properties_dock)

        # right dock – cross section
        from ui.cross_section_panel import CrossSectionPanel
        self.cross_section_panel = CrossSectionPanel(self.viewport, self.lm, self)
        self.cross_section_panel.cross_section_created.connect(self._on_cross_section_created)
        self.cross_section_panel.points_transfer_requested.connect(self._on_cross_section_points_transfer_requested)
        self.cross_section_dock = _CrossSectionDockWidget("Cross Section", self)
        self.cross_section_dock.setObjectName("CrossSectionDock")
        self.cross_section_dock.setWidget(self.cross_section_panel)
        self.cross_section_dock.setMinimumWidth(350)
        self.cross_section_dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cross_section_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self.cross_section_dock.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.cross_section_dock)
        self.splitDockWidget(self.properties_dock, self.cross_section_dock, Qt.Vertical)
        self.cross_section_dock.setVisible(True)

        QTimer.singleShot(0, self._apply_initial_dock_layout)
        QTimer.singleShot(0, self._capture_initial_dock_state)

        self.toolbar = Toolbar(self.lm, self)
        self.toolbar.setObjectName("MainToolbar")
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.toolbar.setAllowedAreas(Qt.TopToolBarArea)
        self.toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.log = LogPanel(self)
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setWidget(self.log)
        self.log_dock.setMaximumHeight(200)
        self.log_dock.setFeatures(QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

        # Pass dock widgets to toolbar for Windows menu (include cross section)
        self.toolbar.set_dock_widgets(self.layers_dock, self.properties_dock, self.log_dock, self.cross_section_dock)

        self.pbar = QProgressBar()
        self.pbar.setMaximumWidth(300)
        self.pbar.setVisible(False)
        self.statusBar().addPermanentWidget(self.pbar)

    def _apply_initial_dock_layout(self):
        if hasattr(self, "cross_section_dock") and self.cross_section_dock.isFloating():
            if hasattr(self, "properties_dock") and self.properties_dock.isVisible():
                self.properties_dock.setMinimumWidth(350)
                self.properties_dock.resize(max(350, self.width() // 4), self.properties_dock.height())
            return
        if hasattr(self, "properties_dock") and hasattr(self, "cross_section_dock") and self.properties_dock.isVisible() and self.cross_section_dock.isVisible():
            self.resizeDocks(
                [self.properties_dock, self.cross_section_dock],
                [380, 320],
                Qt.Vertical,
            )
        elif hasattr(self, "cross_section_dock") and self.cross_section_dock.isVisible():
            self.resizeDocks(
                [self.cross_section_dock],
                [max(320, self.height() - self.log_dock.height())],
                Qt.Vertical,
            )
        if hasattr(self, "layers_dock") and hasattr(self, "properties_dock"):
            self.properties_dock.setMinimumWidth(350)
            self.resizeDocks(
                [self.layers_dock, self.properties_dock],
                [290, 350],
                Qt.Horizontal,
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_initial_dock_layout()

    def _load_startup_test_files(self):
        loaded_count = 0
        missing_files = []
        for path in getattr(self, "_startup_test_files", []):
            if not os.path.isfile(path):
                missing_files.append(path)
                continue
            if self._open_path(path, show_error_dialog=False):
                loaded_count += 1

        if loaded_count:
            self.log.log(f"TESTING startup load: opened {loaded_count} point cloud file(s).")
        if missing_files:
            self.log.log("TESTING startup load missing files: " + ", ".join(missing_files))

    def _capture_initial_dock_state(self):
        try:
            self._initial_dock_state = self.saveState()
        except Exception:
            self._initial_dock_state = None

    def _restore_cross_section_dock_position(self):
        if not hasattr(self, "cross_section_dock"):
            return
        if not self.cross_section_dock.isVisible():
            return
        layers_visible = self.layers_dock.isVisible() if hasattr(self, "layers_dock") else None
        properties_visible = self.properties_dock.isVisible() if hasattr(self, "properties_dock") else None
        log_visible = self.log_dock.isVisible() if hasattr(self, "log_dock") else None
        self.cross_section_dock.hide()
        self.cross_section_dock.setParent(self)
        self.cross_section_dock.setFloating(False)
        if self._initial_dock_state is not None:
            self.restoreState(self._initial_dock_state)
        else:
            self.removeDockWidget(self.cross_section_dock)
            self.addDockWidget(Qt.RightDockWidgetArea, self.cross_section_dock)
            if hasattr(self, "properties_dock") and self.properties_dock.isVisible():
                self.splitDockWidget(self.properties_dock, self.cross_section_dock, Qt.Vertical)
        if layers_visible is not None:
            self.layers_dock.setVisible(layers_visible)
        if properties_visible is not None:
            self.properties_dock.setVisible(properties_visible)
        if log_visible is not None:
            self.log_dock.setVisible(log_visible)
        self.cross_section_dock.setVisible(False)
        self._apply_initial_dock_layout()

    def _build_menus(self):
        fm = self.menuBar().addMenu("&File")
        fm.addAction("Open…", self._open, "Ctrl+O")
        fm.addAction("Export Selected…", self._export_sel, "Ctrl+E")
        fm.addSeparator()
        fm.addAction("Quit", self.close, "Ctrl+Q")

    def _connect(self):
        tb = self.toolbar
        tb.open_requested.connect(self._open)
        tb.undo_requested.connect(self._on_global_undo_requested)
        tb.redo_requested.connect(self._on_global_redo_requested)
        tb.auto_denoise_requested.connect(self._run_auto_denoise)
        tb.pca_requested.connect(self._run_pca)
        tb.poisson_requested.connect(self._run_poisson)
        tb.mesh_filter_requested.connect(self._run_mf)
        tb.noise_removal_requested.connect(self._run_noise)
        tb.export_requested.connect(self._export_sel)
        tb.cross_section_requested.connect(self._show_cross_section_panel)
        tb.font_size_changed.connect(self.set_app_font_size)  # Connect font size changes

        lp = self.layer_panel
        lp.export_requested.connect(self._export_layer)
        lp.delete_requested.connect(self._delete_layer)
        lp.delete_layers_requested.connect(self._delete_layers)
        lp.delete_mask_requested.connect(self._delete_mask)
        lp.combine_layers_requested.connect(self._combine_selected_layers)
        lp.assign_colors_requested.connect(self._assign_colors_to_layers)
        lp.copy_layers_requested.connect(self._copy_layers)
        lp.camera_to_layer_requested.connect(self.viewport.focus_camera_on_layer)
        
        # Connect dock widget visibility changes to toolbar menu
        self.layers_dock.visibilityChanged.connect(self._on_layers_visibility_changed)
        self.properties_dock.visibilityChanged.connect(self._on_properties_visibility_changed)
        self.cross_section_dock.visibilityChanged.connect(self._on_cross_section_visibility_changed)
        self.cross_section_dock.topLevelChanged.connect(self._on_cross_section_top_level_changed)
        self.log_dock.visibilityChanged.connect(self._on_log_visibility_changed)
        self.lm.selection_changed.connect(self.on_layer_selected)
        self.lm.visibility_changed.connect(self._on_layer_visibility_changed)
        self._update_global_undo_redo_state()

    # ── helpers ──────────────────────────────────────────────────

    def _get_or_pick_pc(self):
        layer = self.lm.get_selected_layer()
        if isinstance(layer, PointCloudLayer):
            return layer
        pcs = list(self.lm.point_clouds.values())
        if len(pcs) == 1:
            self.lm.set_selection(pcs[0].id)
            self.log.log(f"Auto-selected: {pcs[0].name}")
            return pcs[0]
        if len(pcs) > 1:
            self.log.log("Multiple point clouds — please select one in the Layers panel.")
        return None

    def _get_or_pick_mesh(self):
        layer = self.lm.get_selected_layer()
        if isinstance(layer, MeshLayer):
            return layer
        meshes = list(self.lm.meshes.values())
        if len(meshes) == 1:
            self.lm.set_selection(meshes[0].id)
            self.log.log(f"Auto-selected: {meshes[0].name}")
            return meshes[0]
        if len(meshes) > 1:
            self.log.log("Multiple meshes — please select one in the Layers panel.")
        return None

    # ── file I/O ─────────────────────────────────────────────────

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open", "",
            "Supported (*.ply *.pcd *.obj *.stl *.xyz *.txt);;All (*)")
        if not path:
            return
        self._open_path(path)

    def dragEnterEvent(self, event):
        mime_data = event.mimeData()
        if mime_data is None or not mime_data.hasUrls():
            event.ignore()
            return

        local_files = [url.toLocalFile() for url in mime_data.urls() if url.isLocalFile()]
        if any(os.path.isfile(path) for path in local_files):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        mime_data = event.mimeData()
        if mime_data is None or not mime_data.hasUrls():
            event.ignore()
            return

        local_files = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if os.path.isfile(path):
                local_files.append(path)

        if not local_files:
            event.ignore()
            return

        loaded_count = 0
        for path in local_files:
            if self._open_path(path, show_error_dialog=False):
                loaded_count += 1

        if loaded_count == 0:
            QMessageBox.warning(self, "Drop Input", "No dropped files could be loaded.")
            event.ignore()
            return

        self.log.log(f"Dropped input: loaded {loaded_count} file(s).")
        event.acceptProposedAction()

    def _open_path(self, path, show_error_dialog=True):
        try:
            from io_utils.ply_io import load_file
            layer = load_file(path)
            if isinstance(layer, PointCloudLayer):
                self.lm.add_point_cloud(layer)
                self.log.log(
                    f"Loaded point cloud: {layer.name} "
                    f"({layer.point_count:,} pts)")
            elif isinstance(layer, MeshLayer):
                self.lm.add_mesh(layer)
                self.log.log(
                    f"Loaded mesh: {layer.name} "
                    f"({layer.face_count:,} faces)")
            self.lm.set_selection(layer.id)
            self.viewport.fit_all()
            return True
        except Exception as e:
            self.log.log(f"ERROR loading: {e}")
            if show_error_dialog:
                QMessageBox.critical(self, "Error", str(e))
            return False

    def _export_sel(self):
        layer = self.lm.get_selected_layer()
        if layer is None:
            QMessageBox.information(self, "Export", "No layer selected.")
            return
        self._export_layer(layer.id, self.lm.selected_sublayer_name)

    def _export_layer(self, lid, sname=None):
        layer = self.lm.get_layer(lid)
        if layer is None:
            return

        base = layer.name + (f"_{sname}" if sname else "")

        if isinstance(layer, PointCloudLayer):
            path, filt = QFileDialog.getSaveFileName(
                self, "Export Point Cloud", base + ".ply",
                "PLY binary (*.ply);;PLY ASCII (*.ply);;XYZ (*.xyz);;TXT (*.txt)")
            if not path:
                return

            # Ensure the chosen filter extension is reflected in the file path.
            if "TXT" in filt:
                desired_ext = ".txt"
            elif "XYZ" in filt:
                desired_ext = ".xyz"
            else:
                desired_ext = ".ply"

            root, ext = os.path.splitext(path)
            if ext.lower() != desired_ext:
                path = root + desired_ext

            binary = "ASCII" not in filt and desired_ext not in (".txt", ".xyz")
            try:
                from io_utils.exporter import export_point_cloud
                export_point_cloud(layer, path, sname, binary)
                n = layer.point_count
                if sname:
                    n = int(self.lm.get_sublayer_mask(layer, sname).sum())
                self.log.log(f"Exported {n:,} pts → {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

        elif isinstance(layer, MeshLayer):
            path, filt = QFileDialog.getSaveFileName(
                self, "Export Mesh", base + ".ply",
                "PLY binary (*.ply);;PLY ASCII (*.ply);;OBJ (*.obj)")
            if not path:
                return
            binary = "ASCII" not in filt and not path.endswith(".obj")
            try:
                from io_utils.exporter import export_mesh
                export_mesh(layer, path, sname, binary)
                self.log.log(f"Exported mesh → {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    # ── delete ───────────────────────────────────────────────────

    def _delete_layer(self, lid):
        layer = self.lm.get_layer(lid)
        if layer is None:
            return
        if QMessageBox.question(
                self, "Delete", f"Delete '{layer.name}'?",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self._delete_layers_with_history([lid])

    def _delete_layers(self, layer_ids):
        layers = []
        seen = set()
        for layer_id in layer_ids:
            if layer_id in seen:
                continue
            layer = self.lm.get_layer(layer_id)
            if layer is None:
                continue
            layers.append(layer)
            seen.add(layer_id)
        if not layers:
            return
        if QMessageBox.question(
                self,
                "Delete",
                f"Delete {len(layers)} selected layers?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self._delete_layers_with_history([layer.id for layer in layers])

    def _delete_layers_with_history(self, layer_ids):
        snapshots = []
        seen = set()
        for layer_id in layer_ids:
            if layer_id in seen:
                continue
            layer = self.lm.get_layer(layer_id)
            if layer is None:
                continue
            snapshots.append(self._snapshot_layer(layer))
            seen.add(layer_id)
        if not snapshots:
            return
        for snapshot in snapshots:
            self.lm.remove_layer(snapshot["id"])
            self.log.log(f"Deleted: {snapshot['name']}")
        self._undo_stack.append({
            "type": "delete",
            "layers": snapshots,
        })
        self._redo_stack.clear()
        self._update_global_undo_redo_state()

    def _delete_mask(self, lid, mgid):
        self.lm.remove_mask_group(lid, mgid)
        self.log.log("Mask group deleted")

    def _assign_colors_to_layers(self, layer_ids):
        unique_ids = []
        seen = set()
        for layer_id in layer_ids:
            if layer_id in seen:
                continue
            layer = self.lm.get_layer(layer_id)
            if layer is None:
                continue
            unique_ids.append(layer_id)
            seen.add(layer_id)
        if not unique_ids:
            return

        total = len(unique_ids)
        for index, layer_id in enumerate(unique_ids):
            layer = self.lm.get_layer(layer_id)
            if layer is None:
                continue
            hue = (index / max(total, 1) + 0.11) % 1.0
            sat = 0.65 + 0.2 * ((index % 2) / 1 if total > 1 else 0.5)
            val = 0.9
            rgb = colorsys.hsv_to_rgb(hue, min(sat, 0.9), val)
            layer.vis_color_scheme = "Solid"
            layer.vis_solid_color = rgb
            layer.render_props["color_mode"] = "solid"
            layer.render_props["solid_color"] = list(rgb)
            self.lm.layer_modified.emit(layer.id)
            self.log.log(f"Assigned colour to: {layer.name}")

    def _copy_layers(self, layer_ids):
        copied_ids = self.lm.copy_layers(layer_ids)
        if not copied_ids:
            return
        copied_layers = [self.lm.get_layer(layer_id) for layer_id in copied_ids]
        copied_layers = [layer for layer in copied_layers if layer is not None]
        if not copied_layers:
            return
        self.lm.set_selected_layers(copied_ids, copied_ids[0], None)
        for layer in copied_layers:
            self.log.log(f"Copied layer: {layer.name}")

    # ── PCA ──────────────────────────────────────────────────────

    def _run_pca(self):
        try:
            layer = self._get_or_pick_pc()
            if layer is None:
                QMessageBox.information(
                    self, "PCA Filter",
                    "Select a point cloud first.\n\n"
                    "Click on a point cloud in the Layers panel, "
                    "then run PCA Filter.")
                return

            self.log.log(f"PCA Filter: opening dialog for '{layer.name}'...")
            print(f"[MainWindow] PCA: layer={layer.name}, "
                  f"points={layer.point_count}", file=sys.stderr)

            from ui.dialogs.pca_dialog import PCADialog
            dlg = PCADialog(self)
            if dlg.exec() != QDialog.Accepted:
                self.log.log("PCA Filter: cancelled.")
                return
            p = dlg.get_params()
            self.log.log(f"PCA Filter: running with radius={p['radius']}, "
                         f"threshold={p['threshold']}...")

            sname = self.lm.selected_sublayer_name
            if sname:
                mask = self.lm.get_sublayer_mask(layer, sname)
                indices = np.where(mask)[0]
                pts = layer.points[indices]
                self.log.log(f"  Operating on sublayer '{sname}' "
                             f"({len(pts):,} pts)")
            else:
                pts = layer.points
                indices = None
                self.log.log(f"  Operating on entire layer ({len(pts):,} pts)")

            from tools.pca_filter import run_pca_filter
            lid = layer.id
            self._launch(run_pca_filter,
                         points=pts, indices=indices,
                         total_count=layer.point_count,
                         radius=p["radius"], threshold=p["threshold"],
                         k_neighbors=p["k_neighbors"],
                         chunk_size=p["chunk_size"],
                         on_done=lambda r, _lid=lid: self._pca_done(_lid, r),
                         loading_title="PCA Filter",
                         loading_message=f"Processing PCA on {layer.name}...")

        except Exception as e:
            msg = f"PCA setup error: {e}\n{traceback.format_exc()}"
            self.log.log(f"ERROR: {msg}")
            print(msg, file=sys.stderr)

    def _pca_done(self, lid, mg):
        if mg is None:
            # Task was cancelled
            self.log.log("PCA Filter: cancelled by user.")
            return
        self.lm.add_mask_group(lid, mg)
        self.log.log(
            f"PCA Filter complete: {mg.positive_count:,} kept, "
            f"{mg.negative_count:,} rejected")

    def _run_auto_denoise(self):
        try:
            layer = self._get_or_pick_pc()
            if layer is None:
                QMessageBox.information(
                    self, "Auto Denoise",
                    "Select a point cloud first.")
                return

            from ui.dialogs.pca_dialog import PCADialog
            pca_dlg = PCADialog(self)
            pca_dlg.setWindowTitle("Auto Denoise - PCA Settings")
            if pca_dlg.exec() != QDialog.Accepted:
                self.log.log("Auto Denoise: cancelled at PCA step.")
                return
            pca_params = pca_dlg.get_params()

            from ui.dialogs.poisson_dialog import PoissonDialog
            poisson_dlg = PoissonDialog(self)
            poisson_dlg.setWindowTitle("Auto Denoise - Poisson Settings")
            if poisson_dlg.exec() != QDialog.Accepted:
                self.log.log("Auto Denoise: cancelled at Poisson step.")
                return
            poisson_params = poisson_dlg.get_params()

            from PySide6.QtWidgets import QInputDialog
            threshold, ok = QInputDialog.getDouble(
                self,
                "Auto Denoise - Noise Removal",
                "Distance Threshold:",
                1.0,
                0.0001,
                1000.0,
                4,
            )
            if not ok:
                self.log.log("Auto Denoise: cancelled at Noise Removal step.")
                return

            self.log.log(f"Auto Denoise: starting on '{layer.name}'...")
            self._start_auto_denoise_pipeline(layer, pca_params, poisson_params, threshold)

        except Exception as e:
            msg = f"Auto denoise setup error: {e}\n{traceback.format_exc()}"
            self.log.log(f"ERROR: {msg}")
            print(msg, file=sys.stderr)

    def _start_auto_denoise_pipeline(self, layer, pca_params, poisson_params, noise_threshold):
        sname = self.lm.selected_sublayer_name
        if sname:
            mask = self.lm.get_sublayer_mask(layer, sname)
            indices = np.where(mask)[0]
            pts = layer.points[indices]
            self.log.log(f"Auto Denoise: PCA on sublayer '{sname}' ({len(pts):,} pts)")
        else:
            pts = layer.points
            indices = None
            self.log.log(f"Auto Denoise: PCA on entire layer ({len(pts):,} pts)")

        from tools.pca_filter import run_pca_filter
        self._launch(
            run_pca_filter,
            points=pts,
            indices=indices,
            total_count=layer.point_count,
            radius=pca_params["radius"],
            threshold=pca_params["threshold"],
            k_neighbors=pca_params["k_neighbors"],
            chunk_size=pca_params["chunk_size"],
            on_done=lambda r, _layer=layer, _pp=poisson_params, _nt=noise_threshold:
                self._auto_denoise_after_pca(_layer, r, _pp, _nt),
            loading_title="Auto Denoise",
            loading_message=f"Running PCA filter on {layer.name}...",
        )

    def _auto_denoise_after_pca(self, layer, mg, poisson_params, noise_threshold):
        if mg is None:
            self.log.log("Auto Denoise: cancelled during PCA step.")
            return

        self.lm.add_mask_group(layer.id, mg)
        self.log.log(
            f"Auto Denoise: PCA complete ({mg.positive_count:,} kept, {mg.negative_count:,} rejected)")

        kept_mask = self.lm.get_sublayer_mask(layer, "pca_kept")
        pts = layer.points[kept_mask]
        clr = layer.colors[kept_mask] if layer.colors is not None else None

        if len(pts) == 0:
            QMessageBox.warning(self, "Auto Denoise", "PCA kept no points.")
            return

        mesh_name = f"{layer.name}_pca_kept_poisson"
        from tools.poisson import run_poisson
        self._launch(
            run_poisson,
            points=pts,
            colors=clr,
            depth=poisson_params["depth"],
            scale=poisson_params["scale"],
            density_quantile=poisson_params["density_quantile"],
            linear_fit=poisson_params["linear_fit"],
            on_done=lambda r, _layer=layer, _name=mesh_name, _nt=noise_threshold:
                self._auto_denoise_after_poisson(_layer, r, _name, _nt),
            loading_title="Auto Denoise",
            loading_message=f"Running Poisson on {layer.name} pca_kept...",
        )

    def _auto_denoise_after_poisson(self, source_layer, result, mesh_name, noise_threshold):
        ml, mg, n_comp, big, small = result
        ml.name = mesh_name
        self.lm.add_mesh(ml)
        self.lm.add_mask_group(ml.id, mg)
        self.log.log(
            f"Auto Denoise: Poisson complete for {ml.name} ({ml.face_count:,} faces). "
            f"Mesh Filter: {n_comp} components, largest {big:,}, small {small:,} faces")

        largest_faces = self.lm.get_sublayer_mask(ml, "largest")
        if not np.any(largest_faces):
            QMessageBox.warning(self, "Auto Denoise", "Largest mesh component is empty.")
            return

        used_v = np.unique(ml.faces[largest_faces].ravel())
        mesh_verts = ml.vertices[used_v]

        from tools.noise_removal import run_noise_removal
        self._launch(
            run_noise_removal,
            points=source_layer.points,
            indices=None,
            total_count=source_layer.point_count,
            mesh_vertices=mesh_verts,
            threshold=noise_threshold,
            on_done=lambda r, _lid=source_layer.id, _mesh_id=ml.id:
                self._auto_denoise_after_noise(_lid, _mesh_id, r),
            loading_title="Auto Denoise",
            loading_message=f"Removing noise from {source_layer.name} using largest mesh...",
        )

    def _auto_denoise_after_noise(self, source_layer_id, mesh_layer_id, mg):
        if mg is None:
            self.log.log("Auto Denoise: cancelled during Noise Removal step.")
            return
        self.lm.add_mask_group(source_layer_id, mg)
        self.lm.set_selection(mesh_layer_id)
        self.log.log(
            f"Auto Denoise complete: {mg.positive_count:,} clean, {mg.negative_count:,} noise")
        self.viewport.fit_all()

    # ── Poisson ──────────────────────────────────────────────────

    def _run_poisson(self):
        try:
            layer = self._get_or_pick_pc()
            if layer is None:
                QMessageBox.information(
                    self, "Poisson",
                    "Select a point cloud first.\n\n"
                    "Click on a point cloud in the Layers panel, "
                    "then run Poisson.")
                return

            self.log.log(
                f"Poisson: opening dialog for '{layer.name}'...")
            print(f"[MainWindow] Poisson: layer={layer.name}, "
                  f"points={layer.point_count}", file=sys.stderr)

            from ui.dialogs.poisson_dialog import PoissonDialog
            dlg = PoissonDialog(self)
            if dlg.exec() != QDialog.Accepted:
                self.log.log("Poisson: cancelled.")
                return
            p = dlg.get_params()
            self.log.log(
                f"Poisson: running with depth={p['depth']}, "
                f"scale={p['scale']}...")

            sname = self.lm.selected_sublayer_name
            if sname:
                mask = self.lm.get_sublayer_mask(layer, sname)
                pts = layer.points[mask]
                clr = layer.colors[mask] if layer.colors is not None else None
                self.log.log(
                    f"  Operating on sublayer '{sname}' ({len(pts):,} pts)")
            else:
                pts = layer.points
                clr = layer.colors
                self.log.log(
                    f"  Operating on entire layer ({len(pts):,} pts)")

            mesh_name = layer.name
            if sname:
                mesh_name += f"_{sname}"
            mesh_name += "_poisson"

            from tools.poisson import run_poisson
            self._launch(run_poisson,
                         points=pts, colors=clr,
                         depth=p["depth"], scale=p["scale"],
                         density_quantile=p["density_quantile"],
                         linear_fit=p["linear_fit"],
                         on_done=lambda r, _n=mesh_name: self._poisson_done(r, _n))

        except Exception as e:
            msg = f"Poisson setup error: {e}\n{traceback.format_exc()}"
            self.log.log(f"ERROR: {msg}")
            print(msg, file=sys.stderr)

    def _poisson_done(self, result, name):
        ml, mg, n_comp, big, small = result
        ml.name = name
        self.lm.add_mesh(ml)
        self.lm.add_mask_group(ml.id, mg)
        self.lm.set_selection(ml.id)
        self.log.log(
            f"Poisson complete: {ml.name} ({ml.face_count:,} faces, "
            f"{ml.vertex_count:,} verts). "
            f"Mesh Filter: {n_comp} components, largest {big:,}, small {small:,} faces")
        self.viewport.fit_all()

    # ── Mesh filter ──────────────────────────────────────────────

    def _run_mf(self):
        try:
            layer = self._get_or_pick_mesh()
            if layer is None:
                QMessageBox.information(
                    self, "Mesh Filter",
                    "Select a mesh first.\n\n"
                    "Click on a mesh in the Layers panel, "
                    "then run Mesh Filter.")
                return

            self.log.log(
                f"Mesh Filter: running on '{layer.name}'...")
            print(f"[MainWindow] MeshFilter: layer={layer.name}, "
                  f"faces={layer.face_count}", file=sys.stderr)

            sname = self.lm.selected_sublayer_name
            if sname:
                fmask = self.lm.get_sublayer_mask(layer, sname)
                fi = np.where(fmask)[0]
                self.log.log(
                    f"  Operating on sublayer '{sname}' ({len(fi):,} faces)")
            else:
                fi = None
                self.log.log(
                    f"  Operating on entire layer ({layer.face_count:,} faces)")

            from tools.mesh_filter import run_mesh_filter
            lid = layer.id
            self._launch(run_mesh_filter,
                         vertices=layer.vertices, faces=layer.faces,
                         face_indices=fi,
                         total_face_count=layer.face_count,
                         on_done=lambda r, _lid=lid: self._mf_done(_lid, r))

        except Exception as e:
            msg = f"Mesh filter setup error: {e}\n{traceback.format_exc()}"
            self.log.log(f"ERROR: {msg}")
            print(msg, file=sys.stderr)

    def _mf_done(self, lid, result):
        mg, n_comp, big, small = result
        self.lm.add_mask_group(lid, mg)
        self.log.log(
            f"Mesh Filter complete: {n_comp} components. "
            f"Largest {big:,}, small {small:,} faces")

    # ── Noise removal ────────────────────────────────────────────

    def _run_noise(self):
        try:
            layer = self._get_or_pick_pc()
            if layer is None:
                QMessageBox.information(
                    self, "Noise Removal",
                    "Select a point cloud first.")
                return

            meshes = list(self.lm.meshes.values())
            if not meshes:
                QMessageBox.information(
                    self, "Noise Removal",
                    "No mesh available as reference.\n"
                    "Run Poisson reconstruction first.")
                return

            preferred_mesh = self._find_preferred_noise_mesh(layer, meshes)

            self.log.log(
                f"Noise Removal: opening dialog for '{layer.name}'...")

            from ui.dialogs.noise_dialog import NoiseDialog
            dlg = NoiseDialog(meshes, preferred_mesh_id=(preferred_mesh.id if preferred_mesh else None), parent=self)
            if dlg.exec() != QDialog.Accepted:
                self.log.log("Noise Removal: cancelled.")
                return
            p = dlg.get_params()
            ref = self.lm.get_layer(p["mesh_id"])
            if ref is None:
                self.log.log("ERROR: reference mesh not found.")
                return

            self.log.log(
                f"Noise Removal: threshold={p['threshold']}, "
                f"ref mesh='{ref.name}'")

            sname = self.lm.selected_sublayer_name
            if sname:
                mask = self.lm.get_sublayer_mask(layer, sname)
                indices = np.where(mask)[0]
                pts = layer.points[indices]
            else:
                pts = layer.points
                indices = None

            mesh_verts = ref.vertices
            if ref.mask_groups and p.get("mesh_sublayer"):
                fm = self.lm.get_sublayer_mask(ref, p["mesh_sublayer"])
                used_v = np.unique(ref.faces[fm].ravel())
                mesh_verts = ref.vertices[used_v]

            from tools.noise_removal import run_noise_removal
            lid = layer.id
            self._launch(run_noise_removal,
                         points=pts, indices=indices,
                         total_count=layer.point_count,
                         mesh_vertices=mesh_verts,
                         threshold=p["threshold"],
                         on_done=lambda r, _lid=lid: self._noise_done(_lid, r))

        except Exception as e:
            msg = f"Noise removal setup error: {e}\n{traceback.format_exc()}"
            self.log.log(f"ERROR: {msg}")
            print(msg, file=sys.stderr)

    def _find_preferred_noise_mesh(self, layer, meshes):
        preferred_names = [
            f"{layer.name}_pca_kept_poisson",
            f"{layer.name}_poisson",
        ]

        for name in preferred_names:
            for mesh in meshes:
                if mesh.name == name:
                    return mesh

        layer_source = getattr(layer, "source_path", None)
        if layer_source:
            for mesh in meshes:
                if getattr(mesh, "source_path", None) == layer_source:
                    return mesh

        return meshes[0] if meshes else None

    def _noise_done(self, lid, mg):
        self.lm.add_mask_group(lid, mg)
        self.log.log(
            f"Noise Removal complete: {mg.positive_count:,} clean, "
            f"{mg.negative_count:,} noise")

    # ── Combine ──────────────────────────────────────────────────

    def _combine_dlg(self):
        from ui.dialogs.combine_dialog import CombineDialog
        dlg = CombineDialog(self.lm, self)
        if dlg.exec() != QDialog.Accepted:
            return
        p = dlg.get_params()
        self._do_combine([p["layer_a_id"], p["layer_b_id"]], p["name"])

    def _combine_two(self, lid_a, lid_b):
        a = self.lm.get_layer(lid_a)
        b = self.lm.get_layer(lid_b)
        if a and b:
            self._do_combine([lid_a, lid_b], f"{a.name}+{b.name}")

    def _combine_selected_layers(self, layer_ids):
        layers = [self.lm.get_layer(layer_id) for layer_id in layer_ids]
        layers = [layer for layer in layers if layer is not None]
        if len(layers) < 2:
            return
        name, ok = QInputDialog.getText(
            self,
            "Combine Layers",
            "Result name:",
            text="",
        )
        if not ok:
            return
        self._do_combine(layer_ids, name.strip())

    def _do_combine(self, layer_ids, name):
        source_layers = []
        for layer_id in layer_ids:
            layer = self.lm.get_layer(layer_id)
            if layer is not None:
                source_layers.append(self._snapshot_layer(layer))
        combined, msg = self.lm.combine_layers(layer_ids, name)
        if combined is None:
            QMessageBox.warning(self, "Combine", msg)
            return
        self._undo_stack.append({
            "type": "combine",
            "source_layers": source_layers,
            "combined_layer": self._snapshot_layer(combined),
        })
        self._redo_stack.clear()
        self._update_global_undo_redo_state()
        self.lm.set_selection(combined.id)
        if isinstance(combined, PointCloudLayer):
            self.log.log(f"Combined: {combined.name} ({combined.point_count:,} pts)")
        elif isinstance(combined, MeshLayer):
            self.log.log(f"Combined: {combined.name} ({combined.face_count:,} faces)")

    # ── task runner ──────────────────────────────────────────────

    def _launch(self, func, on_done, loading_title="Processing", 
                loading_message="Please wait...", **kw):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "A task is already running.")
            return
        
        self.pbar.setVisible(True)
        self.pbar.setValue(0)
        self.toolbar.setEnabled(False)
        self.log.log("Task started...")
        print(f"[MainWindow] Launching worker: {func.__name__}",
              file=sys.stderr)

        # Create and show loading dialog
        self._loading_dialog = LoadingDialog(loading_title, loading_message, self)
        self._loading_dialog.show()

        self._worker = TaskRunner(func, **kw)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished_result.connect(
            lambda r, _cb=on_done: self._task_ok(r, _cb))
        self._worker.cancelled.connect(self._task_cancelled)
        self._worker.error.connect(self._task_err)
        
        # Connect cancel button to worker cancellation
        self._loading_dialog.cancel_clicked.connect(self._worker.request_cancel)
        
        self._worker.start()

    def _on_worker_progress(self, value):
        """Update both progress bar and loading dialog."""
        self.pbar.setValue(value)
        if self._loading_dialog:
            self._loading_dialog.set_progress(value)

    def _task_ok(self, result, cb):
        self._close_loading_dialog()
        self.pbar.setVisible(False)
        self.toolbar.setEnabled(True)
        print(f"[MainWindow] Task finished OK", file=sys.stderr)
        gc.collect()  # Force garbage collection to release RAM
        try:
            cb(result)
        except Exception as e:
            msg = f"Post-task error: {e}\n{traceback.format_exc()}"
            self.log.log(f"ERROR: {msg}")
            print(msg, file=sys.stderr)

    def _task_cancelled(self):
        """Handle task cancellation."""
        self._close_loading_dialog()
        self.pbar.setVisible(False)
        self.toolbar.setEnabled(True)
        print(f"[MainWindow] Task cancelled by user", file=sys.stderr)
        gc.collect()  # Force garbage collection to release RAM
        self.log.log("Task cancelled by user.")

    def _task_err(self, msg):
        self._close_loading_dialog()
        self.pbar.setVisible(False)
        self.toolbar.setEnabled(True)
        self.log.log(f"ERROR: {msg}")
        print(f"[MainWindow] Task FAILED: {msg}", file=sys.stderr)
        gc.collect()  # Force garbage collection to release RAM
        QMessageBox.critical(self, "Task Error", f"Task failed:\n\n{msg}")

    def _show_cross_section_panel(self):
        layer = self._get_or_pick_pc()
        if layer is None:
            QMessageBox.information(
                self, "Cross Section",
                "Select a point cloud first.\n\n"
                "Click on a point cloud in the Layers panel, then open the Cross Section panel.")
            return

        self.cross_section_dock.setVisible(True)
        self.cross_section_dock.raise_()

    def _on_cross_section_top_level_changed(self, _floating):
        if hasattr(self, "cross_section_panel") and self.cross_section_panel is not None:
            if hasattr(self.cross_section_panel, "refresh_preview"):
                self.cross_section_panel.refresh_preview()
            preview_widget = getattr(self.cross_section_panel, "_preview_widget", None)
            if preview_widget is not None and hasattr(preview_widget, "reinitialize_vtk"):
                preview_widget.reinitialize_vtk()

    def _on_cross_section_created(self, new_layer):
        from core.layer import MaskGroup

        if isinstance(new_layer, MaskGroup):
            layer = self.lm.get_selected_layer() or self._get_or_pick_pc()
            if layer is None:
                QMessageBox.warning(self, "Cross Section", "No point cloud selected to apply mask to.")
                return
            self.lm.add_mask_group(layer.id, new_layer)
            self.log.log(
                f"Cross Section mask added to: {layer.name} "
                f"({new_layer.positive_count:,} kept, {new_layer.negative_count:,} rejected)")
            self.viewport.fit_all()
            return

        # fallback: a new layer was provided
        try:
            self.lm.add_point_cloud(new_layer)
            self.lm.set_selection(new_layer.id)
            self.log.log(
                f"Cross Section created: {new_layer.name} "
                f"({new_layer.point_count:,} pts)")
            self.viewport.fit_all()
        except Exception:
            # ignore malformed input
            return

    def _on_cross_section_points_transfer_requested(self, from_layer, to_layer, indices):
        layer = self.cross_section_panel._current_layer if hasattr(self, "cross_section_panel") else None
        if not isinstance(layer, PointCloudLayer):
            return

        ok, message = self.lm.transfer_points_between_sublayers(layer.id, from_layer, to_layer, indices)
        if not ok:
            if message:
                self.log.log(f"Cross Section transfer skipped: {message}")
            return

        moved_count = len(np.asarray(indices).ravel()) if indices is not None else 0
        if moved_count > 0:
            self._undo_stack.append({
                "type": "transfer",
                "layer_id": layer.id,
                "from_layer": from_layer,
                "to_layer": to_layer,
                "indices": np.asarray(indices, dtype=np.int32).copy(),
            })
            self._redo_stack.clear()
            self._update_cross_section_transfer_history_buttons()
            self.log.log(
                f"Cross Section transfer: moved {moved_count:,} pts from '{from_layer}' to '{to_layer}'"
            )
            self.viewport._rebuild(layer.id)
            self.viewport._render()
            if hasattr(self, "cross_section_panel") and self.cross_section_panel is not None:
                self.cross_section_panel.refresh_preview()

    def _on_cross_section_transfer_undo_requested(self):
        if not self._undo_stack or self._undo_stack[-1].get("type") != "transfer":
            return
        entry = self._undo_stack.pop()
        ok, message = self.lm.transfer_points_between_sublayers(
            entry["layer_id"],
            entry["to_layer"],
            entry["from_layer"],
            entry["indices"],
        )
        if not ok:
            if message:
                self.log.log(f"Cross Section undo skipped: {message}")
            self._update_cross_section_transfer_history_buttons()
            return
        self._redo_stack.append(entry)
        self._update_cross_section_transfer_history_buttons()
        self._refresh_cross_section_after_transfer(entry["layer_id"])
        self.log.log(
            f"Cross Section transfer undone: '{entry['to_layer']}' → '{entry['from_layer']}'"
        )

    def _on_cross_section_transfer_redo_requested(self):
        if not self._redo_stack or self._redo_stack[-1].get("type") != "transfer":
            return
        entry = self._redo_stack.pop()
        ok, message = self.lm.transfer_points_between_sublayers(
            entry["layer_id"],
            entry["from_layer"],
            entry["to_layer"],
            entry["indices"],
        )
        if not ok:
            if message:
                self.log.log(f"Cross Section redo skipped: {message}")
            self._update_cross_section_transfer_history_buttons()
            return
        self._undo_stack.append(entry)
        self._update_cross_section_transfer_history_buttons()
        self._refresh_cross_section_after_transfer(entry["layer_id"])
        self.log.log(
            f"Cross Section transfer redone: '{entry['from_layer']}' → '{entry['to_layer']}'"
        )

    def _refresh_cross_section_after_transfer(self, layer_id):
        layer = self.lm.get_layer(layer_id)
        if isinstance(layer, PointCloudLayer):
            self.viewport._rebuild(layer.id)
            self.viewport._render()
            if hasattr(self, "cross_section_panel") and self.cross_section_panel is not None:
                self.cross_section_panel.refresh_preview()

    def _update_cross_section_transfer_history_buttons(self):
        if hasattr(self, "cross_section_panel") and self.cross_section_panel is not None:
            self.cross_section_panel.set_transfer_history_state(
                bool(self._undo_stack and self._undo_stack[-1].get("type") == "transfer"),
                bool(self._redo_stack and self._redo_stack[-1].get("type") == "transfer"),
            )
        self._update_global_undo_redo_state()

    def _on_global_undo_requested(self):
        if not self._undo_stack:
            return
        entry = self._undo_stack[-1]
        if entry.get("type") == "combine":
            self._undo_layer_edit()
        elif entry.get("type") == "delete":
            self._undo_delete_layers()
        elif entry.get("type") == "transfer":
            self._on_cross_section_transfer_undo_requested()

    def _on_global_redo_requested(self):
        if not self._redo_stack:
            return
        entry = self._redo_stack[-1]
        if entry.get("type") == "combine":
            self._redo_layer_edit()
        elif entry.get("type") == "delete":
            self._redo_delete_layers()
        elif entry.get("type") == "transfer":
            self._on_cross_section_transfer_redo_requested()

    def _update_global_undo_redo_state(self):
        if not hasattr(self, "toolbar") or self.toolbar is None:
            return
        can_undo = bool(self._undo_stack)
        can_redo = bool(self._redo_stack)
        for action in self.toolbar.actions():
            text = action.text()
            if text == "↶ Undo":
                action.setEnabled(can_undo)
            elif text == "↷ Redo":
                action.setEnabled(can_redo)

    def _snapshot_layer(self, layer):
        data = {
            "id": layer.id,
            "name": layer.name,
            "visible": layer.visible,
            "modified": layer.modified,
            "display_color": None if layer.display_color is None else tuple(layer.display_color),
            "render_props": dict(layer.render_props),
        }
        if isinstance(layer, PointCloudLayer):
            data.update({
                "kind": "point_cloud",
                "points": np.array(layer.points, copy=True),
                "colors": None if layer.colors is None else np.array(layer.colors, copy=True),
                "normals": None if layer.normals is None else np.array(layer.normals, copy=True),
                "source_path": layer.source_path,
            })
        elif isinstance(layer, MeshLayer):
            data.update({
                "kind": "mesh",
                "vertices": np.array(layer.vertices, copy=True),
                "faces": np.array(layer.faces, copy=True),
                "vertex_colors": None if layer.vertex_colors is None else np.array(layer.vertex_colors, copy=True),
                "face_normals": None if layer.face_normals is None else np.array(layer.face_normals, copy=True),
                "vertex_normals": None if layer.vertex_normals is None else np.array(layer.vertex_normals, copy=True),
                "source_path": layer.source_path,
            })
        return data

    def _restore_layer_from_snapshot(self, snapshot):
        if snapshot["kind"] == "point_cloud":
            layer = PointCloudLayer(
                name=snapshot["name"],
                points=np.array(snapshot["points"], copy=True),
                colors=None if snapshot["colors"] is None else np.array(snapshot["colors"], copy=True),
                normals=None if snapshot["normals"] is None else np.array(snapshot["normals"], copy=True),
                source_path=snapshot.get("source_path"),
                modified=snapshot.get("modified", False),
            )
            self.lm.add_point_cloud(layer)
        else:
            layer = MeshLayer(
                name=snapshot["name"],
                vertices=np.array(snapshot["vertices"], copy=True),
                faces=np.array(snapshot["faces"], copy=True),
                vertex_colors=None if snapshot["vertex_colors"] is None else np.array(snapshot["vertex_colors"], copy=True),
                face_normals=None if snapshot["face_normals"] is None else np.array(snapshot["face_normals"], copy=True),
                vertex_normals=None if snapshot["vertex_normals"] is None else np.array(snapshot["vertex_normals"], copy=True),
                source_path=snapshot.get("source_path"),
                modified=snapshot.get("modified", False),
            )
            self.lm.add_mesh(layer)
        layer.id = snapshot["id"]
        layer.visible = snapshot.get("visible", True)
        layer.display_color = snapshot.get("display_color")
        layer.render_props = dict(snapshot.get("render_props", {}))
        if isinstance(layer, PointCloudLayer):
            self.lm._point_clouds.pop(next(reversed(self.lm._point_clouds)))
            self.lm._point_clouds[layer.id] = layer
        else:
            self.lm._meshes.pop(next(reversed(self.lm._meshes)))
            self.lm._meshes[layer.id] = layer
        self.lm.layer_modified.emit(layer.id)
        return layer

    def _undo_layer_edit(self):
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        if entry.get("type") != "combine":
            return
        combined_snapshot = entry["combined_layer"]
        self.lm.remove_layer(combined_snapshot["id"])
        restored_layers = [self._restore_layer_from_snapshot(snapshot) for snapshot in entry["source_layers"]]
        self._redo_stack.append(entry)
        self._update_global_undo_redo_state()
        if restored_layers:
            self.lm.set_selected_layers([layer.id for layer in restored_layers], restored_layers[0].id)
        self.log.log("Undo combine layer(s)")

    def _redo_layer_edit(self):
        if not self._redo_stack:
            return
        entry = self._redo_stack.pop()
        if entry.get("type") != "combine":
            return
        for snapshot in entry["source_layers"]:
            self.lm.remove_layer(snapshot["id"])
        combined_layer = self._restore_layer_from_snapshot(entry["combined_layer"])
        self._undo_stack.append(entry)
        self._update_global_undo_redo_state()
        self.lm.set_selection(combined_layer.id)
        self.log.log("Redo combine layer(s)")

    def _undo_delete_layers(self):
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        if entry.get("type") != "delete":
            return
        restored_layers = [self._restore_layer_from_snapshot(snapshot) for snapshot in entry["layers"]]
        self._redo_stack.append(entry)
        self._update_global_undo_redo_state()
        if restored_layers:
            self.lm.set_selected_layers([layer.id for layer in restored_layers], restored_layers[0].id)
        self.log.log("Undo delete layer(s)")

    def _redo_delete_layers(self):
        if not self._redo_stack:
            return
        entry = self._redo_stack.pop()
        if entry.get("type") != "delete":
            return
        for snapshot in entry["layers"]:
            self.lm.remove_layer(snapshot["id"])
            self.log.log(f"Deleted: {snapshot['name']}")
        self._undo_stack.append(entry)
        self._update_global_undo_redo_state()
        self.log.log("Redo delete layer(s)")

    def _on_cross_section_visibility_changed(self, visible):
        self.toolbar.cross_section_action.blockSignals(True)
        self.toolbar.cross_section_action.setChecked(visible)
        self.toolbar.cross_section_action.blockSignals(False)
        if visible:
            QTimer.singleShot(0, self._apply_initial_dock_layout)

    def _close_loading_dialog(self):
        """Close and cleanup the loading dialog."""
        if self._loading_dialog:
            self._loading_dialog.close()
            self._loading_dialog = None

    def closeEvent(self, event):
        try:
            if hasattr(self, "cross_section_panel") and self.cross_section_panel is not None:
                shutdown_preview = getattr(self.cross_section_panel, "shutdown_vtk", None)
                if callable(shutdown_preview):
                    shutdown_preview()
        except Exception:
            pass

        try:
            if hasattr(self, "viewport") and self.viewport is not None:
                shutdown_viewport = getattr(self.viewport, "shutdown_vtk", None)
                if callable(shutdown_viewport):
                    shutdown_viewport()
        except Exception:
            pass

        super().closeEvent(event)

    # ── Panel visibility handlers ────────────────────────────────

    def _on_layers_visibility_changed(self, visible):
        """Update Windows menu when Layers panel visibility changes."""
        self.toolbar.layers_action.blockSignals(True)
        self.toolbar.layers_action.setChecked(visible)
        self.toolbar.layers_action.blockSignals(False)

    def _on_properties_visibility_changed(self, visible):
        """Update Windows menu when Properties panel visibility changes."""
        self.toolbar.properties_action.blockSignals(True)
        self.toolbar.properties_action.setChecked(visible)
        self.toolbar.properties_action.blockSignals(False)

    def _on_log_visibility_changed(self, visible):
        """Update Windows menu when Log panel visibility changes."""
        self.toolbar.log_action.blockSignals(True)
        self.toolbar.log_action.setChecked(visible)
        self.toolbar.log_action.blockSignals(False)

    def set_app_font_size(self, size):
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)

    def on_layer_selected(self, layer):
        print("Layer selected:", layer)  # debug
        self.cross_section_panel.set_current_layer(layer)

    def _on_layer_visibility_changed(self, layer_id):
        self.cross_section_panel.refresh_preview()


class _CrossSectionDockWidget(QDockWidget):
    def closeEvent(self, event):
        main_window = self.parent()
        restore_handler = getattr(main_window, "_restore_cross_section_dock_position", None)
        if self.isFloating() and callable(restore_handler):
            try:
                event.ignore()
                restore_handler()
                self.hide()
                return
            except Exception:
                pass
        event.ignore()
        self.hide()
        try:
            widget = self.widget()
            refresh_preview = getattr(widget, "refresh_preview", None)
            if callable(refresh_preview):
                refresh_preview()
        except Exception:
            pass
