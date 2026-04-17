import sys
import traceback
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QFileDialog,
    QMessageBox, QProgressBar,
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from core.layer_manager import LayerManager
from core.layer import PointCloudLayer, MeshLayer
from ui.viewport import Viewport
from ui.layer_panel import LayerPanel
from ui.toolbar import Toolbar
from ui.log_panel import LogPanel
from io_utils.ply_io import load_file
from io_utils.exporter import export_point_cloud, export_mesh
from workers.task_runner import TaskRunner


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Processor")
        self.resize(1400, 900)

        self.lm = LayerManager(self)
        self._worker: TaskRunner | None = None

        self._build_ui()
        self._build_menus()
        self._connect()

    # ── UI setup ─────────────────────────────────────────────────

    def _build_ui(self):
        self.viewport = Viewport(self.lm, self)
        self.setCentralWidget(self.viewport)

        self.layer_panel = LayerPanel(self.lm, self)
        ld = QDockWidget("Layers", self)
        ld.setWidget(self.layer_panel)
        ld.setMinimumWidth(290)
        self.addDockWidget(Qt.LeftDockWidgetArea, ld)

        self.toolbar = Toolbar(self.lm, self)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        self.log = LogPanel(self)
        logd = QDockWidget("Log", self)
        logd.setWidget(self.log)
        logd.setMaximumHeight(200)
        self.addDockWidget(Qt.BottomDockWidgetArea, logd)

        self.pbar = QProgressBar()
        self.pbar.setMaximumWidth(300)
        self.pbar.setVisible(False)
        self.statusBar().addPermanentWidget(self.pbar)

    def _build_menus(self):
        fm = self.menuBar().addMenu("&File")
        fm.addAction("Open…", self._open, "Ctrl+O")
        fm.addAction("Export Selected…", self._export_sel, "Ctrl+E")
        fm.addSeparator()
        fm.addAction("Quit", self.close, "Ctrl+Q")

    def _connect(self):
        tb = self.toolbar
        tb.open_requested.connect(self._open)
        tb.pca_requested.connect(self._run_pca)
        tb.poisson_requested.connect(self._run_poisson)
        tb.mesh_filter_requested.connect(self._run_mf)
        tb.noise_removal_requested.connect(self._run_noise)
        tb.export_requested.connect(self._export_sel)
        tb.combine_requested.connect(self._combine_dlg)

        lp = self.layer_panel
        lp.export_requested.connect(self._export_layer)
        lp.delete_requested.connect(self._delete_layer)
        lp.delete_mask_requested.connect(self._delete_mask)
        lp.combine_requested.connect(self._combine_two)
        lp.run_pca_requested.connect(self._run_pca)
        lp.run_poisson_requested.connect(self._run_poisson)
        lp.run_mesh_filter_requested.connect(self._run_mf)
        lp.run_noise_removal_requested.connect(self._run_noise)

    # ── helpers ──────────────────────────────────────────────────

    def _get_or_pick_pc(self):
        """Get currently selected point cloud, or auto-pick the only one."""
        layer = self.lm.get_selected_layer()
        if isinstance(layer, PointCloudLayer):
            return layer
        # Auto-fallback: if there's exactly one PC, use it
        pcs = list(self.lm.point_clouds.values())
        if len(pcs) == 1:
            self.lm.set_selection(pcs[0].id)
            self.log.log(f"Auto-selected: {pcs[0].name}")
            return pcs[0]
        if len(pcs) > 1:
            self.log.log("Multiple point clouds — please select one in the Layers panel.")
        return None

    def _get_or_pick_mesh(self):
        """Get currently selected mesh, or auto-pick the only one."""
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
            "Supported (*.ply *.pcd *.obj *.stl *.xyz);;All (*)")
        if not path:
            return
        try:
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
            # Auto-select the newly loaded layer
            self.lm.set_selection(layer.id)
            self.viewport.fit_all()
        except Exception as e:
            self.log.log(f"ERROR loading: {e}")
            QMessageBox.critical(self, "Error", str(e))

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
                "PLY binary (*.ply);;PLY ASCII (*.ply)")
            if not path:
                return
            binary = "ASCII" not in filt
            try:
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
            self.lm.remove_layer(lid)
            self.log.log(f"Deleted: {layer.name}")

    def _delete_mask(self, lid, mgid):
        self.lm.remove_mask_group(lid, mgid)
        self.log.log("Mask group deleted")

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
                         on_done=lambda r, _lid=lid: self._pca_done(_lid, r))

        except Exception as e:
            msg = f"PCA setup error: {e}\n{traceback.format_exc()}"
            self.log.log(f"ERROR: {msg}")
            print(msg, file=sys.stderr)

    def _pca_done(self, lid, mg):
        self.lm.add_mask_group(lid, mg)
        self.log.log(
            f"PCA Filter complete: {mg.positive_count:,} kept, "
            f"{mg.negative_count:,} rejected")

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

    def _poisson_done(self, ml, name):
        ml.name = name
        self.lm.add_mesh(ml)
        self.lm.set_selection(ml.id)
        self.log.log(
            f"Poisson complete: {ml.name} ({ml.face_count:,} faces, "
            f"{ml.vertex_count:,} verts)")
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

            self.log.log(
                f"Noise Removal: opening dialog for '{layer.name}'...")

            from ui.dialogs.noise_dialog import NoiseDialog
            dlg = NoiseDialog(meshes, self)
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
        self._do_combine(p["layer_a_id"], p["layer_b_id"], p["name"])

    def _combine_two(self, lid_a, lid_b):
        a = self.lm.get_layer(lid_a)
        b = self.lm.get_layer(lid_b)
        if a and b:
            self._do_combine(lid_a, lid_b, f"{a.name}+{b.name}")

    def _do_combine(self, id_a, id_b, name):
        a = self.lm.get_layer(id_a)
        b = self.lm.get_layer(id_b)
        if a is None or b is None:
            return

        if isinstance(a, PointCloudLayer) and isinstance(b, PointCloudLayer):
            c = PointCloudLayer(
                name=name,
                points=np.vstack([a.points, b.points]),
                modified=True)
            if a.colors is not None and b.colors is not None:
                c.colors = np.vstack([a.colors, b.colors])
            elif a.colors is not None:
                c.colors = np.vstack([
                    a.colors, np.full((len(b.points), 3), 0.5)])
            elif b.colors is not None:
                c.colors = np.vstack([
                    np.full((len(a.points), 3), 0.5), b.colors])
            self.lm.add_point_cloud(c)
            self.lm.set_selection(c.id)
            self.log.log(f"Combined: {c.name} ({c.point_count:,} pts)")

        elif isinstance(a, MeshLayer) and isinstance(b, MeshLayer):
            off = len(a.vertices)
            c = MeshLayer(
                name=name,
                vertices=np.vstack([a.vertices, b.vertices]),
                faces=np.vstack([a.faces, b.faces + off]),
                modified=True)
            if a.vertex_colors is not None and b.vertex_colors is not None:
                c.vertex_colors = np.vstack([
                    a.vertex_colors, b.vertex_colors])
            self.lm.add_mesh(c)
            self.lm.set_selection(c.id)
            self.log.log(f"Combined: {c.name} ({c.face_count:,} faces)")
        else:
            QMessageBox.warning(
                self, "Combine", "Can only combine same-type layers.")

    # ── task runner ──────────────────────────────────────────────

    def _launch(self, func, on_done, **kw):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "A task is already running.")
            return
        self.pbar.setVisible(True)
        self.pbar.setValue(0)
        self.toolbar.setEnabled(False)
        self.log.log("Task started...")
        print(f"[MainWindow] Launching worker: {func.__name__}",
              file=sys.stderr)

        self._worker = TaskRunner(func, **kw)
        self._worker.progress.connect(self.pbar.setValue)
        self._worker.finished_result.connect(
            lambda r, _cb=on_done: self._task_ok(r, _cb))
        self._worker.error.connect(self._task_err)
        self._worker.start()

    def _task_ok(self, result, cb):
        self.pbar.setVisible(False)
        self.toolbar.setEnabled(True)
        print(f"[MainWindow] Task finished OK", file=sys.stderr)
        try:
            cb(result)
        except Exception as e:
            msg = f"Post-task error: {e}\n{traceback.format_exc()}"
            self.log.log(f"ERROR: {msg}")
            print(msg, file=sys.stderr)

    def _task_err(self, msg):
        self.pbar.setVisible(False)
        self.toolbar.setEnabled(True)
        self.log.log(f"ERROR: {msg}")
        print(f"[MainWindow] Task FAILED: {msg}", file=sys.stderr)
        QMessageBox.critical(self, "Task Error", f"Task failed:\n\n{msg}")