import sys
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtCore import Qt, QEvent, QPoint
from PySide6.QtWidgets import QWidget, QVBoxLayout, QColorDialog, QMenu
from PySide6.QtGui import QColor
from tools.gradient_colors import compute_gradient_colors
from core.layer_manager import LayerManager
from core.layer import PointCloudLayer, MeshLayer


class Viewport(QWidget):
    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(parent)
        self.layer_manager = layer_manager
        self._actors: dict[str, list[vtk.vtkActor]] = {}
        self._bg_color = (0.0, 0.0, 0.0)
        self._right_press_pos = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*self._bg_color)
        self.renderer.GradientBackgroundOff()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)

        # orientation axes
        axes = vtk.vtkAxesActor()
        self._orient = vtk.vtkOrientationMarkerWidget()
        self._orient.SetOrientationMarker(axes)
        self._orient.SetInteractor(interactor)
        self._orient.SetViewport(0.0, 0.0, 0.15, 0.15)
        self._orient.EnabledOn()
        self._orient.InteractiveOff()

        # right-click tracking for background color picker
        self.vtk_widget.installEventFilter(self)

        # signals
        lm = self.layer_manager
        lm.layer_added.connect(self._on_change)
        lm.layer_removed.connect(self._on_removed)
        lm.layer_modified.connect(self._on_change)
        lm.layer_renamed.connect(lambda _: None)
        lm.visibility_changed.connect(self._on_change)
        lm.mask_added.connect(lambda lid, mid: self._on_change(lid))
        lm.mask_removed.connect(lambda lid, mid: self._on_change(lid))

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    # ── event filter (right-click → background color picker) ─────

    def eventFilter(self, obj, event):
        if obj is self.vtk_widget:
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    self._right_press_pos = event.position().toPoint()
            elif etype == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.RightButton:
                    if self._right_press_pos is not None:
                        release = event.position().toPoint()
                        dx = abs(release.x() - self._right_press_pos.x())
                        dy = abs(release.y() - self._right_press_pos.y())
                        self._right_press_pos = None
                        if dx + dy < 5:
                            gp = event.globalPosition().toPoint()
                            self._show_bg_menu(gp)
                            return True
        return super().eventFilter(obj, event)

    def _show_bg_menu(self, global_pos):
        menu = QMenu(self)
        pick_action = menu.addAction("Change Background Color…")
        chosen = menu.exec(global_pos)
        if chosen == pick_action:
            self._pick_bg_color()

    def _pick_bg_color(self):
        cur = QColor.fromRgbF(*self._bg_color)
        color = QColorDialog.getColor(cur, self, "Background Color")
        if color.isValid():
            self._bg_color = (color.redF(), color.greenF(), color.blueF())
            self.renderer.SetBackground(*self._bg_color)
            self.renderer.GradientBackgroundOff()
            self._render()

    # ── public ───────────────────────────────────────────────────

    def fit_all(self):
        self.renderer.ResetCamera()
        self._render()

    def rebuild_all(self):
        """Rebuild every visible layer's actors (called when visual props change)."""
        # Clear all existing actors
        for layer_id in list(self._actors.keys()):
            self._clear_actors(layer_id)

        # Rebuild from all known layers
        for layer in self.layer_manager.get_all_layers():
            self._rebuild(layer.id)

        self._render()

    # ── private slots ────────────────────────────────────────────

    def _on_change(self, layer_id):
        self._rebuild(layer_id)
        self._render()

    def _on_removed(self, layer_id):
        self._clear_actors(layer_id)
        self._render()

    # ── actor management ─────────────────────────────────────────

    def _clear_actors(self, layer_id):
        for actor in self._actors.pop(layer_id, []):
            self.renderer.RemoveActor(actor)

    def _rebuild(self, layer_id):
        self._clear_actors(layer_id)
        layer = self.layer_manager.get_layer(layer_id)
        if layer is None or not layer.visible:
            return
        try:
            if isinstance(layer, PointCloudLayer):
                actors = self._build_pc(layer)
            elif isinstance(layer, MeshLayer):
                actors = self._build_mesh(layer)
            else:
                return
            self._actors[layer_id] = actors
            for a in actors:
                self.renderer.AddActor(a)
        except Exception as e:
            print(f"[Viewport] Error rebuilding {layer_id}: {e}",
                  file=sys.stderr)

    def _render(self):
        self.vtk_widget.GetRenderWindow().Render()

    # ── colour resolution (reads vis_* attributes from layer) ────

    def _resolve_pc_colors(self, layer):
        """Return (N, 3) float64 colours using vis_* attributes from properties panel."""
        n = layer.point_count
        scheme = getattr(layer, "vis_color_scheme", "Original")

        if scheme == "Solid":
            sc = getattr(layer, "vis_solid_color", (0.2, 0.6, 1.0))
            return np.tile(np.array(sc, dtype=np.float64), (n, 1))

        if scheme == "Gradient":
            axis = getattr(layer, "vis_gradient_dir", 2)
            flip = getattr(layer, "vis_gradient_flip", False)
            mode = getattr(layer, "vis_gradient_mode", "auto")
            values = layer.points[:, axis]
            
            if mode == "manual":
                mn = getattr(layer, "vis_gradient_min", None)
                mx = getattr(layer, "vis_gradient_max", None)
                if mn is None:
                    mn = float(values.min())
                if mx is None:
                    mx = float(values.max())
            else:
                mn = float(values.min())
                mx = float(values.max())
            
            # Handle flip: swap min/max for negative directions
            if flip:
                mn, mx = mx, mn
            
            return compute_gradient_colors(values, mn, mx).astype(np.float64)

        # "Original" (default)
        if layer.colors is not None and len(layer.colors) == n:
            return layer.colors.copy().astype(np.float64)
        return np.full((n, 3), 0.6, dtype=np.float64)

    def _resolve_mesh_colors(self, layer):
        """Return (V, 3) float64 colours using vis_* attributes from properties panel."""
        nv = layer.vertex_count
        scheme = getattr(layer, "vis_color_scheme", "Original")

        if scheme == "Solid":
            sc = getattr(layer, "vis_solid_color", (0.2, 0.6, 1.0))
            return np.tile(np.array(sc, dtype=np.float64), (nv, 1))

        if scheme == "Gradient":
            axis = getattr(layer, "vis_gradient_dir", 2)
            flip = getattr(layer, "vis_gradient_flip", False)
            mode = getattr(layer, "vis_gradient_mode", "auto")
            values = layer.vertices[:, axis]
            
            if mode == "manual":
                mn = getattr(layer, "vis_gradient_min", None)
                mx = getattr(layer, "vis_gradient_max", None)
                if mn is None:
                    mn = float(values.min())
                if mx is None:
                    mx = float(values.max())
            else:
                mn = float(values.min())
                mx = float(values.max())
            
            # Handle flip: swap min/max for negative directions
            if flip:
                mn, mx = mx, mn
            
            return compute_gradient_colors(values, mn, mx).astype(np.float64)

        # "Original" (default)
        if layer.vertex_colors is not None and len(layer.vertex_colors) == nv:
            return layer.vertex_colors.copy().astype(np.float64)
        return np.full((nv, 3), 0.8, dtype=np.float64)

    # ── point cloud ──────────────────────────────────────────────

    def _build_pc(self, layer: PointCloudLayer):
        if layer.points is None or len(layer.points) == 0:
            return []
        n = len(layer.points)
        colors = self._resolve_pc_colors(layer)
        ps = getattr(layer, "vis_point_size", 2)

        if not layer.mask_groups:
            return [self._make_pc_actor(layer.points, colors, ps)]

        visible = np.zeros(n, dtype=bool)
        any_mask = False
        for mg in layer.mask_groups:
            if mg.mask is None:
                continue
            any_mask = True
            if mg.positive_visible:
                pos_idx = mg.mask
                visible |= pos_idx
                if mg.positive_color is not None:
                    colors[pos_idx] = mg.positive_color
            if mg.negative_visible:
                neg_idx = ~mg.mask
                visible |= neg_idx
                if mg.negative_color is not None:
                    colors[neg_idx] = mg.negative_color
        if not any_mask:
            visible[:] = True
        if not np.any(visible):
            return []
        return [self._make_pc_actor(
            layer.points[visible], colors[visible], ps)]

    def _make_pc_actor(self, points, colors, point_size=2):
        vtk_pts = vtk.vtkPoints()
        arr = numpy_to_vtk(
            np.ascontiguousarray(points, dtype=np.float32),
            deep=True, array_type=vtk.VTK_FLOAT)
        vtk_pts.SetData(arr)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_pts)

        clr = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        vtk_clr = numpy_to_vtk(
            np.ascontiguousarray(clr), deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_clr.SetName("Colors")
        polydata.GetPointData().SetScalars(vtk_clr)

        vgf = vtk.vtkVertexGlyphFilter()
        vgf.SetInputData(polydata)
        vgf.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vgf.GetOutputPort())
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)
        return actor

    # ── mesh ─────────────────────────────────────────────────────

    def _build_mesh(self, layer: MeshLayer):
        if (layer.vertices is None or layer.faces is None
                or len(layer.vertices) == 0 or len(layer.faces) == 0):
            return []
        nv = len(layer.vertices)
        nf = len(layer.faces)
        colors = self._resolve_mesh_colors(layer)

        if not layer.mask_groups:
            return [self._make_mesh_actor(
                layer.vertices, layer.faces, colors)]

        face_vis = np.zeros(nf, dtype=bool)
        any_mask = False
        for mg in layer.mask_groups:
            if mg.mask is None:
                continue
            any_mask = True
            if mg.positive_visible:
                face_vis |= mg.mask
                if mg.positive_color is not None:
                    vi = np.unique(layer.faces[mg.mask].ravel())
                    colors[vi] = mg.positive_color
            if mg.negative_visible:
                face_vis |= ~mg.mask
                if mg.negative_color is not None:
                    vi = np.unique(layer.faces[~mg.mask].ravel())
                    colors[vi] = mg.negative_color
        if not any_mask:
            face_vis[:] = True
        if not np.any(face_vis):
            return []

        vis_faces = layer.faces[face_vis]
        used = np.unique(vis_faces.ravel())
        vmap = np.full(nv, -1, dtype=np.int64)
        vmap[used] = np.arange(len(used))

        return [self._make_mesh_actor(
            layer.vertices[used], vmap[vis_faces], colors[used])]

    def _make_mesh_actor(self, vertices, faces, colors):
        nv = len(vertices)
        nf = len(faces)

        vtk_pts = vtk.vtkPoints()
        arr = numpy_to_vtk(
            np.ascontiguousarray(vertices, dtype=np.float32),
            deep=True, array_type=vtk.VTK_FLOAT)
        vtk_pts.SetData(arr)

        cells = vtk.vtkCellArray()
        offsets = np.arange(0, nf * 3 + 1, 3, dtype=np.int64)
        conn = np.ascontiguousarray(faces.ravel(), dtype=np.int64)
        try:
            cells.SetData(
                numpy_to_vtkIdTypeArray(offsets, deep=True),
                numpy_to_vtkIdTypeArray(conn, deep=True))
        except (TypeError, AttributeError):
            legacy = np.column_stack([
                np.full(nf, 3, dtype=np.int64),
                faces.astype(np.int64)
            ]).ravel()
            cells.SetCells(nf, numpy_to_vtkIdTypeArray(
                np.ascontiguousarray(legacy), deep=True))

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_pts)
        polydata.SetPolys(cells)

        if colors is not None:
            clr = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            vtk_clr = numpy_to_vtk(
                np.ascontiguousarray(clr), deep=True,
                array_type=vtk.VTK_UNSIGNED_CHAR)
            vtk_clr.SetName("Colors")
            polydata.GetPointData().SetScalars(vtk_clr)

        norms = vtk.vtkPolyDataNormals()
        norms.SetInputData(polydata)
        norms.ComputePointNormalsOn()
        norms.ComputeCellNormalsOff()
        norms.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(norms.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor

    def focus_camera_on_layer(self, layer_id):
        """Reset camera to XY-plane view (+Z out of screen), centered on the given layer."""
        actors = self._actors.get(layer_id, [])
        if not actors:
            return

        xmin, xmax = float('inf'), float('-inf')
        ymin, ymax = float('inf'), float('-inf')
        zmin, zmax = float('inf'), float('-inf')
        for actor in actors:
            b = actor.GetBounds()
            xmin = min(xmin, b[0]); xmax = max(xmax, b[1])
            ymin = min(ymin, b[2]); ymax = max(ymax, b[3])
            zmin = min(zmin, b[4]); zmax = max(zmax, b[5])

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cz = (zmin + zmax) / 2.0

        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(cx, cy, cz)
        camera.SetPosition(cx, cy, cz + 1.0)
        camera.SetViewUp(0, 1, 0)

        self.renderer.ResetCamera(xmin, xmax, ymin, ymax, zmin, zmax)
        self._render()