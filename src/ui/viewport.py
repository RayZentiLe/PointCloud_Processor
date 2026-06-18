import sys
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtCore import Qt, QEvent, QPoint, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QColorDialog, QMenu
from PySide6.QtGui import QColor
from tools.gradient_colors import compute_gradient_colors
from core.layer_manager import LayerManager
from core.layer import PointCloudLayer, MeshLayer


class Viewport(QWidget):
    cross_section_point_selected = Signal(object)

    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(parent)
        self.layer_manager = layer_manager
        self._actors: dict[str, list[vtk.vtkActor]] = {}
        self._bg_color = (255.0, 255.0, 255.0)  # Default to white background
        self._right_press_pos = None
        self._picker = vtk.vtkPointPicker()
        self._cross_section_active = False
        self._cross_section_layer_id: str | None = None
        self._cross_section_points: list[np.ndarray] = []
        self._cross_section_reference: np.ndarray | None = None
        self._cross_section_direction_xy: np.ndarray | None = None
        self._cross_section_pick_mode = "direction"
        self._cross_section_line_actor = None
        self._cross_section_confirmed_line_actor = None  # Green line when confirmed
        self._cross_section_polyline_actor = None  # Polyline connecting all confirmed points
        self._cross_section_point_actors: list[vtk.vtkActor] = []
        self._cross_section_ref_actor = None
        self._cross_section_reference_projection_actor = None
        self._cross_section_reference_line_actor = None
        self._cross_section_thickness = 1.0
        self._cross_section_direction_confirmed = False  # Track confirmation state

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

        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press)


    def _on_left_button_press(self, obj, event):
        x, y = obj.GetEventPosition()

        print("Mouse click at:", x, y)  # debug

        self._handle_cross_section_pick(x, y)

        obj.OnLeftButtonDown()

    # ── event filter (right-click → background color picker) ─────

    def eventFilter(self, obj, event):
        if obj is self.vtk_widget:
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton and self._cross_section_active:
                    pos = event.position().toPoint()
                    self._handle_cross_section_pick(pos.x(), pos.y())
                    return True
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

    def enable_cross_section_mode(self, layer):
        if self._cross_section_active and self._cross_section_layer_id == layer.id:
            self._cross_section_pick_mode = "direction"
            self._render()
            return

        self._cross_section_active = True
        self._cross_section_layer_id = layer.id
        self._cross_section_points.clear()
        self._cross_section_reference = None
        self._cross_section_direction_xy = None
        self._cross_section_pick_mode = "direction"
        self._cross_section_direction_confirmed = False
        self._remove_cross_section_actors()
        self._render()

    def disable_cross_section_mode(self):
        self._cross_section_active = False
        self._cross_section_layer_id = None
        self._cross_section_points.clear()
        self._cross_section_reference = None
        self._cross_section_direction_xy = None
        self._cross_section_pick_mode = "direction"
        self._cross_section_direction_confirmed = False
        self._remove_cross_section_actors()
        self._render()

    def set_cross_section_pick_mode(self, mode):
        if mode in ("direction", "reference", "none"):
            self._cross_section_pick_mode = mode
            self._render()

    def set_cross_section_thickness(self, value):
        self._cross_section_thickness = float(value)

    def set_cross_section_direction_confirmed(self, confirmed):
        """Set whether the direction points are confirmed (locked)."""
        self._cross_section_direction_confirmed = confirmed
        self._update_cross_section_preview()
        self._render()

    def clear_cross_section_direction_points(self):
        self._cross_section_points.clear()
        self._cross_section_direction_xy = None
        self._remove_cross_section_actors(point_actors=True, line_actor=True)
        self._render()

    def clear_cross_section_reference_point(self):
        self._cross_section_reference = None
        if self._cross_section_ref_actor is not None:
            self.renderer.RemoveActor(self._cross_section_ref_actor)
            self._cross_section_ref_actor = None
        if self._cross_section_reference_projection_actor is not None:
            self.renderer.RemoveActor(self._cross_section_reference_projection_actor)
            self._cross_section_reference_projection_actor = None
        if self._cross_section_reference_line_actor is not None:
            self.renderer.RemoveActor(self._cross_section_reference_line_actor)
            self._cross_section_reference_line_actor = None
        self._render()

    def get_cross_section_state(self):
        return {
            "active": self._cross_section_active,
            "layer_id": self._cross_section_layer_id,
            "direction_points": list(self._cross_section_points),
            "reference_point": self._cross_section_reference,
            "direction_xy": self._cross_section_direction_xy,
            "thickness": self._cross_section_thickness,
            "mode": self._cross_section_pick_mode,
        }

    def _handle_cross_section_pick(self, x, y):
        if not self._cross_section_active:
            return
        if self._picker.Pick(x, self.vtk_widget.height() - y, 0, self.renderer) == 0:
            return
        point = np.asarray(self._picker.GetPickPosition(), dtype=np.float64)
        point_id = self._picker.GetPointId()
        if point_id < 0:
            return

        if self._cross_section_pick_mode == "direction":
            # Don't allow picking more direction points if already confirmed
            if self._cross_section_direction_confirmed:
                return
            self._cross_section_points.append(point)
            # Bigger cyan spheres for selected points
            actor = self._make_sphere_actor(point, radius=0.03, color=(0.0, 1.0, 1.0))
            self._cross_section_point_actors.append(actor)
            self.renderer.AddActor(actor)
            self._update_cross_section_preview()
            self.cross_section_point_selected.emit({"mode": "direction", "point": point})
        elif self._cross_section_pick_mode == "reference":
            self._cross_section_reference = point
            if self._cross_section_ref_actor is not None:
                self.renderer.RemoveActor(self._cross_section_ref_actor)
            self._cross_section_ref_actor = self._make_sphere_actor(point, radius=0.015, color=(1.0, 0.2, 0.2))
            self.renderer.AddActor(self._cross_section_ref_actor)
            self._update_cross_section_reference_markers()
            self._render()
            self.cross_section_point_selected.emit({"mode": "reference", "point": point})

    def _update_cross_section_preview(self):
        # Remove preview line
        if self._cross_section_line_actor is not None:
            self.renderer.RemoveActor(self._cross_section_line_actor)
            self._cross_section_line_actor = None

        # Remove confirmed line
        if self._cross_section_confirmed_line_actor is not None:
            self.renderer.RemoveActor(self._cross_section_confirmed_line_actor)
            self._cross_section_confirmed_line_actor = None

        # Remove polyline if any
        if self._cross_section_polyline_actor is not None:
            self.renderer.RemoveActor(self._cross_section_polyline_actor)
            self._cross_section_polyline_actor = None

        if len(self._cross_section_points) < 2:
            self._cross_section_direction_xy = None
            self._render()
            return

        points = np.asarray(self._cross_section_points, dtype=np.float64)
        xy = points[:, :2]
        mean_xy = xy.mean(axis=0)
        u, s, vh = np.linalg.svd(xy - mean_xy)
        direction_xy = vh[0]
        if np.linalg.norm(direction_xy) == 0:
            return
        direction_xy = direction_xy / np.linalg.norm(direction_xy)
        if direction_xy[0] < 0:
            direction_xy = -direction_xy
        self._cross_section_direction_xy = direction_xy

        layer = self.layer_manager.get_layer(self._cross_section_layer_id)
        if layer is None or len(layer.points) == 0:
            zmin = 0.0
            zmax = 1.0
            span = 1.0
        else:
            zmin = float(np.min(layer.points[:, 2]))
            zmax = float(np.max(layer.points[:, 2]))
            span = float(np.linalg.norm(layer.points[:, :2].ptp(axis=0)))
        span = max(span, 1.0)
        line_length = span * 1.5

        z0 = float(np.mean(points[:, 2])) if self._cross_section_reference is None else float(self._cross_section_reference[2])
        start = np.array([mean_xy[0] - direction_xy[0] * line_length,
                          mean_xy[1] - direction_xy[1] * line_length,
                          zmin])
        end = np.array([mean_xy[0] + direction_xy[0] * line_length,
                        mean_xy[1] + direction_xy[1] * line_length,
                        zmax])

        # Show the normal direction line from the fit
        if self._cross_section_direction_confirmed:
            self._cross_section_line_actor = self._make_line_actor(
                start, end, color=(0.0, 1.0, 0.0), width=6)
            self.renderer.AddActor(self._cross_section_line_actor)
            # Also optionally draw the selected points polyline for context
            self._cross_section_polyline_actor = self._make_polyline_actor(
                points, color=(0.0, 0.7, 0.7), width=2)
            if self._cross_section_polyline_actor is not None:
                self.renderer.AddActor(self._cross_section_polyline_actor)
        else:
            # Show yellow preview line through all points
            self._cross_section_line_actor = self._make_line_actor(start, end, color=(1.0, 1.0, 0.0), width=4)
            self.renderer.AddActor(self._cross_section_line_actor)

        self._update_cross_section_reference_markers()
        self._render()

    def _update_cross_section_reference_markers(self):
        if self._cross_section_reference is None:
            return
        if self._cross_section_direction_xy is None:
            return
        if len(self._cross_section_points) < 2:
            return

        # Compute closest point on the direction line (in XY) for the selected reference
        points = np.asarray(self._cross_section_points, dtype=np.float64)
        origin_xy = points[:, :2].mean(axis=0)
        direction = self._cross_section_direction_xy
        ref_xy = self._cross_section_reference[:2]
        projected_xy = origin_xy + direction * np.dot(ref_xy - origin_xy, direction)
        projected_point = np.array([
            projected_xy[0], projected_xy[1], float(self._cross_section_reference[2])
        ], dtype=np.float64)

        if self._cross_section_reference_projection_actor is not None:
            self.renderer.RemoveActor(self._cross_section_reference_projection_actor)
            self._cross_section_reference_projection_actor = None
        if self._cross_section_reference_line_actor is not None:
            self.renderer.RemoveActor(self._cross_section_reference_line_actor)
            self._cross_section_reference_line_actor = None

        self._cross_section_reference_projection_actor = self._make_sphere_actor(
            projected_point, radius=0.02, color=(1.0, 0.0, 1.0)
        )
        self.renderer.AddActor(self._cross_section_reference_projection_actor)
        self._cross_section_reference_line_actor = self._make_line_actor(
            self._cross_section_reference, projected_point,
            color=(1.0, 0.2, 1.0), width=2
        )
        self.renderer.AddActor(self._cross_section_reference_line_actor)

    def _remove_cross_section_actors(self, point_actors=False, line_actor=False, ref_actor=False):
        if not any((point_actors, line_actor, ref_actor)):
            point_actors = line_actor = ref_actor = True

        if point_actors:
            for actor in self._cross_section_point_actors:
                self.renderer.RemoveActor(actor)
            self._cross_section_point_actors.clear()

        if line_actor:
            if self._cross_section_line_actor is not None:
                self.renderer.RemoveActor(self._cross_section_line_actor)
                self._cross_section_line_actor = None
            if self._cross_section_confirmed_line_actor is not None:
                self.renderer.RemoveActor(self._cross_section_confirmed_line_actor)
                self._cross_section_confirmed_line_actor = None
            if self._cross_section_polyline_actor is not None:
                self.renderer.RemoveActor(self._cross_section_polyline_actor)
                self._cross_section_polyline_actor = None

        if ref_actor and self._cross_section_ref_actor is not None:
            self.renderer.RemoveActor(self._cross_section_ref_actor)
            self._cross_section_ref_actor = None
        if self._cross_section_reference_projection_actor is not None:
            self.renderer.RemoveActor(self._cross_section_reference_projection_actor)
            self._cross_section_reference_projection_actor = None
        if self._cross_section_reference_line_actor is not None:
            self.renderer.RemoveActor(self._cross_section_reference_line_actor)
            self._cross_section_reference_line_actor = None

    def _make_line_actor(self, start, end, color=(1.0, 1.0, 0.0), width=2):
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(*start.tolist())
        pts.InsertNextPoint(*end.tolist())

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)

        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().SetOpacity(0.9)
        return actor

    def _make_sphere_actor(self, center, radius=0.01, color=(1.0, 1.0, 0.0)):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*center.tolist())
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        sphere.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(0.8)
        return actor

    def _make_polyline_actor(self, points_list, color=(0.0, 1.0, 0.0), width=4):
        """Create a polyline connecting all points in order."""
        if len(points_list) < 2:
            return None
        
        pts = vtk.vtkPoints()
        for pt in points_list:
            pts.InsertNextPoint(*pt.tolist())
        
        # Create polyline (connected line segments)
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(points_list))
        for i in range(len(points_list)):
            polyline.GetPointIds().SetId(i, i)
        
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)
        
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(cells)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().SetOpacity(0.9)
        return actor

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
                # Apply mask color based on color_mode
                colors = self._apply_mask_color_pc(colors, pos_idx, mg, True)
            if mg.negative_visible:
                neg_idx = ~mg.mask
                visible |= neg_idx
                # Apply mask color based on color_mode
                colors = self._apply_mask_color_pc(colors, neg_idx, mg, False)
        if not any_mask:
            visible[:] = True
        if not np.any(visible):
            return []
        return [self._make_pc_actor(
            layer.points[visible], colors[visible], ps)]
    
    def _apply_mask_color_pc(self, colors, mask_idx, mask_group, is_positive):
        """Apply mask colors to a point cloud color array based on color_mode."""
        color_mode = (mask_group.positive_color_mode if is_positive 
                      else mask_group.negative_color_mode)
        solid_color = (mask_group.positive_solid_color if is_positive 
                       else mask_group.negative_solid_color)
        
        if color_mode == "original":
            # Keep parent color - no change needed
            pass
        elif color_mode == "solid":
            # Apply solid color to masked indices
            colors[mask_idx] = np.array(solid_color, dtype=np.float64)
        
        return colors

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
                colors = self._apply_mask_color_mesh(
                    colors, layer.faces, mg.mask, mg, True)
            if mg.negative_visible:
                face_vis |= ~mg.mask
                colors = self._apply_mask_color_mesh(
                    colors, layer.faces, ~mg.mask, mg, False)
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
    
    def _apply_mask_color_mesh(self, colors, faces, face_mask, mask_group, is_positive):
        """Apply mask colors to a mesh color array based on color_mode."""
        color_mode = (mask_group.positive_color_mode if is_positive 
                      else mask_group.negative_color_mode)
        solid_color = (mask_group.positive_solid_color if is_positive 
                       else mask_group.negative_solid_color)
        
        if color_mode == "original":
            # Keep parent color - no change needed
            pass
        elif color_mode == "solid":
            # Apply solid color to vertices of masked faces
            vi = np.unique(faces[face_mask].ravel())
            colors[vi] = np.array(solid_color, dtype=np.float64)
        
        return colors

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