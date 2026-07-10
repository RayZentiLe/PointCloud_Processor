import numpy as np
import importlib
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox,
    QHBoxLayout, QDoubleSpinBox
)
from PySide6.QtCore import Signal, Qt, QEvent
from PySide6.QtGui import QShortcut, QKeySequence

try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_AVAILABLE = True
except Exception as _e:
    VTK_AVAILABLE = False
    _vtk_import_error = _e

from core.layer import PointCloudLayer, MaskGroup


if VTK_AVAILABLE:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    class CrossSectionPreviewStyle(vtk.vtkInteractorStyleUser):
        def __init__(self, preview_widget=None):
            super().__init__()
            self._preview_widget = preview_widget
            self._active_button = None

        def _get_renderer(self):
            renderer = self.GetCurrentRenderer()
            if renderer is None:
                renderer = self.GetDefaultRenderer()
            if renderer is None and self._preview_widget is not None:
                renderer = getattr(self._preview_widget, "renderer", None)
            return renderer

        def _lock_camera_orientation(self):
            renderer = self._get_renderer()
            if renderer is None:
                return
            camera = renderer.GetActiveCamera()
            if camera is None:
                return
            focal_point = camera.GetFocalPoint()
            distance = camera.GetDistance()
            if distance <= 0:
                distance = 1.0
            parallel_scale = camera.GetParallelScale()
            camera.SetPosition(focal_point[0], focal_point[1] - distance, focal_point[2])
            camera.SetFocalPoint(*focal_point)
            camera.SetViewUp(0.0, 0.0, 1.0)
            camera.OrthogonalizeViewUp()
            if parallel_scale > 0:
                camera.SetParallelScale(parallel_scale)
            renderer.ResetCameraClippingRange()

        def OnLeftButtonDown(self):
            self._active_button = "pan"

        def OnLeftButtonUp(self):
            self._active_button = None

        def OnMiddleButtonDown(self):
            self._active_button = "pan"

        def OnMiddleButtonUp(self):
            self._active_button = None

        def OnRightButtonDown(self):
            self._active_button = "zoom"

        def OnRightButtonUp(self):
            self._active_button = None

        def _pan_camera(self):
            interactor = self.GetInteractor()
            renderer = self._get_renderer()
            if interactor is None or renderer is None:
                return
            camera = renderer.GetActiveCamera()
            if camera is None:
                return

            last_x, last_y = interactor.GetLastEventPosition()
            x, y = interactor.GetEventPosition()
            focal_depth = renderer.GetDisplayPoint()[2]

            renderer.SetWorldPoint(*camera.GetFocalPoint(), 1.0)
            renderer.WorldToDisplay()
            focal_depth = renderer.GetDisplayPoint()[2]

            renderer.SetDisplayPoint(last_x, last_y, focal_depth)
            renderer.DisplayToWorld()
            old_pick = renderer.GetWorldPoint()

            renderer.SetDisplayPoint(x, y, focal_depth)
            renderer.DisplayToWorld()
            new_pick = renderer.GetWorldPoint()

            if old_pick[3] == 0.0 or new_pick[3] == 0.0:
                return

            old_pick = np.array(old_pick[:3]) / old_pick[3]
            new_pick = np.array(new_pick[:3]) / new_pick[3]
            motion = old_pick - new_pick

            position = np.array(camera.GetPosition()) + motion
            focal_point = np.array(camera.GetFocalPoint()) + motion
            camera.SetPosition(*position)
            camera.SetFocalPoint(*focal_point)

        def _zoom_camera(self):
            interactor = self.GetInteractor()
            renderer = self._get_renderer()
            if interactor is None or renderer is None:
                return
            camera = renderer.GetActiveCamera()
            if camera is None:
                return

            _, last_y = interactor.GetLastEventPosition()
            _, y = interactor.GetEventPosition()
            dy = y - last_y
            if dy == 0:
                return

            zoom_factor = 1.02 ** abs(dy)
            if dy > 0:
                zoom_factor = 1.0 / zoom_factor

            if camera.GetParallelProjection():
                camera.SetParallelScale(max(camera.GetParallelScale() / zoom_factor, 1e-6))
            else:
                camera.Dolly(zoom_factor)

        def OnMouseWheelForward(self):
            renderer = self._get_renderer()
            if renderer is None:
                return
            camera = renderer.GetActiveCamera()
            if camera is None:
                return
            if camera.GetParallelProjection():
                camera.SetParallelScale(max(camera.GetParallelScale() / 1.1, 1e-6))
            else:
                camera.Dolly(1.1)
            self._lock_camera_orientation()
            self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)

        def OnMouseWheelBackward(self):
            renderer = self._get_renderer()
            if renderer is None:
                return
            camera = renderer.GetActiveCamera()
            if camera is None:
                return
            if camera.GetParallelProjection():
                camera.SetParallelScale(camera.GetParallelScale() * 1.1)
            else:
                camera.Dolly(1.0 / 1.1)
            self._lock_camera_orientation()
            self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)

        def OnChar(self):
            return

        def OnMouseMove(self):
            if self._active_button == "pan":
                self._pan_camera()
                self._lock_camera_orientation()
                self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)
                return
            if self._active_button == "zoom":
                self._zoom_camera()
                self._lock_camera_orientation()
                self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)
                return

    class CrossSectionPreviewWidget(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)

            self._layer_points = np.empty((0, 3), dtype=np.float32)
            self._layer_colors = None
            self._plane_origin = np.zeros(3, dtype=np.float32)
            self._plane_normal_xy = np.array([1.0, 0.0], dtype=np.float32)
            self._thickness = 1.0

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            self.vtk_widget = QVTKRenderWindowInteractor(self)
            layout.addWidget(self.vtk_widget)

            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.1, 0.12)
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
            self._vtk_closed = False

            interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            style = CrossSectionPreviewStyle(self)
            interactor.SetInteractorStyle(style)
            style.SetDefaultRenderer(self.renderer)

            self._build_scene()
            self.vtk_widget.Initialize()
            self.vtk_widget.Start()

        def shutdown_vtk(self):
            if self._vtk_closed:
                return
            self._vtk_closed = True
            try:
                self.hide()
            except Exception:
                pass
            try:
                self.setUpdatesEnabled(False)
            except Exception:
                pass
            try:
                self.renderer.RemoveAllViewProps()
            except Exception:
                pass
            try:
                render_window = self.vtk_widget.GetRenderWindow()
                if render_window is not None:
                    interactor = render_window.GetInteractor()
                    if interactor is not None:
                        try:
                            interactor.Disable()
                        except Exception:
                            pass
                        interactor.SetInteractorStyle(None)
                        interactor.RemoveAllObservers()
                    try:
                        render_window.SetOffScreenRendering(1)
                    except Exception:
                        pass
                    try:
                        render_window.ReleaseGraphicsResources(None)
                    except Exception:
                        pass
                    render_window.Finalize()
            except Exception:
                pass
            try:
                self.vtk_widget.close()
            except Exception:
                pass

        def closeEvent(self, event):
            self.shutdown_vtk()
            super().closeEvent(event)

        def hideEvent(self, event):
            if not self._vtk_closed:
                try:
                    render_window = self.vtk_widget.GetRenderWindow()
                    if render_window is not None:
                        render_window.SetAbortRender(1)
                except Exception:
                    pass
            super().hideEvent(event)

        def set_preview_data(self, layer_points, layer_colors, plane_origin, plane_normal_xy, thickness):
            self._layer_points = np.asarray(layer_points, dtype=np.float32)
            self._layer_colors = None if layer_colors is None else np.asarray(layer_colors, dtype=np.float32)
            self._plane_origin = np.asarray(plane_origin, dtype=np.float32)
            self._plane_normal_xy = np.asarray(plane_normal_xy, dtype=np.float32)
            self._thickness = float(thickness)
            self._rebuild_scene()

        def _build_scene(self):
            preview_points, preview_colors = self._compute_preview_points()
            if len(preview_points) > 0:
                actor = self._make_points_actor(preview_points, colors=preview_colors, point_size=4)
                self.renderer.AddActor(actor)

            origin_actor = self._make_sphere_actor(np.zeros(3, dtype=np.float32), radius=0.03, color=(0.2, 0.8, 1.0))
            self.renderer.AddActor(origin_actor)

            camera = vtk.vtkCamera()
            camera.SetPosition(0.0, -1.0, 0.0)
            camera.SetFocalPoint(0.0, 0.0, 0.0)
            camera.SetViewUp(0.0, 0.0, 1.0)
            camera.SetParallelProjection(True)
            self.renderer.SetActiveCamera(camera)
            self.renderer.ResetCamera()

        def _compute_preview_points(self):
            if len(self._layer_points) == 0:
                return np.empty((0, 3), dtype=np.float32), None

            normal_xy = np.asarray(self._plane_normal_xy, dtype=np.float64)
            norm = np.linalg.norm(normal_xy)
            if norm == 0:
                return np.empty((0, 3), dtype=np.float32), None
            normal_xy = normal_xy / norm

            deltas_xy = self._layer_points[:, :2].astype(np.float64) - self._plane_origin[:2].astype(np.float64)
            signed_distances = deltas_xy @ normal_xy
            mask = np.abs(signed_distances) <= self._thickness
            if not np.any(mask):
                return np.empty((0, 3), dtype=np.float32), None

            selected = self._layer_points[mask].astype(np.float64)
            selected_colors = None
            if self._layer_colors is not None and len(self._layer_colors) == len(self._layer_points):
                selected_colors = self._layer_colors[mask].astype(np.float32)
            selected_deltas_xy = selected[:, :2] - self._plane_origin[:2].astype(np.float64)
            along_plane = -(selected_deltas_xy @ np.array([-normal_xy[1], normal_xy[0]], dtype=np.float64))
            heights = selected[:, 2] - float(self._plane_origin[2])

            preview = np.column_stack([
                along_plane,
                np.zeros_like(along_plane),
                heights,
            ])
            return preview.astype(np.float32), selected_colors

        def _rebuild_scene(self):
            self.renderer.RemoveAllViewProps()
            self._build_scene()
            self.vtk_widget.GetRenderWindow().Render()

        def _make_points_actor(self, points, colors=None, color=(1.0, 1.0, 1.0), point_size=2):
            vtk_pts = vtk.vtkPoints()
            pts = np.ascontiguousarray(points, dtype=np.float32)
            for pt in pts:
                vtk_pts.InsertNextPoint(*pt.tolist())

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_pts)

            verts = vtk.vtkCellArray()
            for i in range(len(points)):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)
            polydata.SetVerts(verts)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if colors is not None and len(colors) == len(points):
                clr = (np.clip(np.ascontiguousarray(colors, dtype=np.float32), 0, 1) * 255).astype(np.uint8)
                vtk_clr = vtk.vtkUnsignedCharArray()
                vtk_clr.SetNumberOfComponents(3)
                vtk_clr.SetName("Colors")
                vtk_clr.SetNumberOfTuples(len(clr))
                for i, rgb in enumerate(clr):
                    vtk_clr.SetTypedTuple(i, rgb.tolist())
                polydata.GetPointData().SetScalars(vtk_clr)
                mapper.ScalarVisibilityOn()
            else:
                actor.GetProperty().SetColor(*color)
                mapper.ScalarVisibilityOff()
            actor.GetProperty().SetPointSize(point_size)
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
            actor.GetProperty().SetOpacity(0.9)
            return actor


class CrossSectionPanel(QWidget):
    cross_section_created = Signal(object)

    def __init__(self, viewport, layer_manager, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self.layer_manager = layer_manager
        self._current_layer = None
        self._normal_pick_mode_active = False
        self._preview_widget = None
        self._plane_offset = 0.0
        

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        if not VTK_AVAILABLE:
            lbl = QLabel("Cross Section requires VTK, which could not be imported.")
            lbl.setWordWrap(True)
            btn = QPushButton("Show error details")
            btn.clicked.connect(self._show_vtk_error)
            layout.addWidget(lbl)
            layout.addWidget(btn)
            layout.addStretch()
            return

        # Normal rich UI (kept minimal here because heavy logic lives in the viewport)

        self._status_label = QLabel(
            "Select a point cloud and then define the cross section."
        )
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # ✅ ADD THIS BUTTON
        self._define_normal_btn = QPushButton("Define Normal")
        self._define_normal_btn.setEnabled(False)  # disabled until valid layer
        layout.addWidget(self._define_normal_btn)

        thickness_row = QHBoxLayout()
        thickness_row.addWidget(QLabel("Distance range:"))
        self._thickness_spin = QDoubleSpinBox(self)
        self._thickness_spin.setDecimals(2)
        self._thickness_spin.setRange(0.0001, 1_000_000.0)
        self._thickness_spin.setSingleStep(0.01)
        self._thickness_spin.setValue(1.0)
        self._thickness_spin.valueChanged.connect(self._on_thickness_value_changed)
        thickness_row.addWidget(self._thickness_spin)
        layout.addLayout(thickness_row)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Move step:"))
        self._step_spin = QDoubleSpinBox(self)
        self._step_spin.setDecimals(2)
        self._step_spin.setRange(0.0001, 1_000_000.0)
        self._step_spin.setSingleStep(0.01)
        self._step_spin.setValue(1.0)
        step_row.addWidget(self._step_spin)
        layout.addLayout(step_row)

        self._offset_label = QLabel("Plane offset: 0.00")
        layout.addWidget(self._offset_label)

        preview_label = QLabel("Preview")
        layout.addWidget(preview_label)

        self._preview_widget = CrossSectionPreviewWidget(self)
        self._preview_widget.setMinimumHeight(280)
        layout.addWidget(self._preview_widget, 1)

        self._define_normal_btn.clicked.connect(self._on_define_normal)
        if hasattr(self.viewport, "cross_section_mode_changed"):
            self.viewport.cross_section_mode_changed.connect(self._on_pick_mode_changed)

        self.setFocusPolicy(Qt.StrongFocus)
        self.installEventFilter(self)
        if self._preview_widget is not None:
            self._preview_widget.installEventFilter(self)
            self._preview_widget.vtk_widget.installEventFilter(self)

        self._forward_shortcut = QShortcut(QKeySequence("W"), self)
        self._forward_shortcut.setContext(Qt.ApplicationShortcut)
        self._forward_shortcut.activated.connect(lambda: self._move_plane(1.0))

        self._backward_shortcut = QShortcut(QKeySequence("S"), self)
        self._backward_shortcut.setContext(Qt.ApplicationShortcut)
        self._backward_shortcut.activated.connect(lambda: self._move_plane(-1.0))

        layout.addStretch()

    def _show_vtk_error(self):
        QMessageBox.critical(self, "VTK Import Error",
                             f"VTK could not be imported:\n{_vtk_import_error}")

    def shutdown_vtk(self):
        if self._preview_widget is not None and hasattr(self._preview_widget, "shutdown_vtk"):
            self._preview_widget.shutdown_vtk()

    def refresh_preview(self):
        if self._preview_widget is None:
            return

        if not isinstance(self._current_layer, PointCloudLayer) or not getattr(self._current_layer, "visible", True):
            self._preview_widget.set_preview_data(
                np.empty((0, 3), dtype=np.float32),
                None,
                np.zeros(3, dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
                self._thickness_spin.value() if hasattr(self, "_thickness_spin") else 1.0,
            )
            return

        state = self.viewport.get_cross_section_state()
        points = state.get("direction_points", [])
        direction_xy = state.get("direction_xy")
        if len(points) < 2 or direction_xy is None:
            return

        self._preview_widget.set_preview_data(
            self._current_layer.points,
            self.viewport.get_cross_section_preview_colors(self._current_layer),
            self._get_shifted_plane_origin(points, direction_xy),
            np.asarray(direction_xy, dtype=np.float32),
            self._thickness_spin.value(),
        )

    # Keep compatibility methods used by MainWindow; the heavy interactivity is
    # implemented in the Viewport, so these methods are intentionally lightweight.
    def open_panel(self):
        return
    
    def set_current_layer(self, layer):
        self._current_layer = layer

        if isinstance(layer, PointCloudLayer):
            self._status_label.setText(f"Selected: {layer.name}")
            self._define_normal_btn.setEnabled(True)
            self.viewport.enable_cross_section_mode(layer)
        else:
            self._status_label.setText("Select a point cloud")
            self._define_normal_btn.setEnabled(False)
            self._normal_pick_mode_active = False
            self._update_define_normal_button()
            self.viewport.disable_cross_section_mode()
            return

        self._normal_pick_mode_active = False
        self._plane_offset = 0.0
        self._update_offset_label()
        self.viewport.set_cross_section_plane_offset(self._plane_offset)
        self._update_define_normal_button()
        
    def _on_define_normal(self):
        if not isinstance(self._current_layer, PointCloudLayer):
            return

        self.viewport.enable_cross_section_mode(self._current_layer)
        self.viewport.begin_cross_section_pick_session(
            mode="direction",
            clear_existing=True,
        )
        self._normal_pick_mode_active = True
        self._plane_offset = 0.0
        self._update_offset_label()
        self.viewport.set_cross_section_plane_offset(self._plane_offset)
        self._status_label.setText(
            "Normal selection mode: click two points to form a normal. Right-click cancels. (noted that you can not click point under orbit view)"
        )
        self._update_define_normal_button()

    def _on_pick_mode_changed(self, mode):
        state = self.viewport.get_cross_section_state()
        self._normal_pick_mode_active = bool(
            state.get("pick_session_active") and mode == "direction"
        )
        has_completed_normal = (
            len(state.get("direction_points", [])) >= 2 and
            state.get("direction_xy") is not None and
            not self._normal_pick_mode_active
        )

        if isinstance(self._current_layer, PointCloudLayer):
            if self._normal_pick_mode_active:
                self._status_label.setText(
                    "Normal selection mode: click two points to form a normal. Right-click cancels. (noted that you can not click point under orbit view)"
                )
            else:
                self._status_label.setText(f"Selected: {self._current_layer.name}")

        self._update_define_normal_button()
        if has_completed_normal:
            self._on_preview_cross_section()

    def _on_preview_cross_section(self):
        if not isinstance(self._current_layer, PointCloudLayer):
            return

        if not getattr(self._current_layer, "visible", True):
            self.refresh_preview()
            return

        state = self.viewport.get_cross_section_state()
        points = state.get("direction_points", [])
        direction_xy = state.get("direction_xy")
        if len(points) < 2 or direction_xy is None:
            QMessageBox.information(
                self,
                "Cross Section Preview",
                "Define the normal line first using two points.",
            )
            return

        direction_xy = np.asarray(direction_xy, dtype=np.float32)
        plane_origin = np.asarray(points[0], dtype=np.float32).copy()
        plane_normal_xy = direction_xy.astype(np.float32)
        plane_origin[:2] += plane_normal_xy * self._plane_offset

        self._preview_widget.set_preview_data(
            self._current_layer.points,
            self.viewport.get_cross_section_preview_colors(self._current_layer),
            plane_origin,
            plane_normal_xy,
            self._thickness_spin.value(),
        )

    def _on_thickness_value_changed(self, value):
        self.viewport.set_cross_section_thickness(value)
        if not isinstance(self._current_layer, PointCloudLayer) or self._preview_widget is None:
            return

        state = self.viewport.get_cross_section_state()
        points = state.get("direction_points", [])
        direction_xy = state.get("direction_xy")
        if len(points) < 2 or direction_xy is None:
            return

        self.refresh_preview()

    def _get_shifted_plane_origin(self, points, direction_xy):
        plane_origin = np.asarray(points[0], dtype=np.float32).copy()
        plane_normal_xy = np.asarray(direction_xy, dtype=np.float32)
        plane_origin[:2] += plane_normal_xy * self._plane_offset
        return plane_origin

    def _update_offset_label(self):
        if hasattr(self, "_offset_label"):
            self._offset_label.setText(f"Plane offset: {self._plane_offset:.2f}")

    def _move_plane(self, direction_sign):
        if not isinstance(self._current_layer, PointCloudLayer):
            return
        state = self.viewport.get_cross_section_state()
        points = state.get("direction_points", [])
        direction_xy = state.get("direction_xy")
        if len(points) < 2 or direction_xy is None:
            return

        self._plane_offset += float(direction_sign) * float(self._step_spin.value())
        self._update_offset_label()
        self.viewport.set_cross_section_plane_offset(self._plane_offset)
        self.refresh_preview()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_W:
            self._move_plane(1.0)
            event.accept()
            return
        if key == Qt.Key_S:
            self._move_plane(-1.0)
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key_W:
                self._move_plane(1.0)
                return True
            if key == Qt.Key_S:
                self._move_plane(-1.0)
                return True
        return super().eventFilter(obj, event)

    def _update_define_normal_button(self):
        if not hasattr(self, "_define_normal_btn"):
            return
        if not isinstance(self._current_layer, PointCloudLayer):
            self._define_normal_btn.setText("Define Normal")
            self._define_normal_btn.setEnabled(False)
            return

        self._define_normal_btn.setEnabled(not self._normal_pick_mode_active)
        self._define_normal_btn.setText(
            "Selecting Normal..." if self._normal_pick_mode_active else "Define Normal"
        )


