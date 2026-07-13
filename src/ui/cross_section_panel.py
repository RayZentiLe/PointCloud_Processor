import numpy as np
import importlib
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox,
    QHBoxLayout, QDoubleSpinBox
)
from PySide6.QtCore import Signal, Qt, QEvent, QRect
from PySide6.QtGui import QShortcut, QKeySequence, QPainter, QColor, QPen

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

    class SelectionOverlay(QWidget):
        def __init__(self, preview_widget, parent=None):
            super().__init__(parent)
            self._preview_widget = preview_widget
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.hide()

        def paintEvent(self, event):
            super().paintEvent(event)
            if self._preview_widget is None:
                return
            if not self._preview_widget._selection_mode_enabled:
                return

            rect = self._preview_widget._selection_rect()
            if rect.isNull():
                return

            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(255, 255, 255), 3, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
            painter.end()

    class CrossSectionPreviewStyle(vtk.vtkInteractorStyleTrackballCamera):
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
            camera.SetParallelProjection(True)
            camera.SetPosition(focal_point[0], focal_point[1], focal_point[2] + distance)
            camera.SetFocalPoint(*focal_point)
            camera.SetViewUp(0.0, 1.0, 0.0)
            camera.OrthogonalizeViewUp()
            if parallel_scale > 0:
                camera.SetParallelScale(parallel_scale)
            renderer.ResetCameraClippingRange()

        def _render_locked(self):
            self._lock_camera_orientation()
            if self._preview_widget is not None:
                self._preview_widget.vtk_widget.GetRenderWindow().Render()

        def OnLeftButtonDown(self):
            if self._preview_widget is not None and self._preview_widget.handle_left_button_down():
                return
            self.FindPokedRenderer(
                self.GetInteractor().GetEventPosition()[0],
                self.GetInteractor().GetEventPosition()[1],
            )
            self.GrabFocus(self.EventCallbackCommand)
            self.StartPan()

        def OnLeftButtonUp(self):
            if self._preview_widget is not None and self._preview_widget.handle_left_button_up():
                return
            self.ReleaseFocus()
            self.EndPan()

        def OnMiddleButtonDown(self):
            self.FindPokedRenderer(
                self.GetInteractor().GetEventPosition()[0],
                self.GetInteractor().GetEventPosition()[1],
            )
            self.GrabFocus(self.EventCallbackCommand)
            self.StartPan()

        def OnMiddleButtonUp(self):
            self.ReleaseFocus()
            self.EndPan()

        def OnRightButtonDown(self):
            self.FindPokedRenderer(
                self.GetInteractor().GetEventPosition()[0],
                self.GetInteractor().GetEventPosition()[1],
            )
            self.GrabFocus(self.EventCallbackCommand)
            self.StartDolly()

        def OnRightButtonUp(self):
            self.ReleaseFocus()
            self.EndDolly()

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
            self._render_locked()
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
            self._render_locked()
            self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)

        def OnChar(self):
            return

        def Rotate(self):
            self._render_locked()
            return

        def Spin(self):
            self._render_locked()
            return

        def OnMouseMove(self):
            if self._preview_widget is not None and self._preview_widget.handle_mouse_move():
                return
            state = self.GetState()
            if state == vtk.VTKIS_PAN:
                self._pan_camera()
                self._render_locked()
                self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)
                return
            if state == vtk.VTKIS_DOLLY:
                self._zoom_camera()
                self._render_locked()
                self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)
                return
            if state in (vtk.VTKIS_ROTATE, vtk.VTKIS_SPIN):
                self._render_locked()
                self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)
                return

        def OnInteraction(self, obj=None, event=None):
            self._render_locked()

    class CrossSectionPreviewWidget(QWidget):
        selection_changed = Signal(object)

        def __init__(self, parent=None):
            super().__init__(parent)

            self._layer_points = np.empty((0, 3), dtype=np.float32)
            self._layer_colors = None
            self._plane_origin = np.zeros(3, dtype=np.float32)
            self._plane_normal_xy = np.array([1.0, 0.0], dtype=np.float32)
            self._thickness = 1.0
            self._selection_mode_enabled = False
            self._selection_drag_active = False
            self._selection_start = None
            self._selection_end = None
            self._selected_preview_indices = np.empty((0,), dtype=np.int32)
            self._highlight_color = np.array([1.0, 0.0, 0.6], dtype=np.float32)
            self._preview_points_cache = np.empty((0, 3), dtype=np.float32)
            self._preview_colors_cache = None
            self._uv_bounds = None
            self._edge_axes_actor = None
            self._edge_axes_label_actors = []
            self._selection_uv_rect = None
            self._selection_corner_label_actors = []
            self._selection_rectangle_actor = None
            self._selection_overlay = None

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            self.vtk_widget = QVTKRenderWindowInteractor(self)
            layout.addWidget(self.vtk_widget)
            self.vtk_widget.setMouseTracking(True)
            self.vtk_widget.installEventFilter(self)
            self._selection_overlay = SelectionOverlay(self, self)
            self._selection_overlay.setGeometry(self.vtk_widget.geometry())
            self._selection_overlay.raise_()
            self._selection_overlay.show()

            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.1, 0.12)
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
            self._vtk_closed = False

            interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            style = CrossSectionPreviewStyle(self)
            interactor.SetInteractorStyle(style)
            style.SetDefaultRenderer(self.renderer)
            interactor.AddObserver(vtk.vtkCommand.InteractionEvent, style.OnInteraction)

            self._build_scene()
            self.vtk_widget.Initialize()
            self.vtk_widget.Start()

        def _selection_rect(self):
            if self._selection_start is None or self._selection_end is None:
                return QRect()
            return QRect(
                self._selection_start[0],
                self._selection_start[1],
                self._selection_end[0] - self._selection_start[0],
                self._selection_end[1] - self._selection_start[1],
            ).normalized()

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
            self.clear_selection()
            self._rebuild_scene()

        def set_selection_mode_enabled(self, enabled):
            enabled = bool(enabled)
            if self._selection_mode_enabled == enabled:
                return
            self._selection_mode_enabled = enabled
            if not enabled:
                self.clear_selection()

        def clear_selection(self):
            self._selection_drag_active = False
            self._selection_start = None
            self._selection_end = None
            self._selected_preview_indices = np.empty((0,), dtype=np.int32)
            self._selection_uv_rect = None
            self._clear_selection_corner_labels()
            self.selection_changed.emit(None)
            self.vtk_widget.update()
            if self._selection_overlay is not None:
                self._selection_overlay.update()
            if hasattr(self, "renderer"):
                self._rebuild_scene()

        def handle_left_button_down(self):
            if not self._selection_mode_enabled:
                return False
            interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            if interactor is None:
                return False
            x, y = interactor.GetEventPosition()
            self._selection_drag_active = True
            self._selection_start = (int(x), int(y))
            self._selection_end = (int(x), int(y))
            self.vtk_widget.update()
            if self._selection_overlay is not None:
                self._selection_overlay.update()
            return True

        def handle_mouse_move(self):
            if not self._selection_mode_enabled or not self._selection_drag_active:
                return False
            interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            if interactor is None:
                return False
            x, y = interactor.GetEventPosition()
            self._selection_end = (int(x), int(y))
            self.vtk_widget.update()
            if self._selection_overlay is not None:
                self._selection_overlay.update()
            return True

        def handle_left_button_up(self):
            if not self._selection_mode_enabled or not self._selection_drag_active:
                return False
            interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            if interactor is None:
                return False
            x, y = interactor.GetEventPosition()
            self._selection_end = (int(x), int(y))
            self._selection_drag_active = False
            self._apply_rectangle_selection()
            self.vtk_widget.update()
            if self._selection_overlay is not None:
                self._selection_overlay.update()
            return True

        def eventFilter(self, obj, event):
            if obj is self.vtk_widget and self._selection_mode_enabled:
                if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                    pos = event.position().toPoint()
                    self._selection_drag_active = True
                    self._selection_start = (pos.x(), pos.y())
                    self._selection_end = (pos.x(), pos.y())
                    self.vtk_widget.update()
                    if self._selection_overlay is not None:
                        self._selection_overlay.update()
                    return True
                if event.type() == QEvent.Type.MouseMove and self._selection_drag_active:
                    pos = event.position().toPoint()
                    self._selection_end = (pos.x(), pos.y())
                    self.vtk_widget.update()
                    if self._selection_overlay is not None:
                        self._selection_overlay.update()
                    return True
                if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton and self._selection_drag_active:
                    pos = event.position().toPoint()
                    self._selection_end = (pos.x(), pos.y())
                    self._selection_drag_active = False
                    self._apply_rectangle_selection()
                    self.vtk_widget.update()
                    if self._selection_overlay is not None:
                        self._selection_overlay.update()
                    return True
            return super().eventFilter(obj, event)

        def _apply_rectangle_selection(self):
            if self._selection_start is None or self._selection_end is None:
                self._selected_preview_indices = np.empty((0,), dtype=np.int32)
                self._rebuild_scene(preserve_camera=True)
                return

            rect = QRect(
                min(self._selection_start[0], self._selection_end[0]),
                min(self._selection_start[1], self._selection_end[1]),
                abs(self._selection_end[0] - self._selection_start[0]),
                abs(self._selection_end[1] - self._selection_start[1]),
            ).normalized()

            preview_points = self._preview_points_cache
            if len(preview_points) == 0 or rect.width() == 0 or rect.height() == 0:
                self._selected_preview_indices = np.empty((0,), dtype=np.int32)
                self._selection_uv_rect = None
                self.selection_changed.emit(None)
                self._rebuild_scene(preserve_camera=True)
                return

            top_left_uv = self._display_to_uv(rect.left(), rect.top())
            bottom_right_uv = self._display_to_uv(rect.right(), rect.bottom())
            u_min = min(float(top_left_uv[0]), float(bottom_right_uv[0]))
            u_max = max(float(top_left_uv[0]), float(bottom_right_uv[0]))
            v_min = min(float(top_left_uv[1]), float(bottom_right_uv[1]))
            v_max = max(float(top_left_uv[1]), float(bottom_right_uv[1]))
            self._selection_uv_rect = {
                "left_up": {
                    "u": u_min,
                    "v": v_max,
                },
                "right_bottom": {
                    "u": u_max,
                    "v": v_min,
                },
            }

            selected_indices = []
            for idx, point in enumerate(preview_points):
                u = float(point[0])
                v = float(point[1])
                if u_min <= u <= u_max and v_min <= v <= v_max:
                    selected_indices.append(idx)

            self._selected_preview_indices = np.asarray(selected_indices, dtype=np.int32)
            self.selection_changed.emit(self._selection_uv_rect)
            self._rebuild_scene(preserve_camera=True)

        def _world_to_display(self, point):
            self.renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
            self.renderer.WorldToDisplay()
            return self.renderer.GetDisplayPoint()

        def _display_to_uv(self, x, y):
            camera = self.renderer.GetActiveCamera()
            if camera is None:
                return np.array([0.0, 0.0], dtype=np.float64)

            focal_u, focal_v, _ = camera.GetFocalPoint()
            parallel_scale = max(float(camera.GetParallelScale()), 1e-6)
            widget_width = max(float(self.vtk_widget.width()), 1.0)
            widget_height = max(float(self.vtk_widget.height()), 1.0)
            aspect = widget_width / widget_height
            half_width = parallel_scale * aspect
            half_height = parallel_scale

            u = (float(x) / widget_width) * (2.0 * half_width) + (focal_u - half_width)
            v = ((widget_height - float(y)) / widget_height) * (2.0 * half_height) + (focal_v - half_height)
            return np.array([u, v], dtype=np.float64)

        def _build_scene(self, preserve_camera=False):
            preview_points, preview_colors = self._compute_preview_points()
            self._preview_points_cache = preview_points
            self._preview_colors_cache = preview_colors
            if len(preview_points) > 0:
                if preview_colors is None:
                    preview_colors = np.full((len(preview_points), 3), 1.0, dtype=np.float32)
                else:
                    preview_colors = np.asarray(preview_colors, dtype=np.float32).copy()
                if len(self._selected_preview_indices) > 0:
                    valid_indices = self._selected_preview_indices[
                        (self._selected_preview_indices >= 0) & (self._selected_preview_indices < len(preview_colors))
                    ]
                    if len(valid_indices) > 0:
                        preview_colors[valid_indices] = self._highlight_color
                actor = self._make_points_actor(preview_points, colors=preview_colors, point_size=4)
                self.renderer.AddActor(actor)

                self._uv_bounds = self._compute_uv_bounds(preview_points)
                self._add_edge_axes_overlay(self._uv_bounds)
            else:
                self._uv_bounds = None

            origin_actor = self._make_sphere_actor(np.zeros(3, dtype=np.float32), radius=0.03, color=(0.2, 0.8, 1.0))
            self.renderer.AddActor(origin_actor)

            if not preserve_camera:
                camera = vtk.vtkCamera()
                camera.SetPosition(0.0, -1.0, 0.0)
                camera.SetFocalPoint(0.0, 0.0, 0.0)
                camera.SetViewUp(0.0, 0.0, 1.0)
                camera.SetParallelProjection(True)
                self.renderer.SetActiveCamera(camera)
                self.renderer.ResetCamera()
                self._apply_parallel_scale_from_bounds()
            else:
                self.renderer.ResetCameraClippingRange()

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
                heights,
                np.zeros_like(along_plane),
            ])
            return preview.astype(np.float32), selected_colors

        def _compute_uv_bounds(self, preview_points):
            if len(preview_points) == 0:
                return None
            u_values = preview_points[:, 0]
            v_values = preview_points[:, 1]
            u_min = float(np.min(u_values))
            u_max = float(np.max(u_values))
            v_min = float(np.min(v_values))
            v_max = float(np.max(v_values))
            if abs(u_max - u_min) < 1e-6:
                u_min -= 0.5
                u_max += 0.5
            if abs(v_max - v_min) < 1e-6:
                v_min -= 0.5
                v_max += 0.5
            padding_u = max((u_max - u_min) * 0.05, 0.1)
            padding_v = max((v_max - v_min) * 0.05, 0.1)
            return {
                "u_min": u_min - padding_u,
                "u_max": u_max + padding_u,
                "v_min": v_min - padding_v,
                "v_max": v_max + padding_v,
            }

        def _apply_parallel_scale_from_bounds(self):
            if not self._uv_bounds:
                return
            camera = self.renderer.GetActiveCamera()
            if camera is None:
                return
            u_center = 0.5 * (self._uv_bounds["u_min"] + self._uv_bounds["u_max"])
            v_center = 0.5 * (self._uv_bounds["v_min"] + self._uv_bounds["v_max"])
            u_span = self._uv_bounds["u_max"] - self._uv_bounds["u_min"]
            v_span = self._uv_bounds["v_max"] - self._uv_bounds["v_min"]
            widget_width = max(float(self.vtk_widget.width()), 1.0)
            widget_height = max(float(self.vtk_widget.height()), 1.0)
            aspect = widget_width / widget_height
            parallel_scale = max(v_span * 0.5, (u_span * 0.5) / max(aspect, 1e-6), 1e-3)
            distance = max(camera.GetDistance(), 1.0)
            camera.SetFocalPoint(u_center, v_center, 0.0)
            camera.SetPosition(u_center, v_center, distance)
            camera.SetViewUp(0.0, 1.0, 0.0)
            camera.SetParallelScale(parallel_scale)
            self.renderer.ResetCameraClippingRange()

        def _add_edge_axes_overlay(self, bounds):
            self._clear_edge_axes_overlay()
            if bounds is None:
                return

            u_axis_start = np.array([bounds["u_min"], bounds["v_min"], 0.0], dtype=np.float32)
            u_axis_end = np.array([bounds["u_max"], bounds["v_min"], 0.0], dtype=np.float32)
            v_axis_start = np.array([bounds["u_min"], bounds["v_min"], 0.0], dtype=np.float32)
            v_axis_end = np.array([bounds["u_min"], bounds["v_max"], 0.0], dtype=np.float32)

            append = vtk.vtkAppendPolyData()
            for start, end in ((u_axis_start, u_axis_end), (v_axis_start, v_axis_end)):
                line_source = vtk.vtkLineSource()
                line_source.SetPoint1(*start.tolist())
                line_source.SetPoint2(*end.tolist())
                line_source.Update()
                append.AddInputData(line_source.GetOutput())
            append.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(append.GetOutputPort())

            self._edge_axes_actor = vtk.vtkActor()
            self._edge_axes_actor.SetMapper(mapper)
            self._edge_axes_actor.GetProperty().SetColor(0.7, 0.7, 0.7)
            self._edge_axes_actor.GetProperty().SetLineWidth(1.5)
            self.renderer.AddActor(self._edge_axes_actor)

            self._edge_axes_label_actors = [
                self._make_axis_label_actor(np.array([bounds["u_max"], bounds["v_min"], 0.0], dtype=np.float32), "U", (0.9, 0.9, 0.9), offset=(-18, 8)),
                self._make_axis_label_actor(np.array([bounds["u_min"], bounds["v_max"], 0.0], dtype=np.float32), "V", (0.9, 0.9, 0.9), offset=(8, -18)),
            ]
            for actor in self._edge_axes_label_actors:
                self.renderer.AddActor2D(actor)

        def _clear_edge_axes_overlay(self):
            if self._edge_axes_actor is not None:
                self.renderer.RemoveActor(self._edge_axes_actor)
                self._edge_axes_actor = None
            for actor in self._edge_axes_label_actors:
                self.renderer.RemoveActor2D(actor)
            self._edge_axes_label_actors = []

        def _clear_selection_corner_labels(self):
            for actor in self._selection_corner_label_actors:
                self.renderer.RemoveActor2D(actor)
            self._selection_corner_label_actors = []

        def _clear_selection_rectangle_actor(self):
            if self._selection_rectangle_actor is not None:
                self.renderer.RemoveActor(self._selection_rectangle_actor)
                self._selection_rectangle_actor = None

        def _make_axis_label_actor(self, point, text, color, offset=(6, 6)):
            coord = self._world_to_display(point)
            label = vtk.vtkTextActor()
            label.SetInput(text)
            label.SetPosition(coord[0] + offset[0], coord[1] + offset[1])
            text_prop = label.GetTextProperty()
            text_prop.SetFontSize(14)
            text_prop.SetColor(*color)
            text_prop.SetBold(True)
            return label

        def _rebuild_scene(self, preserve_camera=False):
            self.renderer.RemoveAllViewProps()
            self._clear_edge_axes_overlay()
            self._clear_selection_corner_labels()
            self._clear_selection_rectangle_actor()
            self._build_scene(preserve_camera=preserve_camera)
            self._add_selection_rectangle()
            self._add_selection_corner_labels()
            self.vtk_widget.GetRenderWindow().Render()

        def _add_selection_rectangle(self):
            if not self._selection_uv_rect:
                return

            left_up = self._selection_uv_rect["left_up"]
            right_bottom = self._selection_uv_rect["right_bottom"]
            u_min = min(float(left_up["u"]), float(right_bottom["u"]))
            u_max = max(float(left_up["u"]), float(right_bottom["u"]))
            v_min = min(float(left_up["v"]), float(right_bottom["v"]))
            v_max = max(float(left_up["v"]), float(right_bottom["v"]))

            points = vtk.vtkPoints()
            rectangle_points = [
                (u_min, v_max, 0.0),
                (u_max, v_max, 0.0),
                (u_max, v_min, 0.0),
                (u_min, v_min, 0.0),
                (u_min, v_max, 0.0),
            ]
            for point in rectangle_points:
                points.InsertNextPoint(*point)

            poly_line = vtk.vtkPolyLine()
            poly_line.GetPointIds().SetNumberOfIds(len(rectangle_points))
            for index in range(len(rectangle_points)):
                poly_line.GetPointIds().SetId(index, index)

            cells = vtk.vtkCellArray()
            cells.InsertNextCell(poly_line)

            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            poly_data.SetLines(cells)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)

            self._selection_rectangle_actor = vtk.vtkActor()
            self._selection_rectangle_actor.SetMapper(mapper)
            self._selection_rectangle_actor.GetProperty().SetColor(1.0, 1.0, 0.2)
            self._selection_rectangle_actor.GetProperty().SetLineWidth(2.5)
            self._selection_rectangle_actor.GetProperty().SetOpacity(0.95)
            self.renderer.AddActor(self._selection_rectangle_actor)

        def _add_selection_corner_labels(self):
            if not self._selection_uv_rect:
                return

            left_up = self._selection_uv_rect["left_up"]
            right_bottom = self._selection_uv_rect["right_bottom"]
            corners = [
                (
                    np.array([left_up["u"], left_up["v"], 0.0], dtype=np.float32),
                    f"LU ({left_up['u']:.3f}, {left_up['v']:.3f})",
                    (8, -22),
                ),
                (
                    np.array([right_bottom["u"], right_bottom["v"], 0.0], dtype=np.float32),
                    f"RB ({right_bottom['u']:.3f}, {right_bottom['v']:.3f})",
                    (-150, 8),
                ),
            ]

            for point, text, offset in corners:
                actor = self._make_axis_label_actor(point, text, (1.0, 1.0, 1.0), offset=offset)
                self._selection_corner_label_actors.append(actor)
                self.renderer.AddActor2D(actor)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if self._selection_overlay is not None:
                self._selection_overlay.setGeometry(self.vtk_widget.geometry())
                self._selection_overlay.raise_()
                self._selection_overlay.update()
            if hasattr(self, "renderer"):
                self._apply_parallel_scale_from_bounds()
                self.vtk_widget.GetRenderWindow().Render()

        def showEvent(self, event):
            super().showEvent(event)
            if self._selection_overlay is not None:
                self._selection_overlay.setGeometry(self.vtk_widget.geometry())
                self._selection_overlay.raise_()
                self._selection_overlay.show()
                self._selection_overlay.update()

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
        self._selection_uv_bounds = None
        

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

        self._select_point_btn = QPushButton("Select Point")
        self._select_point_btn.setCheckable(True)
        self._select_point_btn.setEnabled(False)
        layout.addWidget(self._select_point_btn)

        thickness_row = QHBoxLayout()
        thickness_row.addWidget(QLabel("Distance range:"))
        self._thickness_spin = QDoubleSpinBox(self)
        self._thickness_spin.setDecimals(2)
        self._thickness_spin.setRange(0.0001, 1_000_000.0)
        self._thickness_spin.setSingleStep(0.5)
        self._thickness_spin.setValue(1.0)
        self._thickness_spin.valueChanged.connect(self._on_thickness_value_changed)
        thickness_row.addWidget(self._thickness_spin)
        layout.addLayout(thickness_row)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Move step:"))
        self._step_spin = QDoubleSpinBox(self)
        self._step_spin.setDecimals(2)
        self._step_spin.setRange(0.0001, 1_000_000.0)
        self._step_spin.setSingleStep(0.5)
        self._step_spin.setValue(1.0)
        step_row.addWidget(self._step_spin)
        layout.addLayout(step_row)

        self._offset_label = QLabel("Plane offset: 0.00")
        layout.addWidget(self._offset_label)

        self._selection_uv_label = QLabel("Selection UV: -")
        self._selection_uv_label.setWordWrap(True)
        layout.addWidget(self._selection_uv_label)

        preview_label = QLabel("Preview")
        layout.addWidget(preview_label)

        self._preview_widget = CrossSectionPreviewWidget(self)
        self._preview_widget.setMinimumHeight(280)
        layout.addWidget(self._preview_widget, 1)
        self._preview_widget.selection_changed.connect(self._on_preview_selection_changed)

        self._define_normal_btn.clicked.connect(self._on_define_normal)
        self._select_point_btn.toggled.connect(self._on_select_point_toggled)
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
            self._select_point_btn.setEnabled(True)
            self.viewport.enable_cross_section_mode(layer)
        else:
            self._status_label.setText("Select a point cloud")
            self._define_normal_btn.setEnabled(False)
            self._select_point_btn.setEnabled(False)
            self._select_point_btn.setChecked(False)
            self._normal_pick_mode_active = False
            self._update_define_normal_button()
            self.viewport.disable_cross_section_mode()
            return

        self._normal_pick_mode_active = False
        self._plane_offset = 0.0
        self._update_offset_label()
        self.viewport.set_cross_section_plane_offset(self._plane_offset)
        self._update_define_normal_button()

    def _on_select_point_toggled(self, checked):
        if self._preview_widget is None:
            return
        self._preview_widget.set_selection_mode_enabled(checked)
        if checked:
            self._status_label.setText("Select point mode: left-drag in preview to highlight points inside the rectangle.")
        elif isinstance(self._current_layer, PointCloudLayer) and not self._normal_pick_mode_active:
            self._status_label.setText(f"Selected: {self._current_layer.name}")
            self._selection_uv_label.setText("Selection UV: -")

    def _on_preview_selection_changed(self, selection_uv_rect):
        self._selection_uv_bounds = selection_uv_rect
        if hasattr(self.viewport, "set_cross_section_preview_selection"):
            self.viewport.set_cross_section_preview_selection(selection_uv_rect)
        if not selection_uv_rect:
            self._selection_uv_label.setText("Selection UV: -")
            return

        left_up = selection_uv_rect["left_up"]
        right_bottom = selection_uv_rect["right_bottom"]
        self._selection_uv_label.setText(
            "Selection UV: "
            f"left-up (u={left_up['u']:.3f}, v={left_up['v']:.3f}), "
            f"right-bottom (u={right_bottom['u']:.3f}, v={right_bottom['v']:.3f})"
        )
        
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
            if hasattr(self, "_select_point_btn"):
                self._select_point_btn.setEnabled(False)
            return

        self._define_normal_btn.setEnabled(not self._normal_pick_mode_active)
        self._define_normal_btn.setText(
            "Selecting Normal..." if self._normal_pick_mode_active else "Define Normal"
        )
        if hasattr(self, "_select_point_btn"):
            self._select_point_btn.setEnabled(True)


