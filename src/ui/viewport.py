import sys
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtCore import Qt, QEvent, QPoint, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QColorDialog, QMenu, QPushButton, QHBoxLayout
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from tools.gradient_colors import compute_gradient_colors
from core.layer_manager import LayerManager
from core.layer import PointCloudLayer, MeshLayer


class ZPlanePanStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Camera style that keeps the main viewport locked to a top-down Z view."""

    def __init__(self, viewport):
        super().__init__()
        self._viewport = viewport

    def OnLeftButtonDown(self):
        if getattr(self._viewport, "_fixed_z_plane_view", True):
            self.StartPan()
        else:
            super().OnLeftButtonDown()

    def OnLeftButtonUp(self):
        if getattr(self._viewport, "_fixed_z_plane_view", True):
            self.EndPan()
        else:
            super().OnLeftButtonUp()

    def OnMiddleButtonDown(self):
        self.StartPan()

    def OnMiddleButtonUp(self):
        self.EndPan()

    def OnRightButtonDown(self):
        self.StartDolly()

    def OnRightButtonUp(self):
        self.EndDolly()

    def OnMouseWheelForward(self):
        self.StartDolly()
        self.Dolly()
        self.EndDolly()
        self._viewport._enforce_z_plane_camera()

    def OnMouseWheelBackward(self):
        self.StartDolly()
        self.Dolly()
        self.EndDolly()
        self._viewport._enforce_z_plane_camera()

    def Rotate(self):
        if getattr(self._viewport, "_fixed_z_plane_view", True):
            self._viewport._enforce_z_plane_camera()
            return
        super().Rotate()

    def Spin(self):
        if getattr(self._viewport, "_fixed_z_plane_view", True):
            self._viewport._enforce_z_plane_camera()
            return
        super().Spin()

    def OnMouseMove(self):
        if not getattr(self._viewport, "_fixed_z_plane_view", True):
            super().OnMouseMove()
            return
        state = self.GetState()
        if state == vtk.VTKIS_PAN:
            self.Pan()
            self._viewport._enforce_z_plane_camera()
            self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)
            return
        if state == vtk.VTKIS_DOLLY:
            self.Dolly()
            self._viewport._enforce_z_plane_camera()
            self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)
            return
        self._viewport._enforce_z_plane_camera()
        super().OnMouseMove()


class OrbitViewStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Free orbit camera style with left-drag rotate, middle-drag pan, right-drag zoom."""

    def __init__(self, viewport):
        super().__init__()
        self._viewport = viewport

    def OnLeftButtonDown(self):
        self.FindPokedRenderer(
            self.GetInteractor().GetEventPosition()[0],
            self.GetInteractor().GetEventPosition()[1],
        )
        self.GrabFocus(self.EventCallbackCommand)
        self.StartRotate()

    def OnLeftButtonUp(self):
        self.ReleaseFocus()
        self.EndRotate()

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

    def OnMouseWheelForward(self):
        self.StartDolly()
        self.Dolly()
        self.EndDolly()
        self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)

    def OnMouseWheelBackward(self):
        self.StartDolly()
        self.Dolly()
        self.EndDolly()
        self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)

    def OnMouseMove(self):
        state = self.GetState()
        if state == vtk.VTKIS_ROTATE:
            self.Rotate()
            self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)
            return
        if state == vtk.VTKIS_PAN:
            self.Pan()
            self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)
            return
        if state == vtk.VTKIS_DOLLY:
            self.Dolly()
            self.InvokeEvent(vtk.vtkCommand.InteractionEvent, None)
            return
        super().OnMouseMove()


class Viewport(QWidget):
    cross_section_point_selected = Signal(object)
    cross_section_mode_changed = Signal(str)

    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(parent)
        self.layer_manager = layer_manager
        self._actors: dict[str, list[vtk.vtkActor]] = {}
        self._bg_color = (31 / 255.0, 31 / 255.0, 31 / 255.0)  # Default to #1f1f1f
        self._right_press_pos = None
        self._picker = vtk.vtkPointPicker()
        self._picker.SetTolerance(0.01)
        self._world_picker = vtk.vtkWorldPointPicker()
        self._cross_section_active = False
        self._cross_section_layer_id: str | None = None
        self._cross_section_points: list[np.ndarray] = []
        self._cross_section_reference: np.ndarray | None = None
        self._cross_section_direction_xy: np.ndarray | None = None
        self._cross_section_pick_mode = "direction"
        self._cross_section_line_actor = None
        self._cross_section_preview_line_actor = None
        self._cross_section_preview_plane_actor = None
        self._cross_section_preview_plane_trace_actor = None
        self._cross_section_confirmed_line_actor = None  # Green line when confirmed
        self._cross_section_polyline_actor = None  # Vertical cross-section plane actor
        self._cross_section_point_actors: list[vtk.vtkActor] = []
        self._cross_section_ref_actor = None
        self._cross_section_reference_projection_actor = None
        self._cross_section_reference_line_actor = None
        self._cross_section_thickness = 1.0
        self._cross_section_plane_offset = 0.0
        self._cross_section_direction_confirmed = False  # Track confirmation state
        self._cross_section_pick_session_active = False
        self._cross_section_preview_point: np.ndarray | None = None
        self._cross_section_hover_actor = None
        self._cross_section_hover_point: np.ndarray | None = None
        self._cross_section_hover_label_actor = None
        self._clicked_point_actor = None
        self._clicked_point_label_actor = None
        self._clicked_point: np.ndarray | None = None
        self._cross_section_selected_label_actors: list[vtk.vtkActor2D] = []
        self._cross_section_selected_point_indices = np.empty((0,), dtype=np.int32)
        self._cross_section_selected_point_indices_by_layer: dict[str, np.ndarray] = {}
        self._cross_section_selection_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self._fixed_z_plane_view = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self._view_mode_button = QPushButton("Fixed Z Plane View", self)
        self._view_mode_button.clicked.connect(self._toggle_view_mode)
        top_bar.addWidget(self._view_mode_button)
        layout.addLayout(top_bar)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*self._bg_color)
        self.renderer.GradientBackgroundOff()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        camera = self.renderer.GetActiveCamera()
        if camera is not None:
            camera.SetParallelProjection(True)

        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = ZPlanePanStyle(self)
        interactor.SetInteractorStyle(style)
        self._z_plane_style = style
        self._free_view_style = OrbitViewStyle(self)

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
        interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self._vtk_closed = False

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
            self._remove_cross_section_actors()
        except Exception:
            pass
        try:
            for layer_id in list(self._actors.keys()):
                self._clear_actors(layer_id)
        except Exception:
            pass
        try:
            if hasattr(self, "_orient") and self._orient is not None:
                self._orient.EnabledOff()
                self._orient.SetInteractor(None)
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


    def _on_left_button_press(self, obj, event):
        x, y = obj.GetEventPosition()

        if self._fixed_z_plane_view and self._cross_section_active:
            self._handle_cross_section_pick(x, y)
            return

        self._select_clicked_visible_point(x, y)

    def _on_mouse_move(self, obj, event):
        x, y = obj.GetEventPosition()
        self._update_cross_section_mouse_preview(x, y)

    # ── event filter (right-click → background color picker) ─────

    def eventFilter(self, obj, event):
        if obj is self.vtk_widget:
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress:
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and self._cross_section_active
                    and self._fixed_z_plane_view
                ):
                    pos = event.position().toPoint()
                    self._handle_cross_section_pick(pos.x(), pos.y())
                    return True
                if event.button() == Qt.MouseButton.RightButton:
                    if self._cross_section_pick_session_active:
                        self.end_cross_section_pick_session()
                        return True
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
            self._set_cross_section_pick_mode("direction")
            self._render()
            return

        if self._cross_section_active and len(self._cross_section_points) >= 2 and self._cross_section_direction_xy is not None:
            self._cross_section_active = True
            self._cross_section_layer_id = layer.id
            self._clear_cross_section_hover_actor()
            self._cross_section_pick_session_active = False
            self._set_cross_section_pick_mode("none")
            self._update_cross_section_preview()
            self._render()
            return

        self._cross_section_active = True
        self._cross_section_layer_id = layer.id
        self._cross_section_points.clear()
        self._cross_section_reference = None
        self._cross_section_direction_xy = None
        self._cross_section_preview_point = None
        self._cross_section_plane_offset = 0.0
        self._clear_cross_section_hover_actor()
        self._set_cross_section_pick_mode("direction")
        self._cross_section_direction_confirmed = False
        self._cross_section_pick_session_active = False
        self._remove_cross_section_actors()
        self._render()

    def disable_cross_section_mode(self):
        self._cross_section_active = False
        self._cross_section_layer_id = None
        self._cross_section_points.clear()
        self._cross_section_reference = None
        self._cross_section_direction_xy = None
        self._cross_section_preview_point = None
        self._cross_section_plane_offset = 0.0
        self._clear_cross_section_hover_actor()
        self._set_cross_section_pick_mode("direction")
        self._cross_section_direction_confirmed = False
        self._cross_section_pick_session_active = False
        self._cross_section_selected_point_indices = np.empty((0,), dtype=np.int32)
        self._cross_section_selected_point_indices_by_layer = {}
        self._remove_cross_section_actors()
        self._render()

    def set_cross_section_pick_mode(self, mode):
        if mode in ("direction", "reference", "none"):
            self._set_cross_section_pick_mode(mode)
            self._render()

    def begin_cross_section_pick_session(self, mode="direction", clear_existing=False):
        if mode not in ("direction", "reference"):
            return
        if clear_existing and mode == "direction":
            self.clear_cross_section_direction_points()
            self._cross_section_direction_confirmed = False
        self._cross_section_pick_session_active = True
        self._set_cross_section_pick_mode(mode)
        self._render()

    def end_cross_section_pick_session(self):
        if not self._cross_section_pick_session_active:
            return
        self._cross_section_pick_session_active = False
        self._set_cross_section_pick_mode("none")
        self._render()

    def set_cross_section_thickness(self, value):
        self._cross_section_thickness = float(value)

    def set_cross_section_plane_offset(self, value):
        self._cross_section_plane_offset = float(value)
        self._update_cross_section_preview()

    def set_cross_section_direction_confirmed(self, confirmed):
        """Set whether the direction points are confirmed (locked)."""
        self._cross_section_direction_confirmed = confirmed
        self._update_cross_section_preview()
        self._render()

    def clear_cross_section_direction_points(self):
        self._cross_section_points.clear()
        self._cross_section_direction_xy = None
        self._cross_section_direction_confirmed = False
        self._cross_section_preview_point = None
        self._cross_section_selected_point_indices = np.empty((0,), dtype=np.int32)
        self._cross_section_selected_point_indices_by_layer = {}
        self._clear_cross_section_hover_actor()
        self._remove_cross_section_actors(point_actors=True, line_actor=True)
        if self._cross_section_layer_id is not None:
            self._rebuild(self._cross_section_layer_id)
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
            "plane_offset": self._cross_section_plane_offset,
            "mode": self._cross_section_pick_mode,
            "pick_session_active": self._cross_section_pick_session_active,
        }

    def _set_cross_section_pick_mode(self, mode):
        if self._cross_section_pick_mode == mode:
            return
        self._cross_section_pick_mode = mode
        self.cross_section_mode_changed.emit(mode)

    def _handle_cross_section_pick(self, x, y):
        if not self._cross_section_active:
            return
        point = self._cross_section_hover_point
        if point is None:
            point = self._pick_cross_section_point(x, y)
        if point is None:
            return

        if self._cross_section_pick_mode == "direction":
            if not self._cross_section_pick_session_active:
                return
            if len(self._cross_section_points) >= 2:
                self.clear_cross_section_direction_points()
            layer = self.layer_manager.get_layer(self._cross_section_layer_id)
            if layer is None or len(layer.points) == 0:
                marker_z = float(point[2])
            else:
                marker_z = float(np.max(layer.points[:, 2]))
            marker_point = np.array([point[0], point[1], marker_z], dtype=np.float64)
            self._cross_section_points.append(point)
            actor = self._make_sphere_actor(marker_point, radius=0.06, color=(1.0, 0.0, 0.0))
            self._cross_section_point_actors.append(actor)
            self.renderer.AddActor(actor)
            label_actor = self._make_screen_label_actor(marker_point, (1.0, 0.0, 0.0))
            self._cross_section_selected_label_actors.append(label_actor)
            self.renderer.AddActor2D(label_actor)
            self._cross_section_direction_confirmed = len(self._cross_section_points) == 2
            self._update_cross_section_preview()
            self.cross_section_point_selected.emit({"mode": "direction", "point": point})
            if self._cross_section_direction_confirmed:
                self.end_cross_section_pick_session()
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
        if self._cross_section_line_actor is not None:
            self.renderer.RemoveActor(self._cross_section_line_actor)
            self._cross_section_line_actor = None

        if self._cross_section_preview_line_actor is not None:
            self.renderer.RemoveActor(self._cross_section_preview_line_actor)
            self._cross_section_preview_line_actor = None

        if self._cross_section_preview_plane_actor is not None:
            self.renderer.RemoveActor(self._cross_section_preview_plane_actor)
            self._cross_section_preview_plane_actor = None

        if self._cross_section_preview_plane_trace_actor is not None:
            self.renderer.RemoveActor(self._cross_section_preview_plane_trace_actor)
            self._cross_section_preview_plane_trace_actor = None

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

        p0 = np.asarray(self._cross_section_points[0], dtype=np.float64)
        p1 = np.asarray(self._cross_section_points[1], dtype=np.float64)
        delta_xy = p1[:2] - p0[:2]
        if np.linalg.norm(delta_xy) == 0:
            return
        direction_xy = delta_xy / np.linalg.norm(delta_xy)
        self._cross_section_direction_xy = direction_xy

        layer = self.layer_manager.get_layer(self._cross_section_layer_id)
        if layer is None or len(layer.points) == 0:
            span = 1.0
            max_z = max(float(p0[2]), float(p1[2]))
        else:
            span = float(np.linalg.norm(np.ptp(layer.points[:, :2], axis=0)))
            max_z = float(np.max(layer.points[:, 2]))
        span = max(span, 1.0)
        plane_normal_xy = np.array([-direction_xy[1], direction_xy[0]], dtype=np.float64)
        plane_normal_xy = plane_normal_xy / np.linalg.norm(plane_normal_xy)

        start = np.array([p0[0], p0[1], max_z], dtype=np.float64)
        end = np.array([p1[0], p1[1], max_z], dtype=np.float64)

        self._cross_section_line_actor = self._make_line_actor(
            start,
            end,
            color=(1.0, 1.0, 0.0),
            width=4,
        )
        self.renderer.AddActor(self._cross_section_line_actor)

        plane_half_length = span * 1.5
        plane_center_xy = p0[:2] + direction_xy * self._cross_section_plane_offset
        if layer is None or len(layer.points) == 0:
            min_z = float(min(p0[2], p1[2]))
            max_z = float(max(p0[2], p1[2]))
        else:
            min_z = float(np.min(layer.points[:, 2]))
            max_z = float(np.max(layer.points[:, 2]))
        if abs(max_z - min_z) < 1e-6:
            max_z = min_z + max(span * 0.25, 1.0)

        plane_bottom_start = np.array([
            plane_center_xy[0] - plane_normal_xy[0] * plane_half_length,
            plane_center_xy[1] - plane_normal_xy[1] * plane_half_length,
            min_z,
        ], dtype=np.float64)
        plane_bottom_end = np.array([
            plane_center_xy[0] + plane_normal_xy[0] * plane_half_length,
            plane_center_xy[1] + plane_normal_xy[1] * plane_half_length,
            min_z,
        ], dtype=np.float64)
        plane_top_start = np.array([
            plane_bottom_start[0],
            plane_bottom_start[1],
            max_z,
        ], dtype=np.float64)
        plane_top_end = np.array([
            plane_bottom_end[0],
            plane_bottom_end[1],
            max_z,
        ], dtype=np.float64)

        self._cross_section_polyline_actor = self._make_plane_actor(
            plane_bottom_start,
            plane_bottom_end,
            plane_top_end,
            plane_top_start,
            color=(0.0, 0.7, 0.7),
            opacity=0.5,
        )
        self.renderer.AddActor(self._cross_section_polyline_actor)

        self._cross_section_confirmed_line_actor = self._make_line_actor(
            np.array([
                plane_bottom_start[0],
                plane_bottom_start[1],
                max_z,
            ], dtype=np.float64),
            np.array([
                plane_bottom_end[0],
                plane_bottom_end[1],
                max_z,
            ], dtype=np.float64),
            color=(0.0, 0.7, 0.7),
            width=2,
        )
        self.renderer.AddActor(self._cross_section_confirmed_line_actor)

        self._update_cross_section_reference_markers()
        self._render()

    def _update_cross_section_mouse_preview(self, x, y):
        if self._cross_section_pick_mode != "direction":
            return
        if not self._cross_section_pick_session_active:
            return
        if len(self._cross_section_points) == 0:
            point = self._pick_cross_section_point(x, y)
            self._cross_section_preview_point = point
            self._update_cross_section_hover(point)
            return
        if len(self._cross_section_points) != 1:
            self._cross_section_preview_point = None
            if self._cross_section_preview_line_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_line_actor)
                self._cross_section_preview_line_actor = None
            if self._cross_section_preview_plane_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_plane_actor)
                self._cross_section_preview_plane_actor = None
            if self._cross_section_preview_plane_trace_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_plane_trace_actor)
                self._cross_section_preview_plane_trace_actor = None
                self._render()
            return
        point = self._pick_cross_section_point(x, y)
        if point is None:
            if self._cross_section_preview_line_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_line_actor)
                self._cross_section_preview_line_actor = None
            if self._cross_section_preview_plane_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_plane_actor)
                self._cross_section_preview_plane_actor = None
            if self._cross_section_preview_plane_trace_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_plane_trace_actor)
                self._cross_section_preview_plane_trace_actor = None
            self._update_cross_section_hover(None)
            self._render()
            return
        self._cross_section_preview_point = point
        self._update_cross_section_hover(point)

        if self._cross_section_preview_line_actor is not None:
            self.renderer.RemoveActor(self._cross_section_preview_line_actor)
            self._cross_section_preview_line_actor = None

        if self._cross_section_preview_plane_actor is not None:
            self.renderer.RemoveActor(self._cross_section_preview_plane_actor)
            self._cross_section_preview_plane_actor = None

        if self._cross_section_preview_plane_trace_actor is not None:
            self.renderer.RemoveActor(self._cross_section_preview_plane_trace_actor)
            self._cross_section_preview_plane_trace_actor = None

        layer = self.layer_manager.get_layer(self._cross_section_layer_id)
        if layer is None or len(layer.points) == 0:
            preview_z = max(float(self._cross_section_points[0][2]), float(point[2]))
        else:
            preview_z = float(np.max(layer.points[:, 2]))

        preview_start = np.array([
            self._cross_section_points[0][0],
            self._cross_section_points[0][1],
            preview_z,
        ], dtype=np.float64)
        preview_end = np.array([
            point[0],
            point[1],
            preview_z,
        ], dtype=np.float64)

        self._cross_section_preview_line_actor = self._make_line_actor(
            preview_start,
            preview_end,
            color=(1.0, 1.0, 0.0),
            width=3,
        )
        self.renderer.AddActor(self._cross_section_preview_line_actor)

        p0 = np.asarray(self._cross_section_points[0], dtype=np.float64)
        p1 = np.asarray(point, dtype=np.float64)
        delta_xy = p1[:2] - p0[:2]
        delta_norm = np.linalg.norm(delta_xy)
        if delta_norm > 0:
            direction_xy = delta_xy / delta_norm
            plane_normal_xy = np.array([-direction_xy[1], direction_xy[0]], dtype=np.float64)
            plane_normal_xy = plane_normal_xy / np.linalg.norm(plane_normal_xy)

            if layer is None or len(layer.points) == 0:
                span = max(delta_norm, 1.0)
                min_z = float(min(p0[2], p1[2]))
                max_z = float(max(p0[2], p1[2], preview_z))
            else:
                span = float(np.linalg.norm(np.ptp(layer.points[:, :2], axis=0)))
                span = max(span, 1.0)
                min_z = float(np.min(layer.points[:, 2]))
                max_z = float(np.max(layer.points[:, 2]))

            if abs(max_z - min_z) < 1e-6:
                max_z = min_z + max(span * 0.25, 1.0)

            plane_half_length = span * 1.5
            plane_center_xy = p0[:2] + direction_xy * self._cross_section_plane_offset

            plane_bottom_start = np.array([
                plane_center_xy[0] - plane_normal_xy[0] * plane_half_length,
                plane_center_xy[1] - plane_normal_xy[1] * plane_half_length,
                min_z,
            ], dtype=np.float64)
            plane_bottom_end = np.array([
                plane_center_xy[0] + plane_normal_xy[0] * plane_half_length,
                plane_center_xy[1] + plane_normal_xy[1] * plane_half_length,
                min_z,
            ], dtype=np.float64)
            plane_top_start = np.array([
                plane_bottom_start[0],
                plane_bottom_start[1],
                max_z,
            ], dtype=np.float64)
            plane_top_end = np.array([
                plane_bottom_end[0],
                plane_bottom_end[1],
                max_z,
            ], dtype=np.float64)

            self._cross_section_preview_plane_actor = self._make_plane_actor(
                plane_bottom_start,
                plane_bottom_end,
                plane_top_end,
                plane_top_start,
                color=(0.0, 0.7, 0.7),
                opacity=0.25,
            )
            self.renderer.AddActor(self._cross_section_preview_plane_actor)

            self._cross_section_preview_plane_trace_actor = self._make_line_actor(
                np.array([
                    plane_bottom_start[0],
                    plane_bottom_start[1],
                    max_z,
                ], dtype=np.float64),
                np.array([
                    plane_bottom_end[0],
                    plane_bottom_end[1],
                    max_z,
                ], dtype=np.float64),
                color=(0.0, 0.9, 0.9),
                width=2,
            )
            self.renderer.AddActor(self._cross_section_preview_plane_trace_actor)
        self._render()

    def _update_cross_section_hover(self, point):
        if point is None:
            if self._clear_cross_section_hover_actor():
                self._render()
            return

        point = np.asarray(point, dtype=np.float64)
        if (
            self._cross_section_hover_point is not None and
            np.allclose(self._cross_section_hover_point, point)
        ):
            return

        self._clear_cross_section_hover_actor()
        self._cross_section_hover_actor = self._make_sphere_actor(
            point,
            radius=0.1,
            color=(1.0, 0.5, 0.0),
        )
        self._cross_section_hover_point = point
        self.renderer.AddActor(self._cross_section_hover_actor)
        self._update_hover_label(point)

    def _update_global_hover(self, x, y):
        point = self._pick_any_visible_point(x, y)
        if point is None:
            self._clear_cross_section_hover_actor()
            return
        self._update_cross_section_hover(point)

    def _select_clicked_visible_point(self, x, y):
        point = self._pick_any_visible_point(x, y)
        if point is None:
            self._clear_clicked_point_actor()
            self._render()
            return

        point = np.asarray(point, dtype=np.float64)
        if self._clicked_point is not None and np.allclose(self._clicked_point, point):
            return

        self._clear_clicked_point_actor()
        self._clicked_point_actor = self._make_sphere_actor(
            point,
            radius=0.1,
            color=(1.0, 0.5, 0.0),
        )
        self._clicked_point_label_actor = self._make_screen_label_actor(point, (1.0, 0.5, 0.0))
        self._clicked_point = point
        self.renderer.AddActor(self._clicked_point_actor)
        self.renderer.AddActor2D(self._clicked_point_label_actor)
        self._render()

    def _clear_clicked_point_actor(self):
        removed = False
        if self._clicked_point_actor is not None:
            self.renderer.RemoveActor(self._clicked_point_actor)
            self._clicked_point_actor = None
            removed = True
        if self._clicked_point_label_actor is not None:
            self.renderer.RemoveActor2D(self._clicked_point_label_actor)
            self._clicked_point_label_actor = None
            removed = True
        self._clicked_point = None
        return removed

    def _clear_cross_section_hover_actor(self):
        removed = False
        if self._cross_section_hover_actor is not None:
            self.renderer.RemoveActor(self._cross_section_hover_actor)
            self._cross_section_hover_actor = None
            removed = True
        if self._cross_section_hover_label_actor is not None:
            self.renderer.RemoveActor2D(self._cross_section_hover_label_actor)
            self._cross_section_hover_label_actor = None
            removed = True
        self._cross_section_hover_point = None
        return removed

    def _update_hover_label(self, point):
        if self._cross_section_hover_label_actor is not None:
            self.renderer.RemoveActor2D(self._cross_section_hover_label_actor)
            self._cross_section_hover_label_actor = None
        self._cross_section_hover_label_actor = self._make_screen_label_actor(point, (1.0, 0.5, 0.0))
        self.renderer.AddActor2D(self._cross_section_hover_label_actor)

    def _make_screen_label_actor(self, point, color):
        coord = self._world_to_display(point)
        label = vtk.vtkTextActor()
        label.SetInput(f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
        label.SetPosition(coord[0] + 10, coord[1] + 10)
        text_prop = label.GetTextProperty()
        text_prop.SetFontSize(14)
        text_prop.SetColor(*color)
        text_prop.SetBold(True)
        text_prop.SetBackgroundColor(1.0, 1.0, 1.0)
        text_prop.SetBackgroundOpacity(0.7)
        return label

    def _world_to_display(self, point):
        self.renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        self.renderer.WorldToDisplay()
        return self.renderer.GetDisplayPoint()

    def _pick_cross_section_point(self, x, y):
        layer = self.layer_manager.get_layer(self._cross_section_layer_id)
        if layer is None:
            return None

        display_y = y

        if self._picker.Pick(x, display_y, 0, self.renderer):
            point = np.asarray(self._picker.GetPickPosition(), dtype=np.float64)
            if np.all(np.isfinite(point)):
                return point

        self._world_picker.Pick(x, display_y, 0, self.renderer)
        world_point = np.asarray(self._world_picker.GetPickPosition(), dtype=np.float64)
        if not np.all(np.isfinite(world_point)):
            return None

        if isinstance(layer, PointCloudLayer) and len(layer.points) > 0:
            points = layer.points
        elif isinstance(layer, MeshLayer) and len(layer.vertices) > 0:
            points = layer.vertices
        else:
            return world_point

        deltas = points[:, :2] - world_point[:2]
        nearest_index = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
        snapped_point = np.asarray(points[nearest_index], dtype=np.float64)
        return snapped_point

    def _pick_any_visible_point(self, x, y):
        display_y = y

        if self._picker.Pick(x, display_y, 0, self.renderer):
            point = np.asarray(self._picker.GetPickPosition(), dtype=np.float64)
            if np.all(np.isfinite(point)):
                return point

        self._world_picker.Pick(x, display_y, 0, self.renderer)
        world_point = np.asarray(self._world_picker.GetPickPosition(), dtype=np.float64)
        if not np.all(np.isfinite(world_point)):
            return None

        best_point = None
        best_dist2 = None
        for layer in self.layer_manager.get_all_layers():
            if not getattr(layer, "visible", True):
                continue
            if isinstance(layer, PointCloudLayer) and len(layer.points) > 0:
                points = layer.points
            elif isinstance(layer, MeshLayer) and len(layer.vertices) > 0:
                points = layer.vertices
            else:
                continue

            deltas = points[:, :2] - world_point[:2]
            dist2 = np.einsum("ij,ij->i", deltas, deltas)
            nearest_index = int(np.argmin(dist2))
            nearest_dist2 = float(dist2[nearest_index])
            if best_dist2 is None or nearest_dist2 < best_dist2:
                best_dist2 = nearest_dist2
                best_point = np.asarray(points[nearest_index], dtype=np.float64)

        return best_point

    def _update_cross_section_reference_markers(self):
        if self._cross_section_reference is None:
            return
        if self._cross_section_direction_xy is None:
            return
        if len(self._cross_section_points) < 2:
            return

        # Compute closest point on the cross-section plane trace (in XY) for the selected reference
        origin_xy = np.asarray(self._cross_section_points[0][:2], dtype=np.float64)
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
            for actor in self._cross_section_selected_label_actors:
                self.renderer.RemoveActor2D(actor)
            self._cross_section_selected_label_actors.clear()

        if line_actor:
            if self._cross_section_line_actor is not None:
                self.renderer.RemoveActor(self._cross_section_line_actor)
                self._cross_section_line_actor = None
            if self._cross_section_preview_line_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_line_actor)
                self._cross_section_preview_line_actor = None
            if self._cross_section_preview_plane_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_plane_actor)
                self._cross_section_preview_plane_actor = None
            if self._cross_section_preview_plane_trace_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_plane_trace_actor)
                self._cross_section_preview_plane_trace_actor = None
            if self._cross_section_preview_plane_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_plane_actor)
                self._cross_section_preview_plane_actor = None
            if self._cross_section_preview_plane_trace_actor is not None:
                self.renderer.RemoveActor(self._cross_section_preview_plane_trace_actor)
                self._cross_section_preview_plane_trace_actor = None
            if self._cross_section_confirmed_line_actor is not None:
                self.renderer.RemoveActor(self._cross_section_confirmed_line_actor)
                self._cross_section_confirmed_line_actor = None
            if self._cross_section_polyline_actor is not None:
                self.renderer.RemoveActor(self._cross_section_polyline_actor)
                self._cross_section_polyline_actor = None
            self._clear_cross_section_hover_actor()

        self._clear_clicked_point_actor()

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

    def _make_plane_actor(self, p0, p1, p2, p3, color=(0.0, 0.7, 0.7), opacity=0.5):
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(*p0.tolist())
        pts.InsertNextPoint(*p1.tolist())
        pts.InsertNextPoint(*p2.tolist())
        pts.InsertNextPoint(*p3.tolist())

        quad = vtk.vtkQuad()
        quad.GetPointIds().SetId(0, 0)
        quad.GetPointIds().SetId(1, 1)
        quad.GetPointIds().SetId(2, 2)
        quad.GetPointIds().SetId(3, 3)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(quad)

        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetInterpolationToFlat()
        return actor

    # ── public ───────────────────────────────────────────────────

    def fit_all(self):
        self.renderer.ResetCamera()
        self._enforce_z_plane_camera()
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
        if self._fixed_z_plane_view:
            self._enforce_z_plane_camera()
        self.vtk_widget.GetRenderWindow().Render()

    def _enforce_z_plane_camera(self):
        if not self._fixed_z_plane_view:
            return
        camera = self.renderer.GetActiveCamera()
        if camera is None:
            return

        camera.SetParallelProjection(True)

        focal = np.array(camera.GetFocalPoint(), dtype=np.float64)
        position = np.array(camera.GetPosition(), dtype=np.float64)
        distance = float(abs(position[2] - focal[2]))
        if distance < 1e-6:
            distance = float(camera.GetDistance())
        if distance < 1e-6:
            distance = 1.0

        camera.SetPosition(focal[0], focal[1], focal[2] + distance)
        camera.SetFocalPoint(focal[0], focal[1], focal[2])
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.OrthogonalizeViewUp()

    def _toggle_view_mode(self):
        self._fixed_z_plane_view = not self._fixed_z_plane_view
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        if interactor is not None:
            interactor.SetInteractorStyle(
                self._z_plane_style if self._fixed_z_plane_view else self._free_view_style
            )

        camera = self.renderer.GetActiveCamera()
        if camera is not None:
            if self._fixed_z_plane_view:
                camera.SetParallelProjection(True)
                self._view_mode_button.setText("Fixed Z Plane View")
                self._enforce_z_plane_camera()
            else:
                camera.SetParallelProjection(False)
                camera.OrthogonalizeViewUp()
                self._view_mode_button.setText("Orbit / Free Orbit View")
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

    def get_cross_section_preview_colors(self, layer):
        if not isinstance(layer, PointCloudLayer):
            return None
        if layer.points is None or len(layer.points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        colors = self._resolve_pc_colors(layer)
        if not layer.mask_groups:
            return colors.astype(np.float32)

        visible = np.zeros(layer.point_count, dtype=bool)
        any_mask = False
        for mg in layer.mask_groups:
            if mg.mask is None:
                continue
            any_mask = True
            if mg.positive_visible:
                visible |= mg.mask
                colors = self._apply_mask_color_pc(colors, mg.mask, mg, True)
            if mg.negative_visible:
                neg_mask = ~mg.mask
                visible |= neg_mask
                colors = self._apply_mask_color_pc(colors, neg_mask, mg, False)

        if any_mask:
            colors = colors.copy()
            colors[~visible] = 0.0

        return colors.astype(np.float32)

    def set_cross_section_preview_selection(self, selection_uv_rect):
        layer_id = self._cross_section_layer_id
        layer = self.layer_manager.get_layer(layer_id) if layer_id is not None else None
        if self._cross_section_direction_xy is None or len(self._cross_section_points) < 2:
            self._cross_section_selected_point_indices = np.empty((0,), dtype=np.int32)
            self._cross_section_selected_point_indices_by_layer = {}
            self.rebuild_all()
            return

        if not selection_uv_rect:
            self._cross_section_selected_point_indices = np.empty((0,), dtype=np.int32)
            self._cross_section_selected_point_indices_by_layer = {}
            self.rebuild_all()
            return

        left_up = selection_uv_rect["left_up"]
        right_bottom = selection_uv_rect["right_bottom"]
        u_min = min(float(left_up["u"]), float(right_bottom["u"]))
        u_max = max(float(left_up["u"]), float(right_bottom["u"]))
        v_min = min(float(left_up["v"]), float(right_bottom["v"]))
        v_max = max(float(left_up["v"]), float(right_bottom["v"]))

        plane_origin = np.asarray(self._cross_section_points[0], dtype=np.float64).copy()
        plane_normal_xy = np.asarray(self._cross_section_direction_xy, dtype=np.float64)
        plane_origin[:2] += plane_normal_xy * float(self._cross_section_plane_offset)

        norm = np.linalg.norm(plane_normal_xy)
        if norm == 0:
            self._cross_section_selected_point_indices = np.empty((0,), dtype=np.int32)
            self._cross_section_selected_point_indices_by_layer = {}
            self.rebuild_all()
            return
        plane_normal_xy = plane_normal_xy / norm

        tangent_xy = np.array([-plane_normal_xy[1], plane_normal_xy[0]], dtype=np.float64)
        selected_by_layer: dict[str, np.ndarray] = {}
        active_selection = np.empty((0,), dtype=np.int32)

        for candidate_layer in self.layer_manager.point_clouds.values():
            if candidate_layer.points is None or len(candidate_layer.points) == 0:
                continue

            deltas_xy = candidate_layer.points[:, :2].astype(np.float64) - plane_origin[:2]
            signed_distances = deltas_xy @ plane_normal_xy
            thickness_mask = np.abs(signed_distances) <= float(self._cross_section_thickness)
            along_plane = -(deltas_xy @ tangent_xy)
            heights = candidate_layer.points[:, 2].astype(np.float64) - float(plane_origin[2])

            selection_mask = (
                thickness_mask &
                (along_plane >= u_min) & (along_plane <= u_max) &
                (heights >= v_min) & (heights <= v_max)
            )
            indices = np.flatnonzero(selection_mask).astype(np.int32)
            if len(indices) > 0:
                selected_by_layer[candidate_layer.id] = indices
            if layer_id is not None and candidate_layer.id == layer_id:
                active_selection = indices

        self._cross_section_selected_point_indices_by_layer = selected_by_layer
        self._cross_section_selected_point_indices = active_selection
        self.rebuild_all()

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
            selected_indices = self._cross_section_selected_point_indices_by_layer.get(layer.id, np.empty((0,), dtype=np.int32))
            if len(selected_indices) > 0:
                valid_indices = selected_indices[
                    (selected_indices >= 0) & (selected_indices < n)
                ]
                if len(valid_indices) > 0:
                    colors = colors.copy()
                    colors[valid_indices] = self._cross_section_selection_color
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

        selected_indices = self._cross_section_selected_point_indices_by_layer.get(layer.id, np.empty((0,), dtype=np.int32))
        if len(selected_indices) > 0:
            valid_indices = selected_indices[
                (selected_indices >= 0) & (selected_indices < n)
            ]
            if len(valid_indices) > 0:
                colors = colors.copy()
                colors[valid_indices] = self._cross_section_selection_color

        return [self._make_pc_actor(
            layer.points[visible], colors[visible], ps)]
    
    def _apply_mask_color_pc(self, colors, mask_idx, mask_group, is_positive):
        """Apply mask colors to a point cloud color array based on color_mode."""
        color_mode = (mask_group.positive_color_mode if is_positive 
                      else mask_group.negative_color_mode)
        solid_color = (mask_group.positive_solid_color if is_positive 
                       else mask_group.negative_solid_color)
        gradient_colors = (mask_group.positive_gradient_colors if is_positive
                           else mask_group.negative_gradient_colors)
        
        if color_mode == "original":
            return colors
        elif color_mode == "solid":
            # Apply solid color to masked indices
            colors[mask_idx] = np.array(solid_color, dtype=np.float64)
        elif color_mode == "gradient" and gradient_colors is not None:
            colors[mask_idx] = np.asarray(gradient_colors[mask_idx], dtype=np.float64)
        
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
        self._enforce_z_plane_camera()
        self._render()