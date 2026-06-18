import numpy as np
import importlib
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
)
from PySide6.QtCore import Signal

try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_AVAILABLE = True
except Exception as _e:
    VTK_AVAILABLE = False
    _vtk_import_error = _e

from core.layer import PointCloudLayer, MaskGroup


if VTK_AVAILABLE:
    # Keep the richer preview dialog when VTK is available
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from PySide6.QtWidgets import QDialog

    class CrossSectionPreviewDialog(QDialog):
        def __init__(self, points, line_origin_xy, direction_xy, reference_point, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Cross Section Preview")
            self.resize(700, 520)

            self._points = points.astype(np.float32)
            self._line_origin_xy = np.asarray(line_origin_xy, dtype=np.float32)
            self._direction_xy = np.asarray(direction_xy, dtype=np.float32)
            self._reference_point = np.asarray(reference_point, dtype=np.float32)

            layout = QVBoxLayout(self)
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            layout.addWidget(self.vtk_widget)

            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.1, 0.12)
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

            self._build_scene()
            self.vtk_widget.Initialize()
            self.vtk_widget.Start()

        def _build_scene(self):
            if len(self._points) > 0:
                actor = self._make_points_actor(self._points, color=(0.9, 0.4, 0.1), point_size=4)
                self.renderer.AddActor(actor)

            ref_actor = self._make_sphere_actor(self._reference_point, radius=0.03, color=(0.2, 0.8, 1.0))
            self.renderer.AddActor(ref_actor)

            camera = vtk.vtkCamera()
            camera.SetPosition(0.0, -1.0, 0.0)
            camera.SetFocalPoint(0.0, 0.0, 0.0)
            camera.SetViewUp(0.0, 0.0, 1.0)
            self.renderer.SetActiveCamera(camera)
            self.renderer.ResetCamera()

        def _make_points_actor(self, points, color=(1.0, 1.0, 1.0), point_size=2):
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
            actor.GetProperty().SetColor(*color)
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
        self._status_label = QLabel("Select a point cloud and then define the cross section.")
        layout.addWidget(self._status_label)
        layout.addStretch()

    def _show_vtk_error(self):
        QMessageBox.critical(self, "VTK Import Error",
                             f"VTK could not be imported:\n{_vtk_import_error}")

    # Keep compatibility methods used by MainWindow; the heavy interactivity is
    # implemented in the Viewport, so these methods are intentionally lightweight.
    def open_panel(self):
        return
    
    def set_current_layer(self, layer):
        self._current_layer = layer

        if isinstance(layer, PointCloudLayer):
            self._status_label.setText(f"Selected: {layer.name}")

            # ✅ THIS IS THE MOST IMPORTANT LINE
            self.viewport.enable_cross_section_mode(layer)
        else:
            self._status_label.setText("Select a point cloud")
            self.viewport.disable_cross_section_mode()

