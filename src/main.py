import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

try:
    from ui.main_window import MainWindow
except ImportError as e:
    print("ERROR: Failed to import application modules.")
    print("This often means a required package is missing or blocked.")
    print("ImportError:", e)
    print("\nIf you see a VTK DLL load failure, install vtk or unblock it in Windows application control policies.")
    sys.exit(1)


def main():
    print("Starting Point Cloud Processor...")
    app = QApplication(sys.argv)
    app.setApplicationName("Point Cloud Processor")
    app.setStyle("Fusion")

    window = MainWindow()

    def _shutdown_main_window():
        try:
            if window is not None:
                window.close()
        except Exception:
            pass

    app.aboutToQuit.connect(_shutdown_main_window)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()