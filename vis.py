import sys
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QFileDialog, QToolBar, QComboBox, QLabel
)
import pyqtgraph.opengl as gl

from ui.visual.Octa import Octa
from ui.visual.Bond import Bond as VisBond
from ui.visual.ColorDict import ColorDict
from algorithm.lattice.Lattice import Lattice
from ui.Visualizer import Visualizer
from algorithm.export.Excellent import Excellent


class RunVisualizer:
    def __init__(self):

        self.app = QApplication(sys.argv)

        self.main_window = QMainWindow()
        self.main_window.setWindowTitle("Visualizer")
        self.main_window.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.main_window.setCentralWidget(self.central_widget)

        # initialize the visualizer widget adding it to the layout
        self.vis = Visualizer()
        self.central_layout.addWidget(self.vis)
        self.init_toolbar()

        # data structures
        self.lat_voxels, self.unit_voxels, self.meso_voxels = [], [], []

        self.main_window.show()
        # self.app.exec()

    def init_toolbar(self):
        self.toolbar = QToolBar()
        self.toolbar.setOrientation(Qt.Orientation.Horizontal)
        self.toolbar.addAction("Import", self.import_file)
        # --- dropdown (combo box) ---
        self.toolbar.addWidget(QLabel("View: "))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Unit cell", "Design", "Mesovoxel"])
        self.view_combo.setToolTip("Choose what to visualize")
        self.view_combo.currentIndexChanged.connect(self.on_view_changed)
        self.toolbar.addWidget(self.view_combo)

        self.toolbar.addSeparator()
        self.toolbar.addAction("Clear", self.vis.cleanup)
        self.toolbar.addAction("Exit", self.close)
        self.main_window.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

    def import_file(self):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(
            self.main_window,
            "Open Lattice File",
            "",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        if filepath: 
            self.lat_voxels, self.unit_voxels, self.meso_voxels = Excellent.read_output(filepath)
            self.vis.plot_voxels(self.lat_voxels + self.unit_voxels)

    # ---------- plotting logic ----------
    def on_view_changed(self, idx: int):
        """Re-plot based on dropdown selection."""
        if not self.lat_voxels and not self.unit_voxels and not self.meso_voxels:
            return  # nothing loaded yet

        if idx == 0: # Unit cell
            voxels = self.lat_voxels + self.unit_voxels
        elif idx == 1: # Design
            voxels = self.lat_voxels
        elif idx == 2: # Mesovoxel
            voxels = self.meso_voxels
        else:
            return
        self.vis.cleanup()
        self.vis.plot_voxels(voxels)

    def close(self):
        self.vis.cleanup()
        self.central_layout.removeWidget(self.vis)
        self.vis.setParent(None)
        self.vis.deleteLater()
        self.main_window.close()
        # self.app.quit()

if __name__=="__main__":
    # create the pyqt application instance and run it
    visualizer_app = RunVisualizer()
    sys.exit(visualizer_app.app.exec())