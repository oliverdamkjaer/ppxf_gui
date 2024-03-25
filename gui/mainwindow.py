from PySide6.QtWidgets import QMainWindow, QTabWidget, QLabel
from PySide6.QtGui import QAction

from ppxftab import PpxfTab
from viewertab import ViewerTab
from utils import Separator

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("pPXF GUI")

        self.tab_widget = QTabWidget(self)

        self.tab1 = PpxfTab()
        self.tab2 = ViewerTab()

        self.tab_widget.addTab(self.tab1, "Run pPXF")
        self.tab_widget.addTab(self.tab2, "Viewer")

        self.setCentralWidget(self.tab_widget)
